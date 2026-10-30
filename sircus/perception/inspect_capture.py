#!/usr/bin/env python3
"""Offline depth/PCD inspection; no robot connection and no world-frame claims.

Usage: ../simulation/.venv/bin/python inspect_capture.py data/<timestamp>
Module0.22.6 writes RealSense vertices directly to PCD. We interpret their XYZ
as meters using the RealSense source contract, and report rather than assume
the observed depth-count scale. Cross-RPC comparison is approximate.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import struct
import zlib

import numpy as np


def preview(path, depth):
    """8-bit display only; raw counts remain in NPZ/DEP. Zero stays black."""
    good = depth[depth > 0]
    if not len(good):
        return
    lo, hi = np.percentile(good, [2,98])
    gray = np.clip((depth.astype(float)-lo)/max(hi-lo,1)*254+1,1,255).astype('u1')
    gray[depth == 0] = 0
    h,w = gray.shape
    def chunk(kind, data):
        return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
    raw = b''.join(b'\0'+row.tobytes() for row in gray)
    path.write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',w,h,8,0,0,0,0))+
                     chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b''))


def pcd(path):
    raw = gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
    end = raw.find(b'DATA ')
    stop = raw.find(b'\n', end) + 1
    if end < 0 or stop < 1:
        raise ValueError('Missing PCD DATA header')
    header = {}
    for line in raw[:stop].decode('ascii').splitlines():
        if line and not line.startswith('#'):
            k, *v = line.split()
            header[k] = v
    if header['DATA'] != ['binary']:
        raise ValueError('Only uncompressed binary PCD is implemented; original payload is preserved')
    fields = header['FIELDS']
    sizes = [int(x) for x in header['SIZE']]
    kinds = header['TYPE']
    counts = [int(x) for x in header.get('COUNT', ['1'] * len(fields))]
    codes = {'F': 'f', 'U': 'u', 'I': 'i'}
    dtype = np.dtype([(name, '<' + codes[kind] + str(size), (count,)) if count != 1
                      else (name, '<' + codes[kind] + str(size))
                      for name, size, kind, count in zip(fields, sizes, kinds, counts)])
    points = np.frombuffer(raw, dtype=dtype, offset=stop)
    if len(points) != int(header['POINTS'][0]):
        raise ValueError('PCD point count does not match payload')
    xyz = np.column_stack([points[k] for k in ('x','y','z')])
    return header, xyz


def main(path):
    meta = json.loads((path/'metadata.json').read_text())
    report = {'dataset': str(path), 'scope': 'offline camera-frame consistency, not metric/extrinsic calibration',
              'world_frame_poses_calibrated': False, 'frames': [],
              'units': {'raw_depth': 'unscaled device counts',
                        'pcd_xyz': 'meters per exact module0.22.6 RealSense vertex encoding; PCD header alone is unitless'},
              'joint_delta_degrees': meta.get('joint_delta_degrees'),
              'synchronization_limits': meta.get('synchronization_limits')}
    arrays = []
    for file in sorted(path.glob('frame_*_depth_raw.npz')):
        depth = np.load(file)['depth_raw_counts']
        arrays.append(depth)
        preview(path/(file.stem+'_preview.png'),depth)
        good = depth[depth > 0]
        report['frames'].append({'file': file.name, 'shape': list(depth.shape),
            'valid_fraction': float(np.mean(depth > 0)),
            'raw_count_percentiles_5_50_95': np.percentile(good, [5,50,95]).tolist() if good.size else None})
    if len(arrays) >= 2 and arrays[0].shape == arrays[-1].shape:
        valid = (arrays[0] > 0) & (arrays[-1] > 0)
        delta = np.abs(arrays[-1].astype(float) - arrays[0])
        report['temporal_depth_difference_counts'] = {'common_valid_pixels': int(valid.sum()),
            'median_abs': float(np.median(delta[valid])) if valid.any() else None,
            'p95_abs': float(np.percentile(delta[valid],95)) if valid.any() else None}
    cloud_path = path/'pointcloud.pcd.gz'
    if cloud_path.exists():
        header, xyz = pcd(cloud_path)
        valid = np.isfinite(xyz).all(axis=1) & (xyz[:,2] > 0)
        good = xyz[valid]
        report['pointcloud'] = {'header': header, 'points': len(xyz), 'positive_finite_points': int(valid.sum()),
            'valid_fraction': float(valid.mean()),
            'xyz_m_p05_p50_p95': np.percentile(good,[5,50,95],axis=0).tolist() if len(good) else None,
            'binary_byte_order': 'little-endian, consistent with observed physical Linux/AMD64 host'}
        prop_path = path/'camera_properties.json'
        if prop_path.exists() and arrays and len(good):
            prop = json.loads(prop_path.read_text())
            intr = prop.get('intrinsic_parameters', {})
            depth = arrays[-1]
            needed = ['focal_x_px','focal_y_px','center_x_px','center_y_px']
            if all(k in intr for k in needed):
                sample = good[::8].astype(float)
                u = np.rint(sample[:,0]/sample[:,2]*intr['focal_x_px'] + intr['center_x_px']).astype(int)
                v = np.rint(sample[:,1]/sample[:,2]*intr['focal_y_px'] + intr['center_y_px']).astype(int)
                keep = (u>=0)&(u<depth.shape[1])&(v>=0)&(v<depth.shape[0])
                u,v,sample = u[keep],v[keep],sample[keep]
                counts = depth[v,u].astype(float)
                keep = counts > 0
                ratio = sample[keep,2] / counts[keep]
                report['depth_scale_crosscheck'] = {'matched_projected_samples': len(ratio),
                    'meters_per_count_p05_p50_p95': np.percentile(ratio,[5,50,95]).tolist() if len(ratio) else None,
                    'interpretation': 'Empirical scale consistency only. Separate RPC frames, rounded pixels and uncorrected distortion; no known-distance reference.',
                    'does_not_establish': 'metric accuracy, physical mounting, camera-to-robot extrinsics or exact pixel alignment'}
    report['alignment'] = {'configured_align_color_depth': 'Record separately from Viam configuration',
                           'empirically_calibrated': False,
                           'note': 'Matching dimensions and point-cloud projection consistency alone do not validate RGB/depth registration.'}
    (path/'inspection.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',type=Path)
    main(parser.parse_args().dataset)
