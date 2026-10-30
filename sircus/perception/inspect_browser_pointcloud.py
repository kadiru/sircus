#!/usr/bin/env python3
"""Offline inspection of a separately downloaded PCD; never contacts hardware.

Usage: python inspect_browser_pointcloud.py BROWSER_DATASET --compare EARLIER_RGBD
The optional comparison is cross-time evidence, not synchronized calibration.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from inspect_capture import pcd


def inspect(folder, earlier=None):
    cloud = folder / 'pointcloud.pcd.gz'
    raw = gzip.decompress(cloud.read_bytes())
    header, xyz = pcd(cloud)
    valid = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0)
    good = xyz[valid].astype(float)
    report = {
        'scope': 'offline browser point-cloud followup, independent of earlier RGB-D capture',
        'pointcloud_retrieval': 'successful browser download; earlier Python SDK retrieval failure remains separate evidence',
        'capture_timestamp_utc': None,
        'download_click_utc': '2026-10-30T00:25:49.121Z',
        'timing_note': 'Click time is not exposure time; no paired RGB-D or same-time arm joint bracket.',
        'header': header,
        'raw_bytes': len(raw),
        'raw_sha256': hashlib.sha256(raw).hexdigest(),
        'compressed_sha256': hashlib.sha256(cloud.read_bytes()).hexdigest(),
        'points': len(xyz),
        'positive_finite_points': int(valid.sum()),
        'zero_xyz_points': int(np.all(xyz == 0, axis=1).sum()),
        'nonfinite_points': int((~np.isfinite(xyz).all(axis=1)).sum()),
        'positive_finite_fraction': float(valid.mean()),
        'xyz_unit': 'meters, supported by exact module 0.22.6 unchanged RealSense vertex encoding',
        'coordinate_frame': 'module camera optical/color frame per source; not world coordinates',
        'xyz_m_percentiles_0_5_50_95_100': np.percentile(good, [0, 5, 50, 95, 100], axis=0).tolist(),
        'metric_accuracy_verified_with_known_reference': False,
        'camera_to_robot_extrinsics_calibrated': False,
        'world_frame_poses': None,
        'ball_center_camera_meters': None,
        'ball_center_status': 'unknown: no accepted robust sphere fit; primary RGB-D showed severe ball depth dropout. A later PCD does not make sparse foreground returns a validated sphere center.',
        'board_impact_surface_camera': None,
        'bucket_opening_center_camera': None,
        'bucket_opening_normal_camera': None,
        'target_localization_note': 'No supported board or bucket localization established from this followup.',
    }
    if earlier is not None:
        intr = json.loads((earlier / 'camera_properties.json').read_text())['intrinsic_parameters']
        depth = np.load(earlier / 'frame_00_depth_raw.npz')['depth_raw_counts']
        u = np.rint(good[:, 0] / good[:, 2] * intr['focal_x_px'] + intr['center_x_px']).astype(int)
        v = np.rint(good[:, 1] / good[:, 2] * intr['focal_y_px'] + intr['center_y_px']).astype(int)
        inside = (u >= 0) & (u < depth.shape[1]) & (v >= 0) & (v < depth.shape[0])
        sample, u, v = good[inside], u[inside], v[inside]
        counts = depth[v, u]
        usable = counts > 0
        ratio = sample[usable, 2] / counts[usable]
        report['cross_time_scale_consistency'] = {
            'reference_dataset': earlier.name,
            'reference_depth': 'frame_00_depth_raw.npz',
            'paired_observations': False,
            'projected_in_image': int(inside.sum()),
            'nonzero_depth_comparisons': int(usable.sum()),
            'meters_per_count_percentiles_1_5_50_95_99': np.percentile(ratio, [1, 5, 50, 95, 99]).tolist(),
            'fraction_within_1_percent_of_0_001_m_per_count': float(np.mean(np.abs(ratio / .001 - 1) <= .01)),
            'fraction_within_5_percent_of_0_001_m_per_count': float(np.mean(np.abs(ratio / .001 - 1) <= .05)),
            'interpretation': 'Empirical cross-time data strongly supports approximately 0.001 m/count for this observation; the device scale was not directly queried and metric accuracy was not checked against a known distance.',
            'limitations': [
                'Different acquisitions separated by minutes; camera and scene constancy not established by a joint bracket.',
                'Projection uses earlier reported color intrinsics, rounded pixels, and zero reported distortion.',
                'Table-dominated consistency does not verify ball depth, full RGB/depth registration, or camera-to-robot calibration.',
                'Long-range isolated returns may be stereo outliers; raw points are preserved without filtering.'
            ],
        }
    (folder / 'inspection.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--compare', type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect(args.dataset, args.compare), indent=2, allow_nan=False))
