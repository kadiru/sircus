#!/usr/bin/env python3
"""Bounded read-only RGB-D capture. No motion, tool, tuning, or config methods.

Run with the existing SDK environment, e.g.:
  ../simulation/.venv/bin/python capture_stationary.py --credentials .private/sfhack10.json
Credentials JSON: {"address": "...", "api_key_id": "...", "api_key": "..."}; mode0600.
Raw depth is stored as COUNTS: device scale is not exposed by GetProperties.
The separate inspect_capture.py compares these counts with source-defined PCD units.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import gzip
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import stat
import time

os.environ.setdefault('RUST_LOG', 'off')

import numpy as np
from google.protobuf.json_format import MessageToDict
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.robot.client import RobotClient
from viam.rpc.dial import Credentials, DialOptions

ROOT = Path(__file__).resolve().parent


def utc():
    return datetime.now(timezone.utc).isoformat()


def proto(value):
    return MessageToDict(value, preserving_proto_field_name=True)


def safe_error(exc, secrets):
    message = str(exc)
    for secret in secrets:
        if secret:
            message = message.replace(secret, '[REDACTED]')
    return {'type': type(exc).__name__, 'message': message[:800]}


async def timed(call, timeout, secrets):
    start, mono = utc(), time.monotonic()
    try:
        value = await asyncio.wait_for(call(), timeout=timeout)
        return value, {'ok': True, 'request_started_utc': start, 'response_received_utc': utc(),
                       'elapsed_s': time.monotonic() - mono}
    except Exception as exc:
        return None, {'ok': False, 'request_started_utc': start, 'response_received_utc': utc(),
                      'elapsed_s': time.monotonic() - mono, 'error': safe_error(exc, secrets)}


async def joint_snapshot(robot, timeout, secrets):
    async def one(name):
        arm = Arm.from_robot(robot, name)
        positions, timing = await timed(lambda: arm.get_joint_positions(timeout=timeout), timeout + 1, secrets)
        moving, moving_timing = await timed(arm.is_moving, timeout + 1, secrets)
        return name, {'joint_degrees': list(positions.values) if positions is not None else None,
                      'joint_timing': timing, 'is_moving': moving, 'moving_timing': moving_timing,
                      'provenance': 'physical arm RPC readback, not a command'}
    return dict(await asyncio.gather(one('arm'), one('arm-board')))


def decode_depth(raw, byte_order):
    if len(raw) < 24:
        raise ValueError('Depth payload is shorter than Viam DEP header')
    width, height = int.from_bytes(raw[8:16], 'big'), int.from_bytes(raw[16:24], 'big')
    if not (0 < width <= 8192 and 0 < height <= 8192) or len(raw) != 24 + width * height * 2:
        raise ValueError('Depth dimensions or payload length do not match Viam DEP format')
    dtype = '>u2' if byte_order == 'big' else '<u2'
    return np.frombuffer(raw, dtype=dtype, offset=24).reshape(height, width).astype(np.uint16)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


async def capture(args):
    info = args.credentials.stat()
    if args.credentials.is_symlink() or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
        raise ValueError('Credentials file must be owned by current user, nonsymlink, mode0600')
    credential = json.loads(args.credentials.read_text())
    address = credential['address']
    # Explicit physical-machine scope. A different target requires an intentional script edit.
    if address != 'sfhack10-main.ty1cywh7tq.viam.cloud':
        raise ValueError('This script is scoped only to the observed sfhack10 main address')
    secrets = [credential['api_key'], credential['api_key_id']]
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = args.output or ROOT / 'data' / stamp
    out.mkdir(parents=True, exist_ok=False)
    metadata = {
        'schema': 'sircus-stationary-rgbd-v1', 'capture_started_utc': utc(),
        'machine': 'sfhack10', 'camera': args.camera, 'arms': ['arm', 'arm-board'],
        'transport': 'direct TLS' if args.direct else 'SDK default WebRTC/TLS',
        'mutating_robot_methods_called': [], 'physical_motion_commanded': False,
        'depth_raw_unit': 'device counts; scale is not asserted by this capture script',
        'depth_byte_order': args.depth_byte_order,
        'pointcloud_raw_unit': 'module0.22.6 encodes RealSense vertices directly; inspect source/scale before use',
        'synchronization_limits': [
            'RGB/depth in one GetImages response share a frameset according to the driver; individual imager timestamps are not returned.',
            'Only response captured_at and local request/response times are available; host/camera clock synchronization is not verified.',
            'Point cloud uses a separate RPC/latest frameset; it is not guaranteed to match any saved RGB/depth pair.',
            'Arm readbacks bracket observations and are not hardware synchronized with camera exposure.',
            'Joint constancy cannot establish camera mounting or calibrated camera-to-robot transforms.'
        ], 'frames': []
    }
    robot = None
    try:
        options = RobotClient.Options(dial_options=DialOptions(disable_webrtc=args.direct,
            auth_entity=credential['api_key_id'], credentials=Credentials(type='api-key', payload=credential['api_key'])))
        robot, connection_timing = await timed(lambda: RobotClient.at_address(address, options), args.timeout + 15, secrets)
        metadata['connection'] = connection_timing
        if robot is None:
            write_json(out / 'metadata.json', metadata)
            print(json.dumps({'dataset': str(out), 'connected': False, 'error_type': connection_timing['error']['type']}))
            return 1
        metadata['resources'] = [proto(x) for x in robot.resource_names]
        required = {('camera', args.camera), ('arm', 'arm'), ('arm', 'arm-board')}
        present = {(x.subtype, x.name) for x in robot.resource_names}
        if not required.issubset(present):
            raise ValueError('Expected physical camera and both named arm resources are missing')
        camera = Camera.from_robot(robot, args.camera)
        status, status_timing = await timed(lambda: camera.get_status(timeout=args.timeout), args.timeout + 1, secrets)
        metadata['camera_status_timing'] = status_timing
        if status is not None:
            write_json(out / 'camera_status.json', status)
        properties, prop_timing = await timed(lambda: camera.get_properties(timeout=args.timeout), args.timeout + 1, secrets)
        metadata['properties_timing'] = prop_timing
        if properties is not None:
            write_json(out / 'camera_properties.json', proto(properties))
        frames, frame_timing = await timed(robot.get_frame_system_config, args.timeout + 1, secrets)
        metadata['frame_system_timing'] = frame_timing
        if frames is not None:
            write_json(out / 'configured_frames.json', [proto(x) for x in frames])
        metadata['joints_before'] = await joint_snapshot(robot, args.timeout, secrets)
        if any(x['is_moving'] is True for x in metadata['joints_before'].values()):
            metadata['capture_aborted_reason'] = 'An arm reported moving; no stop or motion command was sent.'
            write_json(out / 'metadata.json', metadata)
            print(json.dumps({'dataset': str(out), 'capture_aborted_reason': metadata['capture_aborted_reason']}))
            return 1
        for index in range(args.frames):
            pair, timing = await timed(lambda: camera.get_images(timeout=args.timeout), args.timeout + 1, secrets)
            record = {'index': index, 'rpc': timing, 'images': []}
            if pair is not None:
                images, response_metadata = pair
                record['response_metadata'] = proto(response_metadata)
                for image in images:
                    name = re.sub(r'[^a-zA-Z0-9_-]', '_', image.name) or 'unnamed'
                    mime = getattr(image.mime_type, 'value', str(image.mime_type))
                    suffix = {'image/jpeg': '.jpg', 'image/png': '.png', 'image/vnd.viam.dep': '.dep.gz'}.get(mime, '.bin')
                    filename = f'frame_{index:02d}_{name}{suffix}'
                    if suffix.endswith('.gz'):
                        (out / filename).write_bytes(gzip.compress(image.data, mtime=0))
                    else:
                        (out / filename).write_bytes(image.data)
                    item = {'source_name': image.name, 'mime_type': mime, 'file': filename,
                            'width': image.width, 'height': image.height, 'raw_bytes': len(image.data),
                            'raw_sha256': hashlib.sha256(image.data).hexdigest()}
                    if mime == 'image/vnd.viam.dep':
                        try:
                            depth = decode_depth(image.data, args.depth_byte_order)
                            np.savez_compressed(out / f'frame_{index:02d}_{name}_raw.npz', depth_raw_counts=depth)
                            valid = depth[depth > 0]
                            item['depth_raw_counts'] = {'valid_fraction': float(np.mean(depth > 0)),
                                'min': int(valid.min()) if valid.size else None,
                                'median': float(np.median(valid)) if valid.size else None,
                                'max': int(valid.max()) if valid.size else None}
                        except Exception as exc:
                            item['decode_error'] = safe_error(exc, secrets)
                    record['images'].append(item)
            metadata['frames'].append(record)
            write_json(out / 'metadata.json', metadata)
            if index + 1 < args.frames:
                await asyncio.sleep(0.2)
        metadata['joints_after_images'] = await joint_snapshot(robot, args.timeout, secrets)
        metadata['image_joint_delta_degrees'] = {}
        for name in ['arm', 'arm-board']:
            a, b = metadata['joints_before'][name]['joint_degrees'], metadata['joints_after_images'][name]['joint_degrees']
            metadata['image_joint_delta_degrees'][name] = [y-x for x,y in zip(a,b)] if a is not None and b is not None else None
        write_json(out / 'metadata.json', metadata)
        if args.skip_pointcloud:
            metadata['pointcloud'] = {'skipped': True, 'reason': 'Explicit image-only capture; earlier PCD failure preserved separately'}
            metadata['joints_after'] = metadata['joints_after_images']
        else:
            cloud, cloud_timing = await timed(lambda: camera.get_point_cloud(timeout=args.timeout), args.timeout + 1, secrets)
            metadata['pointcloud'] = {'rpc': cloud_timing}
            if cloud is not None:
                raw, mime = cloud
                (out / 'pointcloud.pcd.gz').write_bytes(gzip.compress(raw, mtime=0))
                metadata['pointcloud'].update({'file': 'pointcloud.pcd.gz', 'mime_type': mime,
                    'raw_bytes': len(raw), 'raw_sha256': hashlib.sha256(raw).hexdigest()})
            metadata['joints_after'] = await joint_snapshot(robot, args.timeout, secrets)
        metadata['joint_delta_degrees'] = {}
        for name in ['arm', 'arm-board']:
            a, b = metadata['joints_before'][name]['joint_degrees'], metadata['joints_after'][name]['joint_degrees']
            metadata['joint_delta_degrees'][name] = [y-x for x,y in zip(a,b)] if a is not None and b is not None else None
        metadata['capture_finished_utc'] = utc()
    except Exception as exc:
        metadata['capture_error'] = safe_error(exc, secrets)
    finally:
        if robot is not None:
            _, metadata['disconnect'] = await timed(robot.close, 8, secrets)
        write_json(out / 'metadata.json', metadata)
        manifest = [{'path': p.name, 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                    for p in sorted(out.iterdir()) if p.is_file()]
        write_json(out / 'MANIFEST.json', {'files': manifest})
    print(json.dumps({'dataset': str(out), 'frames': len(metadata['frames']),
                      'pointcloud_ok': metadata.get('pointcloud', {}).get('rpc', {}).get('ok', False),
                      'capture_error_type': metadata.get('capture_error', {}).get('type'),
                      'joint_delta_degrees': metadata.get('joint_delta_degrees')}))
    return 1 if metadata.get('capture_error') else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--credentials', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--camera', default='cam')
    parser.add_argument('--frames', type=int, choices=range(1, 6), default=3)
    parser.add_argument('--timeout', type=float, default=25)
    parser.add_argument('--direct', action='store_true', help='Use authenticated direct TLS instead of WebRTC; no security checks disabled')
    parser.add_argument('--skip-pointcloud', action='store_true', help='Only RGB/depth and bracketed arm readbacks; no PCD request')
    parser.add_argument('--depth-byte-order', choices=['big', 'little'], default='big')
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    try:
        raise SystemExit(asyncio.run(capture(args)))
    except Exception as exc:
        # Never echo connection arguments or a credential-containing exception.
        print(json.dumps({'failed': True, 'error_type': type(exc).__name__}))
        raise SystemExit(1)
