"""Offline ArUco inspection only; no robot connection or metric pose claims.

Example:
  python inspect_fiducials.py color.png --out fiducials.json
Requires opencv-contrib-python-headless (tested 4.12.0.88).
"""
import argparse
import json
from pathlib import Path
import cv2


def inspect(image_path, dictionary_name, max_width=2400):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError('Image could not be decoded')
    scale = min(1.0, max_width / image.shape[1])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if scale != 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    params = cv2.aruco.DetectorParameters()
    params.detectInvertedMarker = True
    params.errorCorrectionRate = 0
    corners, ids, rejected = cv2.aruco.ArucoDetector(dictionary, params).detectMarkers(gray)
    observations = []
    if ids is not None:
        for marker_id, points in zip(ids.reshape(-1), corners):
            points = points.reshape(4, 2) / scale
            observations.append({'id': int(marker_id), 'corners_px': points.tolist(),
                                 'center_px': points.mean(axis=0).tolist()})
    return {'source_image': str(image_path), 'opencv': cv2.__version__,
            'dictionary': dictionary_name, 'image_size_wh': [image.shape[1], image.shape[0]],
            'detector_scale': scale, 'error_correction_rate': 0,
            'detect_inverted_marker': True, 'detected_count': len(observations),
            'rejected_candidates': len(rejected), 'observations': observations,
            'metric_pose': None, 'metric_pose_status': 'missing measured square/marker dimensions and verified layout/intrinsics',
            'scope': '2D image observations only; dictionary supersets can share the same marker IDs'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--dictionary', default='DICT_5X5_100')
    args = parser.parse_args()
    result = inspect(args.image, args.dictionary)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(f"Detected {result['detected_count']} markers; wrote {args.out}; no metric pose inferred.")
