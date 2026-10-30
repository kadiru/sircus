"""Offline color/relative-depth inspection. No robot connection or metric scale assumption.

The free-radius sphere fits are diagnostics, not accepted object poses. They
are compared with RGB silhouette rays and several mask erosions. Depth counts
must not be labeled mm until a separate scale check establishes that mapping.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np


def sphere_fit(points):
    origin = points.mean(axis=0)
    q = points - origin
    a = np.column_stack((2*q, np.ones(len(q))))
    b = (q*q).sum(axis=1)
    solution = np.linalg.lstsq(a, b, rcond=None)[0]
    center = solution[:3] + origin
    radius = np.sqrt(max(0, solution[3] + solution[:3] @ solution[:3]))
    # Robust reweighted algebraic fit; center/radius stay free, no supplied diameter.
    for _ in range(30):
        residual = np.linalg.norm(points-center, axis=1)-radius
        cutoff = max(1.0, 1.5 * 1.4826 * np.median(np.abs(residual-np.median(residual))))
        weights = np.minimum(1.0, cutoff / np.maximum(np.abs(residual), 1e-9))
        solution = np.linalg.lstsq(a*np.sqrt(weights[:,None]),b*np.sqrt(weights),rcond=None)[0]
        new_center = solution[:3] + origin
        new_radius = np.sqrt(max(0, solution[3] + solution[:3] @ solution[:3]))
        change = np.linalg.norm(new_center-center)+abs(new_radius-radius)
        center, radius = new_center, new_radius
        if change < 1e-6:
            break
    residual = np.linalg.norm(points-center, axis=1)-radius
    return center, float(radius), residual


def main(dataset):
    props = json.loads((dataset/'camera_properties.json').read_text())
    k = props['intrinsic_parameters']
    fx, fy, cx, cy = [k[key] for key in ('focal_x_px','focal_y_px','center_x_px','center_y_px')]
    output = {'dataset': str(dataset), 'scope': 'camera-frame RGB observations and raw-count depth consistency only',
              'depth_unit': 'raw device counts; no metric conversion applied',
              'depth_alignment_assumption': 'configured aligned-to-color; empirical alignment not yet established',
              'distortion_used': 'reported coefficients are all zero',
              'color_mask_hsv_opencv': {'lower':[75,140,35], 'upper':[105,255,255]},
              'mask_selection': 'largest interior component >3000 px; fill outer contour; exclude components touching image edges',
              'frames': [], 'board_impact_surface': None, 'board_surface_status':'not visible in captured view',
              'bucket_opening_center':None, 'bucket_opening_normal':None, 'bucket_status':'not visible in captured view',
              'ball_center_camera_raw_counts':None, 'ball_center_camera_meters':None,
              'world_frame_poses':None, 'world_frame_calibrated':False}
    for image_path in sorted(dataset.glob('frame_*_color.jpg')):
        stem = image_path.name.replace('_color.jpg','')
        image = cv2.imread(str(image_path))
        depth = np.load(dataset/(stem+'_depth_raw.npz'))['depth_raw_counts']
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        binary = cv2.inRange(hsv, np.array([75,140,35]), np.array([105,255,255]))
        count, labels, stats, centers = cv2.connectedComponentsWithStats(binary)
        height, width = depth.shape
        candidates = [i for i in range(1,count) if stats[i,4]>3000 and stats[i,0]>0 and stats[i,1]>0
                      and stats[i,0]+stats[i,2]<width and stats[i,1]+stats[i,3]<height]
        if not candidates:
            output['frames'].append({'frame':stem,'ball_status':'no suitable color component'})
            continue
        chosen = max(candidates,key=lambda i:stats[i,4])
        mask = (labels==chosen).astype(np.uint8)
        contour = max(cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0],key=cv2.contourArea)
        ellipse = cv2.fitEllipse(contour)
        cv2.drawContours(mask,[contour],-1,1,-1)
        uv = np.array(ellipse[0])
        direction = np.array([(uv[0]-cx)/fx,(uv[1]-cy)/fy,1.])
        unit = direction / np.linalg.norm(direction)
        ring = (cv2.dilate(mask,np.ones((61,61),np.uint8))>0) & (mask==0) & (depth>0)
        table_median = float(np.median(depth[ring]))
        valid = (mask>0)&(depth>0)
        edge = contour.reshape(-1,2).astype(float)[::max(1,len(contour)//150)]
        rays = np.column_stack(((edge[:,0]-cx)/fx,(edge[:,1]-cy)/fy,np.ones(len(edge))))
        rays /= np.linalg.norm(rays,axis=1)[:,None]
        row = {'frame':stem, 'color_component_label':'green/turquoise ball candidate',
               'ellipse_center_px':uv.tolist(), 'ellipse_axes_px':list(ellipse[1]),'ellipse_angle_deg':ellipse[2],
               'silhouette_bearing_unit_xyz':unit.tolist(),
               'bearing_note':'approximate ray through apparent silhouette center, not a metrically established 3D sphere center',
               'image_occlusion_status':'complete colored silhouette inside image; depth is largely missing',
               'mask_pixels':int(mask.sum()), 'valid_depth_fraction_in_mask':float(valid.sum()/mask.sum()),
               'valid_depth_percentiles_0_5_50_95_100':np.percentile(depth[valid],[0,5,50,95,100]).tolist(),
               'surrounding_table_median_raw_counts':table_median, 'free_radius_candidates':[]}
        for erosion in (5,10,20):
            eroded = cv2.erode(mask,np.ones((2*erosion+1,2*erosion+1),np.uint8))>0
            # Reject depths at/behind the nearby tabletop; this is an exploratory filter.
            use = eroded & (depth>0) & (depth<table_median-3)
            v,u = np.where(use)
            z = depth[v,u].astype(float)
            points = np.column_stack(((u-cx)*z/fx,(v-cy)*z/fy,z))
            if len(points)<30:
                continue
            center,radius,residual = sphere_fit(points)
            distance = np.linalg.norm(center)
            angle = np.arcsin(np.clip(radius/distance,-1,1)) if distance else np.nan
            silhouette_residual = np.arccos(np.clip(rays@(center/distance),-1,1))-angle
            row['free_radius_candidates'].append({'mask_erosion_px':erosion,
                'support_points':len(points), 'support_fraction_of_color_mask':float(len(points)/mask.sum()),
                'center_xyz_raw_count_units':center.tolist(), 'radius_raw_counts':radius,
                'radial_rms_raw_counts':float(np.sqrt(np.mean(residual**2))),
                'radial_median_abs_raw_counts':float(np.median(np.abs(residual))),
                'rgb_silhouette_angular_rms_deg':float(np.sqrt(np.mean(silhouette_residual**2))*180/np.pi),
                'accepted_as_pose':False})
        output['frames'].append(row)
    output['ball_center_status'] = ('missing: color bearing is observed, but depth support and free-radius/silhouette '
                                    'consistency require review; no known-radius fit or metric center is substituted')
    output['limitations'] = ['Color thresholds target this visible green/turquoise ball only.',
                            'Raw count deprojection assumes the saved depth grid is registered to the reported color intrinsics.',
                            'Sparse near-table depths may be foreground mismatches, flying pixels, or invalid surface returns.',
                            'The single-surface fits are diagnostic; small residual on a sparse patch is not full-sphere validation.',
                            'No manually supplied ball diameter was used. No camera-to-robot/world transform was applied.']
    target = dataset/'initial_localization.json'
    target.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(target)
    for row in output['frames']:
        print(row['frame'],'center_px',row.get('ellipse_center_px'),'valid_depth_fraction',row.get('valid_depth_fraction_in_mask'))
        for fit in row.get('free_radius_candidates',[]):
            print(' diagnostic sphere:',fit['center_xyz_raw_count_units'],'radius',fit['radius_raw_counts'],
                  'RMS',fit['radial_rms_raw_counts'],'silhouette_RMS_deg',fit['rgb_silhouette_angular_rms_deg'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',type=Path)
    args=parser.parse_args()
    main(args.dataset)
