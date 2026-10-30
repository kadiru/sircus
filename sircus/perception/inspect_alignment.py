"""Offline stationary table-edge consistency check; not calibration.

ROI x=1000:1170, rows60:660 was selected from this dataset's visible right table
edge. RGB uses peak grayscale derivative; depth uses a near/far threshold from
local medians and requires five consecutive far pixels. This one edge cannot
validate full image registration or camera extrinsics.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np


def inspect(dataset):
    results=[]
    for image_path in sorted(dataset.glob('frame_*_color.jpg')):
        stem=image_path.name.replace('_color.jpg','')
        image=cv2.imread(str(image_path))
        depth=np.load(dataset/(stem+'_depth_raw.npz'))['depth_raw_counts']
        gray=cv2.GaussianBlur(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(5,5),0).astype(float)
        samples=[]
        for y in range(60,661,20):
            rgb=1000+int(np.argmax(np.abs(np.diff(gray[y,1000:1170]))))+.5
            row=depth[y,1000:1170]
            near=np.median(row[:30][row[:30]>0]);far=np.median(row[-30:][row[-30:]>0])
            runs=np.where(np.convolve((row>(near+far)/2).astype(int),np.ones(5,int),'valid')==5)[0]
            if not len(runs):
                continue
            dep=1000+int(runs[0])
            samples.append({'row':y,'rgb_edge_x':rgb,'depth_transition_x':dep,'offset_px':dep-rgb})
        offset=np.array([r['offset_px'] for r in samples])
        results.append({'frame':stem,'samples':samples,'sample_count':len(samples),
                        'median_signed_offset_px':float(np.median(offset)),
                        'median_absolute_offset_px':float(np.median(abs(offset))),
                        'p95_absolute_offset_px':float(np.percentile(abs(offset),95)),
                        'maximum_absolute_offset_px':float(max(abs(offset)))})
    output={'method':__doc__,'scope':'heuristic one-edge RGB/depth consistency, not calibrated alignment',
            'full_registration_verified':False,'world_extrinsics_verified':False,'frames':results,
            'limitations':['Silhouette occlusion, edge matching and depth discontinuity artifacts can affect these offsets.',
                           'No ground-truth registered target or measured depth reference was used.',
                           'The board and bucket are outside the recorded view.']}
    (dataset/'alignment_edge_check.json').write_text(json.dumps(output,indent=2)+'\n')
    for r in results:
        print(r['frame'],r['sample_count'],'median abs',r['median_absolute_offset_px'],
              'p95 abs',r['p95_absolute_offset_px'],'max abs',r['maximum_absolute_offset_px'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',type=Path)
    inspect(parser.parse_args().dataset)
