"""Curate credential-free numerical evidence; never copies raw teachers or traces."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from sircus.studies import front_study as study
from sircus.studies.run_front_study import write

RESULT_FIELDS=('success','outcome','capture_class','scoring_version','physics_version','executed',
    'board_bounce_t_s','whole_ball_entry_t_s','bucket_retained_1s','max_continuous_containment_s',
    'full_exits','bounce_outs','impact_uv_mm','h_m','actual_h_m','center_error_mm','max_support_tilt_deg')


def score(row):
    check=row.get('feasibility') or {}
    return {'scene_id':row['scene_id'],'layout_sha256':row['layout_sha256'],'split':row['split'],
        'seed':row['seed'],'role':row.get('role'),'action':row['action'],
        'checkpoint_id':row.get('checkpoint_id'),'result':{k:row['result'][k] for k in RESULT_FIELDS if k in row['result']},
        'endpoint_ik':{'arm10_reachable':check.get('arm_reachable',False),'arm14_reachable':check.get('board_arm_reachable',False)},
        'data_origin':'synthetic','physically_verified':False,'native_paths_checked':False}


def copy_json(source,destination):
    # Parse first, ensuring only explicitly allowlisted JSON artifacts are copied.
    value=json.loads(Path(source).read_text());write(destination,value)


def publish(curve_dir,initial_dir,destination):
    curve_dir,initial_dir,destination=map(Path,(curve_dir,initial_dir,destination))
    report=json.loads((curve_dir/'report.json').read_text())
    expected_extensions={'sircus-contact-intervention-v1','sircus-post-study-diagnostics-v1'}
    if report['status']!='complete' or not expected_extensions.issubset({e.get('schema') for e in report.get('extensions',[])}):raise ValueError('Complete core, contact extension and post-study diagnostics required')
    if len((curve_dir/'report.json').read_bytes())>=32*1024*1024:raise ValueError('Report exceeds dashboard limit')
    cp=json.loads((curve_dir/'checkpoint.json').read_text());study.validate_checkpoint(cp)
    destination.mkdir(parents=True,exist_ok=False)
    for name in ('checkpoint.json','report.json','report-core.json','report-pre-diagnostic.json','manifest.json','learning-curve.json','independent-model-verification.json'):
        copy_json(curve_dir/name,destination/name)
    copy_json(curve_dir/'summary.json',destination/'core-summary.json')
    copy_json(curve_dir/'challenge-summary.json',destination/'core-challenge-summary.json')
    write(destination/'summary.json',report['summary'])
    write(destination/'challenge-summary.json',report['challenges'])
    validation=json.loads((curve_dir/'validation.json').read_text())
    write(destination/'validation-summary.json',[{**{k:r[k] for k in ('seed','train_scenes')},
        'candidates':[{k:c[k] for k in ('degree','regularization','summary')} for c in r['candidates']]} for r in validation])
    for source in sorted(curve_dir.glob('seed-*/checkpoint-n*.json')):
        value=json.loads(source.read_text());study.validate_checkpoint(value)
        write(destination/'checkpoints'/f'{source.parent.name}-{source.name}',value)
    results=json.loads((curve_dir/'curve-results.json').read_text())
    write(destination/'test-scorecard.json',[{'seed':r['seed'],'train_scenes':r['train_scenes'],
        'rows':[score(row) for row in r['rows']]} for r in results])
    comparators=json.loads((curve_dir/'comparators.json').read_text())
    write(destination/'comparator-scorecard.json',{k:[score(r) for r in comparators[k]] for k in ('fixed_baseline','held_out_search_oracle')})
    for name in ('manifest.json','summary.json'):
        copy_json(curve_dir/'contact-intervention'/name,destination/'contact-intervention'/name)
    for name in ('manifest.json','summary.json'):
        copy_json(curve_dir/'post-study-diagnostics'/name,destination/'post-study-diagnostics'/name)
    diagnostic=json.loads((curve_dir/'post-study-diagnostics/results.json').read_text())
    write(destination/'post-study-diagnostics/scorecard.json',[{'label':g['label'],'method':g['method'],
        'rows':[score(r) for r in g['rows']]} for g in diagnostic])
    initial=destination/'initial-24-6-8'
    for name in ('checkpoint.json','episodes.json','experiment.json','summary.json','preview-scene.json','preview.json'):
        copy_json(initial_dir/name,initial/name)
    raw=json.loads((initial_dir/'test.json').read_text())
    write(initial/'test-scorecard.json',{k:[score(r) for r in raw[k]] for k in ('frozen_policy','fixed_baseline','held_out_search_oracle')})
    write(destination/'publication.json',{'schema':'sircus-front-study-publication-v1','profile_id':study.PROFILE_ID,
        'default_checkpoint_id':cp['checkpoint_id'],'source_contract':cp['contract'],
        'source_baseline_commit':'1fb2e5cf003950854bfd80bdb322b8bc9fb23c38',
        'data_origin':'synthetic','physically_verified':False,'native_paths_checked':False,
        'excluded':['raw teacher candidate traces','raw training labels','raw trial traces','runtime logs','host calibration','credentials']})
    checksums(destination)
    return destination


def checksums(destination):
    destination=Path(destination)
    files=sorted(p for p in destination.rglob('*') if p.is_file() and p.name!='SHA256SUMS')
    (destination/'SHA256SUMS').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(destination).as_posix()}\n' for p in files))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--curve-dir',type=Path,required=True)
    p.add_argument('--initial-dir',type=Path,required=True);p.add_argument('--destination',type=Path,required=True)
    a=p.parse_args();publish(a.curve_dir,a.initial_dir,a.destination)
