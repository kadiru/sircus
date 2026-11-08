import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from sircus.mjlab import arm_clearance_pipeline as pipeline
from sircus.mjlab import arm_clearance_study as study
from sircus.mjlab.arm_clearance_selector import RULE

class PipelineTests(unittest.TestCase):
    def test_forward_all_input_paths_to_worker_and_audit(self):
        args={k:Path('/proof')/k for k in (*pipeline.INPUT_KEYS,'output','gpu_lock','display')}
        args.update(queue_timeout=86400,wall_time_limit=3600);args=SimpleNamespace(**args)
        worker=pipeline.worker_command(args,Path('/source'));audit=pipeline.audit_command(args,Path('/source'))
        for command in (worker,audit):
            for key in pipeline.INPUT_KEYS:
                flag='--'+key.replace('_','-');self.assertEqual(command[command.index(flag)+1],str(getattr(args,key)))
        parsed=pipeline.parser().parse_args([x for key in (*pipeline.INPUT_KEYS,'output','gpu_lock','display') for x in ('--'+key.replace('_','-'),str(getattr(args,key)))])
        self.assertEqual(parsed.gap_study,args.gap_study);self.assertEqual(parsed.gap_audit,args.gap_audit)
    def test_actual_publisher_pending_and_failure_preserve_previous_keys(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);out=root/'study';out.mkdir();display=root/'display.json';display.write_text('{"older_result":{"captures":24}}')
            spec=dict(version=study.VERSION,protocol=study.protocol(),maximum_requests=112,new_training_requests=0,clearance_rule=RULE)
            spec['fingerprint']=study.digest(spec)
            report=dict(version=study.VERSION,protocol=study.protocol(),contract_fingerprint=spec['fingerprint'],status='queued',phase='waiting_for_4090',scope=study.SCOPE,counts=dict(requested=0),queue=dict(wait_s=1))
            study.atomic(out/'contract.json',spec);study.atomic(out/'report.json',report)
            result=pipeline.publish(out,display)
            self.assertEqual(result['status'],'queued');self.assertEqual(result['comparisons'],[])
            data=json.loads(display.read_text());self.assertEqual(data['older_result'],dict(captures=24))
            pipeline.publish(out,display,pipeline=dict(status='failed',error='own worker stopped'))
            data=json.loads(display.read_text());self.assertEqual(data['arm_clearance_comparison']['status'],'failed')
            self.assertEqual(data['older_result'],dict(captures=24))

if __name__=='__main__':unittest.main()
