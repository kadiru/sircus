import copy
from collections import Counter
import unittest
import numpy as np
from sircus.mjlab import corrected_imitation_contract as C
from sircus.mjlab.tests.test_coverage_recovery_contract import score_fields


def fixture():
    context='recovery-test-r1-c1-n1';anchor=dict(id='anchor',bucket_xy_m=[.6,-.2],values=[.6,-.2,.3,28.,.2])
    source=dict(id='original',context_id=context,method='recovery_bc',phase='test',role='test',bucket_xy_m=[.7,-.25],
        source_hint=anchor,effective_source_hint=anchor,source_teacher_id='anchor',context_resolution_file='resolution.json',
        context_resolution_sha256='a'*64,context_home_sha256='h'*64,context_layout_eligible=True,
        context_anchor_admitted=True,context_hard_reject=None,observation=C.observation([.7,-.25],anchor).tolist())
    rows=[]
    for i in range(3):
        r=dict(copy.deepcopy(source),id='correction-'+str(i),phase='diagnostic',role='diagnostic',repeat=i,variant='parent_xy',
            source_record_id=source['id'],source_record_sha256=C.digest(source),values=[.61,-.21,.31,29.,.21],
            q10=[.1]*6,q14=[.2]*6,connection_sha256='c'*64,native_release=True,route_success=True,
            invalid=None,invalid_physics=False,normalized_residual=[-.9]*5,route_quality=1,reward=4.)
        r.update(score_fields());rows.append(r)
    return source,rows


class ContractTests(unittest.TestCase):
    def test_corrected_teacher_rebuilds_residual_and_records_roles_without_repeat_weighting(self):
        source,rows=fixture();before=copy.deepcopy(rows)
        teacher,group=C.corrected_teacher(rows,source['context_id'],'parent_xy',source)
        self.assertEqual(rows,before);self.assertEqual(len(group),3)
        np.testing.assert_allclose(teacher['normalized_residual'],[.25,-.25,.2,.125,1/12])
        self.assertEqual(teacher['role'],'train');self.assertEqual(teacher['original_role'],'test')
        self.assertEqual(teacher['source_correction_role'],'diagnostic');self.assertTrue(teacher['development_reclassified'])
        self.assertEqual(teacher['source_correction_record_sha256'],C.digest(rows[0]))
        self.assertEqual(teacher['correction_repeat_ids'],[r['id'] for r in rows])

    def test_failed_missing_or_duplicate_confirmation_rejected(self):
        for mutate in (lambda rows:rows.pop(),lambda rows:rows[2].update(route_success=False),lambda rows:rows[2].update(repeat=1)):
            source,rows=fixture();mutate(rows)
            with self.assertRaises(ValueError):C.corrected_teacher(rows,source['context_id'],'parent_xy',source)

    def test_changed_branch_or_action_rejected(self):
        for field,value in (('q10',[.3]*6),('connection_sha256','d'*64),('values',[.6,-.2,.3,28.,.2])):
            source,rows=fixture();rows[2][field]=value
            with self.assertRaises(ValueError):C.corrected_teacher(rows,source['context_id'],'parent_xy',source)

    def test_mutated_parent_lineage_or_anchor_rejected(self):
        source,rows=fixture();rows[0]['source_record_sha256']='b'*64
        with self.assertRaises(ValueError):C.corrected_teacher(rows,source['context_id'],'parent_xy',source)
        source,rows=fixture();rows[0]['effective_source_hint']['id']='wrong'
        with self.assertRaises(ValueError):C.corrected_teacher(rows,source['context_id'],'parent_xy',source)

    def test_out_of_support_target_refuses_clipping(self):
        source,rows=fixture()
        for row in rows:row['values'][0]=.65
        with self.assertRaises(ValueError):C.corrected_teacher(rows,source['context_id'],'parent_xy',source)

    def test_split_is_fresh_and_stratified_without_outcome_filtering(self):
        excluded=[[.1,-.65],[.4,-.4],[.6,-.2],[1.,0.]]
        first=C.test_split(excluded);self.assertEqual(first,C.test_split(excluded))
        self.assertEqual(len(first),70);self.assertEqual(Counter(tuple(r['grid_cell']) for r in first),Counter({(x,y):2 for y in range(5) for x in range(7)}))
        points=np.asarray([r['bucket_xy_m'] for r in first]);distance=np.linalg.norm(points[:,None]-np.asarray(excluded)[None],axis=2)
        self.assertGreaterEqual(distance.min(),.005)
        another=C.test_split(excluded+[first[0]['bucket_xy_m']]);self.assertNotEqual(first[0],another[0])
        self.assertTrue(all(r['split']=='test' and r['id'].startswith('corrected-imitation-test-') for r in first))

    def test_empty_or_nonfinite_history_rejected(self):
        for excluded in ([],[[float('nan'),0.]]):
            with self.assertRaises(ValueError):C.test_split(excluded)

    def test_protocol_has_no_search_or_ppo_and_all_requested_slots_count(self):
        p=C.protocol();self.assertEqual((p['base_teacher_count'],p['correction_teacher_count'],p['teacher_count']),(92,5,97))
        self.assertEqual((p['test_requests'],p['control_requests'],p['maximum_requests']),(140,18,158))
        self.assertEqual((p['actor_steps'],p['critic_steps'],p['substeps']),(800,0,64))
        self.assertEqual((p['worker_limit_s'],p['queue_limit_s'],p['global_limit_s']),(3600,900,6300))
        self.assertEqual(len(C.CORRECTIONS),len({x[1] for x in C.CORRECTIONS}));self.assertEqual(C.METHODS,('corrected_bc','current_bc'))

if __name__=='__main__':unittest.main()
