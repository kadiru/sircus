"""Direct-first admission, immutable context and full accepted-path controls."""
import ast
from collections import deque
import copy
from pathlib import Path
import tempfile
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import numpy as np
    from sircus.mjlab import direct_path_first as direct
    from sircus.mjlab import calibrated_admission as legacy
except ModuleNotFoundError as error:
    if error.name != 'torch':
        raise
    direct = None


class FakePlanner:
    def __init__(self, results=()):
        self.results = list(results)
        self.events = []
        self.gate = SimpleNamespace(bucket_xy=None)

    def check_home(self, bucket):
        self.events.append('home')
        return dict(passed=True, hard_reject=False, reason=None)

    _serial_points = staticmethod(lambda *args: legacy.SerializedPlanner._serial_points(*args))

    def _audit_path(self, points):
        self.events.append('direct')
        passed = self.results.pop(0) if self.results else True
        edges = []
        for a,b in zip(points, points[1:]):
            n=max(1, int(np.ceil(np.max(np.abs(b-a))/legacy.SAMPLE_STEP_RAD)))+1
            edges.append(dict(passed=passed, minimum_clearance_m=.01 if passed else .004,
                total_samples=n, maximum_actual_joint_sample_step_rad=float(np.max(np.abs(b-a))/(n-1))))
        return dict(passed=passed, waypoints=[q.tolist() for q in points], edges=edges,
            reverse_edges=list(reversed(copy.deepcopy(edges))), scope=legacy.SCOPE,
            minimum_clearance_m=.01 if passed else .004,
            max_joint_sample_step_rad=legacy.SAMPLE_STEP_RAD)


def fixture(directory, outcomes=()):
    base=object.__new__(legacy.GeometryResolver)
    base.directory=Path(directory)
    base.home=dict(sha256='home', q10=[0.]*6, q14=[0.]*6)
    base.sources=[dict(id='source', bucket_xy_m=[.6,-.4], values=[.4,-.2,.3,28.,.4], q10=[.1]*6,q14=[.2]*6)]
    base.bounds=dict(low=[.1,-.8,.1,1.,.01],high=[1.3,.4,1.,80.,1.])
    base.planner=FakePlanner(outcomes)
    base._resolved={}; base._connected={}
    resolver=object.__new__(direct.DirectPathResolver)
    resolver.base=base; resolver.contexts=base._connected
    resolver.connections={}; resolver.pending=deque()
    resolver.last_direct_attempts=[]; resolver.last_fallback=None
    return resolver


def proposal():
    return dict(id='candidate', values=[.4,-.2,.3,28.,.4], board_heading_offset_deg=0., board_twist_deg=0.)


def save_candidate(resolver, anchor, target):
    p=proposal()
    paths=resolver._paths(anchor, target)
    candidate=dict(values=p['values'],q10=target[:6].tolist(),q14=target[6:].tolist(),endpoint=dict(reason=None))
    return resolver._save_branch(p,anchor,dict(board_world=np.eye(4)),candidate,paths)


@unittest.skipIf(direct is None,'Pinned CPU environment requires Torch')
class DirectFirstTests(unittest.TestCase):
    def test_context_is_explicitly_deferred_and_does_not_search_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            resolver=fixture(directory)
            resolver.base.resolve=lambda _:self.fail('Reference search ran during context creation')
            anchor=resolver.resolve([.8,-.4])
            self.assertTrue(anchor['reference_search_deferred'])
            self.assertFalse(anchor['anchor_admitted'])
            self.assertEqual(anchor['attempts'],[])
            self.assertEqual(resolver.planner.events,['home'])
            self.assertTrue(direct.verify_resolution(anchor,resolver.sources,resolver.home,resolver.bounds))
            with self.assertRaises(ValueError):
                legacy.verify_resolution(anchor,resolver.sources,resolver.home)
            self.assertTrue(resolver.resolve([.8,-.4])['cache_hit'])
            self.assertEqual(resolver.planner.events,['home'])

    def test_first_accepted_direct_path_never_searches_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            resolver=fixture(directory,[True])
            resolver.base.resolve=lambda _:self.fail('Unneeded reference search')
            anchor=resolver.resolve([.8,-.4]); target=np.r_[np.full(6,.1),np.full(6,.2)]
            record=save_candidate(resolver,anchor,target)
            self.assertIsNone(record['fallback_resolution'])
            self.assertEqual(len(record['direct_path_attempts']),1)
            self.assertEqual(record['direct_path_attempts'][0]['order'],[14,10])
            self.assertEqual(resolver.planner.events,['home','direct'])
            self.assertTrue(direct.verify_connection(record,anchor,resolver.home,proposal()['values']))
            self.assertEqual(len(record['proof']['edges']),2)
            self.assertEqual(len(record['proof']['reverse_edges']),2)

    def test_second_direct_order_retains_first_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            resolver=fixture(directory,[False,True])
            resolver.base.resolve=lambda _:self.fail('Unneeded reference search')
            anchor=resolver.resolve([.8,-.4]); target=np.r_[np.full(6,.1),np.full(6,.2)]
            record=save_candidate(resolver,anchor,target)
            self.assertEqual([a['order'] for a in record['direct_path_attempts']],[[14,10],[10,14]])
            self.assertFalse(record['path_attempts'][0]['passed'])
            self.assertTrue(record['path_attempts'][1]['passed'])
            for mutation in ('samples','order','early_accept','endpoint','seed'):
                altered=copy.deepcopy(record)
                if mutation=='samples':altered['proof']['edges'][0]['total_samples']-=1
                elif mutation=='order':altered['direct_path_attempts'][0]['order']=[10,14]
                elif mutation=='early_accept':altered['direct_path_attempts'][0]['proof']['passed']=True
                elif mutation=='endpoint':altered['q10'][0]+=.01
                else:altered['ik_seed']['q10'][0]+=.01
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                    direct.verify_connection(altered,anchor,resolver.home,proposal()['values'])

    def test_reference_is_only_requested_after_both_direct_orders_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            resolver=fixture(directory,[False,False])
            anchor=resolver.resolve([.8,-.4]); target=np.r_[np.full(6,.1),np.full(6,.2)]
            reference=dict(version=legacy.VERSION,file='historical.json',sha256='old')
            def resolve(bucket):
                self.assertEqual(resolver.planner.events,['home','direct','direct'])
                resolver.planner.events.append('reference')
                return reference
            resolver.base.resolve=resolve
            with patch.object(legacy,'verify_resolution') as verify, \
                 patch.object(resolver.base,'_check_resolution_bytes') as check, \
                 patch.object(direct.HeadingResolver,'_paths',return_value=[dict(passed=True)]) as paths:
                resolver._paths(anchor,target)
            verify.assert_called_once_with(reference,resolver.sources,resolver.home)
            check.assert_called_once_with(reference)
            paths.assert_called_once()
            self.assertEqual(resolver.last_fallback,reference)
            self.assertEqual(len(resolver.last_direct_attempts),2)

    def test_deferred_context_rejects_false_search_claim_and_changed_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            resolver=fixture(directory);anchor=resolver.resolve([.8,-.4])
            for mutation in ('passed','attempt','deferred','source','home'):
                altered=copy.deepcopy(anchor)
                if mutation=='passed':altered['passed']=True
                elif mutation=='attempt':altered['attempts']=[{}]
                elif mutation=='deferred':altered['reference_search_deferred']=False
                elif mutation=='source':altered['source_hint']['q10'][0]=1.
                else:altered['home_geometry']['passed']=False
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                    direct.verify_resolution(altered,resolver.sources,resolver.home,resolver.bounds)
            (Path(directory)/anchor['file']).write_text('{}')
            with self.assertRaises(ValueError):resolver.resolve([.8,-.4])

    def test_native_hydration_cannot_start_new_search(self):
        with tempfile.TemporaryDirectory() as directory:
            resolver=fixture(directory);engine=SimpleNamespace(adapter=SimpleNamespace(resolver=resolver))
            with self.assertRaisesRegex(ValueError,'refusing native replanning'):
                direct.hydrate_direct_resolution(engine,[.8,-.4])
            anchor=resolver.resolve([.8,-.4]);resolver.contexts.clear()
            saved=direct.hydrate_direct_resolution(engine,[.8,-.4])
            self.assertEqual(saved['sha256'],anchor['sha256'])
            self.assertEqual(resolver.planner.events,['home'])

    def test_native_evaluator_has_unchanged_data_and_physics_logic(self):
        root=Path(__file__).resolve().parents[1]
        def node(path,cls):
            tree=ast.parse(path.read_text())
            found=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
            return next(n for n in found.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate')
        before=ast.unparse(node(root/'calibrated_adapter.py','CalibratedAdapter'))
        after=ast.unparse(node(root/'direct_path_first.py','DirectPathAdapter'))
        self.assertEqual(before.replace('adapter_version=VERSION','adapter_version=ADAPTER_VERSION'),after)

if __name__=='__main__':unittest.main()
