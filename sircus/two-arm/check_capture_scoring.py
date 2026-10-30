"""Focused numerical and synthetic-event tests; no servers, SDK calls or archives."""
import copy
import json
import unittest
from pathlib import Path

from capture_scoring import CaptureTracker, SCORING_VERSION
from physics import bucket_contact, bucket_contains_sphere, resolve_bucket_contacts, simulate
from shot_rollout import first_contact, prepare_scene, release_coordinates, rollout, selection_key

SCENE = json.loads((Path(__file__).parent/'scene.json').read_text())
ACTION = {'u_mm':0.,'v_mm':0.,'gap_mm':140.}


class ScoringEvents(unittest.TestCase):
    def tracker(self, kind='complete_shot', bounced=True):
        tracker = CaptureTracker(SCENE, episode_kind=kind)
        tracker.update(0,[680,245,250], board_bounce=bounced,
                       board_contact_xyz=[600,20,384] if bounced else None)
        tracker.update(.1,[680,245,200])
        return tracker

    def test_full_entry_and_uninterrupted_one_second(self):
        t=self.tracker();t.update(.2,[680,245,180])
        self.assertFalse(t.entry)
        t.update(.3,[680,245,164]);t.update(1.299,[680,245,64.5])
        self.assertFalse(t.success)
        t.update(1.3,[680,245,64.5]);self.assertTrue(t.success)
        self.assertEqual(t.result(1.3)['capture_class'],'clean')

    def test_rim_contact_does_not_veto_capture(self):
        t=self.tracker();t.note_contact('bucket-rim',.15)
        t.update(.2,[680,245,164]);t.update(1.2,[680,245,64.5])
        self.assertEqual(t.result(1.2)['capture_class'],'rim_assisted')

    def test_partial_exit_reentry_restarts_full_timer(self):
        t=self.tracker();t.update(.2,[680,245,164]);t.update(.8,[680,245,180])
        self.assertIsNone(t.inside_since);self.assertEqual(t.bounce_outs,1)
        t.update(.9,[680,245,164]);t.update(1.899,[680,245,64.5])
        self.assertFalse(t.success)
        t.update(1.9,[680,245,64.5]);self.assertTrue(t.success)
        self.assertEqual(len(t.crossings),1)

    def test_complete_exit_requires_valid_new_crossing(self):
        t=self.tracker();t.update(.2,[680,245,164]);t.update(.8,[680,245,250])
        self.assertFalse(t.valid_opening);self.assertEqual(t.full_exits,1)
        t.update(.9,[800,245,200]);t.update(1.,[680,245,100]);t.update(2.1,[680,245,64.5])
        self.assertFalse(t.success)

    def test_no_crossing_no_entry_no_invented_distance(self):
        t=CaptureTracker(SCENE);t.update(0,[680,245,100]);t.update(4.,[680,245,100])
        result=t.result(4.)
        self.assertEqual(result['outcome'],'no_entry')
        self.assertIsNone(result['signed_miss_xy_mm'])

    def test_missing_board_bounce_and_late_bounce_do_not_count(self):
        t=self.tracker(bounced=False);t.update(.2,[680,245,164])
        t.update(.8,[680,245,100],board_bounce=True,board_contact_xyz=[600,20,384])
        t.update(1.2,[680,245,64.5])
        self.assertFalse(t.success)
        self.assertEqual(t.result(1.2)['outcome'],'no_board_bounce')

    def test_contact_fixture_cannot_be_complete_shot_success(self):
        t=self.tracker(kind='contact_fixture');t.update(.2,[680,245,164]);t.update(1.2,[680,245,64.5])
        r=t.result(1.2);self.assertFalse(r['success']);self.assertTrue(r['bucket_retained_1s'])
        self.assertEqual(r['outcome'],'fixture_retained')

    def test_short_horizon_is_timeout_and_model_limit_overrides_capture(self):
        t=self.tracker();t.update(.2,[680,245,164]);t.update(.7,[680,245,64.5])
        self.assertEqual(t.result(.7)['outcome'],'timeout')
        self.assertTrue(t.result(.7)['truncated'])
        t.update(1.2,[680,245,64.5])
        r=t.result(1.2,model_limit='contact correction')
        self.assertFalse(r['success']);self.assertEqual(r['outcome'],'model_limit')

    def test_late_entry_gets_full_retention_time(self):
        t=CaptureTracker(SCENE)
        t.update(3.8,[680,245,250],board_bounce=True,board_contact_xyz=[600,20,384])
        t.update(3.9,[680,245,200]);t.update(3.95,[680,245,164]);t.update(4.,[680,245,100])
        self.assertFalse(t.done(4.))
        t.update(4.949,[680,245,64.5]);self.assertFalse(t.success)
        t.update(4.95,[680,245,64.5]);self.assertTrue(t.success)

    def test_bounce_out_class_distinct_from_timeout(self):
        t=self.tracker();t.update(.2,[680,245,164]);t.update(.8,[680,245,180]);t.update(4.,[900,245,70])
        r=t.result(4.);self.assertEqual(r['outcome'],'bounce_out');self.assertFalse(r['truncated'])


class ContactRollouts(unittest.TestCase):
    def test_corner_resolves_both_surfaces_without_energy_gain(self):
        c=prepare_scene(SCENE);p=[.7203,.245,.06449];v=[.1,0,-.1]
        hits=resolve_bucket_contacts(p,v,c)
        self.assertEqual({h['object'] for h in hits},{'bucket-floor','bucket-inner-wall'})
        self.assertIsNone(bucket_contact([x*1000 for x in p],37.5,c))
        self.assertTrue(bucket_contains_sphere(c,[x*1000 for x in p],37.5))
        for h in hits:
            self.assertLessEqual(h['kinetic_after_j'],h['kinetic_before_j']+1e-12)
            if h['incoming_normal_m_s']<0:
                self.assertAlmostEqual(h['outgoing_normal_m_s'],-.35*h['incoming_normal_m_s'])

    def test_clean_and_rim_shots_and_outside_miss(self):
        for v,expected in [(-40,'clean'),(0,'rim_assisted'),(30,None)]:
            action={**ACTION,'v_mm':v,'gap_mm':240 if v==30 else 140}
            r=rollout(SCENE,action,42)
            self.assertEqual(r['capture_class'],expected)
            self.assertEqual(r['success'],expected is not None)
            self.assertNotIn('retained',r)
            if expected:
                self.assertTrue(r['valid_board_bounce']);self.assertGreaterEqual(r['max_continuous_containment_s'],1.-1e-9)
            else:self.assertEqual(r['outcome'],'no_entry')

    def test_wall_floor_regression_and_reset(self):
        c=copy.deepcopy(SCENE)
        first=rollout(c,ACTION,42,frames=True)
        corner=rollout(c,{**ACTION,'v_mm':-20,'gap_mm':400},42)
        self.assertTrue(corner['success'])
        again=rollout(c,ACTION,42,frames=True)
        self.assertEqual(first,again);self.assertEqual(c,SCENE)

    def test_rim_capture_at_three_timesteps(self):
        for dt in (.0005,.00025,.000125):
            c=prepare_scene(SCENE);c['simulation']['dt_s']=dt
            r=simulate(c,ACTION,42,capture_mode=SCORING_VERSION)
            self.assertTrue(r['success']);self.assertEqual(r['capture_class'],'rim_assisted')
            self.assertLess(r['max_bucket_projection_mm'],2.)
            for f in r['frames']:
                contact=bucket_contact(f['ball_xyz_mm'],37.5,c)
                self.assertTrue(contact is None or contact[1][0]<1e-5)

    def test_isolated_escape_and_partial_reentry(self):
        c=prepare_scene(SCENE)
        escaped=simulate(c,ACTION,42,capture_mode=SCORING_VERSION,episode_kind='contact_fixture',
                         release_override=[635,245,250],initial_velocity_m_s=[5,0,-8])
        self.assertEqual(escaped['outcome'],'bounce_out');self.assertFalse(escaped['success'])
        self.assertTrue(escaped['whole_ball_entry']);self.assertLess(escaped['max_continuous_containment_s'],1.)
        reentered=simulate(c,ACTION,42,capture_mode=SCORING_VERSION,episode_kind='contact_fixture',
                          release_override=[725,245,1000])
        self.assertEqual(reentered['outcome'],'fixture_retained');self.assertFalse(reentered['success'])
        self.assertEqual(reentered['bounce_outs'],1);self.assertEqual(len(reentered['opening_crossings']),1)

    def test_coarse_high_speed_contact_is_model_limit(self):
        c=prepare_scene(SCENE);c['simulation']['dt_s']=.0005
        r=simulate(c,ACTION,42,capture_mode=SCORING_VERSION,episode_kind='contact_fixture',
                   release_override=[635,245,250],initial_velocity_m_s=[5,0,-8])
        self.assertEqual(r['outcome'],'model_limit');self.assertFalse(r['success'])
        self.assertEqual(r['termination_reason'],'model_limit')

    def test_short_rollout_does_not_promote_partial_retention(self):
        c=prepare_scene(SCENE);c['simulation']['duration_s']=1.
        r=simulate(c,ACTION,42,capture_mode=SCORING_VERSION)
        self.assertTrue(r['whole_ball_entry']);self.assertTrue(r['valid_board_bounce'])
        self.assertGreater(r['current_containment_s'],.2)
        self.assertLess(r['current_containment_s'],1.)
        self.assertEqual(r['outcome'],'timeout');self.assertTrue(r['truncated'])
        self.assertFalse(r['success']);self.assertEqual(r['duration_s'],1.)
        self.assertEqual(r['frames'][-1]['t'],1.)
        for actual,displayed in zip(r['final_ball_xyz_mm'],r['frames'][-1]['ball_xyz_mm']):
            self.assertAlmostEqual(actual,displayed,places=4)

    def test_overrides_and_unknown_versions_rejected(self):
        with self.assertRaises(ValueError):simulate(SCENE,ACTION,capture_mode='unknown')
        with self.assertRaises(ValueError):simulate(prepare_scene(SCENE),ACTION,capture_mode=SCORING_VERSION,initial_velocity_m_s=[1,0,0])
        r=rollout(SCENE,{**ACTION,'gap_mm':-1})
        self.assertEqual(r['outcome'],'invalid_action');self.assertFalse(r['executed'])
        with self.assertRaises(ValueError):selection_key({'retained':True,'reward':1000})
        tilted=prepare_scene(SCENE);tilted['bin']['rpy_deg']=[0,0,0]
        with self.assertRaises(ValueError):simulate(tilted,ACTION,capture_mode=SCORING_VERSION)

    def test_finite_board_height_and_legacy_mode(self):
        self.assertIsNone(first_contact(SCENE,[900,0]))
        coords=release_coordinates(SCENE,{**ACTION,'v_mm':30,'gap_mm':240})
        self.assertAlmostEqual(coords['h_m'],.240)
        self.assertEqual(coords['commanded_release_ball_center_mm'][:2],[570,20])
        legacy=simulate(SCENE,ACTION,42,frames=False)
        self.assertTrue(legacy['retained']);self.assertNotIn('success',legacy)
        self.assertNotIn('scoring_version',legacy)


if __name__ == '__main__':unittest.main(verbosity=2)
