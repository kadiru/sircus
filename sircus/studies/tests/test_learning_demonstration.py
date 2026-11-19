"""Offline movie evidence checks. No encoder, services, network or hardware."""
import copy
import hashlib
import json
import math
from pathlib import Path
import unittest

from sircus.studies import render_learning_demonstration as movie

ROOT=Path(__file__).resolve().parents[3]
BUNDLE=ROOT/'sircus/simulation/web/media/demo-reel/showcase.json'


class LearningDemonstrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle=json.loads(BUNDLE.read_text());cls.digest=movie.sha(BUNDLE)
        cls.geometry=movie.prepare(cls.bundle,cls.digest)

    def test_same_layout_failures_are_preserved_and_success_is_separate(self):
        rows=movie.selected_clips(self.bundle)
        self.assertEqual([r['train_scenes'] for r in rows[:4]],[8,24,64,128])
        self.assertEqual(len({r['scene_id'] for r in rows[:4]}),1)
        self.assertTrue(all(not r['result']['success'] for r in rows[:4]))
        self.assertTrue(rows[4]['result']['success'])
        self.assertNotEqual(rows[4]['scene_id'],rows[0]['scene_id'])

    def test_illustrated_fk_reaches_saved_endpoints_and_moves_both_arms(self):
        for row in movie.selected_clips(self.bundle):
            frames=self.geometry['clips'][row['id']];first,last=frames[0],frames[-1]
            self.assertLess(math.dist(last['board_center_mm'],row['board_pose']['center_mm']),1e-7)
            self.assertLess(math.dist(last['ball_xyz_mm'],row['release_coordinates']['commanded_release_ball_center_mm']),2)
            self.assertNotEqual(first['arm14_boxes'],last['arm14_boxes'])
            self.assertNotEqual(first['arm10_boxes'],last['arm10_boxes'])
            self.assertFalse(self.geometry['native_paths_checked'])

    def test_evidence_bundle_and_actual_ball_frames_are_unchanged(self):
        before=hashlib.sha256(json.dumps(self.bundle,sort_keys=True).encode()).hexdigest()
        movie.validate_geometry(self.bundle,self.geometry,self.digest)
        for row in movie.selected_clips(self.bundle):
            frames=row['result']['frames'];chosen=movie.frame_at(frames,.193)
            self.assertIn(chosen,frames);self.assertLessEqual(chosen['t'],.193)
        after=hashlib.sha256(json.dumps(self.bundle,sort_keys=True).encode()).hexdigest()
        self.assertEqual(before,after)

    def test_incompatible_or_physical_input_is_rejected(self):
        with self.assertRaises(ValueError):movie.validate_geometry(self.bundle,self.geometry,'wronghash')
        changed=copy.deepcopy(self.bundle);changed['physically_verified']=True
        with self.assertRaises(ValueError):movie.selected_clips(changed)
        changed=copy.deepcopy(self.bundle);changed['learning']['same_layout_clips'].pop()
        with self.assertRaises(ValueError):movie.selected_clips(changed)

    def test_soundtrack_contacts_have_saved_evidence_and_unknown_stays_silent(self):
        rows=movie.selected_clips(self.bundle);chapters=[{'kind':'paired_checkpoint','start_s':i*10} for i in range(5)]
        events=movie.soundtrack_events(rows,chapters)
        self.assertEqual(sum(e['asset']=='drum_roll.wav' for e in events),15)
        by_trial={r['trial_id']:r for r in rows}
        for e in events:
            if 'contact_index' in e:
                contact=by_trial[e['trial_id']]['result']['contacts'][e['contact_index']]
                self.assertEqual(contact['object'],e['contact_object'])
        changed=copy.deepcopy(rows);changed[0]['result']['outcome']='unknown'
        cues=movie.soundtrack_events(changed,chapters)
        self.assertFalse(any(e['trial_id']==changed[0]['trial_id'] and e['asset'] in ('capture.wav','miss.wav') for e in cues))


if __name__=='__main__':unittest.main()
