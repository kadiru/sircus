"""Adversarial accounting checks for exploratory versus confirmation evidence."""
import copy
import unittest

from sircus.mjlab.benchmark_verify import Checks
from sircus.mjlab.board_pose_recovery_audit import (
    captured, exact_repeat, independent_summary, ranked_diverse,
    search_success_ids, verify_confirmations, verify_request_keys,
)
from sircus.mjlab.board_pose_recovery_study import schedule, repeat_schedule


def command(identity):
    return dict(id=identity, values=[.5, -.2, .4, 25., .3], bucket_xy_m=[1., 0.],
                board_heading_offset_deg=0., board_twist_deg=0., selected_q10=[0.]*6,
                selected_q14=[0.]*6, native_prepared=True, prepared_connection_sha256='original-proof')


def plan():
    candidate = command('candidate')
    return dict(cases=[dict(id='case', directory='native-case', original=command('parent'),
        candidates=[candidate], admitted=[candidate], proposals=[candidate], records=[{}],
        predictions={'candidate': {'score_m': .01}})])


def observation(variant='search', repeat=0, success=True, invalid=False, robot=False):
    return dict(id=f'{variant}-{repeat}', context_id='case', candidate_id='candidate',
                variant=variant, repeat=repeat, native_release=True, invalid_physics=invalid,
                route_success=success, shot_route='direct' if success else None,
                experimental_route={'robot_contact_steps': int(robot), 'robot_assisted_capture': robot and not success})


class AuditTests(unittest.TestCase):
    def test_omitted_original_or_candidate_is_rejected(self):
        p = plan(); slots = schedule(p)
        verify_request_keys(p, slots, Checks())
        for index in (0, len(slots)-1):
            with self.assertRaises(ValueError):
                verify_request_keys(p, slots[:index]+slots[index+1:], Checks())

    def test_repeat_changes_to_release_or_proof_rejected(self):
        source = command('candidate')
        for field, value in (('values', [.5, -.2, .4, 25., .4]), ('selected_q14', [1.]*6),
                             ('prepared_connection_sha256', 'different-proof')):
            request = copy.deepcopy(source); request[field] = value
            with self.assertRaises(ValueError):
                exact_repeat(source, request, Checks(), 'changed')

    def test_only_initial_valid_standard_capture_qualifies(self):
        self.assertFalse(captured(observation(invalid=True)))
        self.assertFalse(captured(observation(success=False, robot=True)))
        self.assertEqual(search_success_ids([observation('repeat')]), [])
        self.assertEqual(search_success_ids([observation()]), ['candidate'])
        with self.assertRaises(ValueError):
            search_success_ids([observation(), observation()])

    def test_repeat_every_success_three_times_and_no_failed_candidate(self):
        p = plan(); rows = [observation()] + [observation('repeat', i) for i in (1, 2, 3)]
        verify_confirmations(p, rows, Checks())
        verify_request_keys(p, repeat_schedule(p, rows), Checks(), confirmations=True, eligible={'candidate'})
        with self.assertRaises(ValueError):
            verify_confirmations(p, rows[:-1], Checks())
        with self.assertRaises(ValueError):
            verify_confirmations(p, [observation(success=False)]+rows[1:], Checks())

    def test_one_failed_confirmation_is_not_a_confirmed_setting(self):
        p = plan(); rows = [observation()]+[observation('repeat', i, success=i != 3) for i in (1, 2, 3)]
        result = independent_summary(p, rows)[0]
        self.assertEqual(result['results']['search']['captures'], 1)
        self.assertEqual(result['results']['repeat']['captures'], 2)
        self.assertEqual(result['confirmed_settings'], [])
        self.assertEqual(result['repeated_settings'][0]['repeat_captures'], 2)
        self.assertEqual(result['results']['original']['requested'], 3)

    def test_diversity_precedes_redundant_second_best(self):
        a, b, c = [command(x) for x in ('a', 'b', 'c')]; c['values'][0] += .1
        predictions = {'a': {'score_m': .1}, 'b': {'score_m': .09}, 'c': {'score_m': .08}}
        self.assertEqual([r['id'] for r in ranked_diverse([c, b, a], predictions, 2)], ['a', 'c'])


if __name__ == '__main__':
    unittest.main()
