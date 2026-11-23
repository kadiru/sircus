"""Adversarial provenance and menu checks; no native engine or GPU."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from sircus.mjlab.benchmark_verify import Checks, digest, sha
from sircus.mjlab.nearby_transfer_audit import (check_source_trace, independent_candidates,
    independent_order, qualified, require_teacher_rows, verify_proposals)
from sircus.mjlab.nearby_transfer_plan import proposal_menu, preparation_selection, candidate_selection


def fixture():
    request=dict(id='teacher',values=[.62,.1,.4,25.,.23],bucket_xy_m=[.95,-.03],
        board_heading_offset_deg=4.,board_twist_deg=180.,selected_q10=[.1]*6,selected_q14=[.2]*6,
        intended_route='table')
    bank=[dict(id='recovery:teacher',source_request=request,bucket_xy_m=request['bucket_xy_m'],
               strength='confirmed_3_of_3')]
    case=dict(id='parallel-board-miss-03',original=dict(request,bucket_xy_m=[.955,-.04]),
              bounds=dict(low=[.3,-.5,.12,8.,.05],high=[1.04,.35,.6,60.,.65]))
    case['proposals']=proposal_menu(case,bank,case['bounds']['low'],case['bounds']['high'])
    return case,bank


def native_row(i=1,**changed):
    return dict(dict(id='row-'+str(i),repeat=i,native_release=True,invalid_physics=False,
                     route_success=True,shot_route='direct'),**changed)


class SourceAuditTests(unittest.TestCase):
    def test_only_observed_valid_direct_or_table_capture_is_teacher(self):
        self.assertTrue(qualified(native_row()))
        for changes in ({'native_release':False},{'invalid_physics':True},{'shot_route':None},
                        {'route_success':False},{'shot_route':'arm_assisted_capture'}):
            self.assertFalse(qualified(native_row(**changes)))

    def test_three_means_three_distinct_fresh_confirmations(self):
        rows=[native_row(i) for i in (1,2,3)]
        require_teacher_rows(rows,repeats=3,checks=Checks(),label='source')
        for bad in (rows[:-1],[rows[0],rows[0],rows[2]],rows[:2]+[native_row(4)],
                    rows[:2]+[native_row(3,route_success=False)]):
            with self.assertRaises(ValueError):
                require_teacher_rows(bad,repeats=3,checks=Checks(),label='bad')

    def test_native_source_packet_and_proof_bytes_are_authenticated(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary);(path/'native').mkdir();(path/'requests').mkdir()
            for filename in ('resolution.json','connection.json','trace.npz'):
                (path/'native'/filename).write_text(filename)
            request=dict(id='source',values=[.6,.1,.4,25.,.2],bucket_xy_m=[1.,0.],
                board_heading_offset_deg=0.,board_twist_deg=180.,selected_q10=[.1]*6,selected_q14=[.2]*6,
                context_resolution_file='resolution.json',context_resolution_sha256=sha(path/'native/resolution.json'),
                prepared_connection_file='connection.json',prepared_connection_sha256=sha(path/'native/connection.json'))
            packet=path/'requests/batch.json';packet.write_text(json.dumps(dict(rows=[request])))
            row=dict(id='source',values=request['values'],bucket_xy_m=request['bucket_xy_m'],q10=[.1]*6,q14=[.2]*6,
                board_heading_offset_deg=0.,board_twist_deg=180.,adapter_directory='native',
                context_resolution_file='resolution.json',context_resolution_sha256=request['context_resolution_sha256'],
                connection_file='connection.json',connection_sha256=request['prepared_connection_sha256'],
                native_cache_file='trace.npz',native_cache_sha256=sha(path/'native/trace.npz'),
                request_packet='requests/batch.json',request_packet_sha256=sha(packet))
            source=dict(study=path,manifest={p.relative_to(path).as_posix():sha(p) for p in path.rglob('*') if p.is_file()})
            check_source_trace(source,request,row,Checks())
            changed=copy.deepcopy(row);changed['q14'][0]+=.1
            with self.assertRaises(ValueError):check_source_trace(source,request,changed,Checks())
            old=(path/'native/connection.json').read_text();(path/'native/connection.json').write_text('altered')
            with self.assertRaises(ValueError):check_source_trace(source,request,row,Checks())
            (path/'native/connection.json').write_text(old)
            altered=copy.deepcopy(request);altered['values'][4]+=.1
            packet.write_text(json.dumps(dict(rows=[altered])))
            source['manifest']['requests/batch.json']=row['request_packet_sha256']=sha(packet)
            with self.assertRaises(ValueError):check_source_trace(source,request,row,Checks())


class TransferAuditTests(unittest.TestCase):
    def test_changed_geometry_source_joint_or_omitted_proposal_rejected(self):
        case,bank=fixture();verify_proposals(case,bank,Checks())
        for key in ('values','seed_q14','source_distance_m'):
            changed=copy.deepcopy(case)
            if key=='source_distance_m':changed['proposals'][0][key]+=.1
            else:changed['proposals'][0][key][0]+=.1
            with self.assertRaises(ValueError):verify_proposals(changed,bank,Checks())
        case['proposals'].pop()
        with self.assertRaises(ValueError):verify_proposals(case,bank,Checks())

    def test_three_transforms_have_measurable_meaning(self):
        case,bank=fixture();old=bank[0]['source_request'];target=case['original']['bucket_xy_m']
        initial={p['transfer_mode']:p for p in case['proposals'] if p['delta_index']==0}
        self.assertEqual(initial['retain']['values'][:2],old['values'][:2])
        for axis in (0,1):
            delta=target[axis]-old['bucket_xy_m'][axis]
            self.assertAlmostEqual(initial['translate']['values'][axis]-old['values'][axis],delta)
            self.assertAlmostEqual(initial['blend']['values'][axis]-old['values'][axis],delta/2)
        self.assertTrue(all('selected_q14' not in p and 'prepared_connection_file' not in p for p in case['proposals']))

    def test_independent_source_balanced_order_and_reserved_candidates(self):
        case,bank=fixture()
        case['wrist_reach']={p['id']:dict(passed=True) for p in case['proposals']}
        case['nominal_predictions']={p['id']:dict(score_m=float(i%7)/100) for i,p in enumerate(case['proposals'])}
        expected=preparation_selection(case,bank,case['proposals'],case['nominal_predictions'],case['wrist_reach'])
        self.assertEqual(independent_order(case,bank),[p['id'] for p in expected])
        predictions=case['nominal_predictions']
        selected=independent_candidates(case['proposals'],predictions)
        self.assertEqual(selected,candidate_selection(case['proposals'],predictions))
        self.assertEqual(len(selected),20)
        self.assertEqual({p['transfer_mode'] for p in selected if p['delta_index']==0},{'retain','blend','translate'})


if __name__=='__main__':unittest.main()
