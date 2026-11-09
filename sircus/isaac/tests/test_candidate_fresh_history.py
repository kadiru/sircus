"""Coordinate-only historical discovery checks; no outcome/model/native calls."""

import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import candidate_fresh_history as history


class HistoryTests(unittest.TestCase):
    def test_nested_coordinates_ignore_outcomes_and_unset_ui_goal(self):
        value = dict(bucket_xy_m=None, cases=[dict(bucket_xy_m=[.4,-.2], captured=True),
            dict(bucket_xy_m=[.5,-.3], captured=False), dict(bucket_xy_m=[.4,-.2], reward=-100)])
        expected = [(.4,-.2),(.5,-.3)]
        self.assertEqual(history.coordinates_only(value), expected)
        value['cases'][0]['captured'] = False
        value['cases'][1]['reward'] = 1e9
        self.assertEqual(history.coordinates_only(value), expected)

    def test_invalid_recorded_coordinates_fail(self):
        for invalid in ([True,0], [float('nan'),0], [1], 'unknown', {'x':1}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                history.coordinates_only({'bucket_xy_m':invalid})

    def test_git_blobs_deduplicate_refs_and_retain_original_pool(self):
        blob = json.dumps({'records':[{'bucket_xy_m':[.4,-.2],'outcome':'ignored'}]}).encode()
        oid = 'a'*40
        def git(repo,*args,input=None):
            if args[0]=='for-each-ref': return b'refs/remotes/origin/a\nrefs/remotes/origin/b\n'
            if args[0]=='ls-tree': return f'100644 blob {oid}\tbaselines/example/result.json\0'.encode()
            if args[0]=='cat-file' and args[1].startswith('--batch-check'):
                return f'{oid} blob {len(blob)}\n'.encode()
            if args==('cat-file','--batch'): return f'{oid} blob {len(blob)}\n'.encode()+blob+b'\n'
            self.fail(args)
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'old.json'
            path.write_text(json.dumps(dict(coordinates=[[.1,-.1]],sources=[dict(added_coordinates=1)])))
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.object(history,'_git',side_effect=git):
                receipt,inventory=history.discover_history(temporary,path,expected_original_pool_sha256=digest)
            self.assertEqual(receipt['coordinates'],[[.1,-.1],[.4,-.2]])
            self.assertEqual(inventory['refs_examined'],3)
            self.assertEqual(inventory['unique_baseline_json_blobs'],1)
            self.assertEqual(inventory['discovered_union_coordinates'],2)
            self.assertFalse(inventory['outcome_fields_used'])
            with self.assertRaisesRegex(ValueError,'bytes changed'):
                history.discover_history(temporary,path,expected_original_pool_sha256='f'*64)

    def test_modules_have_no_training_or_native_dispatch_calls(self):
        source=Path(history.__file__).parent
        for name in ('candidate_fresh_history.py','candidate_fresh_parent.py'):
            tree=ast.parse((source/name).read_text())
            calls=[node.func.attr for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)]
            self.assertFalse(set(calls)&{'backward','step','shot_batch','home','manual_seed'})
            imports=[node.module or '' for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
            self.assertFalse(any('updates' in name or 'native' in name for name in imports))


if __name__=='__main__': unittest.main()
