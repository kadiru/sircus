"""Startup order and cleanup are tested with mock clients and no bound sockets."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server


class HardwareStartupTests(unittest.TestCase):
    def run_main(self, mode='hardware', *, bind_error=None, lease_error=None,
                 physical_error=None, close_result=True, audio_close_error=None,
                 offline=True):
        calls=[]
        def record(name, result=None, error=None):
            def invoke(*args, **kwargs):
                calls.append(name)
                if error:raise error
                return result
            return invoke
        http=Mock();http.serve_forever.side_effect=record('serve');http.server_close.side_effect=record('http_close')
        lease=Mock();lease.close.side_effect=record('lease_close')
        audio=Mock();audio.close.side_effect=record('audio_close',error=audio_close_error)
        physical=Mock();physical.close.side_effect=record('physical_close',close_result)
        simulation=Mock();simulation.close.side_effect=record('simulation_close',True)
        with ExitStack() as stack:
            stack.enter_context(patch.multiple(server,OFFLINE=False,BRIDGE=None,PHYSICAL=None,
                AUDIO=None,SHOW_ENABLED=False,SHOW_AUDIO=None,OPERATIONS=None,TRAINING_JOBS=None))
            stack.enter_context(patch.object(server,'ThreadingHTTPServer',side_effect=record('bind',http,bind_error)))
            stack.enter_context(patch.object(server,'acquire_hardware_lease',side_effect=record('lease',lease,lease_error)))
            stack.enter_context(patch.object(server,'optional_audio_observer',side_effect=record('audio',audio)))
            stack.enter_context(patch.object(server,'PhysicalBridge',side_effect=record('physical',physical,physical_error)))
            stack.enter_context(patch.object(server,'ViamBridge',side_effect=record('simulation',simulation)))
            stack.enter_context(patch.object(server.signal,'signal'))
            argv=['server','--physical-mode',mode,'--port','9999']
            if offline:argv.append('--offline')
            stack.enter_context(patch.object(sys,'argv',argv))
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            error=None
            try:server.main()
            except Exception as exc:error=exc
        return calls,error

    def test_bind_and_lease_precede_every_client_and_release_after_cleanup(self):
        calls,error=self.run_main()
        self.assertIsNone(error)
        self.assertEqual(calls,['bind','lease','audio','physical','serve','http_close',
                                'physical_close','audio_close','lease_close'])

    def test_busy_http_port_never_acquires_lease_connects_or_mutates_journal(self):
        calls,error=self.run_main(bind_error=OSError('address already in use'))
        self.assertIsInstance(error,OSError)
        self.assertEqual(calls,['bind'])

    def test_held_or_unsafe_lease_closes_reserved_port_before_any_client(self):
        calls,error=self.run_main(lease_error=RuntimeError('owned'))
        self.assertEqual(str(error),'owned')
        self.assertEqual(calls,['bind','lease','http_close'])

    def test_constructor_failure_releases_port_audio_and_lease(self):
        calls,error=self.run_main(physical_error=ValueError('invalid calibration'))
        self.assertIsInstance(error,ValueError)
        self.assertEqual(calls,['bind','lease','audio','physical','http_close','audio_close','lease_close'])

    def test_nonoffline_partial_failure_also_closes_simulation_bridge(self):
        calls,error=self.run_main(physical_error=ValueError('invalid calibration'),offline=False)
        self.assertIsInstance(error,ValueError)
        self.assertEqual(calls,['bind','lease','audio','simulation','physical','http_close',
                                'simulation_close','audio_close','lease_close'])

    def test_disabled_and_dry_run_never_acquire_hardware_lease(self):
        for mode in ('disabled','dry_run'):
            with self.subTest(mode=mode):
                calls,error=self.run_main(mode)
                self.assertIsNone(error)
                self.assertNotIn('lease',calls)
                self.assertNotIn('lease_close',calls)
                self.assertEqual('physical' in calls,mode=='dry_run')

    def test_incomplete_sdk_loop_cleanup_keeps_lease_until_process_exit(self):
        calls,error=self.run_main(close_result=False)
        self.assertIsNone(error)
        self.assertIn('audio_close',calls)
        self.assertNotIn('lease_close',calls)

    def test_audio_cleanup_failure_cannot_skip_hardware_lease_release(self):
        calls,error=self.run_main(audio_close_error=RuntimeError('audio unavailable'))
        self.assertIsNone(error)
        self.assertEqual(calls[-1],'lease_close')

    def test_mock_bridge_closes_loop_without_any_actuation(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(server,'AUDIO',None):
            bridge=server.PhysicalBridge('dry_run',None,folder)
            bridge.startup.result(timeout=3)
            adapter=bridge.controller.adapter
            self.assertTrue(bridge.close())
            self.assertFalse(bridge.thread.is_alive())
            self.assertTrue(bridge.loop.is_closed())
            self.assertEqual(adapter.commands,[])
            self.assertTrue(bridge.close())
            with self.assertRaisesRegex(RuntimeError,'closed'):bridge.call('action',{'action':'stop'})

    def test_simulation_bridge_closes_only_transport(self):
        bridge=server.ViamBridge()
        robot=Mock();robot.close=AsyncMock()
        bridge.robot=robot
        self.assertTrue(bridge.close())
        robot.close.assert_awaited_once_with()
        self.assertEqual(robot.mock_calls, [unittest.mock.call.close()])
        self.assertFalse(bridge.thread.is_alive())
        self.assertTrue(bridge.loop.is_closed())
        self.assertTrue(bridge.close())
        with self.assertRaisesRegex(RuntimeError,'closed'):bridge.command({'action':'run'})

    def test_bad_calibration_fails_before_starting_a_background_loop(self):
        loops=[];factory=server.asyncio.new_event_loop
        def loop_factory():
            value=factory();loops.append(value);return value
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder)/'calibration.json').write_text('not json')
            with patch.object(server.asyncio,'new_event_loop',side_effect=loop_factory), \
                 patch.object(server.threading,'Thread') as thread:
                with self.assertRaises(ValueError):server.PhysicalBridge('dry_run',None,folder)
                thread.assert_not_called()
            self.assertTrue(loops[0].is_closed())


if __name__=='__main__':unittest.main()
