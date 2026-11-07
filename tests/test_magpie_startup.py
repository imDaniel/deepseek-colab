"""Magpie cold starts return JSON retries without allocating real Colab sessions."""
import concurrent.futures
import http.client
import json
import sys
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock

if 'test_magpie_colab' in sys.modules:  # unittest discover imports this sibling without the tests prefix
    llm, controller = sys.modules['test_magpie_colab'].llm, sys.modules['test_magpie_colab'].controller
else:
    from tests.test_magpie_colab import llm, controller


class StartupTests(unittest.TestCase):
    def test_http_cold_start_then_progress_then_ready(self):
        manager = llm.Manager()

        def begin(new):
            manager.state, manager.begun = 'starting', time.time()

        class Handler(llm.Handler):
            held = controller.retry_held
            def authed(self): return True
            def host_ok(self): return True
            def forward(self, body, sse=False):
                self.send_json(200, json.loads(llm.with_effort(body)))

        server = llm.Server(('127.0.0.1', 0), Handler)
        server.mgr = manager
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def request(stream, effort=None):
            body = {'model': 'deepseek-v4.1-flash-sec', 'stream': stream, 'messages': [{'role': 'user', 'content': 'hi'}]}
            if effort: body['reasoning_effort'] = effort
            c = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=3)
            c.request('POST', '/v1/chat/completions', json.dumps(body))
            r = c.getresponse(); data = json.loads(r.read()); headers = dict(r.getheaders()); c.close()
            return r.status, data, headers

        try:
            with mock.patch.object(manager, '_begin', side_effect=begin) as start, \
                    mock.patch.object(controller, 'startup_timings', return_value={}):
                t0 = time.monotonic()
                rc, data, headers = request(True)
                self.assertLess(time.monotonic() - t0, 1)
                self.assertEqual(rc, 429)
                self.assertEqual(headers['Content-Type'], 'application/json')
                self.assertEqual(headers['Retry-After'], '10')
                self.assertEqual(data['server_status']['phase'], 'allocating')
                self.assertFalse(data['server_status']['vm_created'])
                self.assertEqual(data['error']['code'], 'model_starting')
                with manager.cond:
                    self.assertTrue(manager.cond.wait_for(lambda: manager.active == manager.inflight == 0, timeout=2))
                manager.steps = [('allocated', time.time(), None), ('uploaded', time.time(), None)]
                manager.begun -= 30
                rc, data, _ = request(False)
                self.assertEqual(rc, 429)
                self.assertTrue(data['server_status']['vm_created'])
                self.assertEqual(data['server_status']['phase'], 'downloading')
                self.assertLessEqual(data['server_status']['estimated_remaining_seconds'], 150)
                start.assert_called_once_with(True)
                manager.state, manager.tunnel_ok = 'ready', True
                for effort in (None, 'low', 'high', 'max'):
                    rc, data, _ = request(False, effort)
                    self.assertEqual(rc, 200)
                    self.assertEqual(data['reasoning_effort'], effort or 'max')
                with manager.cond:
                    self.assertTrue(manager.cond.wait_for(lambda: manager.active == manager.inflight == 0, timeout=2))
        finally:
            server.shutdown(); server.server_close(); thread.join(5)

    def test_parallel_retries_start_once(self):
        manager = llm.Manager()
        def begin(new): manager.state, manager.begun = 'starting', time.time()
        def request():
            h = object.__new__(llm.Handler); h.server = mock.Mock(mgr=manager); h.send_json = mock.Mock()
            controller.retry_held(h, b'{}')
            self.assertEqual(h.send_json.call_args.args[0], 429)
        with mock.patch.object(manager, '_begin', side_effect=begin) as start, \
                mock.patch.object(controller, 'startup_timings', return_value={}):
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                for result in [pool.submit(request) for _ in range(8)]:
                    result.result(timeout=3)
            start.assert_called_once_with(True)

    def test_failures_cooldown_and_stopping_do_not_start_again(self):
        for state, cooldown, expected in (('off', 60, 503), ('stopping', 0, 429), ('ready', 0, 429)):
            manager = llm.Manager(); manager.state = state; manager.fail_until = time.time() + cooldown
            h = object.__new__(llm.Handler); h.server = mock.Mock(mgr=manager); h.send_json = mock.Mock()
            with mock.patch.object(manager, '_begin') as start:
                controller.retry_held(h, b'{}')
                self.assertEqual(h.send_json.call_args.args[0], expected)
                start.assert_not_called()

    def test_timing_history_is_bounded_private_and_marks_overruns(self):
        with tempfile.TemporaryDirectory() as root, \
                mock.patch.object(controller, 'STARTUP_TIMINGS', Path(root) / 'timings.json'):
            manager = llm.Manager(); manager.state = 'ready'; manager.begun = time.time() - 110
            for _ in range(7): controller.remember_startup(manager)
            history = controller.startup_timings()
            self.assertEqual(len(history[controller.timing_key()]), 5)
            self.assertEqual(controller.STARTUP_TIMINGS.stat().st_mode & 0o777, 0o600)
            manager.state = 'starting'; manager.begun = time.time() - 50
            status = controller.startup_status(manager)
            self.assertEqual(status['estimate_source'], 'recent_successful_starts')
            self.assertAlmostEqual(status['estimated_remaining_seconds'], 60, delta=1)
            manager.begun = time.time() - 200
            self.assertIsNone(controller.startup_status(manager)['estimated_remaining_seconds'])

    def test_catalog_declares_levels_separately_from_default(self):
        h = object.__new__(llm.Handler)
        data = h.models()['data'][0]
        self.assertTrue(data['reasoning'])
        self.assertEqual(data['reasoning_levels'], ['low', 'high', 'max'])
        self.assertEqual(data['supported_reasoning_levels'], [{'effort': s} for s in ('low', 'high', 'max')])


if __name__ == '__main__': unittest.main()
