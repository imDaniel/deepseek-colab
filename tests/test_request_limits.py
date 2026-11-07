"""Request admission and output budgets must reject excess work before waking Colab."""
import concurrent.futures
import http.client
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location('llm_request_limits', Path(__file__).resolve().parents[1] / 'llm.py')
llm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(llm)
from integrations.magpie_colab import config


class RequestLimits(unittest.TestCase):
    def test_output_defaults_and_explicit_smaller_budgets(self):
        for path, field in (('/v1/chat/completions', 'max_tokens'), ('/v1/completions', 'max_tokens'), ('/v1/responses', 'max_output_tokens')):
            self.assertEqual(json.loads(llm.with_output_limit(b'{}', path))[field], 32768)
            self.assertEqual(json.loads(llm.with_output_limit(json.dumps({field: 1024}).encode(), path))[field], 1024)
        value = json.loads(llm.with_output_limit(b'{"max_completion_tokens":8192}', '/v1/chat/completions'))
        self.assertEqual(value, {'max_tokens': 8192})
        self.assertEqual(json.loads(llm.with_output_limit(b'{"n_predict":256}', '/v1/chat/completions')), {'max_tokens': 256})
        self.assertEqual(llm.with_output_limit(b'invalid', '/v1/embeddings'), b'invalid')

    def test_removed_model_does_not_enter_or_wake(self):
        for path in ('/v1/chat/completions', '/v1/completions', '/v1/responses'):
            h = object.__new__(llm.Handler)
            h.server = mock.Mock(mgr=llm.Manager())
            h.path, h.command = path, 'POST'
            h.host_ok = h.authed = lambda: True
            h.fail, h.held = mock.Mock(), mock.Mock()
            h.route(b'{"model":"ds41-max"}')
            self.assertEqual(h.fail.call_args.args[0], 400)
            h.held.assert_not_called()
            self.assertEqual((h.server.mgr.state, h.server.mgr.inflight), ('off', 0))

    def test_ask_max_uses_explicit_effort_on_single_model(self):
        for extra in ([], ['--max']):
            response = mock.MagicMock(status=200)
            response.__iter__.return_value = iter([b'data: [DONE]\n'])
            with mock.patch.object(llm, 'serve_running', return_value={'port': 18080}), \
                    mock.patch.object(llm, 'api_key', return_value='test-only'), \
                    mock.patch.object(llm, 'http_call', return_value=(mock.Mock(), response)) as call, \
                    mock.patch('sys.stdout', new_callable=io.StringIO):
                llm.cmd_ask(['hello'] + extra)
            body = json.loads(call.call_args.args[3])
            self.assertEqual(body['model'], 'deepseek-v4.1-flash-sec')
            self.assertEqual(body.get('reasoning_effort'), 'max' if extra else None)
            self.assertEqual(body['messages'][0]['content'], 'hello')

    def test_invalid_budget_does_not_enter_or_wake(self):
        for payload in ({'max_tokens': 65536}, {'max_tokens': 0}, {'max_tokens': True}, {'max_tokens': 1.5},
                        {'max_tokens': None}, {'max_tokens': 5, 'max_completion_tokens': 5}, {'n_predict': 65536}, []):
            h = object.__new__(llm.Handler)
            h.server = mock.Mock(mgr=llm.Manager())
            h.path, h.command = '/v1/chat/completions', 'POST'
            h.host_ok = h.authed = lambda: True
            h.fail, h.held = mock.Mock(), mock.Mock()
            h.route(json.dumps(payload).encode())
            self.assertEqual(h.fail.call_args.args[0], 400)
            h.held.assert_not_called()
            self.assertEqual((h.server.mgr.state, h.server.mgr.inflight), ('off', 0))

    def test_four_waiters_admitted_fifth_rejected_and_slots_return(self):
        entered, release = threading.Event(), threading.Event()
        count, lock = [0], threading.Lock()
        mgr = llm.Manager()

        class Handler(llm.Handler):
            def host_ok(self): return True
            def authed(self): return True
            def held(self, body):
                self.asserted_budget = json.loads(body)['max_tokens']
                with lock:
                    count[0] += 1
                    if count[0] == 4: entered.set()
                release.wait(5)
                self.send_json(200, {'ok': True})

        server = llm.Server(('127.0.0.1', 0), Handler)
        server.mgr = mgr
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        def request():
            c = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=8)
            c.request('POST', '/v1/chat/completions', body=b'{}', headers={'Content-Type': 'application/json'})
            r = c.getresponse(); data = json.loads(r.read()); c.close()
            return r.status, data
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(request) for _ in range(4)]
                self.assertTrue(entered.wait(4))
                self.assertEqual(mgr.inflight, 4)
                self.assertEqual(request()[0], 429)
                self.assertEqual(count[0], 4)
                release.set()
                self.assertTrue(all(f.result()[0] == 200 for f in futures))
            with mgr.cond:
                self.assertTrue(mgr.cond.wait_for(lambda: mgr.inflight == 0, timeout=2))
            self.assertEqual(request()[0], 200)
            with mgr.cond:
                self.assertTrue(mgr.cond.wait_for(lambda: mgr.inflight == 0, timeout=2))
        finally:
            release.set(); server.shutdown(); server.server_close(); thread.join(5)

    def test_forwarding_exception_releases_admission(self):
        h = object.__new__(llm.Handler); h.server = mock.Mock(mgr=llm.Manager())
        h.path, h.command = '/v1/chat/completions', 'POST'
        h.host_ok = h.authed = lambda: True
        h.held = mock.Mock(side_effect=ConnectionResetError())
        with self.assertRaises(ConnectionResetError): h.route(b'{}')
        self.assertEqual(h.server.mgr.inflight, 0)

    def test_context_tiers_preserve_saved_million_and_legacy_snapshots(self):
        self.assertEqual(config.DEFAULTS['context'], 524288)
        self.assertEqual(config.CHOICES['context'], (262144, 524288, 1048576))
        for legacy, desired in ((32768, 262144), (65536, 262144), (131072, 262144)):
            with tempfile.TemporaryDirectory() as temp:
                p = Path(temp) / 'settings.json'; p.write_text(json.dumps(dict(config.DEFAULTS, context=legacy)))
                self.assertEqual(config.read(p)['context'], legacy)
                self.assertEqual(config.read_desired(p)['context'], desired)
                with self.assertRaises(ValueError): config.write(p, dict(config.DEFAULTS, context=legacy))
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / 'settings.json'
            self.assertEqual(config.read_desired(p)['context'], 524288)
            config.write(p, dict(config.DEFAULTS, context=1048576))
            self.assertEqual(config.read_desired(p)['context'], 1048576)

if __name__ == '__main__': unittest.main()
