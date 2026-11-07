"""Offline tests for settings validation, snapshots and the loopback page API."""
import http.client
import importlib.util
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

spec = importlib.util.spec_from_file_location('llm_magpie_test', Path(__file__).resolve().parent.parent / 'llm.py')
llm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(llm)
with mock.patch.dict(sys.modules, {'llm': llm}):
    from integrations.magpie_colab import config, controller, panel, refresh


class SettingsTests(unittest.TestCase):
    def test_refresh_updates_running_gateway_catalog_and_keeps_key_on_loopback(self):
        with mock.patch.dict(os.environ, {'MAGPIE_WEB_KEY': 'fixture-key', 'COLAB_MAGPIE_WEB_URL': 'http://127.0.0.1:3430'}, clear=True), \
                mock.patch.object(refresh, 'urlopen') as fetch, mock.patch.object(refresh.subprocess, 'run') as cli:
            fetch.return_value.__enter__.return_value.status = 200
            self.assertTrue(refresh.refresh_models())
            request = fetch.call_args.args[0]
            self.assertEqual(request.full_url, 'http://127.0.0.1:3430/api/provider/models')
            self.assertEqual(json.loads(request.data), {'id': refresh.PROVIDER})
            self.assertEqual(request.get_header('Cookie'), 'magpie_web_3430=fixture-key')
            cli.assert_not_called()
            os.environ['COLAB_MAGPIE_WEB_URL'] = 'https://example.com'
            fetch.reset_mock()
            with self.assertRaises(ValueError): refresh.refresh_models()
            fetch.assert_not_called()

    def test_dead_backend_is_stopped_only_after_repeated_idle_failures(self):
        mgr = llm.Manager()
        mgr.state, mgr.tunnel_ok, mgr.port, mgr.attempt = 'ready', True, 12345, 1
        with mock.patch.object(llm, 'tunnel_health', return_value=False), mock.patch.object(mgr, 'request_stop') as stop:
            mgr._check_idle_health(1, 12345)
            mgr._check_idle_health(1, 12345)
            stop.assert_not_called()
            mgr._check_idle_health(1, 12345)
            stop.assert_called_once_with('backend health check failed', only_inactive=True)
            stop.reset_mock()
            mgr.active = 1
            mgr._check_idle_health(1, 12345)
            stop.assert_not_called()
            self.assertEqual(mgr.health_failures, 0)
        mgr.active = 1
        mgr.request_stop('backend health check failed', only_inactive=True)
        self.assertEqual(mgr.state, 'ready')

    def test_backend_recovery_resets_failures_and_idle_failure_stops(self):
        mgr = llm.Manager()
        mgr.state, mgr.tunnel_ok, mgr.port, mgr.attempt = 'ready', True, 12345, 2
        stopped = threading.Event()
        idle_since = mgr.last_activity
        with mock.patch.object(llm, 'tunnel_health', side_effect=[False, False, True, False, False, False]), \
                mock.patch.object(mgr, '_stop', side_effect=lambda reason: stopped.set()) as stop:
            for _ in range(3):
                mgr._check_idle_health(2, 12345)
            self.assertEqual(mgr.health_failures, 0)
            self.assertEqual(mgr.state, 'ready')
            self.assertEqual(mgr.last_activity, idle_since)
            for _ in range(3):
                mgr._check_idle_health(2, 12345)
            self.assertEqual(mgr.state, 'stopping')
            self.assertTrue(stopped.wait(1))
            stop.assert_called_once_with('backend health check failed')

    def test_catalog_refresh_tracks_active_settings_until_session_stops(self):
        active = dict(config.DEFAULTS, context=1048576)
        desired = dict(config.DEFAULTS, context=32768)
        status = {'controller': {'state': 'ready'}, 'active_settings': active, 'desired': desired}
        self.assertEqual(refresh.effective(status), active)
        status['controller']['state'] = 'off'
        self.assertEqual(refresh.effective(status), desired)
        status['controller']['state'] = 'unavailable'
        self.assertIsNone(refresh.effective(status))

    def test_measured_ista_cache_does_not_change_other_profiles(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith('LLM_')}
        cases = [
            ({'model': 'original', 'uncensor': True, 'context': 1048576, 'images': True, 'draft': True}, '68864'),
            ({'model': 'abiray', 'uncensor': False, 'context': 1048576, 'images': True, 'draft': True}, '64000'),
            ({'model': 'original', 'uncensor': False, 'context': 1048576, 'images': True, 'draft': True}, '64000'),
            ({'model': 'original', 'uncensor': True, 'context': 524288, 'images': True, 'draft': True}, '73000'),
            ({'model': 'abiray', 'uncensor': False, 'context': 524288, 'images': True, 'draft': True}, '64000'),
            ({'model': 'original', 'uncensor': False, 'context': 524288, 'images': True, 'draft': True}, '64000'),
            ({'model': 'original', 'uncensor': True, 'context': 524288, 'images': False, 'draft': True}, '64000'),
            ({'model': 'original', 'uncensor': True, 'context': 524288, 'images': True, 'draft': False}, '64000'),
            ({'model': 'original', 'uncensor': True, 'context': 1048576, 'images': False, 'draft': True}, '64000'),
            ({'model': 'original', 'uncensor': True, 'context': 1048576, 'images': True, 'draft': False}, '64000'),
        ]
        with mock.patch.dict(os.environ, env, clear=True):
            for changes, expected in cases:
                with self.subTest(changes=changes), mock.patch.object(llm, 'SETTINGS', dict(config.DEFAULTS, **changes)):
                    self.assertEqual(llm.profile()['MOE_CACHE_MIB'], expected)

    def test_long_context_reserves_one_full_million_token_slot(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith('LLM_')}
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(llm, 'SETTINGS', dict(llm.DEFAULTS, context=1048576, images=True)):
            self.assertEqual(llm.context_value('1M'), 1048576)
            info = llm.profile()
            self.assertEqual((info['CTX'], info['SLOTS'], info['MMPROJ']), ('1048576', '1', '1'))
            self.assertEqual(config.validate(dict(config.DEFAULTS, context=1048576))['context'], 1048576)
            self.assertEqual((info['MOE_CACHE_MIB'], info['UBATCH'], info['DYN']), ('68864', '512', '80'))
            handler = object.__new__(llm.Handler)
            self.assertEqual(handler.models()['data'][0]['context_length'], 1048576)
            self.assertEqual(handler.models()['data'][0]['modalities']['input'], ['text', 'image'])

    def test_tuned_256k_preserves_two_slots_and_other_feature_profiles(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith('LLM_')}
        cases = [
            ({}, ('73000', '1024', '166')),
            ({'model': 'abiray'}, ('70430', '2048', '110')),
            ({'uncensor': False}, ('70430', '2048', '110')),
            ({'images': False}, ('70430', '2048', '110')),
            ({'draft': False}, ('70430', '2048', '110')),
            ({'context': 131072}, ('70430', '2048', '110')),
        ]
        with mock.patch.dict(os.environ, env, clear=True):
            for changes, expected in cases:
                settings = dict(config.DEFAULTS, context=262144)
                settings.update(changes)
                with self.subTest(changes=changes), mock.patch.object(llm, 'SETTINGS', settings):
                    info = llm.profile()
                    self.assertEqual(tuple(info[key] for key in ('MOE_CACHE_MIB', 'UBATCH', 'DYN')), expected)
                    self.assertEqual((info['SLOTS'], info['DMA']), ('2', '0'))

    def test_tuned_512k_preserves_one_slot_and_scopes_dynamic_cache(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith('LLM_')}
        cases = [
            ({}, ('73000', '512', '166')),
            ({'model': 'abiray'}, ('64000', '512', '50')),
            ({'uncensor': False}, ('64000', '512', '50')),
            ({'images': False}, ('64000', '512', '50')),
            ({'draft': False}, ('64000', '512', '50')),
        ]
        with mock.patch.dict(os.environ, env, clear=True):
            for changes, expected in cases:
                settings = dict(config.DEFAULTS, context=524288)
                settings.update(changes)
                with self.subTest(changes=changes), mock.patch.object(llm, 'SETTINGS', settings):
                    info = llm.profile()
                    self.assertEqual(tuple(info[key] for key in ('MOE_CACHE_MIB', 'UBATCH', 'DYN')), expected)
                    self.assertEqual((info['CTX'], info['SLOTS'], info['DMA']), ('524288', '1', '0'))

    def test_million_index_chunk_reaches_boot_env_only_for_the_tuned_profile(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith('LLM_')}
        key = 'LLAMA_DSV41_IDX_CHUNK_MIB'
        cases = [({}, '128'), ({'context': 262144}, None), ({'context': 524288}, None),
                 ({'context': 786432}, None), ({'model': 'abiray'}, None),
                 ({'uncensor': False}, None), ({'images': False}, None), ({'draft': False}, None)]
        with mock.patch.dict(os.environ, env, clear=True):
            for changes, expected in cases:
                settings = dict(config.DEFAULTS, context=1048576)
                settings.update(changes)
                with self.subTest(changes=changes), mock.patch.object(llm, 'SETTINGS', settings):
                    self.assertEqual(llm.boot_vars().get(key), expected)
            with mock.patch.object(llm, 'SETTINGS', dict(config.DEFAULTS, context=1048576)):
                self.assertIn(key + "='128'", llm.boot_env())
                with mock.patch.dict(os.environ, {'LLM_BOOT_ENV': key + '=64'}):
                    self.assertEqual(llm.boot_vars()[key], '64')

    def test_first_request_uses_settings_applied_at_wake(self):
        handler = object.__new__(llm.Handler)
        manager = mock.Mock()
        manager.port = 12345
        manager.tunnel_failed.return_value = False
        handler.server = mock.Mock(mgr=manager)
        handler.path, handler.command, handler.headers = '/v1/chat/completions', 'POST', {}
        handler.host_ok, handler.authed, handler.fail = mock.Mock(return_value=True), mock.Mock(return_value=True), mock.Mock()
        forwarded = []

        def on_acquire(body):
            llm.SETTINGS['effort'] = 'high'
            handler.forward(body)

        def capture(port, method, path, body, headers, timeout):
            forwarded.append(json.loads(body))
            raise OSError('test upstream')

        handler.held = on_acquire
        with mock.patch.object(llm, 'SETTINGS', dict(llm.SETTINGS, effort='low')), \
                mock.patch.object(llm, 'http_call', side_effect=capture):
            handler.route(json.dumps({'model': 'deepseek-v4.1-flash-sec', 'messages': []}).encode())
        self.assertEqual(forwarded[0]['reasoning_effort'], 'high')

    def test_invalid_values_are_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'settings.json'
            config.write(path, dict(config.DEFAULTS))
            before = path.read_bytes()
            for key, value in [('idle_min', 1), ('max_hours', 100), ('model', 'other'),
                               ('context', 1), ('draft', 'false'), ('min_balance', -1)]:
                with self.subTest(key=key), self.assertRaises(ValueError):
                    config.write(path, dict(config.DEFAULTS, **{key: value}))
                self.assertEqual(path.read_bytes(), before)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_model_options_map_to_correct_boot_configuration(self):
        keys = ['LLM_MODEL', 'LLM_IMAGES', 'LLM_CONTEXT', 'LLM_PROFILE', 'UNCENSOR', 'DRAFT', 'LLM_BOOT_ENV']
        env = {k: v for k, v in os.environ.items() if k not in keys}
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(llm, 'SETTINGS', dict(llm.SETTINGS)), \
                mock.patch.object(llm, 'CHILD_ENV', dict(llm.CHILD_ENV)), \
                mock.patch.object(llm, 'IDLE_MIN'), mock.patch.object(llm, 'MAX_HOURS'), \
                mock.patch.object(llm, 'MIN_BALANCE'):
            controller.apply_settings(dict(config.DEFAULTS, model='abiray', draft=False))
            values = controller.boot_vars()
            self.assertEqual((values['MODEL_VARIANT'], values['UNCENSOR'], values['DRAFT']), ('abiray', '0', '0'))
            controller.apply_settings(dict(config.DEFAULTS, model='original', uncensor=True, draft=True))
            values = controller.boot_vars()
            self.assertEqual((values['MODEL_VARIANT'], values['UNCENSOR'], values['DRAFT']), ('original', '1', '1'))

    def test_pending_changes_wait_for_next_launch_and_snapshot_for_recovery(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith('LLM_')}
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(controller, 'DESIRED', Path(root) / 'desired.json'), \
                mock.patch.object(controller, 'ACTIVE', Path(root) / 'active.json'), \
                mock.patch.object(llm, 'load_session', return_value=None), \
                mock.patch.object(llm, 'SETTINGS', dict(llm.SETTINGS)), \
                mock.patch.object(llm, 'CHILD_ENV', dict(llm.CHILD_ENV)), \
                mock.patch.object(llm, 'IDLE_MIN'), mock.patch.object(llm, 'MAX_HOURS'), \
                mock.patch.object(llm, 'MIN_BALANCE'), mock.patch.object(llm, 'boot_vars', llm.boot_vars), \
                mock.patch.object(llm.Handler, 'models', llm.Handler.models), \
                mock.patch.object(llm.Handler, 'held', llm.Handler.held), \
                mock.patch.object(llm.Manager, '_run_start', llm.Manager._run_start), \
                mock.patch.object(llm.Manager, '_begin', autospec=True) as begin:
            config.write(controller.DESIRED, dict(config.DEFAULTS, model='abiray'))
            controller.install()
            manager = llm.Manager()
            manager._begin(True)
            begin.assert_called_once_with(manager, True)
            self.assertEqual(config.read(controller.ACTIVE)['model'], 'abiray')
            config.write(controller.DESIRED, dict(config.DEFAULTS, model='original', idle_min=2))
            self.assertEqual(llm.SETTINGS['model'], 'abiray')
            self.assertEqual(llm.IDLE_MIN, 10)
            manager._begin(True)
            self.assertEqual(llm.SETTINGS['model'], 'original')
            self.assertEqual(llm.IDLE_MIN, 2)
            self.assertEqual(config.read(controller.ACTIVE)['model'], 'original')

    def test_restart_adopts_running_snapshot_not_pending_selection(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith('LLM_')}
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(controller, 'DESIRED', Path(root) / 'desired.json'), \
                mock.patch.object(controller, 'ACTIVE', Path(root) / 'active.json'), \
                mock.patch.object(llm, 'load_session', return_value={'model_variant': 'abiray'}), \
                mock.patch.object(llm, 'SETTINGS', dict(llm.SETTINGS)), \
                mock.patch.object(llm, 'CHILD_ENV', dict(llm.CHILD_ENV)), \
                mock.patch.object(llm, 'IDLE_MIN'), mock.patch.object(llm, 'MAX_HOURS'), \
                mock.patch.object(llm, 'MIN_BALANCE'), mock.patch.object(llm, 'boot_vars', llm.boot_vars), \
                mock.patch.object(llm.Handler, 'models', llm.Handler.models), \
                mock.patch.object(llm.Handler, 'held', llm.Handler.held), \
                mock.patch.object(llm.Manager, '_run_start', llm.Manager._run_start), \
                mock.patch.object(llm.Manager, '_begin', autospec=True):
            config.write(controller.ACTIVE, dict(config.DEFAULTS, model='abiray', idle_min=2))
            config.write(controller.DESIRED, dict(config.DEFAULTS, model='original', idle_min=30))
            controller.install()
            self.assertEqual(llm.SETTINGS['model'], 'abiray')
            self.assertEqual(llm.IDLE_MIN, 2)
            self.assertEqual(llm.CHILD_ENV['LLM_IDLE_MIN'], '2')

    def test_model_catalog_uses_desired_when_off_and_active_when_running(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith('LLM_')}
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(controller, 'DESIRED', Path(root) / 'desired.json'), \
                mock.patch.object(llm, 'load_session', return_value=None), \
                mock.patch.object(llm, 'SETTINGS', dict(llm.SETTINGS)), \
                mock.patch.object(llm, 'CHILD_ENV', dict(llm.CHILD_ENV)), \
                mock.patch.object(llm, 'IDLE_MIN'), mock.patch.object(llm, 'MAX_HOURS'), \
                mock.patch.object(llm, 'MIN_BALANCE'), mock.patch.object(llm, 'boot_vars', llm.boot_vars), \
                mock.patch.object(llm.Handler, 'models', llm.Handler.models), \
                mock.patch.object(llm.Handler, 'held', llm.Handler.held), \
                mock.patch.object(llm.Manager, '_run_start', llm.Manager._run_start), \
                mock.patch.object(llm.Manager, '_begin', autospec=True):
            config.write(controller.DESIRED, dict(config.DEFAULTS, context=524288))
            controller.install()
            manager = llm.Manager()
            handler = object.__new__(llm.Handler)
            handler.server = mock.Mock(mgr=manager)
            self.assertEqual(handler.models()['data'][0]['context_length'], 524288)
            manager.state = 'ready'
            config.write(controller.DESIRED, dict(config.DEFAULTS, context=262144, images=False))
            active = handler.models()['data'][0]
            self.assertEqual(active['context_length'], 524288)
            self.assertEqual(active['modalities']['input'], ['text', 'image'])
            manager.state = 'off'
            desired = handler.models()['data'][0]
            self.assertEqual(desired['context_length'], 262144)
            self.assertEqual(desired['modalities']['input'], ['text'])
            config.write(controller.DESIRED, dict(config.DEFAULTS, context=1048576))
            self.assertEqual(handler.models()['data'][0]['context_length'], 1048576)
            self.assertEqual(config.read_desired(controller.DESIRED)['context'], 1048576)


class PanelTests(unittest.TestCase):
    def test_logs_endpoint_tails_and_redacts_without_contacting_colab(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.object(panel, 'STATE', Path(root)), \
                mock.patch.object(panel, 'COLAB', '/nonexistent/colab'):
            state = Path(root)
            (state / 'api_key').write_text('my-secret-controller-key')
            lines = ['entry %d' % i for i in range(350)]
            lines += ['Bearer example-secret', 'api_key=example-secret', 'hf_abcdefghijklmnop', '"token":"json-secret"',
                      'my-secret-controller-key', 'https://example.test/data?token=private', '<script>not html</script>']
            (state / 'llm.log').write_text('\n'.join(lines))
            server = panel.ThreadingHTTPServer(('127.0.0.1', 0), panel.Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
            try:
                c = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=5)
                c.request('GET', '/api/logs'); response = c.getresponse(); data = json.loads(response.read()); c.close()
                self.assertEqual(response.status, 200)
                self.assertTrue(data['available']); self.assertTrue(data['truncated'])
                self.assertLessEqual(len(data['text'].splitlines()), panel.LOG_TAIL_LINES)
                self.assertNotIn('entry 0\n', data['text'])
                for secret in ('example-secret', 'hf_abcdefghijklmnop', 'json-secret', 'my-secret-controller-key', 'token=private'):
                    self.assertNotIn(secret, data['text'])
                self.assertIn('entry 349', data['text'])
                self.assertIn('<script>not html</script>', data['text'])
                self.assertIn('textContent', panel.PAGE.decode())
            finally:
                server.shutdown(); server.server_close(); thread.join(timeout=5)

    def test_log_snapshot_limits_bytes_and_handles_absent_file(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.object(panel, 'STATE', Path(root)):
            self.assertEqual(panel.log_snapshot(), {'available': False, 'text': '', 'truncated': False})
            (Path(root) / 'llm.log').write_text('old\n' + 'x' * panel.LOG_TAIL_BYTES + '\nlatest')
            data = panel.log_snapshot()
            self.assertTrue(data['available']); self.assertTrue(data['truncated'])
            self.assertEqual(data['text'], 'latest')
            (Path(root) / 'llm.log').write_text('y' * (panel.LOG_TAIL_BYTES + 20))
            self.assertEqual(len(panel.log_snapshot()['text']), panel.LOG_TAIL_BYTES)

    def test_settings_endpoint_and_nonsecret_status(self):
        with tempfile.TemporaryDirectory() as root, \
                mock.patch.object(panel, 'DESIRED', Path(root) / 'settings.json'), \
                mock.patch.object(panel, 'ACTIVE', Path(root) / 'active.json'), \
                mock.patch.object(panel, 'controller_status', return_value={'state': 'off', 'has_error': False}), \
                mock.patch.object(panel, 'balance_snapshot', return_value={'value': 20}):
            server = panel.ThreadingHTTPServer(('127.0.0.1', 0), panel.Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                c = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=5)
                c.request('GET', '/api/status'); r = c.getresponse(); data = json.loads(r.read())
                self.assertEqual(r.status, 200)
                self.assertEqual(data['desired']['model'], 'original')
                self.assertEqual(data['desired']['context'], 524288)
                self.assertTrue(data['desired']['uncensor'])
                self.assertEqual(data['balance']['value'], 20)
                c.close()
                value = dict(config.DEFAULTS, model='original', idle_min=2, context=1048576)
                c = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=5)
                c.request('POST', '/api/config', body=json.dumps(value), headers={'Content-Type': 'application/json'})
                r = c.getresponse(); r.read(); self.assertEqual(r.status, 200); c.close()
                self.assertEqual(config.read(panel.DESIRED), value)
                c = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=5)
                c.request('POST', '/api/config', body=json.dumps(dict(value, idle_min=1)))
                r = c.getresponse(); r.read(); self.assertEqual(r.status, 400); c.close()
                self.assertEqual(config.read(panel.DESIRED), value)
            finally:
                server.shutdown(); server.server_close(); thread.join(timeout=5)


if __name__ == '__main__':
    unittest.main()
