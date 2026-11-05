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
    def test_catalog_refresh_tracks_active_settings_until_session_stops(self):
        active = dict(config.DEFAULTS, context=1048576)
        desired = dict(config.DEFAULTS, context=32768)
        status = {'controller': {'state': 'ready'}, 'active_settings': active, 'desired': desired}
        self.assertEqual(refresh.effective(status), active)
        status['controller']['state'] = 'off'
        self.assertEqual(refresh.effective(status), desired)
        status['controller']['state'] = 'unavailable'
        self.assertIsNone(refresh.effective(status))

    def test_long_context_reserves_one_full_million_token_slot(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith('LLM_')}
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(llm, 'SETTINGS', dict(llm.DEFAULTS, context=1048576, images=True)):
            self.assertEqual(llm.context_value('1M'), 1048576)
            info = llm.profile()
            self.assertEqual((info['CTX'], info['SLOTS'], info['MMPROJ']), ('1048576', '1', '1'))
            self.assertEqual(config.validate(dict(config.DEFAULTS, context=1048576))['context'], 1048576)
            handler = object.__new__(llm.Handler)
            self.assertEqual(handler.models()['data'][0]['context_length'], 1048576)
            self.assertEqual(handler.models()['data'][0]['modalities']['input'], ['text', 'image'])

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
            handler.route(json.dumps({'model': 'ds41', 'messages': []}).encode())
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
                mock.patch.object(llm.Manager, '_begin', autospec=True) as begin:
            config.write(controller.DESIRED, dict(config.DEFAULTS))
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
                mock.patch.object(llm.Manager, '_begin', autospec=True):
            config.write(controller.ACTIVE, dict(config.DEFAULTS, idle_min=2))
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
                mock.patch.object(llm.Manager, '_begin', autospec=True):
            config.write(controller.DESIRED, dict(config.DEFAULTS, context=1048576))
            controller.install()
            manager = llm.Manager()
            handler = object.__new__(llm.Handler)
            handler.server = mock.Mock(mgr=manager)
            self.assertEqual(handler.models()['data'][0]['context_length'], 1048576)
            manager.state = 'ready'
            config.write(controller.DESIRED, dict(config.DEFAULTS, context=32768, images=False))
            active = handler.models()['data'][0]
            self.assertEqual(active['context_length'], 1048576)
            self.assertEqual(active['modalities']['input'], ['text', 'image'])
            manager.state = 'off'
            desired = handler.models()['data'][0]
            self.assertEqual(desired['context_length'], 32768)
            self.assertEqual(desired['modalities']['input'], ['text'])


class PanelTests(unittest.TestCase):
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
                self.assertEqual(data['desired']['model'], 'abiray')
                self.assertEqual(data['balance']['value'], 20)
                c.close()
                value = dict(config.DEFAULTS, model='original', idle_min=2)
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
