"""Offline checks for model selection and VM download manifests."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
VM = ROOT / 'vm'
spec = importlib.util.spec_from_file_location('llm_model_test', ROOT / 'llm.py')
llm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(llm)


class ModelVariants(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, {}, clear=False)
        self.env.start()
        for k in ('LLM_MODEL', 'MODEL_VARIANT', 'UNCENSOR', 'LLM_BOOT_ENV', 'LLM_PROFILE', 'LLM_IMAGES', 'LLM_CONTEXT'):
            os.environ.pop(k, None)
        self.settings = mock.patch.dict(llm.SETTINGS, dict(llm.DEFAULTS), clear=True)
        self.settings.start()
        self.addCleanup(self.env.stop)
        self.addCleanup(self.settings.stop)

    def test_original_keeps_saved_uncensor_behavior(self):
        self.assertEqual(llm.boot_vars()['MODEL_VARIANT'], 'original')
        self.assertEqual(llm.boot_vars()['UNCENSOR'], '1')
        llm.SETTINGS['uncensor'] = False
        self.assertEqual(llm.boot_vars()['UNCENSOR'], '0')

    def test_abiray_skips_patch_and_rejects_conflicts(self):
        os.environ['LLM_MODEL'] = 'abiray'
        self.assertEqual((llm.boot_vars()['MODEL_VARIANT'], llm.boot_vars()['UNCENSOR']), ('abiray', '0'))
        os.environ['UNCENSOR'] = '1'
        with self.assertRaises(llm.StartError):
            llm.boot_vars()
        os.environ.pop('UNCENSOR')
        os.environ['LLM_BOOT_ENV'] = 'MODEL_VARIANT=original'
        with self.assertRaises(llm.StartError):
            llm.boot_vars()

    def test_cli_override_is_inherited_by_detached_process(self):
        llm.SETTINGS['model'] = 'original'
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(llm, 'STATE', temp), \
                mock.patch.object(llm, 'cmd_up') as up, mock.patch.object(sys, 'argv', ['llm.py', 'up', '--model', 'abiray']), \
                mock.patch.dict(llm.CHILD_ENV, {}, clear=False):
            llm.main()
            up.assert_called_once()
            self.assertEqual(llm.model_variant(), 'abiray')
            self.assertEqual(llm.CHILD_ENV['LLM_MODEL'], 'abiray')

    def test_invalid_variant_rejected_before_up(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(llm, 'STATE', temp), \
                mock.patch.object(llm, 'cmd_up') as up, mock.patch.object(sys, 'argv', ['llm.py', 'up', '--model', 'unknown']):
            with self.assertRaises(SystemExit):
                llm.main()
            up.assert_not_called()

    def test_saved_model_can_be_overridden(self):
        llm.SETTINGS['model'] = 'abiray'
        self.assertEqual(llm.model_variant(), 'abiray')
        os.environ['LLM_MODEL'] = 'original'
        self.assertEqual(llm.model_variant(), 'original')

    def test_jobs_pin_sources_and_exact_sizes(self):
        with open(VM / 'models.json') as f:
            profiles = json.load(f)
        with tempfile.TemporaryDirectory() as temp:
            for variant in ('original', 'abiray'):
                env = dict(os.environ, MODEL_VARIANT=variant)
                paths = {}
                for kind in ('backbone', 'engram', 'draft'):
                    path = Path(temp) / (variant + kind + '.json')
                    subprocess.run([sys.executable, '-I', str(VM / 'jobs.py'), kind, temp, str(path)],
                                   env=env, check=True, capture_output=True)
                    paths[kind] = json.loads(path.read_text())
                self.assertEqual([j['size'] for j in paths['backbone']], profiles[variant]['sizes'])
                self.assertTrue(all(j['url'].startswith(profiles[variant]['backbone_url']) for j in paths['backbone']))
                self.assertIn('ISTA-DASLab/', paths['engram'][0]['url'])
                self.assertIn('unsloth/', paths['draft'][0]['url'])
                if variant == 'abiray':
                    self.assertEqual(profiles[variant]['sizes'],
                                     [5780704, 39661282624, 39851072320, 39111455584, 19951637632])

    def test_shard_validation_uses_selected_sizes(self):
        with tempfile.TemporaryDirectory() as temp, open(VM / 'models.json') as f:
            profiles = json.load(f)
            # Sparse files test the actual VM checker without downloading hundreds of GB.
            name = 'DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-%05d-of-00006.gguf'
            for variant in ('original', 'abiray'):
                sizes = profiles[variant]['sizes'] + [110595290816]
                for i, size in enumerate(sizes, 1):
                    with open(Path(temp) / (name % i), 'wb') as shard:
                        shard.truncate(size)
                env = dict(os.environ, MODEL_VARIANT=variant)
                result = subprocess.run([sys.executable, '-I', str(VM / 'checksizes.py'), temp], env=env, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stdout)
                other = 'abiray' if variant == 'original' else 'original'
                result = subprocess.run([sys.executable, '-I', str(VM / 'checksizes.py'), temp],
                                        env=dict(env, MODEL_VARIANT=other), capture_output=True)
                self.assertNotEqual(result.returncode, 0)

    def test_running_different_variant_is_not_reused(self):
        os.environ['LLM_MODEL'] = 'abiray'
        with mock.patch.object(llm, 'setup', return_value=100), \
                mock.patch.object(llm, 'serve_running', return_value={'model_variant': 'original'}):
            with self.assertRaisesRegex(SystemExit, 'stop it first'):
                llm.cmd_up()

    def test_running_unknown_session_is_not_adopted_as_abiray(self):
        os.environ['LLM_MODEL'] = 'abiray'
        with mock.patch.object(llm.os.path, 'exists', return_value=True), \
                mock.patch.object(llm, 'load_session', return_value=None), \
                mock.patch.object(llm, 'session_alive', return_value=True), \
                mock.patch.object(llm, 'take_lock') as take_lock:
            with self.assertRaisesRegex(SystemExit, 'unknown model'):
                llm.serve()
            take_lock.assert_not_called()

    def test_vm_rejects_double_uncensor_before_work(self):
        env = dict(os.environ, MODEL_VARIANT='abiray', UNCENSOR='1', CTX='1', UBATCH='1', MOE_CACHE_MIB='1', DYN='1', MMPROJ='0')
        result = subprocess.run(['bash', str(VM / 'boot.sh')], env=env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('already includes the uncensor edit', result.stderr)


if __name__ == '__main__':
    unittest.main()
