"""Focused regression checks for safe cache-failure behavior."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('github_cache_restore', HERE / 'vm' / 'github-cache-restore.py')
restore = importlib.util.module_from_spec(spec)
spec.loader.exec_module(restore)


class FailureBehavior(unittest.TestCase):
    def test_network_failure_falls_back_without_logging_redirect_secret(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            stage = root / 'stage'
            stage.mkdir()
            report = root / 'report.json'
            output = io.StringIO()
            with mock.patch.object(restore, 'REPORT', report), \
                 mock.patch.object(restore.tempfile, 'mkdtemp', return_value=str(stage)), \
                 mock.patch.object(restore.requests, 'get', side_effect=restore.requests.ConnectionError('sensitive-redirect-value')), \
                 contextlib.redirect_stdout(output):
                self.assertEqual(restore.main(), 1)
            result = json.loads(report.read_text())
            self.assertEqual(result['mode'], 'fallback')
            self.assertEqual(result['phase'], 'download')
            self.assertNotIn('sensitive-redirect-value', report.read_text() + output.getvalue())
            self.assertFalse(stage.exists())


if __name__ == '__main__':
    unittest.main()
