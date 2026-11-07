"""Independent functional grader for the SQLite queue budget comparison.

Run inside an OS sandbox: generated Python is not trusted host code.
Usage: python check.py /path/to/generated.py
"""
import concurrent.futures
import importlib.util
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
import unittest

MODULE = str(Path(sys.argv.pop(1)).resolve())
spec = importlib.util.spec_from_file_location('candidate', MODULE)
candidate = importlib.util.module_from_spec(spec)
sys.modules['candidate'] = candidate
spec.loader.exec_module(candidate)


def process_claim(path, worker):
    q = candidate.JobQueue(path)
    out = []
    while True:
        lease = q.claim(worker, now=0, lease_seconds=100)
        if lease is None:
            return out
        out.append(lease['key'])
        if not q.ack(lease['key'], lease['token'], now=1, result=worker):
            raise AssertionError('fresh process lease was not authorized')


class QueueChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = str(Path(self.tmp.name) / 'jobs.sqlite')
        self.q = candidate.JobQueue(self.path)

    def add(self, key='a', **kw):
        self.assertTrue(self.q.enqueue(key, {'name': key}, **kw))

    def take(self, key='a', now=0, worker='w', seconds=10):
        value = self.q.claim(worker, now=now, lease_seconds=seconds)
        self.assertIsNotNone(value)
        self.assertEqual(value['key'], key)
        self.assertEqual(value['worker'], worker)
        return value

    def test_01_json_idempotency_and_reopen(self):
        payload = {'中文': [None, True, {'x': 1}]}
        self.assertTrue(self.q.enqueue('a', payload, priority=3, max_attempts=5))
        payload['中文'].append('changed')
        self.assertFalse(self.q.enqueue('a', {'replacement': True}, priority=99))
        row = candidate.JobQueue(self.path).get('a')
        self.assertEqual(row['payload'], {'中文': [None, True, {'x': 1}]})
        self.assertEqual((row['priority'], row['max_attempts'], row['attempts'], row['status']), (3, 5, 0, 'waiting'))
        for k in ('worker', 'token', 'lease_until', 'result', 'error'):
            self.assertIsNone(row[k])
        self.assertIsNone(self.q.get('absent'))

    def test_02_priority_delay_and_lexicographic_ties(self):
        for key, priority, at in [('z', 20, 20), ('d', 2, 0), ('c', 2, 0), ('b', 2, -1), ('a', 1, -5)]:
            self.add(key, priority=priority, run_at=at)
        for key in ('b', 'c', 'd', 'a'):
            self.take(key)
        self.assertIsNone(self.q.claim('w', now=0, lease_seconds=10))
        self.take('z', now=20)

    def test_03_expiry_exact_boundary_and_stale_token_fencing(self):
        self.add(); first = self.take()
        self.assertEqual(first['attempt'], 1)
        self.assertFalse(self.q.ack('a', first['token'], now=10))
        self.assertFalse(self.q.heartbeat('a', first['token'], now=10, lease_seconds=10))
        second = self.take(now=10, worker='next')
        self.assertNotEqual(first['token'], second['token'])
        self.assertEqual(second['attempt'], 2)
        self.assertFalse(self.q.ack('a', first['token'], now=11, result='stale'))
        self.assertFalse(self.q.fail('a', first['token'], now=11, error='stale'))
        self.assertTrue(self.q.ack('a', second['token'], now=11, result='fresh'))
        self.assertEqual(self.q.get('a')['result'], 'fresh')

    def test_04_retry_backoff_and_attempt_exhaustion(self):
        self.add(max_attempts=2); first = self.take()
        self.assertTrue(self.q.fail('a', first['token'], now=1, error='retry', retry_delay=5))
        self.assertEqual(self.q.get('a')['attempts'], 1)
        self.assertIsNone(self.q.claim('w', now=5, lease_seconds=10))
        second = self.take(now=6)
        self.assertTrue(self.q.fail('a', second['token'], now=7, error='terminal'))
        self.assertEqual(self.q.get('a')['status'], 'failed')
        self.assertEqual(self.q.get('a')['error'], 'terminal')
        self.assertIsNone(self.q.claim('w', now=99, lease_seconds=10))

    def test_05_expired_last_attempt_is_terminal(self):
        self.add(max_attempts=1); self.take()
        self.assertIsNone(self.q.claim('w', now=10, lease_seconds=10))
        row = self.q.get('a')
        self.assertEqual((row['status'], row['attempts']), ('failed', 1))
        self.assertIsNone(row['token'])

    def test_06_heartbeat_never_shortens_or_revives(self):
        self.add(); lease = self.take()
        self.assertTrue(self.q.heartbeat('a', lease['token'], now=1, lease_seconds=2))
        self.assertEqual(self.q.get('a')['lease_until'], 10)
        self.assertTrue(self.q.heartbeat('a', lease['token'], now=9, lease_seconds=10))
        self.assertEqual(self.q.get('a')['lease_until'], 19)
        self.assertFalse(self.q.heartbeat('a', lease['token'], now=19, lease_seconds=10))
        self.assertEqual(self.q.get('a')['attempts'], 1)

    def test_07_dependency_ordering_and_diamond(self):
        self.add('root'); self.add('a', depends_on=['root', 'root']); self.add('b', depends_on=['root'])
        self.add('leaf', depends_on=['b', 'a'], priority=100)
        root = self.take('root'); self.assertIsNone(self.q.claim('w', now=0, lease_seconds=10))
        self.assertTrue(self.q.ack('root', root['token'], now=1))
        a = self.take('a', now=1); b = self.take('b', now=1)
        self.assertTrue(self.q.ack('a', a['token'], now=2)); self.assertTrue(self.q.ack('b', b['token'], now=2))
        self.take('leaf', now=2)
        self.assertEqual(self.q.get('leaf')['depends_on'], ['a', 'b'])

    def test_08_cycle_edit_rollback_and_unknown_dependency(self):
        self.add('a'); self.add('b', depends_on=['a']); self.add('c', depends_on=['b'])
        with self.assertRaises(ValueError): self.q.set_dependencies('a', ['c'])
        self.assertEqual(self.q.get('a')['depends_on'], [])
        with self.assertRaises(ValueError): self.q.set_dependencies('b', ['missing'])
        self.assertEqual(self.q.get('b')['depends_on'], ['a'])
        with self.assertRaises(ValueError): self.q.enqueue('bad', {}, depends_on=['bad'])
        self.assertIsNone(self.q.get('bad'))
        self.assertFalse(self.q.set_dependencies('unknown', []))
        self.take('a'); self.assertFalse(self.q.set_dependencies('a', []))

    def test_09_terminal_dependency_propagates_transitively(self):
        self.add('a'); self.add('b', depends_on=['a']); self.add('c', depends_on=['b']); self.add('independent')
        self.assertTrue(self.q.cancel('a')); self.take('independent')
        for key in ('b', 'c'):
            row = self.q.get(key)
            self.assertEqual((row['status'], row['attempts']), ('failed', 0))

    def test_10_cancel_and_idempotent_ack(self):
        self.add(); lease = self.take(); self.assertTrue(self.q.cancel('a'))
        self.assertFalse(self.q.ack('a', lease['token'], now=1)); self.assertFalse(self.q.cancel('a'))
        self.add('b'); lease = self.take('b'); self.assertTrue(self.q.ack('b', lease['token'], now=1, result=[1, 2]))
        self.assertFalse(self.q.ack('b', lease['token'], now=1)); self.assertFalse(self.q.cancel('b'))

    def test_11_invalid_numbers_and_json_roll_back(self):
        for value in (True, float('nan'), float('inf'), '1'):
            with self.assertRaises(ValueError): self.q.enqueue('bad', {}, run_at=value)
            self.assertIsNone(self.q.get('bad'))
        for opts in ({'max_attempts': 0}, {'max_attempts': True}, {'priority': False}):
            with self.assertRaises(ValueError): self.q.enqueue('bad', {}, **opts)
        for payload in ({'x': float('nan')}, object()):
            with self.assertRaises(ValueError): self.q.enqueue('bad', payload)
        self.add(); lease = self.take()
        for seconds in (0, -1, True, math.inf):
            with self.assertRaises(ValueError): self.q.heartbeat('a', lease['token'], now=1, lease_seconds=seconds)
        with self.assertRaises(ValueError): self.q.ack('a', lease['token'], now=1, result={'x': math.inf})
        self.assertEqual(self.q.get('a')['status'], 'running')
        self.assertTrue(self.q.ack('a', lease['token'], now=1))

    def test_12_threads_share_one_instance_without_duplicate_leases(self):
        for n in range(40): self.add('%03d' % n)
        def work(n):
            out = []
            while True:
                lease = self.q.claim('thread-%s' % n, now=0, lease_seconds=100)
                if lease is None: return out
                out.append(lease['key'])
                if not self.q.ack(lease['key'], lease['token'], now=1): raise AssertionError('fresh lease rejected')
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            found = sum(list(pool.map(work, range(8))), [])
        self.assertEqual(len(found), 40); self.assertEqual(len(set(found)), 40)

    def test_13_processes_serialize_claims(self):
        for n in range(48): self.add('%03d' % n)
        context = multiprocessing.get_context('fork')
        with concurrent.futures.ProcessPoolExecutor(max_workers=4, mp_context=context) as pool:
            futures = [pool.submit(process_claim, self.path, 'process-%d' % n) for n in range(4)]
            found = sum([f.result(timeout=20) for f in futures], [])
        self.assertEqual(len(found), 48); self.assertEqual(len(set(found)), 48)
        self.assertTrue(all(self.q.get('%03d' % n)['status'] == 'succeeded' for n in range(48)))

    def test_14_reads_are_fresh_and_fencing_survives_reopen(self):
        self.add(); row = self.q.get('a'); row['payload']['name'] = 'changed'
        lease = self.take(); other = candidate.JobQueue(self.path)
        self.assertEqual(other.get('a')['payload']['name'], 'a')
        self.assertTrue(other.ack('a', lease['token'], now=1, result={'nested': [1]}))
        row = self.q.get('a'); row['result']['nested'].append(2)
        self.assertEqual(other.get('a')['result'], {'nested': [1]})

    def test_15_unknown_stale_and_terminal_operations_are_false(self):
        self.assertFalse(self.q.ack('missing', 'token', now=0))
        self.assertFalse(self.q.fail('missing', 'token', now=0, error='x'))
        self.assertFalse(self.q.heartbeat('missing', 'token', now=0, lease_seconds=1))
        self.assertFalse(self.q.cancel('missing'))
        self.add(); lease = self.take(); self.assertFalse(self.q.ack('a', 'wrong', now=0))
        self.assertTrue(self.q.ack('a', lease['token'], now=1))
        self.assertFalse(self.q.fail('a', lease['token'], now=2, error='x'))

    def test_16_replace_dependencies_and_failure_after_expiry(self):
        self.add('a', max_attempts=1); self.add('b', max_attempts=1); self.add('c', depends_on=['b'])
        self.assertTrue(self.q.set_dependencies('c', ['a', 'a']))
        self.assertEqual(self.q.get('c')['depends_on'], ['a'])
        self.take('a'); self.take('b')
        self.assertIsNone(self.q.claim('w', now=10, lease_seconds=10))
        self.assertEqual(self.q.get('c')['status'], 'failed')


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(QueueChecks)
    total = suite.countTestCases()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print('GRADER_JSON ' + json.dumps({'total': total, 'ran': result.testsRun,
          'passed': result.testsRun - len(result.failures) - len(result.errors),
          'failures': [case.id().split('.')[-1] for case, _ in result.failures],
          'errors': [case.id().split('.')[-1] for case, _ in result.errors]}))
    raise SystemExit(0 if result.wasSuccessful() else 1)
