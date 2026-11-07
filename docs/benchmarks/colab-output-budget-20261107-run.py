#!/usr/bin/env python3
"""Fixed-task output-budget screen; run on an isolated, already-loaded Colab.

No raw reasoning or credentials are written to the results. Generated code is
saved as an untrusted artifact and must be graded in a separate OS sandbox.
"""
import argparse
import hashlib
import http.client
import json
import re
import subprocess
import threading
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--task', required=True)
p.add_argument('--out', required=True)
p.add_argument('--port', type=int, default=18081)
p.add_argument('--key-file', default='/content/secrets/key')
a = p.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
key = Path(a.key_file).read_text().strip()
prompt = Path(a.task).read_text()
phase = 'init'
stop = threading.Event()
rows = []
result = {'schema_version': 1, 'context': 524288, 'model': 'deepseek-v4.1-flash-sec',
          'effort': 'max', 'temperature': 0.6, 'top_p': 1, 'seed': 42,
          'task_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
          'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'transport': 'isolated Colab engine loopback', 'cases': [], 'complete': False}


def save():
    path = out / 'results.json'; temp = out / '.results.tmp'
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n'); temp.replace(path)


def api(path, body=None, timeout=20):
    c = http.client.HTTPConnection('127.0.0.1', a.port, timeout=timeout)
    c.request('GET' if body is None else 'POST', path, None if body is None else json.dumps(body).encode(),
              {'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    r = c.getresponse(); data = json.loads(r.read()); c.close()
    if r.status != 200: raise RuntimeError('HTTP %d on %s' % (r.status, path))
    return data


def sample():
    while not stop.is_set():
        try:
            lines = Path('/proc/meminfo').read_text().splitlines()
            host = int(next(x for x in lines if x.startswith('MemAvailable:')).split()[1])
            r = subprocess.run(['nvidia-smi', '--query-gpu=memory.free,memory.used,utilization.gpu,temperature.gpu',
                                '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
            values = [float(s.strip()) for s in r.stdout.strip().split(',')]
            row = {'at': time.time(), 'phase': phase, 'host_available_kib': host,
                   'gpu_free_mib': values[0], 'gpu_used_mib': values[1], 'gpu_util_pct': values[2], 'gpu_temp_c': values[3]}
            rows.append(row)
            with (out / 'resources.jsonl').open('a') as f: f.write(json.dumps(row) + '\n')
        except (OSError, ValueError, StopIteration, subprocess.SubprocessError):
            pass
        stop.wait(5)


def request(budget):
    # This model does not implement slot erasure. Prime the identical prefix
    # before every case so first-token latency is consistently a prefix hit.
    primed = api('/v1/chat/completions', {'model': result['model'],
                 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': 1,
                 'reasoning_effort': 'max', 'temperature': 0.6, 'top_p': 1, 'seed': 42,
                 'stream': False}, timeout=180)
    before = api('/slots')
    assert len(before) == 1 and not before[0]['is_processing']
    body = {'model': result['model'], 'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': budget, 'reasoning_effort': 'max', 'temperature': 0.6,
            'top_p': 1, 'seed': 42, 'stream': True, 'stream_options': {'include_usage': True}}
    c = http.client.HTTPConnection('127.0.0.1', a.port, timeout=180)
    t0 = time.monotonic(); first = visible = last = None
    c.request('POST', '/v1/chat/completions', json.dumps(body).encode(),
              {'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    r = c.getresponse(); usage = {}; timings = {}; text = []; digest = hashlib.sha256()
    reasoning_chars = 0; finish = None; done = False; error = None; saved_at = t0
    if r.status != 200: raise RuntimeError('generation HTTP %d' % r.status)
    while True:
        line = r.readline()
        if not line: break
        if time.monotonic() - t0 > 3600: raise TimeoutError('case exceeded one hour')
        if not line.startswith(b'data: '): continue
        data = line[6:].strip()
        if data == b'[DONE]': done = True; break
        event = json.loads(data)
        if event.get('error'): error = str(event['error']); break
        usage = event.get('usage') or usage; timings = event.get('timings') or timings
        for choice in event.get('choices', []):
            finish = choice.get('finish_reason') or finish
            delta = choice.get('delta') or {}
            think = delta.get('reasoning_content') or delta.get('reasoning') or ''
            content = delta.get('content') or ''
            if think or content:
                now = time.monotonic(); first = first or now; last = now
                if content and visible is None: visible = now
                reasoning_chars += len(think); text.append(content); digest.update((think + content).encode())
        now = time.monotonic()
        if now - saved_at >= 10:
            result['progress'] = {'budget': budget, 'elapsed_s': round(now-t0, 2),
                                  'reasoning_chars': reasoning_chars, 'visible_chars': sum(map(len, text))}
            save(); saved_at = now
    c.close(); elapsed = time.monotonic()-t0; answer = ''.join(text)
    (out / ('answer-%d.txt' % budget)).write_text(answer)
    blocks = re.findall(r'```(?:python|py)?\s*\n(.*?)```', answer, re.S)
    code = max(blocks, key=len) if blocks else (answer if answer.lstrip().startswith(('import ', 'from ', '#', 'class ')) else '')
    if code: (out / ('candidate-%d.py' % budget)).write_text(code)
    ct = usage.get('completion_tokens')
    samples = [x for x in rows if x['phase'] == str(budget)]
    return {'budget': budget, 'ok': done and error is None and first is not None, 'finish_reason': finish,
            'error': error, 'completion_tokens': ct, 'prompt_tokens': usage.get('prompt_tokens'),
            'cached_tokens': (usage.get('prompt_tokens_details') or {}).get('cached_tokens'),
            'elapsed_s': round(elapsed, 3), 'ttft_s': round(first-t0, 3) if first else None,
            'first_visible_s': round(visible-t0, 3) if visible else None,
            'decode_tps': round((ct-1)/(last-first), 3) if ct and first and last and last>first else None,
            'timings': timings, 'reasoning_chars': reasoning_chars, 'visible_chars': len(answer),
            'answer_sha256': hashlib.sha256(answer.encode()).hexdigest(), 'all_output_sha256': digest.hexdigest(),
            'candidate_sha256': hashlib.sha256(code.encode()).hexdigest() if code else None,
            'cache_condition': 'identical prefix primed before every case',
            'prime_completion_tokens': primed.get('usage', {}).get('completion_tokens'), 'resource_samples': len(samples),
            'min_host_available_kib': min(x['host_available_kib'] for x in samples) if samples else None,
            'min_gpu_free_mib': min(x['gpu_free_mib'] for x in samples) if samples else None}


thread = threading.Thread(target=sample, daemon=True); thread.start()
try:
    props = api('/props'); assert props['model_alias'] == result['model']
    assert props['default_generation_settings']['n_ctx'] == result['context']
    result['engine_build'] = props.get('build_info')
    result['chat_template_sha256'] = hashlib.sha256(props['chat_template'].encode()).hexdigest()
    for budget in (32768, 65536, 131072, 262144):
        phase = str(budget); result['phase'] = phase; save()
        row = request(budget); result['cases'].append(row); save()
        print(json.dumps({k:row[k] for k in ('budget','finish_reason','completion_tokens','elapsed_s','decode_tps','visible_chars')}), flush=True)
        if not row['ok']: raise RuntimeError('incomplete transport for budget %d' % budget)
    result['complete'] = True; result['phase'] = 'complete'; result.pop('progress', None); save()
except BaseException as exc:
    result['error'] = type(exc).__name__ + ': ' + str(exc)[:500]; save(); raise
finally:
    stop.set(); thread.join(6)
