"""Loopback-only Colab settings/status service, proxied by Magpie's page token."""
import http.client
import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from integrations.magpie_colab import config  # noqa: E402
from llm import output_policy  # noqa: E402

STATE = Path(os.environ.get('LLM_STATE', '/var/lib/colab-deepseek/state'))
DESIRED = STATE / 'magpie-settings.json'
ACTIVE = STATE / 'magpie-active-settings.json'
PAGE = Path(__file__).with_name('panel.html').read_bytes()
COLAB = os.environ.get('LLM_COLAB', '/opt/colab-deepseek/venv312/bin/colab')
PORT = int(os.environ.get('COLAB_PANEL_PORT', '18082'))
balance_lock = threading.Lock()
balance = {'checked_at': 0.0, 'running': False, 'value': None, 'rate': None, 'assignments': None}
LOG_TAIL_BYTES = 65536
LOG_TAIL_LINES = 300


def log_snapshot():
    """Read a bounded tail of the local controller log; never contact or wake Colab."""
    try:
        with (STATE / 'llm.log').open('rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            offset = max(0, size - LOG_TAIL_BYTES)
            f.seek(offset)
            data = f.read(LOG_TAIL_BYTES)
        if offset and b'\n' in data:
            data = data.partition(b'\n')[2]  # discard a partial first line
        lines = data.decode('utf-8', 'replace').splitlines()
        truncated = bool(offset or len(lines) > LOG_TAIL_LINES)
        text = '\n'.join(lines[-LOG_TAIL_LINES:])
        # The controller also redacts its own key before writing; apply a second
        # pass before showing any log through the authenticated page service.
        try:
            key = (STATE / 'api_key').read_text().strip()
            if key:
                text = text.replace(key, '[REDACTED]')
        except OSError:
            pass
        text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', text)
        text = re.sub(r'(?i)(\bBearer\s+)[^\s,;]+', r'\1[REDACTED]', text)
        text = re.sub(r'''(?i)(\b(?:api[_-]?key|(?:hf[_-]?)?token|access[_-]?token|refresh[_-]?token|password|secret)["']?\s*[:=]\s*["']?)[^\s,;\]}"']+''',
                      r'\1[REDACTED]', text)
        text = re.sub(r'\b(?:hf_|sk-|ya29\.)[A-Za-z0-9_.-]{8,}', '[REDACTED]', text)
        text = re.sub(r'(https?://[^\s?#]+)\?[^\s]*', r'\1?[REDACTED]', text)
        return {'available': True, 'text': text, 'truncated': truncated}
    except OSError:
        return {'available': False, 'text': '', 'truncated': False}


def refresh_balance():
    try:
        p = subprocess.run([COLAB, '--auth', 'oauth2', 'usage'], stdin=subprocess.DEVNULL,
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=35)
        def find(pattern):
            m = re.search(pattern, p.stdout, re.I)
            return float(m.group(1)) if m and p.returncode == 0 else None
        value = find(r'balance:\s*([0-9.]+)')
        rate = find(r'rate:\s*([0-9.]+)')
        assignments = find(r'assignments:\s*([0-9.]+)')
    except (OSError, subprocess.SubprocessError):
        value, rate, assignments = None, None, None
    with balance_lock:
        balance.update(checked_at=time.time(), running=False, value=value, rate=rate, assignments=assignments)


def balance_snapshot():
    with balance_lock:
        if not balance['running'] and time.time() - balance['checked_at'] > 60:
            balance['running'] = True
            threading.Thread(target=refresh_balance, daemon=True).start()
        return {key: balance[key] for key in ('checked_at', 'value', 'rate', 'assignments')}


def controller_status():
    try:
        key = (STATE / 'api_key').read_text().strip()
        c = http.client.HTTPConnection('127.0.0.1', 18080, timeout=4)
        c.request('GET', '/_llm/status', headers={'Authorization': 'Bearer ' + key})
        r = c.getresponse()
        body = json.loads(r.read())
        c.close()
        if r.status != 200:
            raise ValueError('controller unavailable')
        # Never expose the controller's raw errors or credentials to the page.
        keys = ('state', 'stage', 'active', 'inflight', 'max_inflight', 'decode_slots', 'served', 'ran', 'cu', 'idle_s',
                'idle_stop_min', 'max_hours', 'started', 'last_stop')
        result = {k: body.get(k) for k in keys}
        result['has_error'] = bool(body.get('last_error'))
        return result
    except (OSError, ValueError, KeyError, http.client.HTTPException):
        return {'state': 'unavailable', 'has_error': True}


class Handler(BaseHTTPRequestHandler):
    server_version = 'ColabPanel'
    sys_version = ''

    def log_message(self, *args):
        pass

    def send(self, status, data, content_type='application/json; charset=utf-8'):
        body = data if isinstance(data, bytes) else json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Access-Control-Allow-Origin', 'null')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send(200, b'')

    def do_GET(self):
        path = self.path.split('?', 1)[0]
        if path == '/':
            return self.send(200, PAGE, 'text/html; charset=utf-8')
        if path == '/api/logs':
            return self.send(200, log_snapshot())
        if path != '/api/status':
            return self.send(404, {'error': 'not found'})
        try:
            desired = config.read_desired(DESIRED)
            current = controller_status()
            active = config.read(ACTIVE, default=False) if current['state'] not in ('off', 'unavailable') else None
            return self.send(200, {'desired': desired, 'active_settings': active,
                                   'output_profiles': {str(ctx): output_policy(ctx) for ctx in config.CHOICES['context']},
                                   'pending': active is not None and active != desired,
                                   'controller': current, 'balance': balance_snapshot()})
        except (OSError, ValueError, json.JSONDecodeError):
            return self.send(503, {'error': 'settings unavailable'})

    def do_POST(self):
        if self.path.split('?', 1)[0] != '/api/config':
            return self.send(404, {'error': 'not found'})
        try:
            n = int(self.headers.get('Content-Length', '0'))
            if not 0 < n <= 16384:
                raise ValueError('invalid request size')
            body = json.loads(self.rfile.read(n))
            settings = config.write(DESIRED, body)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return self.send(400, {'error': str(exc)[:120]})
        return self.send(200, {'saved': True, 'desired': settings})


if __name__ == '__main__':
    ThreadingHTTPServer(('127.0.0.1', PORT), Handler).serve_forever()
