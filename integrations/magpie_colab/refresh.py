"""Keep Magpie's model catalog aligned with the controller's effective settings.

Run as the same unprivileged user and environment as Magpie, not as the Colab
controller. Reads the local page API and invokes Magpie only when its model
capabilities change; a failed refresh is retried without starting a GPU.
"""
import json
import os
import subprocess
import time
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

PANEL = os.environ.get('COLAB_PANEL_URL', 'http://127.0.0.1:18082/api/status')
MAGPIE = os.environ.get('MAGPIE_BIN', '/var/lib/magpie/bin/magpie')
PROVIDER = os.environ.get('COLAB_MAGPIE_PROVIDER', 'colab-ds41')


def management_key():
    key = os.environ.get('MAGPIE_WEB_KEY')
    service = os.environ.get('COLAB_MAGPIE_SERVICE')
    if not key and service:
        # A catalog follower can reuse its same-user Magpie process credentials
        # without copying a management secret into another environment file.
        pid = subprocess.check_output(['systemctl', 'show', service, '-p', 'MainPID', '--value'],
                                      text=True, timeout=5).strip()
        if not pid.isdigit() or int(pid) <= 0:
            raise ValueError('Magpie is not running')
        values = Path('/proc/' + pid + '/environ').read_bytes().split(b'\0')
        key = next((v.split(b'=', 1)[1].decode() for v in values if v.startswith(b'MAGPIE_WEB_KEY=')), None)
        if not key:
            raise ValueError('Magpie management key unavailable')
    return key


def refresh_models():
    key = management_key()
    if key:
        base = os.environ.get('COLAB_MAGPIE_WEB_URL', 'http://127.0.0.1:3430').rstrip('/')
        url = urlsplit(base)
        if url.scheme != 'http' or url.hostname not in ('127.0.0.1', 'localhost', '::1') or url.username or url.password or url.path or url.query or url.fragment:
            raise ValueError('COLAB_MAGPIE_WEB_URL must be a loopback HTTP origin')
        request = Request(base + '/api/provider/models', data=json.dumps({'id': PROVIDER}).encode(),
                          headers={'Cookie': 'magpie_web_%d=%s' % (url.port or 80, key), 'Content-Type': 'application/json'})
        with urlopen(request, timeout=45) as response:
            response.read()
            return response.status == 200
    # Standalone CLI processes do not invalidate a running gateway's in-memory
    # catalog. Keep this fallback for installations without the management key.
    p = subprocess.run([MAGPIE, 'provider', 'refresh', PROVIDER],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       stdin=subprocess.DEVNULL, timeout=45)
    return p.returncode == 0


def effective(status):
    state = (status.get('controller') or {}).get('state')
    if state in ('starting', 'ready', 'stopping'):
        return status.get('active_settings')
    if state == 'off':
        return status.get('desired')
    return None


def signature(settings):
    return tuple(settings.get(k) for k in ('context', 'images', 'effort', 'model', 'uncensor'))


def current_signature():
    model = os.environ.get('COLAB_UPSTREAM_MODEL')
    if not model:
        with urlopen(PANEL, timeout=5) as response:
            settings = effective(json.load(response))
        return signature(settings) if settings is not None else None
    root = os.environ.get('XDG_CONFIG_HOME', os.path.expanduser('~/.config'))
    path = Path(os.environ.get('MAGPIE_PROVIDER_CONFIG', os.path.join(root, 'magpie', 'providers.json')))
    provider = next(p for p in json.loads(path.read_text())['providers'] if p['id'] == PROVIDER)
    base = provider['chat'].rstrip('/')
    url = urlsplit(base)
    if (url.scheme != 'https' and not (url.scheme == 'http' and url.hostname in ('127.0.0.1', 'localhost', '::1'))) or url.username or url.password or url.query or url.fragment:
        raise ValueError('upstream must use HTTPS or loopback HTTP')
    request = Request(base + '/models', headers={'Authorization': 'Bearer ' + provider['key']})
    with urlopen(request, timeout=10) as response:
        row = next(m for m in json.load(response)['data'] if m['id'] == model)
    for key in ('context_length', 'max_output_tokens'):
        if type(row.get(key)) is not int or row[key] <= 0:
            raise ValueError('upstream model capacity unavailable')
    return json.dumps({k: row.get(k) for k in ('context_length', 'max_output_tokens', 'modalities',
                                               'reasoning', 'supported_reasoning_levels')}, sort_keys=True)


def main():
    last = None
    while True:
        try:
            current = current_signature()
            if current is not None and current != last:
                if refresh_models():
                    last = current
                    print('Magpie model capabilities refreshed', flush=True)
                else:
                    print('Magpie model refresh failed; will retry', flush=True)
        except (OSError, ValueError, KeyError, StopIteration, subprocess.SubprocessError):
            print('Magpie model refresh unavailable; will retry', flush=True)
        time.sleep(10)


if __name__ == '__main__':
    main()
