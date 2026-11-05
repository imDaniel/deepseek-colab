"""Keep Magpie's model catalog aligned with the controller's effective settings.

Run as the same unprivileged user and environment as Magpie, not as the Colab
controller. Reads the local page API and invokes Magpie only when its model
capabilities change; a failed refresh is retried without starting a GPU.
"""
import json
import os
import subprocess
import time
from urllib.request import urlopen

PANEL = os.environ.get('COLAB_PANEL_URL', 'http://127.0.0.1:18082/api/status')
MAGPIE = os.environ.get('MAGPIE_BIN', '/var/lib/magpie/bin/magpie')
PROVIDER = os.environ.get('COLAB_MAGPIE_PROVIDER', 'colab-ds41')


def effective(status):
    state = (status.get('controller') or {}).get('state')
    if state in ('starting', 'ready', 'stopping'):
        return status.get('active_settings')
    if state == 'off':
        return status.get('desired')
    return None


def signature(settings):
    return tuple(settings.get(k) for k in ('context', 'images', 'effort', 'model', 'uncensor'))


def main():
    last = None
    while True:
        try:
            with urlopen(PANEL, timeout=5) as response:
                settings = effective(json.load(response))
            if settings is not None and signature(settings) != last:
                p = subprocess.run([MAGPIE, 'provider', 'refresh', PROVIDER],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   stdin=subprocess.DEVNULL, timeout=45)
                if p.returncode == 0:
                    last = signature(settings)
                    print('Magpie model capabilities refreshed', flush=True)
                else:
                    print('Magpie model refresh failed; will retry', flush=True)
        except (OSError, ValueError, subprocess.SubprocessError):
            print('Magpie model refresh unavailable; will retry', flush=True)
        time.sleep(10)


if __name__ == '__main__':
    main()
