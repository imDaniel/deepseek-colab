#!/usr/bin/env python3
"""Restore a verified Colab G4 build from a pinned public release asset."""
import hashlib
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import time

import requests

URL = ('https://github.com/imDaniel/deepseek-colab/releases/download/'
       'v0.2.0/'
       'llama-b11963-sm120-cuda13-test.tar.gz')
SHA256 = 'fe01a69712365f81331b71e9e1409f27182a92b808fcbe123df5f1c48de10aa8'
SIZE = 57044889
SOURCE_COMMIT = '39febd2adb947b70da35e95f9351b3bf58ad170a'
REPORT = Path('/content/res/github-cache-report.json')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def command(args, env=None):
    return subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, env=env, timeout=45, check=True).stdout


def main():
    started = time.monotonic()
    stage = Path(tempfile.mkdtemp(prefix='github-cache-', dir='/content'))
    phase = 'download'
    try:
        archive = stage / 'cache.tar.gz'
        h = hashlib.sha256()
        total = 0
        with requests.get(URL, stream=True, timeout=(15, 30)) as response:
            response.raise_for_status()
            with archive.open('xb') as output:
                for chunk in response.iter_content(4 * 1024 * 1024):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > SIZE or time.monotonic() - started > 120:
                        raise ValueError('download size or time limit exceeded')
                    h.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
        downloaded = time.monotonic() - started
        if total != SIZE or h.hexdigest() != SHA256:
            raise ValueError('pinned archive size or checksum mismatch')

        phase = 'compatibility'
        with tarfile.open(archive, 'r:gz') as bundle:
            manifest = json.load(bundle.extractfile('manifest.json'))
            patches = Path('/content/vm/patches')
            files = {p.name: digest(p) for p in sorted(patches.glob('*.py'))
                     if not p.name.startswith('._')}
            checks = {
                'commit': manifest['commit'] == SOURCE_COMMIT,
                'arch': manifest['cuda_architecture'] == '120',
                'series': manifest['patch_series_sha256'] == digest(patches / 'series'),
                'build_script': manifest['build_script_sha256'] == digest(Path('/content/vm/build.sh')),
                'patches': manifest['patches'] == files,
                'cuda': manifest['nvcc'] == command(['/usr/local/cuda/bin/nvcc', '--version']).splitlines()[-1],
                'os': manifest['os_release'] == platform.platform(),
                'libc': manifest['libc'] == list(platform.libc_ver()),
                'gpu_driver': manifest['build_gpu'] == command([
                    'nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader']).strip(),
            }
            if not all(checks.values()):
                raise ValueError('incompatible fields: ' + ','.join(k for k, v in checks.items() if not v))
            phase = 'extract'
            bundle.extractall(stage / 'unpacked', filter='data')

        phase = 'dependencies'
        binary_dir = stage / 'unpacked' / 'bin'
        env = dict(os.environ, LD_LIBRARY_PATH=str(binary_dir) + ':/usr/local/cuda/lib64:/usr/lib64-nvidia')
        ldd = command(['ldd', str(binary_dir / 'llama-server')], env)
        if 'not found' in ldd:
            raise ValueError('required shared library not found')
        command([str(binary_dir / 'llama-server'), '--version'], env)
        destination = Path('/content/llama/bin')
        if destination.exists():
            raise ValueError('binary destination already exists')
        os.rename(binary_dir, destination)
        elapsed = time.monotonic() - started
        REPORT.write_text(json.dumps({'mode': 'cache', 'seconds': round(elapsed, 3),
                                      'download_seconds': round(downloaded, 3),
                                      'bytes': total, 'checks': checks, 'sha256': SHA256}))
        print('GITHUB_CACHE_RESTORED total=%.2fs download=%.2fs bytes=%d' %
              (elapsed, downloaded, total), flush=True)
        return 0
    except Exception as exc:
        # Do not print network exception messages: redirects can contain signed URLs.
        detail = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
        REPORT.write_text(json.dumps({'mode': 'fallback', 'seconds': round(time.monotonic() - started, 3),
                                      'phase': phase, 'reason': detail}))
        print('GITHUB_CACHE_FALLBACK phase=%s reason=%s' % (phase, detail), flush=True)
        return 1
    finally:
        shutil.rmtree(stage, ignore_errors=True)


if __name__ == '__main__':
    raise SystemExit(main())
