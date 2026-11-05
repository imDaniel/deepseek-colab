"""Validated, atomic settings shared by the local controller and Magpie page."""
import json
import os
import tempfile
from pathlib import Path

DEFAULTS = {
    'idle_min': 10, 'max_hours': 6, 'model': 'abiray', 'images': True,
    'context': 32768, 'effort': 'max', 'uncensor': True, 'draft': True,
    'min_balance': 5,
}
CHOICES = {
    'idle_min': (2, 10, 30, 60), 'max_hours': (1, 2, 6, 12, 24),
    'model': ('original', 'abiray'), 'context': (32768, 65536, 131072, 262144, 524288, 1048576),
    'effort': ('low', 'high', 'max'),
}


def validate(value):
    if not isinstance(value, dict) or set(value) != set(DEFAULTS):
        raise ValueError('settings fields do not match the supported options')
    for key, options in CHOICES.items():
        if isinstance(value[key], bool) or value[key] not in options:
            raise ValueError('invalid ' + key)
    for key in ('images', 'uncensor', 'draft'):
        if not isinstance(value[key], bool):
            raise ValueError('invalid ' + key)
    number = value['min_balance']
    if isinstance(number, bool) or not isinstance(number, (int, float)) or not 0 <= number <= 100000:
        raise ValueError('invalid min_balance')
    return {key: value[key] for key in DEFAULTS}


def read(path, default=True):
    try:
        with Path(path).open(encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        if default:
            return dict(DEFAULTS)
        raise
    return validate(data)


def write(path, value):
    value = validate(value)
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.settings-', dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return value
