"""Start the existing controller with an immutable settings snapshot per run."""
import os
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import llm  # noqa: E402
from integrations.magpie_colab import config  # noqa: E402

DESIRED = Path(llm.STATE) / 'magpie-settings.json'
ACTIVE = Path(llm.STATE) / 'magpie-active-settings.json'
STARTUP_TIMINGS = Path(llm.STATE) / 'startup-timings.json'
STARTUP_RETRY_S = 10
STARTUP_ESTIMATE_S = 180


def timing_key():
    return json.dumps([llm.SETTINGS.get(k) for k in ('model', 'uncensor', 'images', 'context', 'draft')])


def startup_timings():
    try:
        value = json.loads(STARTUP_TIMINGS.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def remember_startup(manager):
    if manager.state != 'ready' or not manager.begun:
        return
    duration = round(time.time() - manager.begun)
    if not 1 <= duration <= llm.BOOT_TIMEOUT + 600:
        return
    history = startup_timings()
    key = timing_key()
    samples = history.get(key, [])
    history[key] = (samples if isinstance(samples, list) else [])[-4:] + [duration]
    history = dict(list(history.items())[-24:])
    try:
        fd, name = tempfile.mkstemp(prefix='.startup-timings-', dir=STARTUP_TIMINGS.parent)
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(history, f)
            os.replace(name, STARTUP_TIMINGS)
        finally:
            if os.path.exists(name):
                os.unlink(name)
    except OSError:
        pass  # timing history must never prevent serving a model


def startup_status(manager):
    """Only public lifecycle facts; no session IDs, keys, balance or raw errors."""
    done = {key for key, at, label in manager.steps}
    created = 'allocated' in done
    state = manager.state
    if state == 'ready':
        phase = 'reconnecting'
    elif state == 'stopping':
        phase = 'stopping'
    elif not created:
        phase = 'allocating'
    elif 'uploaded' not in done:
        phase = 'uploading'
    elif 'model' not in done:
        phase = 'downloading'
    elif 'build' not in done:
        phase = 'building'
    else:
        phase = 'loading'
    samples = startup_timings().get(timing_key(), [])
    samples = [v for v in samples if isinstance(v, (int, float)) and not isinstance(v, bool) and 1 <= v <= llm.BOOT_TIMEOUT + 600] if isinstance(samples, list) else []
    total = round(sum(samples) / len(samples)) if samples else STARTUP_ESTIMATE_S
    elapsed = max(0, round(time.time() - manager.begun)) if manager.begun and state == 'starting' else 0
    remaining = max(1, total - elapsed) if state == 'starting' and elapsed < total else None
    return {'state': state, 'phase': phase, 'vm_created': created, 'model_ready': False,
            'elapsed_seconds': elapsed, 'estimated_remaining_seconds': remaining,
            'estimate_source': 'recent_successful_starts' if samples else 'initial_estimate',
            'estimate_is_approximate': True, 'retry_after_seconds': STARTUP_RETRY_S}


def retry_held(handler, body):
    """Wake once in the background; never hold a client or send a 200 SSE prelude."""
    manager = handler.server.mgr
    with manager.cond:
        if manager.enter():
            acquired, status, failed = True, None, False
        else:
            acquired = False
            failed = manager.closing or (manager.state == 'off' and time.time() < manager.fail_until)
            if manager.state == 'off' and not failed:
                manager._begin(True)
            status = startup_status(manager)
    if acquired:
        try:
            return handler.forward(body)
        finally:
            manager.leave()
    if failed:
        code, kind = 503, 'model_unavailable'
        message = 'Colab 暂时无法启动，请检查管理页状态和日志后重试。'
    else:
        code, kind = 429, 'model_starting'
        labels = {'allocating': '正在创建 Colab 虚拟机', 'uploading': '虚拟机已创建，正在上传启动文件',
                  'downloading': '虚拟机已创建，正在下载模型', 'building': '虚拟机已创建，正在准备推理引擎',
                  'loading': '虚拟机已创建，正在加载或预热模型', 'stopping': 'Colab 正在停止，请稍后重试',
                  'reconnecting': '模型连接正在恢复'}
        remaining = status['estimated_remaining_seconds']
        message = labels[status['phase']]
        message += '；预计还需约 %d 秒（估算）' % remaining if remaining is not None else '；剩余时间暂无法可靠估计'
        message += '；请在 %d 秒后重试。此条消息尚未执行。' % STARTUP_RETRY_S
    payload = {'error': {'code': kind, 'type': 'rate_limit_error' if code == 429 else 'unavailable_error',
                         'message': message}, 'server_status': status}
    return handler.send_json(code, payload, {'Retry-After': STARTUP_RETRY_S, 'Cache-Control': 'no-store'})


def apply_settings(settings):
    settings = config.validate(settings, legacy=True)
    llm.SETTINGS = dict(settings)
    llm.IDLE_MIN = float(settings['idle_min'])
    llm.MAX_HOURS = float(settings['max_hours'])
    llm.MIN_BALANCE = float(settings['min_balance'])
    # The watchdog is a separate process running llm.py. Pin it to the same
    # settings even if the user saves different settings while the GPU runs.
    llm.CHILD_ENV.update({
        'LLM_MODEL': settings['model'], 'LLM_IMAGES': '1' if settings['images'] else '0',
        'LLM_CONTEXT': str(settings['context']), 'LLM_IDLE_MIN': str(settings['idle_min']),
        'LLM_MAX_HOURS': str(settings['max_hours']), 'LLM_MIN_BALANCE': str(settings['min_balance']),
    })


original_boot_vars = llm.boot_vars


def boot_vars():
    values = original_boot_vars()
    values['DRAFT'] = '1' if llm.SETTINGS['draft'] else '0'
    return values


def install():
    # A restart with an existing session must adopt its original settings,
    # not a pending model selection saved in the management page.
    settings = config.read(ACTIVE, default=False) if llm.load_session() else config.read_desired(DESIRED)
    apply_settings(settings)
    llm.boot_vars = boot_vars
    original_begin = llm.Manager._begin

    def begin(manager, new):
        if new:
            settings = config.read_desired(DESIRED)
            apply_settings(settings)
            config.write(ACTIVE, settings)
        return original_begin(manager, new)

    llm.Manager._begin = begin

    original_run_start = llm.Manager._run_start

    def run_start(manager, attempt, new):
        original_run_start(manager, attempt, new)
        if new:
            remember_startup(manager)

    llm.Manager._run_start = run_start
    llm.Handler.held = retry_held

    def request_output_policy(handler):
        manager = handler.server.mgr
        with manager.cond:
            context = config.read_desired(DESIRED)['context'] if manager.state == 'off' else llm.choice()[1]
            return llm.output_policy(context)

    llm.Handler.output_policy = request_output_policy

    original_models = llm.Handler.models

    def models(handler):
        manager = handler.server.mgr
        with manager.cond:
            if manager.state == 'off':
                apply_settings(config.read_desired(DESIRED))
            return original_models(handler)

    llm.Handler.models = models


if __name__ == '__main__':
    install()
    llm.main()
