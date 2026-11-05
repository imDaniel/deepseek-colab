"""Start the existing controller with an immutable settings snapshot per run."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import llm  # noqa: E402
from integrations.magpie_colab import config  # noqa: E402

DESIRED = Path(llm.STATE) / 'magpie-settings.json'
ACTIVE = Path(llm.STATE) / 'magpie-active-settings.json'


def apply_settings(settings):
    settings = config.validate(settings)
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
    settings = config.read(ACTIVE, default=False) if llm.load_session() else config.read(DESIRED)
    apply_settings(settings)
    llm.boot_vars = boot_vars
    original_begin = llm.Manager._begin

    def begin(manager, new):
        if new:
            settings = config.read(DESIRED)
            apply_settings(settings)
            config.write(ACTIVE, settings)
        return original_begin(manager, new)

    llm.Manager._begin = begin

    original_models = llm.Handler.models

    def models(handler):
        manager = handler.server.mgr
        with manager.cond:
            if manager.state == 'off':
                apply_settings(config.read(DESIRED))
            return original_models(handler)

    llm.Handler.models = models


if __name__ == '__main__':
    install()
    llm.main()
