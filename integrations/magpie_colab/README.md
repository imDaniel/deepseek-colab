# Magpie Colab DeepSeek integration

This integration adds a Magpie page for the on-demand `llm.py` controller. It displays the controller state and Colab CLI balance and saves settings for the next GPU session. It requires a working Colab CLI OAuth login, a separately configured Magpie OpenAI-compatible provider pointing at the controller's loopback endpoint, and a local service manager. Installation of this plugin alone does not configure those dependencies or authenticate Colab.

## Components

- `controller.py serve` wraps the project controller, reads `magpie-settings.json`, and freezes its settings in `magpie-active-settings.json` for the life of each session. It starts a GPU only for an authenticated inference request.
- `panel.py` listens on `127.0.0.1:18082` and serves status and settings. Run it under the same unprivileged account and `LLM_STATE` as the controller. Magpie's page service proxies access to this endpoint; do not expose this port publicly.
- Install the plugin folder with `magpie plugin add /absolute/path/to/integrations/magpie_colab` and enable it. Its host entry is needed by Magpie's page-service proxy. Launch the controller and panel under your service manager and arrange the CLI binary and runtime state with `LLM_COLAB`, `LLM_STATE` and `HOME` as needed. Restart the panel after updating its HTML.

The page lets you choose the ISTA base, the project's AtomicChat-patched Uncensored model, or Abiray's pre-abliterated model. Existing requests keep the active session settings; saved changes apply when the next session starts. The idle timer starts after the last request completes; the maximum runtime is a local forced-stop threshold and can interrupt active requests. Google may end the session earlier. A 24-hour selection does not guarantee the account or Colab runtime is eligible to run for that duration. The minimum balance is checked at startup and is not a spending cap.

For the page-service design, see [Magpie's plugin documentation](https://usemagpie.ai/docs/plugins) and the [community chat page](https://github.com/Magiczerowxy/magpie-chat). Treat the Colab OAuth token and all gateway keys as secrets; do not include them in plugin configuration or repository files.
