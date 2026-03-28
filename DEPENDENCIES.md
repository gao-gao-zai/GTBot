# Dependencies

## Current source of truth

- Runtime dependencies are now managed in `pyproject.toml`.
- `requirements.txt` is retained only as a compatibility shim and delegates to `pyproject.toml`.

## Scope

- The dependency set covers the main bot, `GTBot`, `restart_plugin`, `log_backup`, `status.py`, and root scripts.
- `plugins/chatai` is intentionally excluded from dependency management in this project file.

## Optional groups

- `cli`: installs `prompt-toolkit` for the interactive CLI experience in `scripts/llm_cli.py`.
- `langchain-adapters`: installs optional LangChain adapters used by conditional code paths such as Anthropic, Gemini, and DashScope chat model integrations.
- `providers`: installs provider SDKs and includes `langchain-adapters`.

## Install examples

- Minimal runtime: `pip install -e .`
- Only optional LangChain adapters: `pip install -e ".[langchain-adapters]"`
- Common local setup: `pip install -e ".[cli,providers]"`
- Compatibility path: `pip install -r requirements.txt`
