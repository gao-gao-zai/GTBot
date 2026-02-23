# Dependencies

## Notes

- `pyproject.toml` is referenced by `bot.py` (`nonebot.load_from_toml("pyproject.toml")`) but the file is not present in this workspace path.
- The lists below are best-effort summaries based on:
  - `README.md` installation commands
  - Static analysis of Python `import` statements across `*.py`

## Explicitly documented in README.md

- `nonebot2`
- `chromadb`
- `ollama`
- `openai`
- `aiosqlite`

## Detected third-party Python modules (from imports)

- `PIL`
- `aiofiles`
- `aiohttp`
- `aiosqlite`
- `chromadb`
- `deepdiff`
- `fastapi`
- `httpx`
- `langchain`
- `langchain_core`
- `langchain_openai`
- `langgraph`
- `nonebot`
- `nonebot_plugin_apscheduler`
- `numpy`
- `openai`
- `prompt_toolkit`
- `pydantic`
- `qdrant_client`
- `requests`
- `rich`
- `sqlalchemy`
- `sympy`
- `tabulate`
- `tiktoken`
- `toml`
- `tomli`
- `tqdm`
- `uvicorn`
- `yaml`

## Potentially local/project modules (not treated as external deps)

These names appeared in `import` statements but look like local packages/modules in this repo (or ambiguous), so they are not counted as external dependencies here:

- `plugins`
- `services`

