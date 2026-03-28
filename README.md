# GGZ Bot

基于 NoneBot2 的 QQ 机器人项目，当前以 `GTBot` 为主插件集合进行维护。

这个仓库现在按“可发布项目”整理：

- 依赖统一由 `pyproject.toml` 管理
- 真实密钥与本地参数继续走配置文件
- 仓库只提交示例配置，不提交真实配置
- `plugins/chatai` 作为历史目录保留，但不纳入当前发布依赖与包发现范围

## 主要组成

- `bot.py`：NoneBot 启动入口
- `plugins/GTBot/`：当前主要功能插件
- `plugins/restart_plugin/`：重启与停机控制
- `plugins/log_backup/`：日志备份插件
- `scripts/llm_cli.py`：独立调试脚本

## 环境要求

- Python 3.10+
- OneBot V11 运行环境

## 安装

基础依赖：

```bash
pip install -e .
```

启用交互式 CLI：

```bash
pip install -e ".[cli]"
```

启用可选 LangChain 适配器与额外提供商支持：

```bash
pip install -e ".[cli,providers]"
```

## 配置

本项目默认使用“配置文件优先”的方式管理密钥和本地参数。发布仓库只保留模板文件，请先复制示例文件再填写真实配置。

建议至少检查这些模板：

- `.env.example` -> `.env`
- `.env.prod.example` -> `.env.prod`
- `plugins/GTBot/config/config.json.example` -> `plugins/GTBot/config/config.json`
- `plugins/GTBot/config/api_config.json.example` -> `plugins/GTBot/config/api_config.json`
- `plugins/GTBot/tools/long_memory/config.json.example` -> `plugins/GTBot/tools/long_memory/config.json`
- `plugins/GTBot/tools/comfyui_draw/config.json.example` -> `plugins/GTBot/tools/comfyui_draw/config.json`
- `plugins/log_backup/config.json.example` -> `plugins/log_backup/config.json`
- `plugins/restart_plugin/permission_config.json.example` -> `plugins/restart_plugin/permission_config.json`
- `plugins/GTBot/tools/tavily_search_plugin.config.json.example` -> `plugins/GTBot/tools/tavily_search_plugin.config.json`

说明：

- 真实配置文件不会提交到 Git。
- 缺失部分插件配置时，插件会按默认值生成或回退到示例配置。
- 密钥仍建议写入你本地的配置文件，而不是提交到仓库。

## 启动

```bash
python bot.py
```

或：

```bash
nb run
```

## 发布前建议

- 确认 `.env`、各类 `config.json`、数据库文件、日志和压缩包没有被 Git 跟踪
- 更换任何曾经提交过的真实密钥
- 优先提交示例配置、文档和代码，不提交本地运行状态

## License

Apache License 2.0，见 `LICENSE`。
