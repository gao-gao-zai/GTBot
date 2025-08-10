

# QQ Chat Bot

一个基于 NoneBot2 的 QQ 机器人项目，支持群聊消息处理、AI 对话、消息记录和向量数据库功能。

## 功能特点

- **AI 对话**：集成 ChatGPT 支持的多种 AI 模型，可实现自然语言处理和对话回复。
- **权限管理**：支持群主、管理员、黑名单等权限控制。
- **消息记录**：使用 SQLite 保存群聊记录，便于后续检索和分析。
- **向量数据库**：通过 ChromaDB 和 Ollama 提供嵌入和相似性搜索功能，支持 RAG（Retrieval-Augmented Generation）机制。
- **插件系统**：模块化设计，支持插件扩展功能。
- **支持表情点赞、戳一戳、消息撤回等 QQ 特性**。

## 目录结构

- `bot.py`: 主程序入口。
- `plugins/`: 存放插件代码。
- `plugins/chatai/`: AI 相关插件，包括权限控制、消息记录、向量数据库集成和主处理逻辑。
- `plugins/chatai/SQLiteManager.py`: SQLite 数据库管理模块。
- `plugins/chatai/VectorDatabaseSystem.py`: 向量数据库管理系统，集成 ChromaDB 和 Ollama。
- `plugins/chatai/chatgpt.py`: ChatGPT 对话模块。
- `plugins/chatai/main.py`: 消息处理和 AI 回复逻辑。
- `plugins/chatai/config_manager.py`: 配置管理模块。
- `plugins/chatai/Permissions.py`: 权限控制模块。

## 安装与配置

### 依赖安装

请确保已安装 Python 和 `nonebot2`，然后执行以下命令：

```bash
pip install nonebot2
pip install chromadb
pip install ollama
pip install openai
pip install aiosqlite
```

### 配置文件

- `.env` 和 `.env.prod`: 环境变量配置，如 API 密钥、数据库路径等。
- `plugins/chatai/api_config.json`: AI 模型的 API 配置。
- `plugins/chatai/permission_config.json`: 权限控制配置，如群主、管理员、黑名单等。
- `plugins/chatai/config.toml`: 主配置文件，包含 AI 模型、向量数据库等设置。

## 使用说明

- 运行机器人：`nonebot run bot:bot`
- 配置 AI 模型：修改 `plugins/chatai/api_config.json` 中的模型信息。
- 配置权限：在 `plugins/chatai/permission_config.json` 中设置群组权限。
- 修改机器人行为：在 `plugins/chatai/config.toml` 中调整相关参数。

## 开发者指南

- `plugins/chatai/record.py`: 消息记录初始化逻辑。
- `plugins/chatai/SQLiteManager.py`: SQLite 数据库管理类，用于消息存储。
- `plugins/chatai/VectorDatabaseSystem.py`: 向量数据库相关代码，包括嵌入生成、相似性搜索等。
- `plugins/chatai/fun.py`: 工具类函数，如发送图片、语音，获取用户信息等。

## 贡献者

欢迎提交 PR 或 Issues。请遵循 PEP8 编码规范。

## 开源协议

MIT License，详见 `LICENSE` 文件。