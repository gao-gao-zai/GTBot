# GTBot 配置模板说明

## 快速开始

此目录包含 GTBot 配置系统的模板文件。按照以下步骤配置 GTBot：

### 1. 复制示例文件

将以下示例文件复制/重命名为实际配置文件（真实配置不要提交到仓库）：

- `config.json.example` -> `config.json`
- `api_config.json.example` -> `api_config.json`

### 2. 编辑配置文件

根据实际需求修改各配置文件中的参数。

#### config.json

主配置文件，指定其他配置文件的位置和默认配置组。

#### api_config.json

存储 LLM 服务提供商的 API 密钥和模型配置。

#### config_group.toml

定义不同的使用场景配置组。

### 3. 启动 GTBot

配置完成后，重启 NoneBot 即可加载新配置。

## 配置文件详说

详见同目录下的 `配置指南.md` 文件。
