# GTBot 配置模板说明

## 快速开始

此目录包含GTBot配置系统的模板文件。按照以下步骤配置GTBot：

### 1. 复制模板文件

将以下模板文件重命名为实际配置文件：

- `config.json.template` -> `config.json`
- `api_config.json.template` -> `api_config.json`
- `config_group.json.template` -> `config_group.json`

### 2. 编辑配置文件

根据实际需求修改各配置文件中的参数。

#### config.json
主配置文件，指定其他配置文件的位置和默认配置组。

#### api_config.json
存储LLM服务提供商的API密钥和模型配置。

#### config_group.toml
定义不同的使用场景配置组。

### 3. 启动GTBot

配置完成后，重启NoneBot即可加载新配置。

## 配置文件详说

详见同目录下的 `配置指南.md` 文件。
