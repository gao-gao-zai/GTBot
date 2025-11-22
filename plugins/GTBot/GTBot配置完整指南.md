# GTBot 完整配置指南

## 概述

GTBot 是一个功能丰富的聊天机器人系统，支持多提供商API、多模型配置、灵活的配置组管理和自定义数据存储。本文档详细介绍了所有配置选项和使用方法。

## 配置文件结构

```
GTBot/config/
├── config.json          # 主配置文件
├── api_config.json      # API提供商和模型配置
├── config_group.json    # 配置组定义
└── prompts/            # 提示词目录
    └── 角色提示词/
        └── 默认.txt     # 角色提示词文件

GTBot/data/             # 数据目录
├── data.db            # SQLite数据库
└── ...                # 其他数据文件
```

## 1. 主配置文件 (`config/config.json`)

### 当前配置项

```json
{
  "api_config_path": "./config/api_config.json",
  "config_groups_path": "./config/config_group.json", 
  "default_config_group": "default",
  "prompt_dir_path": "./config/prompts",
  "data_dir_path": "./data"
}
```

### 配置说明

| 配置项 | 类型 | 说明 | 默认值 |
|--------|------|------|--------|
| `api_config_path` | string | API配置文件路径 | `"./config/api_config.json"` |
| `config_groups_path` | string | 配置组文件路径 | `"./config/config_group.json"` |
| `default_config_group` | string | 默认配置组名称 | `"default"` |
| `prompt_dir_path` | string | 提示词目录路径 | `"./config/prompts"` |
| `data_dir_path` | string | **数据目录路径** | `"./data"` |

## 2. API配置文件 (`config/api_config.json`)

### 配置结构

```json
{
  "provider_name": {
    "base_url": "API基础URL",
    "api_key": "API密钥",
    "llm_models": {
      "model_alias": {
        "model": "实际模型名称",
        "max_input_tokens": 最大输入token数,
        "supports_vision": 是否支持视觉,
        "supports_audio": 是否支持音频,
        "parameters": {
          "temperature": 温度参数
        }
      }
    }
  }
}
```

### 当前配置示例

```json
{
  "openai": {
    "base_url": "https://deepseek-ai.cloudns.org/v1",
    "api_key": "你的API密钥",
    "llm_models": {
      "gemini-2.5-flash-lite": {
        "model": "GeminiP/gemini-2.5-flash-lite",
        "max_input_tokens": 32768,
        "supports_vision": true,
        "supports_audio": true,
        "parameters": {
          "temperature": 1.2
        }
      }
    }
  }
}
```

### API配置说明

#### 提供商级别配置
- `base_url`: API服务的基础URL
- `api_key`: 用于认证的API密钥
- `llm_models`: 该提供商下的所有可用模型

#### 模型级别配置
- `model`: 上游API的实际模型名称
- `max_input_tokens`: 模型支持的最大输入token数
- `supports_vision`: 是否支持图像理解
- `supports_audio`: 是否支持音频处理
- `parameters`: 模型参数（如temperature等）

## 3. 配置组文件 (`config/config_group.json`)

### 配置结构

```json
{
  "group_name": {
    "chat_model": {
      "model": "提供商/模型别名",
      "maximum_number_of_incoming_messages": 最大消息数,
      "behavioral_prompt": "行为提示词文件路径",
      "character_prompt": "角色提示词文件路径"
    }
  }
}
```

### 当前配置

```json
{
  "default": {
    "chat_model": {
      "model": "openai/gemini-2.5-flash-lite",
      "maximum_number_of_incoming_messages": 10,
      "behavioral_prompt": "角色提示词/默认.txt",
      "character_prompt": "角色提示词/默认.txt"
    }
  }
}
```

### 配置组说明

- `model`: 格式为 `"提供商名称/模型别名"`
- `maximum_number_of_incoming_messages`: 对话上下文中保留的最大消息数
- `behavioral_prompt`: 定义AI行为模式的提示词文件
- `character_prompt`: 定义AI角色特征的提示词文件

## 4. 数据目录配置 ✨ 新功能

### 功能特性

- **自动创建**: 指定目录不存在时自动创建
- **路径灵活性**: 支持相对路径和绝对路径
- **环境变量**: 支持 `$HOME`、`~` 等变量展开
- **错误处理**: 完善的错误处理和回退机制

### 使用示例

```python
from GTBot.ConfigManager import TotalConfiguration

# 初始化配置
config = TotalConfiguration.init()

# 获取数据目录路径
data_dir = config.get_data_dir_path()

# 使用数据目录
log_file = data_dir / "logs" / "app.log"
database_file = data_dir / "data.db"
```

## 5. 编程接口

### 配置初始化

```python
from GTBot.ConfigManager import TotalConfiguration

# 使用默认配置文件
config = TotalConfiguration.init()

# 使用自定义配置文件
config = TotalConfiguration.init("./custom_config.json")
```

### 配置组管理

```python
# 获取当前配置组
current_group = config.get_current_group_name()

# 获取所有可用配置组
available_groups = config.get_available_config_groups()

# 切换配置组
config.switch_config_group("high_performance")

# 重载配置（保持当前配置组）
config.reload(keep_current_group=True)

# 重载配置（使用默认配置组）
config.reload(keep_current_group=False)
```

### 路径访问

```python
# 获取数据目录
data_dir = config.get_data_dir_path()

# 获取配置文件路径
config_file = config.get_config_file_path()
```

## 6. 配置扩展指南

### 添加新的API提供商

1. 在 `api_config.json` 中添加新提供商：

```json
{
  "anthropic": {
    "base_url": "https://api.anthropic.com",
    "api_key": "your_anthropic_key",
    "llm_models": {
      "claude-3": {
        "model": "claude-3-opus-20240229",
        "max_input_tokens": 200000,
        "supports_vision": true,
        "supports_audio": false,
        "parameters": {
          "temperature": 0.7,
          "max_tokens": 4096
        }
      }
    }
  }
}
```

### 添加新的配置组

1. 在 `config_group.json` 中添加新配置组：

```json
{
  "default": { ... },
  "creative": {
    "chat_model": {
      "model": "anthropic/claude-3",
      "maximum_number_of_incoming_messages": 20,
      "behavioral_prompt": "角色提示词/创意助手.txt",
      "character_prompt": "角色提示词/创意助手.txt"
    }
  }
}
```

### 添加新的提示词

1. 在 `config/prompts/角色提示词/` 目录下创建新文件
2. 在配置组中引用新的提示词文件

## 7. 故障排除

### 常见问题

1. **配置文件不存在**
   - 确保所有配置文件都存在于正确位置
   - 检查文件路径是否正确

2. **API密钥无效**
   - 验证API密钥是否正确
   - 检查API服务是否可用

3. **权限问题**
   - 确保应用有权限读取配置文件
   - 确保应用有权限在数据目录创建文件

4. **路径配置错误**
   - 检查路径分隔符（Windows使用`\`，Unix使用`/`）
   - 验证相对路径是否正确

### 调试命令

```bash
# 运行配置测试
python ConfigManager.py

# 测试数据目录配置
python test_data_config.py
```

## 8. 最佳实践

1. **备份配置**: 修改前备份配置文件
2. **环境分离**: 为不同环境使用不同的配置组
3. **安全性**: 不要在代码仓库中提交API密钥
4. **监控**: 定期检查配置文件的有效性
5. **文档**: 为自定义配置组添加说明文档

## 9. 更新日志

### v2.0 (2025-11-22)
- ✅ 添加 `data_dir_path` 配置项
- ✅ 支持数据目录自动创建  
- ✅ 更新 model.py 使用配置化数据路径
- ✅ 添加完整的配置测试
- ✅ 完善配置文档

### v1.0 
- ✅ 基础配置系统
- ✅ 多提供商API支持
- ✅ 配置组管理
- ✅ 提示词系统
