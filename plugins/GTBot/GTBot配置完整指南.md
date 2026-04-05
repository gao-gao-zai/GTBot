# GTBot 配置完整指南

本文档是 GTBot 配置系统的完整说明，覆盖主配置、API 提供商配置、配置组、提示词以及多提供商聊天模型的使用方式。

## 配置目录结构

```text
GTBot/config/
├── config.json
├── api_config.json
├── config_group.json
└── prompts/

GTBot/data/
├── data.db
└── ...
```

## 1. 主配置文件 `config/config.json`

示例：

```json
{
  "api_config_path": "./config/api_config.json",
  "config_groups_path": "./config/config_group.json",
  "default_config_group": "default",
  "prompt_dir_path": "./config/prompts",
  "data_dir_path": "./data",
  "plugin_dir": "./tools",
  "user_cache_update_interval_sec": 3600,
  "user_cache_expire_sec": 604800
}
```

字段说明：

- `api_config_path`: API 配置文件路径
- `config_groups_path`: 配置组文件路径
- `default_config_group`: 默认启用的配置组
- `prompt_dir_path`: 提示词目录
- `data_dir_path`: 数据目录
- `plugin_dir`: GTBot 工具插件目录
- `user_cache_update_interval_sec`: 用户缓存刷新间隔
- `user_cache_expire_sec`: 用户缓存过期时间

## 2. API 配置文件 `config/api_config.json`

### 2.1 结构

```json
{
  "provider_name": {
    "provider_type": "openai_compatible",
    "base_url": "API基础URL",
    "api_key": "API密钥",
    "llm_models": {
      "model_alias": {
        "model": "实际模型名称",
        "max_input_tokens": 32768,
        "supports_vision": true,
        "supports_audio": false,
        "parameters": {
          "temperature": 0.7
        }
      }
    }
  }
}
```

### 2.2 Provider 字段

- `provider_type`
  底层提供商类型。支持：
  `openai_compatible`、`openai_responses`、`anthropic`、`gemini`、`dashscope`
- `process_tool_call_deltas`
  GTBot 内部开关，可写在 provider 根级作为默认值，控制是否处理流式工具调用增量
- `base_url`
  API 基础地址
- `api_key`
  API 密钥
- `llm_models`
  当前 provider 下的模型列表

### 2.3 模型字段

- `model`
  上游实际模型 ID
- `max_input_tokens`
  最大输入 token 数
- `supports_vision`
  是否支持视觉输入
- `supports_audio`
  是否支持音频输入
- `parameters`
  模型参数，例如 `temperature`、`top_p`、`stream`；也支持 GTBot 内部参数 `process_tool_call_deltas`

### 2.4 示例

```json
{
  "openai_main": {
    "provider_type": "openai_compatible",
    "base_url": "https://api.openai.com/v1",
    "api_key": "sk-xxx",
    "llm_models": {
      "gpt4": {
        "model": "gpt-4-turbo",
        "max_input_tokens": 4096,
        "supports_vision": true,
        "supports_audio": false,
        "parameters": {
          "temperature": 0.7,
          "process_tool_call_deltas": true
        }
      }
    }
  },
  "anthropic_main": {
    "provider_type": "anthropic",
    "base_url": "https://api.anthropic.com",
    "api_key": "sk-ant-xxx",
    "llm_models": {
      "claude-sonnet": {
        "model": "claude-3-7-sonnet-latest",
        "max_input_tokens": 200000,
        "supports_vision": true,
        "supports_audio": false,
        "parameters": {
          "temperature": 0.5
        }
      }
    }
  }
}
```

### 2.5 `provider_type` 的意义

`provider/model_alias` 里的 provider 名称只是你自己起的名字，不代表底层一定是 OpenAI。

真正决定运行时创建哪种聊天客户端的是 `provider_type`：

- `openai_compatible`: `ChatOpenAI`
- `openai_responses`: `ChatOpenAI` + Responses API
- `anthropic`: `ChatAnthropic`
- `gemini`: Google Gemini 适配器
- `dashscope`: `ChatTongyi`

这意味着：

- 你可以把 provider 名称写成 `ds`、`my_gemini`、`prod_llm`
- 但它最终走哪种 SDK，取决于该 provider 的 `provider_type`

## 3. 配置组文件 `config/config_group.json`

### 3.1 结构

```json
{
  "default": {
    "chat_model": {
      "model": "openai_main/gpt4",
      "maximum_number_of_incoming_messages": 40,
      "max_message_length": 200,
      "behavioral_prompt": "行为提示词/默认.txt",
      "character_prompt": "角色提示词/默认.txt",
      "max_concurrent_responses_per_group": 1,
      "max_total_concurrent_responses": 5,
      "max_tool_calls_per_turn": 20,
      "recursion_limit": 50,
      "api_timeout_sec": 60,
      "memory": {
        "notepad_max_entries": 15,
        "notepad_retention_seconds": 600
      }
    },
    "user_profile": {
      "max_descriptions": 10,
      "max_description_char_length": 500
    },
    "message_format_placeholder": "[[$time_M]-[$time_d] [$time_h]:[$time_m]:[$time_s]] [$user_name]([$user_id], [$message_id]):[$message]"
  }
}
```

### 3.2 关键字段

- `chat_model.model`
  格式必须为 `provider/model_alias`
- `maximum_number_of_incoming_messages`
  上下文保留消息条数
- `max_message_length`
  单条消息截断长度，`0` 表示不限制
- `behavioral_prompt`
  行为提示词文件
- `character_prompt`
  角色提示词文件
- `max_concurrent_responses_per_group`
  单群并发响应上限
- `max_total_concurrent_responses`
  全局并发响应上限
- `max_tool_calls_per_turn`
  单回合工具调用上限
- `recursion_limit`
  agent 递归上限
- `api_timeout_sec`
  模型请求超时
- `memory`
  记事本配置

## 4. 多提供商聊天模型支持

GTBot 现在支持在聊天主链路里按 `provider_type` 创建不同模型提供商客户端。

支持范围：

- OpenAI 兼容接口
- OpenAI Responses API
- Anthropic
- Gemini
- DashScope / Tongyi

注意事项：

- 不同 provider 的工具调用、流式输出、视觉能力支持程度可能不同
- `supports_vision` 和 `supports_audio` 只是 GTBot 配置声明，真正是否可用还取决于上游模型
- 如果切到非 OpenAI 兼容 provider，请确认对应依赖已安装
- `process_tool_call_deltas` 默认为 `true`，可写在 provider 根级做默认值，也可写在某个模型的 `parameters` 里单独覆盖
- 关闭后会恢复更接近原始流式行为，也不会再对阿里 DashScope OpenAI 兼容地址自动切回非流式

## 5. 提示词文件

`behavioral_prompt` 与 `character_prompt` 都是纯文本文件，运行时会被读取并拼接为最终 system prompt。

相对路径基于 `config.json` 中的 `prompt_dir_path` 解析。

## 6. 消息格式模板 `message_format_placeholder`

该字段定义历史消息格式化样式。

常见占位符：

- `[$time_Y]`
- `[$time_M]`
- `[$time_d]`
- `[$time_h]`
- `[$time_m]`
- `[$time_s]`
- `[$user_id]`
- `[$user_name]`
- `[$group_id]`
- `[$message_id]`
- `[$message]`

示例：

```text
[[$time_M]-[$time_d] [$time_h]:[$time_m]:[$time_s]] [$user_name]([$user_id]):[$message]
```

## 7. 配置加载与热重载

- 启动时自动加载全部配置
- 支持 `total_config.reload()`
- `reload(keep_current_group=True)` 尝试保留当前配置组
- `reload(keep_current_group=False)` 回退到默认配置组

## 8. 常见问题

### 8.1 provider 名称和 provider_type 有什么区别？

- provider 名称是你在 `api_config.json` 里定义的 key
- provider_type 是 GTBot 运行时选择底层客户端的依据

### 8.2 为什么配置没错但工具调用失败？

可能原因：

- 当前模型本身不支持工具调用
- 当前 provider 的 LangChain 适配器对工具调用支持有限
- 模型参数不兼容
- 如果你显式把 `process_tool_call_deltas` 设为 `false`，那 GTBot 不会再帮你处理这类流式工具调用增量问题

### 8.3 哪些 provider_type 必须写 `base_url`？

通常：

- `openai_compatible`
- `openai_responses`
- `anthropic`

这三类最好保证 `base_url` 非空。

### 8.4 路径是相对于哪里解析的？

- `api_config_path`、`config_groups_path`、`prompt_dir_path` 相对 `ConfigManager.py`
- `behavioral_prompt`、`character_prompt` 相对 `prompt_dir_path`

## 9. 最佳实践

- 不要把真实 `api_key` 提交到仓库
- 不同环境使用不同 provider 名称，便于切换
- provider 名称和模型别名都不要包含 `/`
- 变更配置后用热重载或重启验证一次
