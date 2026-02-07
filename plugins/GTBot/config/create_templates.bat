@echo off
REM GTBot 配置文件模板生成脚本
REM 此脚本用于创建GTBot所需的模板配置文件

chcp 65001 >nul
setlocal enabledelayedexpansion

echo.
echo ======================================
echo   GTBot 配置模板创建工具
echo ======================================
echo.

REM 获取当前脚本所在目录
set SCRIPT_DIR=%~dp0

REM 检查是否为开发环境
if not exist "config_group.toml" (
    echo 检测到新环境，开始创建模板文件...
) else (
    echo 检测到现有配置文件
)

REM 创建 config.json 模板
echo 创建 config.json 模板...
(
    echo {
    echo   // GTBot 主配置文件
    echo   // API配置文件的路径
    echo   "api_config_path": "./api_config.json",
    echo   // 配置组文件的路径
    echo   "config_groups_path": "./config_group.json",
    echo   // 默认使用的配置组名称
    echo   "default_config_group": "default",
    echo   // 提示词目录的路径
    echo   "prompt_dir_path": "../../prompts"
    echo }
) > "%SCRIPT_DIR%config.json.template"
if %errorlevel% equ 0 (
    echo ✓ config.json.template 创建成功
) else (
    echo ✗ config.json.template 创建失败
)

REM 创建 api_config.json 模板
echo 创建 api_config.json 模板...
(
    echo {
    echo   // API 配置文件 - 存储所有LLM服务提供商的配置
    echo   "openai": {
    echo     "base_url": "https://api.openai.com/v1",
    echo     "api_key": "sk-your-api-key-here",
    echo     "llm_models": {
    echo       "gpt4": {
    echo         "model": "gpt-4-turbo",
    echo         "max_input_tokens": 4096,
    echo         "supports_vision": true,
    echo         "supports_audio": false,
    echo         "parameters": {
    echo           "temperature": 0.7
    echo         }
    echo       },
    echo       "gpt35": {
    echo         "model": "gpt-3.5-turbo",
    echo         "max_input_tokens": 2048,
    echo         "supports_vision": false,
    echo         "supports_audio": false,
    echo         "parameters": {
    echo           "temperature": 0.7
    echo         }
    echo       }
    echo     }
    echo   },
    echo   "anthropic": {
    echo     "base_url": "https://api.anthropic.com",
    echo     "api_key": "sk-ant-your-api-key-here",
    echo     "llm_models": {
    echo       "claude3": {
    echo         "model": "claude-3-opus-20240229",
    echo         "max_input_tokens": 8192,
    echo         "supports_vision": true,
    echo         "supports_audio": false,
    echo         "parameters": {
    echo           "temperature": 0.7
    echo         }
    echo       }
    echo     }
    echo   }
    echo }
) > "%SCRIPT_DIR%api_config.json.template"
if %errorlevel% equ 0 (
    echo ✓ api_config.json.template 创建成功
) else (
    echo ✗ api_config.json.template 创建失败
)

REM 创建 config_group.json 模板
echo 创建 config_group.json 模板...
(
    echo {
    echo   // GTBot 配置组文件 - 定义不同的机器人行为配置
    echo   "default": {
    echo     "chat_model": {
    echo       "model": "openai/gpt4",
    echo       "maximum_number_of_incoming_messages": 10,
    echo       "memory": {
    echo         "notepad_max_entries": 15,
    echo         "notepad_retention_seconds": 300
    echo       },
    echo       "behavioral_prompt": "角色提示词/默认.txt",
    echo       "character_prompt": "角色提示词/默认.txt"
    echo     }
    echo   },
    echo   "casual": {
    echo     "description": "休闲聊天配置组",
    echo     "chat_model": {
    echo       "model": "openai/gpt35",
    echo       "maximum_number_of_incoming_messages": 5,
    echo       "memory": {
    echo         "notepad_max_entries": 15,
    echo         "notepad_retention_seconds": 300
    echo       },
    echo       "behavioral_prompt": "角色提示词/默认.txt",
    echo       "character_prompt": "角色提示词/默认.txt"
    echo     }
    echo   }
    echo }
) > "%SCRIPT_DIR%config_group.json.template"
if %errorlevel% equ 0 (
    echo ✓ config_group.json.template 创建成功
) else (
    echo ✗ config_group.json.template 创建失败
)

REM 创建 README.md 说明文件
echo 创建 README.md 说明文件...
(
    echo # GTBot 配置模板说明
    echo.
    echo ## 快速开始
    echo.
    echo 此目录包含GTBot配置系统的模板文件。按照以下步骤配置GTBot：
    echo.
    echo ### 1. 复制模板文件
    echo.
    echo 将以下模板文件重命名为实际配置文件：
    echo.
    echo - `config.json.template` ^-^> `config.json`
    echo - `api_config.json.template` ^-^> `api_config.json`
    echo - `config_group.json.template` ^-^> `config_group.json`
    echo.
    echo ### 2. 编辑配置文件
    echo.
    echo 根据实际需求修改各配置文件中的参数。
    echo.
    echo #### config.json
    echo 主配置文件，指定其他配置文件的位置和默认配置组。
    echo.
    echo #### api_config.json
    echo 存储LLM服务提供商的API密钥和模型配置。
    echo.
    echo #### config_group.toml
    echo 定义不同的使用场景配置组。
    echo.
    echo ### 3. 启动GTBot
    echo.
    echo 配置完成后，重启NoneBot即可加载新配置。
    echo.
    echo ## 配置文件详说
    echo.
    echo 详见同目录下的 `配置指南.md` 文件。
) > "%SCRIPT_DIR%README_SETUP.md"
if %errorlevel% equ 0 (
    echo ✓ README_SETUP.md 创建成功
) else (
    echo ✗ README_SETUP.md 创建失败
)

echo.
echo ======================================
echo   模板文件创建完成！
echo ======================================
echo.
echo 下一步：
echo 1. 查看各 .template 文件的内容
echo 2. 根据实际情况复制并修改为实际配置文件
echo 3. 填入必要的API密钥等信息
echo.
pause
