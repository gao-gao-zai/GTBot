from __future__ import annotations

import html
import hashlib
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_BUILTIN_THEME_CSS: dict[str, str] = {
    "default": """
      body {
        background:
          radial-gradient(circle at top left, #f4f8ff 0, transparent 34%),
          linear-gradient(180deg, #fbfcfe 0%, #f2f5f9 100%);
      }
      #capture {
        background: #ffffff;
        border: 1px solid rgba(15, 23, 42, 0.06);
        box-shadow: 0 18px 48px rgba(15, 23, 42, 0.08);
      }
      blockquote {
        border-left-color: #9ecbff;
        background: #edf6ff;
      }
      th {
        background: #f7f8fa;
      }
    """,
    "notebook": """
      body {
        background:
          linear-gradient(0deg, rgba(41, 98, 255, 0.06), rgba(41, 98, 255, 0.06)),
          repeating-linear-gradient(
            180deg,
            #fffef8 0,
            #fffef8 31px,
            #e4edf9 32px
          );
      }
      #capture {
        background: rgba(255, 255, 255, 0.96);
        border: 1px solid rgba(47, 84, 150, 0.10);
        box-shadow: 0 20px 60px rgba(42, 56, 88, 0.10);
      }
      h1, h2, h3, h4, h5, h6 {
        color: #213547;
      }
      blockquote {
        border-left-color: #ffb703;
        background: #fff5d6;
        color: #5f4b1c;
      }
      table {
        box-shadow: 0 0 0 1px #d8e2f0;
      }
      th {
        background: #eef4fb;
      }
    """,
    "glass": """
      body {
        background:
          radial-gradient(circle at 20% 15%, rgba(255, 255, 255, 0.75), transparent 24%),
          radial-gradient(circle at 80% 0%, rgba(179, 229, 252, 0.8), transparent 30%),
          linear-gradient(135deg, #dbeafe 0%, #eff6ff 40%, #f5f3ff 100%);
      }
      #capture {
        background: rgba(255, 255, 255, 0.86);
        border: 1px solid rgba(255, 255, 255, 0.55);
        box-shadow: 0 22px 70px rgba(31, 41, 55, 0.12);
        backdrop-filter: blur(12px);
      }
      blockquote {
        border-left-color: #7c3aed;
        background: rgba(124, 58, 237, 0.08);
        color: #4c1d95;
      }
      th {
        background: rgba(219, 234, 254, 0.7);
      }
    """,
}

_PYGMENTS_STYLE_ALIASES: dict[str, str] = {
    "default": "github-dark",
    "github": "github-dark",
    "light": "xcode",
    "dark": "github-dark",
}


@dataclass(frozen=True, slots=True)
class MarkdownRenderResult:
    """描述一次 Markdown 图片渲染的输出结果。

    该结构只保存当前插件在后续发送与文件注册阶段真正需要的关键信息，
    避免工具层重新解析截图文件。截图尺寸由 Playwright 基于目标元素的真实
    布局计算得出，因此适合作为日志、调试和后续扩展字段。

    Attributes:
        image_path: 已写入本地磁盘的 PNG 文件路径。
        width: 截图区域最终宽度，单位为 CSS 像素。
        height: 截图区域最终高度，单位为 CSS 像素。
    """

    image_path: Path
    width: int
    height: int


def _normalize_fixed_width(width: int | None) -> int | None:
    """规范化固定宽度参数。

    Args:
        width: 调用方显式传入的固定宽度。

    Returns:
        合法的固定宽度；未指定时返回 `None`。

    Raises:
        ValueError: 当 `width` 超出允许范围时抛出。
    """

    if width is None:
        return None
    normalized = int(width)
    if normalized < 480:
        raise ValueError("width 不能小于 480")
    if normalized > 2000:
        raise ValueError("width 不能大于 2000")
    return normalized


def _normalize_auto_width_bounds(min_width: int | None, max_width: int | None) -> tuple[int, int]:
    """规范化自动宽度模式的最小和最大边界。

    Args:
        min_width: 自动宽度模式下的最小宽度。
        max_width: 自动宽度模式下的最大宽度。

    Returns:
        一个 `(min_width, max_width)` 元组。

    Raises:
        ValueError: 当范围非法时抛出。
    """

    normalized_min = 560 if min_width is None else int(min_width)
    normalized_max = 1200 if max_width is None else int(max_width)
    if normalized_min < 360:
        raise ValueError("min_width 不能小于 360")
    if normalized_max > 2000:
        raise ValueError("max_width 不能大于 2000")
    if normalized_min > normalized_max:
        raise ValueError("min_width 不能大于 max_width")
    return normalized_min, normalized_max


def _normalize_scale(scale: float | None) -> float:
    """规范化截图缩放倍率。

    Args:
        scale: 调用方传入的设备缩放倍率。

    Returns:
        经校验后的缩放倍率。

    Raises:
        ValueError: 当 `scale` 不大于 0 时抛出。
    """

    normalized = 2.0 if scale is None else float(scale)
    if normalized <= 0:
        raise ValueError("scale 必须大于 0")
    return normalized


def _normalize_padding(padding: int | None) -> int:
    """规范化页面内边距。

    Args:
        padding: 调用方传入的页面内边距。

    Returns:
        经校验后的内边距像素值。

    Raises:
        ValueError: 当 `padding` 为负数时抛出。
    """

    normalized = 32 if padding is None else int(padding)
    if normalized < 0:
        raise ValueError("padding 不能小于 0")
    return normalized


def _resolve_theme_file_path(theme: str, *, theme_base_dir: Path | None) -> Path | None:
    """把主题字符串解析为本地文件路径。

    Args:
        theme: 调用方传入的主题字符串。
        theme_base_dir: 解析相对路径时使用的基准目录。

    Returns:
        命中时返回已存在的本地文件路径，否则返回 `None`。
    """

    raw = str(theme or "").strip()
    if not raw:
        return None
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate if candidate.exists() and candidate.is_file() else None
    if theme_base_dir is not None:
        resolved = (theme_base_dir / candidate).resolve()
        if resolved.exists() and resolved.is_file():
            return resolved
    resolved_from_cwd = candidate.resolve()
    if resolved_from_cwd.exists() and resolved_from_cwd.is_file():
        return resolved_from_cwd
    return None


def _compile_scss_file_to_css(theme_file: Path) -> str:
    """把本地 SCSS/Sass 文件编译为 CSS 文本。

    Args:
        theme_file: 待编译的 SCSS/Sass 文件路径。

    Returns:
        编译成功后的 CSS 文本。

    Raises:
        RuntimeError: 当本机缺少 Sass 编译器，或编译过程失败时抛出。
    """

    sass_exe = shutil.which("sass")
    use_npx = False
    if not sass_exe:
        npx_exe = shutil.which("npx")
        if npx_exe:
            sass_exe = npx_exe
            use_npx = True
        else:
            raise RuntimeError("主题文件是 SCSS/Sass，但当前机器未安装 `sass` 或 `npx`")

    cmd = [sass_exe]
    if use_npx:
        cmd.append("sass")
    cmd.extend(["--no-source-map", str(theme_file)])
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(theme_file.parent),
    )
    if proc.returncode != 0:
        message = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"主题 SCSS 编译失败: {message}")
    return proc.stdout


def _resolve_theme_css(
    theme: str | None,
    *,
    theme_base_dir: Path | None,
) -> str:
    """解析内置主题名或主题文件并返回最终 CSS。

    Args:
        theme: 调用方传入的主题名称或文件路径。
        theme_base_dir: 解析相对主题文件路径时使用的基准目录。

    Returns:
        最终会注入到页面中的主题 CSS。

    Raises:
        ValueError: 当主题既不是受支持的内置主题，也不是已存在的本地文件时抛出。
        RuntimeError: 当主题文件存在但无法读取，或 SCSS 编译失败时抛出。
    """

    raw = str(theme or "default").strip()
    normalized = raw.lower() or "default"
    builtin_css = _BUILTIN_THEME_CSS.get(normalized)
    if builtin_css is not None:
        return builtin_css

    theme_file = _resolve_theme_file_path(raw, theme_base_dir=theme_base_dir)
    if theme_file is None:
        raise ValueError(
            "theme 必须是内置主题名，或一个存在的本地 .css/.scss/.sass 文件路径"
        )

    suffix = theme_file.suffix.lower()
    if suffix == ".css":
        return theme_file.read_text(encoding="utf-8")
    if suffix in {".scss", ".sass"}:
        return _compile_scss_file_to_css(theme_file)
    raise ValueError("主题文件仅支持 .css、.scss 或 .sass")


def _resolve_pygments_style_name(style_name: str | None) -> str:
    """解析代码高亮主题名。

    Args:
        style_name: 调用方传入的代码高亮主题名。

    Returns:
        可以直接交给 `Pygments` 的样式名。
    """

    normalized = str(style_name or "default").strip().lower() or "default"
    return _PYGMENTS_STYLE_ALIASES.get(normalized, normalized)


def _build_html_document(
    *,
    body_html: str,
    auto_width: bool,
    width: int,
    min_width: int,
    max_width: int,
    padding: int,
    theme_css: str,
    code_theme_css: str,
    custom_css: str,
) -> str:
    """构造用于截图的完整 HTML 文档。

    当前模板只依赖内联样式，不引用远程字体、脚本或主题文件，避免机器人
    在无外网、CDN 波动或工作目录切换时出现白图、超时或样式不一致的问题。

    Args:
        body_html: 已由 Markdown 转换后的 HTML 片段。
        auto_width: 是否启用自动宽度布局。
        width: 固定宽度模式的页面主内容宽度。
        min_width: 自动宽度模式的最小宽度。
        max_width: 自动宽度模式的最大宽度。
        padding: 页面内边距。
        theme_css: 当前内置主题 CSS。
        code_theme_css: 代码高亮主题 CSS。
        custom_css: 调用方追加的自定义 CSS。

    Returns:
        可直接交给浏览器 `set_content()` 的完整 HTML 文本。
    """

    width_css = (
        "width: fit-content; min-width: var(--page-min-width); max-width: var(--page-max-width);"
        if auto_width
        else "width: var(--page-width);"
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <link
    rel="stylesheet"
    href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css"
  />
  <style>
    :root {{
      color-scheme: light;
      --page-width: {int(width)}px;
      --page-min-width: {int(min_width)}px;
      --page-max-width: {int(max_width)}px;
      --page-padding: {int(padding)}px;
      --line: #d9dde5;
      --text: #1f2328;
      --muted: #57606a;
      --code: #eef2f6;
      --accent: #0969da;
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      padding: 24px;
      color: var(--text);
      font: 16px/1.75 "Microsoft YaHei", "PingFang SC", "Segoe UI", sans-serif;
    }}
    #capture {{
      {width_css}
      margin: 0 auto;
      padding: var(--page-padding);
      border-radius: 20px;
      overflow: hidden;
    }}
    #capture > *:first-child {{
      margin-top: 0;
    }}
    #capture > *:last-child {{
      margin-bottom: 0;
    }}
    h1, h2, h3, h4, h5, h6 {{
      margin: 1.25em 0 0.6em;
      line-height: 1.3;
      color: #101828;
    }}
    h1 {{
      font-size: 2em;
      padding-bottom: 0.3em;
      border-bottom: 1px solid var(--line);
    }}
    h2 {{
      font-size: 1.5em;
      padding-bottom: 0.25em;
      border-bottom: 1px solid var(--line);
    }}
    p, ul, ol, blockquote, table, pre {{
      margin: 0 0 1em;
    }}
    ul, ol {{
      padding-left: 1.5em;
    }}
    li + li {{
      margin-top: 0.35em;
    }}
    a {{
      color: var(--accent);
      text-decoration: none;
    }}
    code {{
      padding: 0.12em 0.35em;
      border-radius: 6px;
      background: var(--code);
      font: 0.92em/1.4 "Cascadia Code", "Consolas", monospace;
    }}
    pre {{
      padding: 16px 18px;
      border-radius: 14px;
      border: 1px solid rgba(15, 23, 42, 0.06);
      overflow-x: auto;
      line-height: 1.6;
    }}
    pre code {{
      padding: 0;
      background: transparent;
      color: inherit;
      font-size: 0.92em;
    }}
    .codehilite {{
      margin: 0 0 1em;
      border-radius: 14px;
      overflow: hidden;
    }}
    .codehilite pre {{
      margin: 0;
      border: 0;
      border-radius: 0;
    }}
    blockquote {{
      padding: 12px 16px;
      border-left: 4px solid;
      border-radius: 0 12px 12px 0;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      overflow: hidden;
      border-radius: 12px;
      border-style: hidden;
      box-shadow: 0 0 0 1px var(--line);
    }}
    th, td {{
      padding: 10px 12px;
      border: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}
    th {{
      font-weight: 700;
    }}
    img {{
      max-width: 100%;
      height: auto;
      border-radius: 12px;
    }}
    hr {{
      border: 0;
      border-top: 1px solid var(--line);
      margin: 1.5em 0;
    }}
    {theme_css}
    {code_theme_css}
    {custom_css}
  </style>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
  <script defer>
    function sleep(ms) {{
      return new Promise(resolve => setTimeout(resolve, ms));
    }}

    async function waitFor(predicate, timeoutMs) {{
      const start = Date.now();
      while (Date.now() - start < timeoutMs) {{
        try {{
          if (predicate()) return true;
        }} catch (_err) {{
          // ignore
        }}
        await sleep(50);
      }}
      return false;
    }}

    async function renderMath() {{
      const ready = await waitFor(
        () => typeof window.renderMathInElement === "function",
        10000
      );
      if (!ready) return;
      try {{
        window.renderMathInElement(document.getElementById("capture"), {{
          delimiters: [
            {{ left: "$$", right: "$$", display: true }},
            {{ left: "$", right: "$", display: false }}
          ],
          throwOnError: false
        }});
      }} catch (_err) {{
        // ignore
      }}
    }}

    async function renderMermaid() {{
      const blocks = Array.from(document.querySelectorAll("pre.mermaid"));
      if (blocks.length === 0) return;
      const ready = await waitFor(
        () => typeof window.mermaid !== "undefined",
        10000
      );
      if (!ready) return;
      try {{
        window.mermaid.initialize({{ startOnLoad: false }});
        const result = window.mermaid.run({{ querySelector: "pre.mermaid" }});
        if (result && typeof result.then === "function") {{
          await result;
        }}
      }} catch (_err) {{
        // ignore
      }}
    }}

    window.addEventListener("DOMContentLoaded", async () => {{
      try {{
        await renderMath();
        await renderMermaid();
      }} finally {{
        window.__M2I_READY__ = true;
      }}
    }});
  </script>
</head>
<body>
  <article id="capture">{body_html}</article>
</body>
</html>
"""


def _build_output_path(*, output_dir: Path, markdown_text: str) -> Path:
    """根据输入内容生成稳定但不冲突的输出文件路径。

    相同 Markdown 内容在同一目录下会尽量复用文件名，便于排查缓存与人工检查；
    但文件名只使用内容哈希前缀，不暴露原文片段，避免把敏感文本直接写入路径。

    Args:
        output_dir: 输出目录。
        markdown_text: 原始 Markdown 文本。

    Returns:
        最终 PNG 文件路径。
    """

    digest = hashlib.sha1(markdown_text.encode("utf-8")).hexdigest()[:16]
    return output_dir / f"markdown_image_{digest}.png"


def _replace_mermaid_fences(markdown_text: str) -> str:
    """把 Mermaid fenced code 预处理为可直接渲染的 HTML 块。

    Args:
        markdown_text: 原始 Markdown 文本。

    Returns:
        已完成 Mermaid 代码块替换的 Markdown 文本。
    """

    pattern = re.compile(r"```mermaid\s*\r?\n(.*?)\r?\n```", flags=re.DOTALL | re.IGNORECASE)

    def _replace(match: re.Match[str]) -> str:
        mermaid_source = match.group(1).strip("\r\n")
        return f'<pre class="mermaid">{html.escape(mermaid_source)}</pre>'

    return pattern.sub(_replace, markdown_text)


def _coerce_box_number(box: Any | None, key: str, default: int) -> int:
    """从截图边界对象中安全读取数值字段。

    Playwright 的 `bounding_box()` 返回类型在静态分析里不是普通 `dict`，直接把
    `box["width"]` 或 `box.get(...)` 交给 `int()` 会让类型检查器认为其中仍可能
    是 `None`。这里统一先做一次显式判空和数值转换，避免把类型噪音扩散到主流程。

    Args:
        box: `bounding_box()` 返回的对象，可能为 `None`。
        key: 需要读取的字段名，如 `width` 或 `height`。
        default: 读取失败时使用的默认整数值。

    Returns:
        成功读取并转换后的整数值；若字段缺失或无法转换，则返回 `default`。
    """

    getter = getattr(box, "get", None)
    if not callable(getter):
        return int(default)
    value = getter(key, default)
    if value is None:
        return int(default)
    if not isinstance(value, (int, float, str, bytes, bytearray)):
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


async def render_markdown_to_image(
    markdown_text: str,
    *,
    output_dir: Path,
    theme_base_dir: Path | None = None,
    width: int | None = None,
    auto_width: bool = True,
    min_width: int | None = None,
    max_width: int | None = None,
    padding: int | None = None,
    scale: float | None = None,
    theme: str | None = None,
    custom_css: str | None = None,
    code_theme: str | None = None,
) -> MarkdownRenderResult:
    """把 Markdown 文本渲染成 PNG 图片。

    渲染流程分为三步：先用 Python Markdown 生成 HTML，再由 Playwright 在内存
    中加载文档并截图，最后把结果保存到插件自己的数据目录。函数不会尝试发送
    图片，也不会向 GT 文件系统注册映射，这些副作用统一留给工具层处理。

    Args:
        markdown_text: 待渲染的 Markdown 文本。
        output_dir: 图片输出目录。
        theme_base_dir: 解析相对主题文件路径时使用的基准目录。
        width: 固定宽度模式下的内容区域宽度。
        auto_width: 是否根据内容自动收缩宽度。
        min_width: 自动宽度模式的最小宽度。
        max_width: 自动宽度模式的最大宽度。
        padding: 内容区域内边距。
        scale: 截图时的设备缩放倍率。
        theme: 内置页面主题名称。
        custom_css: 追加到最终文档中的自定义 CSS 文本。
        code_theme: 代码高亮主题名称。

    Returns:
        包含输出文件路径和截图尺寸的渲染结果。

    Raises:
        RuntimeError: 当 Markdown、Pygments 或 Playwright 依赖缺失，或 Chromium 不可用时抛出。
        ValueError: 当宽度、主题、缩放倍率或内边距不合法时抛出。
    """

    normalized_width = _normalize_fixed_width(width)
    normalized_min_width, normalized_max_width = _normalize_auto_width_bounds(min_width, max_width)
    normalized_padding = _normalize_padding(padding)
    normalized_scale = _normalize_scale(scale)
    theme_css = _resolve_theme_css(theme, theme_base_dir=theme_base_dir)
    normalized_custom_css = str(custom_css or "").strip()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = _build_output_path(output_dir=output_dir, markdown_text=markdown_text)

    try:
        from markdown import markdown
    except ImportError as exc:
        raise RuntimeError("缺少 Markdown 依赖，请先安装项目依赖后再使用该工具") from exc

    try:
        from pygments.formatters import HtmlFormatter
        from pygments.styles import get_all_styles
    except ImportError as exc:
        raise RuntimeError("缺少 Pygments 依赖，请先安装项目依赖后再使用该工具") from exc

    try:
        from playwright.async_api import Error as PlaywrightError
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise RuntimeError("缺少 Playwright 依赖，请先安装项目依赖后再使用该工具") from exc

    pygments_style = _resolve_pygments_style_name(code_theme)
    available_styles = set(get_all_styles())
    if pygments_style not in available_styles:
        raise ValueError(f"code_theme 不存在: {pygments_style}")
    code_theme_css = HtmlFormatter(style=pygments_style).get_style_defs(".codehilite")

    processed_markdown = _replace_mermaid_fences(markdown_text)
    body_html = markdown(
        processed_markdown,
        extensions=[
            "fenced_code",
            "tables",
            "nl2br",
            "sane_lists",
            "codehilite",
        ],
        extension_configs={
            "codehilite": {
                "guess_lang": False,
                "use_pygments": True,
                "noclasses": False,
            },
        },
    )
    html = _build_html_document(
        body_html=body_html,
        auto_width=bool(auto_width and normalized_width is None),
        width=normalized_width or normalized_max_width,
        min_width=normalized_min_width,
        max_width=normalized_max_width,
        padding=normalized_padding,
        theme_css=theme_css,
        code_theme_css=code_theme_css,
        custom_css=normalized_custom_css,
    )

    initial_viewport_width = (
        normalized_max_width if normalized_width is None else normalized_width
    ) + normalized_padding * 2 + 96
    box: Any | None = None
    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            try:
                page = await browser.new_page(
                    viewport={"width": int(initial_viewport_width), "height": 800},
                    device_scale_factor=normalized_scale,
                )
                await page.set_content(html, wait_until="load")
                await page.wait_for_function("() => window.__M2I_READY__ === true", timeout=15000)
                locator = page.locator("#capture")
                box = await locator.bounding_box()
                await locator.screenshot(path=str(output_path))
            finally:
                await browser.close()
    except PlaywrightError as exc:
        message = str(exc)
        if "Executable doesn't exist" in message or "browserType.launch" in message:
            raise RuntimeError("Playwright Chromium 未安装，请执行 `playwright install chromium`") from exc
        raise RuntimeError(f"Markdown 图片渲染失败: {message}") from exc

    rendered_width = _coerce_box_number(box, "width", normalized_width or normalized_max_width)
    rendered_height = _coerce_box_number(box, "height", 0)
    return MarkdownRenderResult(
        image_path=output_path,
        width=rendered_width,
        height=rendered_height,
    )
