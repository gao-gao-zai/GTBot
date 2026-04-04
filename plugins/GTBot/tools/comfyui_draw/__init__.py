from __future__ import annotations

import importlib

from ...services.help import HelpCommandSpec, register_help
from ...services.permission import PermissionRole

def register(registry) -> None:  # noqa: ANN001
    cfg_mod = importlib.import_module(__name__ + ".config")
    getattr(cfg_mod, "get_comfyui_draw_plugin_config")()
    mod = importlib.import_module(__name__ + ".tool")
    registry.add_tool(getattr(mod, "comfyui_draw_image"))


try:
    from nonebot import on_command
    from nonebot.adapters.onebot.v11.event import GroupMessageEvent
except Exception:  # noqa: BLE001
    on_command = None
    GroupMessageEvent = None


if on_command is not None:
    manager_mod = importlib.import_module(__name__ + ".manager")
    get_draw_queue_manager = getattr(manager_mod, "get_draw_queue_manager")

    QueryDrawTasks = on_command(
        "绘图任务",
        aliases={"画图任务", "draw_tasks"},
        priority=-5,
        block=True,
    )

    register_help(
        HelpCommandSpec(
            name="绘图任务",
            aliases=("画图任务", "draw_tasks"),
            category="绘图服务",
            summary="查看当前绘图队列和运行中的任务。",
            description="展示 ComfyUI 绘图队列的运行数、排队数，以及最近的运行中和排队任务摘要。",
            examples=(
                "/绘图任务",
            ),
            required_role=PermissionRole.USER,
            audience="群聊",
            sort_key=10,
        )
    )

    @QueryDrawTasks.handle()
    async def _handle_query_draw_tasks(event: GroupMessageEvent) -> None:  # type: ignore[valid-type]
        manager = get_draw_queue_manager()
        snap = await manager.snapshot()

        running = snap.get("running") or []
        queued = snap.get("queued") or []
        running_count = int(snap.get("running_count") or 0)
        queued_count = int(snap.get("queued_count") or 0)
        queue_max = int(snap.get("queue_max") or 0)

        lines: list[str] = []
        lines.append(f"running={running_count} queued={queued_count}/{queue_max}")

        def _fmt(item: dict) -> str:
            job_id = item.get("job_id", "")
            status = item.get("status", "")
            w = item.get("width", "")
            h = item.get("height", "")
            seed = item.get("seed", "")
            target = item.get("target_user_id", "")
            prompt = item.get("prompt_preview", "")
            return f"- {job_id} {status} {w}x{h} seed={seed} target={target} prompt={prompt}"

        if running:
            lines.append("[running]")
            for it in running[:10]:
                if isinstance(it, dict):
                    lines.append(_fmt(it))

        if queued:
            lines.append("[queued]")
            for it in queued[:10]:
                if isinstance(it, dict):
                    lines.append(_fmt(it))

        await QueryDrawTasks.finish("\n".join(lines))
