from __future__ import annotations

import unittest
from dataclasses import dataclass
from types import ModuleType
from types import SimpleNamespace
from typing import Any
from uuid import uuid4


def _load_longmemory_package(longmemory_dir: str) -> str:
    package_name = f"_longmemory_group_profile_unit_testpkg_{uuid4().hex}"
    pkg = ModuleType(package_name)
    pkg.__path__ = [longmemory_dir]  # type: ignore[attr-defined]
    import sys

    sys.modules[package_name] = pkg
    return package_name


def _load_module_from_path(module_qualname: str, file_path: str) -> ModuleType:
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(module_qualname, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法创建 spec: {module_qualname} -> {file_path}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_qualname] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_manager_cls() -> type:
    from pathlib import Path
    import sys
    from typing import Protocol, runtime_checkable

    longmemory_dir = str(Path(__file__).resolve().parents[1])
    pkg = _load_longmemory_package(longmemory_dir)

    _load_module_from_path(f"{pkg}.model", str(Path(longmemory_dir) / "model.py"))

    # 为了让该单测尽量不依赖外部环境（例如 pydantic/aiohttp），这里不加载真实的 VectorGenerator.py。
    # GroupProfileQdrant 只需要 `VectorGenerator` 协议用于类型标注，因此注入一个最小实现即可。
    vec_mod_name = f"{pkg}.VectorGenerator"
    vec_mod = ModuleType(vec_mod_name)

    @runtime_checkable
    class _VectorGenerator(Protocol):
        async def embed_query(self, text: str):  # noqa: ANN001
            ...

        async def embed_documents(self, texts: list[str]):  # noqa: ANN001
            ...

    setattr(vec_mod, "VectorGenerator", _VectorGenerator)
    sys.modules[vec_mod_name] = vec_mod

    gp_mod = _load_module_from_path(f"{pkg}.GroupProfileQdrant", str(Path(longmemory_dir) / "GroupProfileQdrant.py"))
    cls = getattr(gp_mod, "QdrantGroupProfileManager", None)
    if cls is None:
        raise RuntimeError("无法加载 QdrantGroupProfileManager")
    return cls


_VECTOR_DIM = 8


class _FakeVectorGenerator:
    async def embed_query(self, text: str):  # noqa: ANN001
        vecs = await self.embed_documents([text])
        return vecs[0]

    async def embed_documents(self, texts: list[str]):  # noqa: ANN001
        import numpy as np

        if not texts:
            raise ValueError("texts 不能为空")

        out = np.zeros((len(texts), _VECTOR_DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            b = str(t).encode("utf-8", errors="ignore")
            acc = np.zeros((_VECTOR_DIM,), dtype=np.float32)
            for j, v in enumerate(b):
                acc[j % _VECTOR_DIM] += float(v)
            norm = float(np.linalg.norm(acc))
            if norm > 0:
                acc /= norm
            out[i] = acc
        return out


@dataclass
class _StoredPoint:
    point_id: Any
    vector: list[float]
    payload: dict[str, Any]


class _FakeAsyncQdrantClient:
    def __init__(self) -> None:
        self._points: dict[str, _StoredPoint] = {}

    async def upsert(self, *, collection_name: str, points: list[Any]) -> None:  # noqa: ARG002
        for p in points:
            key = str(getattr(p, "id"))
            payload = dict(getattr(p, "payload") or {})
            vector = list(getattr(p, "vector"))
            self._points[key] = _StoredPoint(point_id=getattr(p, "id"), vector=vector, payload=payload)

    async def scroll(
        self,
        *,
        collection_name: str,  # noqa: ARG002
        scroll_filter: Any | None = None,
        limit: int = 256,
        offset: Any = None,
        with_payload: bool = True,  # noqa: FBT001, ARG002
        with_vectors: bool = False,  # noqa: FBT001, ARG002
    ) -> tuple[list[Any], Any]:
        required: dict[str, Any] = {}
        if scroll_filter is not None:
            must = getattr(scroll_filter, "must", None) or []
            for cond in must:
                key = getattr(cond, "key", "")
                match = getattr(cond, "match", None)
                value = getattr(match, "value", None) if match is not None else None
                if key:
                    required[str(key)] = value

        items: list[_StoredPoint] = []
        for sp in self._points.values():
            ok = True
            for k, v in required.items():
                if sp.payload.get(k) != v:
                    ok = False
                    break
            if ok:
                items.append(sp)

        items.sort(key=lambda x: str(x.point_id))

        start = int(offset or 0)
        end = min(start + int(limit), len(items))
        batch = items[start:end]
        next_offset = None if end >= len(items) else end

        out = []
        for sp in batch:
            out.append(
                SimpleNamespace(
                    id=sp.point_id,
                    payload=dict(sp.payload),
                    vector=list(sp.vector) if with_vectors else None,
                )
            )
        return out, next_offset

    async def retrieve(
        self,
        *,
        collection_name: str,  # noqa: ARG002
        ids: list[Any],
        with_payload: bool = True,  # noqa: FBT001
        with_vectors: bool = False,  # noqa: FBT001
    ) -> list[Any]:
        out = []
        for raw_id in ids:
            key = str(raw_id)
            sp = self._points.get(key)
            if sp is None:
                continue
            payload = dict(sp.payload) if with_payload else None
            vector = list(sp.vector) if with_vectors else None
            out.append(SimpleNamespace(id=sp.point_id, payload=payload, vector=vector))
        return out

    async def delete(self, *, collection_name: str, points_selector: Any) -> None:  # noqa: ARG002
        points = getattr(points_selector, "points", None) or []
        for raw_id in list(points):
            self._points.pop(str(raw_id), None)

    async def query_points(
        self,
        *,
        collection_name: str,  # noqa: ARG002
        query: list[float],
        limit: int,
        query_filter: Any | None = None,
        with_payload: bool = True,  # noqa: FBT001
        with_vectors: bool = False,  # noqa: FBT001, ARG002
    ) -> Any:
        import numpy as np

        required: dict[str, Any] = {}
        if query_filter is not None:
            must = getattr(query_filter, "must", None) or []
            for cond in must:
                key = getattr(cond, "key", "")
                match = getattr(cond, "match", None)
                value = getattr(match, "value", None) if match is not None else None
                if key:
                    required[str(key)] = value

        q = np.asarray(list(query), dtype=np.float32)
        qn = float(np.linalg.norm(q))
        if qn > 0:
            q /= qn

        scored: list[tuple[float, _StoredPoint]] = []
        for sp in self._points.values():
            ok = True
            for k, v in required.items():
                if sp.payload.get(k) != v:
                    ok = False
                    break
            if not ok:
                continue

            v = np.asarray(sp.vector, dtype=np.float32)
            vn = float(np.linalg.norm(v))
            if vn > 0:
                v = v / vn
            score = float(np.dot(q, v))
            scored.append((score, sp))

        scored.sort(key=lambda x: x[0], reverse=True)
        out_points: list[Any] = []
        for score, sp in scored[: int(limit)]:
            payload = dict(sp.payload) if with_payload else None
            out_points.append(SimpleNamespace(id=sp.point_id, payload=payload, score=score))

        return SimpleNamespace(points=out_points)


class TestQdrantGroupProfileManagerUnit(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        try:
            import numpy  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            raise unittest.SkipTest(f"跳过：缺少依赖 numpy: {exc}")

        try:
            manager_cls = _load_manager_cls()
        except Exception as exc:  # noqa: BLE001
            raise unittest.SkipTest(f"跳过：无法加载 QdrantGroupProfileManager: {exc}")

        self.client = _FakeAsyncQdrantClient()
        self.vector_generator = _FakeVectorGenerator()
        self.mgr = manager_cls(
            collection_name="unit_test",
            client=self.client,  # type: ignore[arg-type]
            vector_generator=self.vector_generator,  # type: ignore[arg-type]
        )

    async def test_add_get_count_update_delete_search(self) -> None:
        group_id = 123

        ids = await self.mgr.add_group_profile(group_id=group_id, profile_texts=["猫", "狗"], category="pet")
        self.assertEqual(len(ids), 2)

        got = await self.mgr.get_group_profiles(group_id, limit=10, sort_by="last_updated", sort_order="desc", touch_read_time=False)
        self.assertEqual(got.id, group_id)
        self.assertEqual(len(got.description), 2)

        c_all = await self.mgr.count_group_profile_descriptions(group_id)
        self.assertEqual(c_all, 2)

        c_cat = await self.mgr.count_group_profile_descriptions(group_id, category="pet")
        self.assertEqual(c_cat, 2)

        c_other = await self.mgr.count_group_profile_descriptions(group_id, category="other")
        self.assertEqual(c_other, 0)

        target_id = got.description[0].doc_id
        updated = await self.mgr.update_by_doc_id(target_id, description="猫猫")
        self.assertTrue(updated)

        got2 = await self.mgr.get_group_profiles(group_id, limit=10, sort_by="last_updated", sort_order="desc", touch_read_time=False)
        item = next((x for x in got2.description if x.doc_id == target_id), None)
        self.assertIsNotNone(item)
        assert item is not None
        self.assertEqual(item.text, "猫猫")

        hits = await self.mgr.search_group_profiles("猫猫", group_id=group_id, n_results=3)
        self.assertTrue(hits)
        self.assertTrue(any(h.doc_id == target_id for h in hits))

        deleted = await self.mgr.delete_many_by_doc_id(group_id, [target_id])
        self.assertEqual(deleted, 1)

        c_after = await self.mgr.count_group_profile_descriptions(group_id)
        self.assertEqual(c_after, 1)

    async def test_delete_many_by_doc_id_should_not_delete_other_group(self) -> None:
        gid1 = 1
        gid2 = 2

        ids1 = await self.mgr.add_group_profile(group_id=gid1, profile_texts="a")
        ids2 = await self.mgr.add_group_profile(group_id=gid2, profile_texts="b")
        self.assertEqual(len(ids1), 1)
        self.assertEqual(len(ids2), 1)

        deleted = await self.mgr.delete_many_by_doc_id(gid1, [ids2[0]])
        self.assertEqual(deleted, 0)

        c1 = await self.mgr.count_group_profile_descriptions(gid1)
        c2 = await self.mgr.count_group_profile_descriptions(gid2)
        self.assertEqual(c1, 1)
        self.assertEqual(c2, 1)


if __name__ == "__main__":
    unittest.main()
