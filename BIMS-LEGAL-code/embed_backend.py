"""嵌入后端：Ollama HTTP 或进程内 GPU SentenceTransformer（高 batch、高 GPU 利用率）。"""
from __future__ import annotations

import logging
import os
import sqlite3
import struct
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import requests

from config import (
    VECTOR_DIM,
    OLLAMA_BASE_URL,
    EMBEDDING_MODEL,
    EMBED_MAX_RETRIES,
    EMBED_BATCH_SIZE,
    EMBED_OLLAMA_WORKERS,
    EMBED_DISK_CACHE_DIR,
    EMBED_HF_MODEL,
    EMBED_GPU_BATCH_SIZE,
    EMBED_GPU_FP16,
)

logger = logging.getLogger(__name__)

_PACK = struct.Struct(f"{VECTOR_DIM}f")


class EmbedDiskCache:
    """磁盘向量缓存（以空间换时间，跨实例/跨 run 复用）。"""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS emb (h TEXT PRIMARY KEY, vec BLOB NOT NULL)"
        )
        self.conn.commit()

    def get_many(self, keys: List[str]) -> dict[str, np.ndarray]:
        if not keys:
            return {}
        out: dict[str, np.ndarray] = {}
        chunk = 500
        for i in range(0, len(keys), chunk):
            part = keys[i:i + chunk]
            placeholders = ",".join("?" * len(part))
            rows = self.conn.execute(
                f"SELECT h, vec FROM emb WHERE h IN ({placeholders})", part
            ).fetchall()
            for h, blob in rows:
                if len(blob) == _PACK.size:
                    out[h] = np.array(_PACK.unpack(blob), dtype=np.float32)
        return out

    def put_many(self, items: List[tuple[str, np.ndarray]]) -> None:
        if not items:
            return
        rows = [(h, _PACK.pack(*v.astype(np.float32).tolist())) for h, v in items]
        self.conn.executemany("INSERT OR REPLACE INTO emb(h, vec) VALUES (?, ?)", rows)
        self.conn.commit()


class BaseEmbedBackend(ABC):
    api_calls = 0

    @abstractmethod
    def encode_batch(self, texts: List[str], *, is_query: bool = False) -> List[np.ndarray]:
        ...


class OllamaEmbedBackend(BaseEmbedBackend):
    def encode_batch(self, texts: List[str], *, is_query: bool = False) -> List[np.ndarray]:
        if not texts:
            return []
        out: List[np.ndarray] = []
        batch_size = EMBED_BATCH_SIZE
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            vecs = None
            for attempt in range(EMBED_MAX_RETRIES):
                try:
                    self.api_calls += 1
                    resp = requests.post(
                        f"{OLLAMA_BASE_URL}/api/embed",
                        json={"model": EMBEDDING_MODEL, "input": chunk},
                        timeout=max(180, 60 + len(chunk) * 2),
                    )
                    if resp.status_code == 200:
                        embs = resp.json().get("embeddings", [])
                        if len(embs) == len(chunk):
                            vecs = [
                                np.array(e, dtype="float32") if len(e) == VECTOR_DIM
                                else np.zeros(VECTOR_DIM, dtype="float32")
                                for e in embs
                            ]
                            break
                    logger.warning(
                        "Ollama batch embed status=%s len=%s attempt=%s",
                        resp.status_code, len(chunk), attempt + 1,
                    )
                except Exception as e:
                    logger.error("Ollama batch embed failed attempt=%s: %s", attempt + 1, e)
            if vecs is None:
                vecs = [self._one(t) for t in chunk]
            out.extend(vecs)
        return out

    def _one(self, text: str) -> np.ndarray:
        for attempt in range(EMBED_MAX_RETRIES):
            try:
                self.api_calls += 1
                resp = requests.post(
                    f"{OLLAMA_BASE_URL}/api/embeddings",
                    json={"model": EMBEDDING_MODEL, "prompt": text},
                    timeout=60,
                )
                if resp.status_code == 200:
                    emb = resp.json().get("embedding", [])
                    if len(emb) == VECTOR_DIM:
                        return np.array(emb, dtype="float32")
            except Exception as e:
                logger.error("Ollama embed failed attempt=%s: %s", attempt + 1, e)
        return np.zeros(VECTOR_DIM, dtype="float32")


class OllamaParallelEmbedBackend(OllamaEmbedBackend):
    """多线程并发 Ollama batch 请求（无需 HF，比单线程 batch 再快 ~2x）。"""

    def encode_batch(self, texts: List[str], *, is_query: bool = False) -> List[np.ndarray]:
        if not texts:
            return []
        from concurrent.futures import ThreadPoolExecutor

        batch_size = max(EMBED_BATCH_SIZE, 32)
        workers = EMBED_OLLAMA_WORKERS
        chunks = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        results: List[List[np.ndarray]] = [[] for _ in chunks]

        def _run(idx_chunk):
            idx, chunk = idx_chunk
            return idx, super(OllamaParallelEmbedBackend, self).encode_batch(chunk, is_query=is_query)

        pool_workers = min(workers, max(len(chunks), 1))
        with ThreadPoolExecutor(max_workers=pool_workers) as ex:
            for idx, vecs in ex.map(_run, enumerate(chunks)):
                results[idx] = vecs
        out: List[np.ndarray] = []
        for part in results:
            out.extend(part)
        return out


class GpuLocalEmbedBackend(BaseEmbedBackend):
    """进程内 GPU 推理：大 batch + fp16，显著提高 GPU 利用率。"""

    _inst: Optional["GpuLocalEmbedBackend"] = None

    @classmethod
    def get(cls) -> "GpuLocalEmbedBackend":
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    def __init__(self) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        device = os.environ.get("EMBED_GPU_DEVICE", "cuda:0")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("EMBED_BACKEND=gpu_local 但 CUDA 不可用")
        logger.info(
            "加载 GPU 嵌入模型 %s device=%s batch=%s fp16=%s",
            EMBED_HF_MODEL, device, EMBED_GPU_BATCH_SIZE, EMBED_GPU_FP16,
        )
        self._model = SentenceTransformer(EMBED_HF_MODEL, trust_remote_code=True, device=device)
        if EMBED_GPU_FP16 and device.startswith("cuda"):
            try:
                self._model.half()
            except Exception as e:
                logger.warning("fp16 转换失败，使用 fp32: %s", e)
        self._batch_size = EMBED_GPU_BATCH_SIZE
        self._device = device

    def encode_batch(self, texts: List[str], *, is_query: bool = False) -> List[np.ndarray]:
        if not texts:
            return []
        prompt = "query" if is_query else "passage"
        out: List[np.ndarray] = []
        for start in range(0, len(texts), self._batch_size):
            chunk = texts[start:start + self._batch_size]
            self.api_calls += 1
            embs = self._model.encode(
                chunk,
                batch_size=len(chunk),
                prompt_name=prompt,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
            for row in embs:
                v = np.asarray(row, dtype=np.float32)
                if v.shape[0] != VECTOR_DIM:
                    v = np.zeros(VECTOR_DIM, dtype=np.float32)
                out.append(v)
        return out


def get_embed_backend() -> BaseEmbedBackend:
    name = os.environ.get("EMBED_BACKEND", "ollama").lower()
    if name in ("gpu", "gpu_local", "local"):
        return GpuLocalEmbedBackend.get()
    if name in ("ollama_parallel", "parallel"):
        return OllamaParallelEmbedBackend()
    return OllamaEmbedBackend()


_disk_cache: Optional[EmbedDiskCache] = None


def get_disk_cache() -> Optional[EmbedDiskCache]:
    global _disk_cache
    if os.environ.get("USE_EMBED_DISK_CACHE", "0").lower() not in ("1", "true", "yes"):
        return None
    if _disk_cache is None:
        _disk_cache = EmbedDiskCache(EMBED_DISK_CACHE_DIR)
    return _disk_cache


def stable_text_hash(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
