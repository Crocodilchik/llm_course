from __future__ import annotations
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os


class Embedder:
    """Bounded CPU worker: model loading/inference never blocks the event loop."""
    dimension = 384

    def __init__(self, settings):
        self.settings = settings
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="embeddings")
        self.model = None

    def _encode(self, texts: list[str], query: bool):
        if self.model is None:
            cache = self.settings.model_cache
            cache.mkdir(parents=True, exist_ok=True)
            os.environ["HF_HOME"] = str(cache)
            os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            from sentence_transformers import SentenceTransformer
            import torch
            torch.set_num_threads(2)
            self.model = SentenceTransformer(self.settings.embedding_model,
                revision=self.settings.embedding_revision, cache_folder=str(cache),
                device="cpu", token=False, trust_remote_code=False)
        prefix = "query: " if query else "passage: "
        vectors = self.model.encode([prefix + text for text in texts], normalize_embeddings=True,
                                    batch_size=32, show_progress_bar=False).tolist()
        if any(len(v) != self.dimension for v in vectors):
            raise ValueError("embedding_dimension_mismatch")
        return vectors

    async def encode(self, texts: list[str], query: bool = False):
        return await asyncio.get_running_loop().run_in_executor(self.executor, self._encode, texts, query)

    def close(self):
        self.executor.shutdown(wait=False, cancel_futures=True)
