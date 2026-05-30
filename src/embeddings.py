import os
from pathlib import Path

import numpy as np
import pandas as pd
from ollama import Client
from openai import OpenAI
from tqdm import tqdm

from config import load_env


class EmbeddingService:
    """Generate embeddings using a local Ollama instance or an OpenAI-compatible API."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        load_env()
        self.model = model or os.getenv("EMBEDDING_MODEL", "bge-m3")
        self.base_url = base_url or os.getenv("BASE_URL", "http://localhost:11434")
        self.api_key = api_key or os.getenv("API_KEY") or None

        if self.api_key:
            self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        else:
            self.client = Client(host=self.base_url)

    def embed(self, text: str) -> list[float]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("`text` must be a non-empty string")

        try:
            if self.api_key:
                response = self.client.embeddings.create(model=self.model, input=text)
                return response.data[0].embedding

            response = self.client.embed(model=self.model, input=text)
            return response.embeddings[0]

        except Exception as e:
            provider = "OpenAI-compatible API" if self.api_key else "Ollama"
            raise RuntimeError(f"{provider} embedding failed: {e}") from e

    def _embed_batch_openai(self, texts: list[str]) -> list[list[float]]:
        """Batch embedding for OpenAI-compatible clients only."""
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]

    def attach_embeddings_to_dataframe(
        self,
        df: pd.DataFrame,
        cache_path: Path,
        batch_size: int = 100,
    ) -> pd.DataFrame:
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Load cache
        if cache_path.exists():
            cache = np.load(cache_path, allow_pickle=True)
            embedding_by_id = {
                int(id_): emb
                for id_, emb in zip(cache["ids"], cache["embeddings"])
            }
        else:
            embedding_by_id = {}

        # Compute embeddings for rows not in cache
        missing_df = df[df["Id"].apply(lambda i: int(i) not in embedding_by_id)]

        if not missing_df.empty:
            if self.api_key:
                for start in tqdm(range(0, len(missing_df), batch_size), desc="Generating embeddings", unit="batch"):
                    chunk = missing_df.iloc[start : start + batch_size]
                    ids = chunk["Id"].astype(int).tolist()
                    embeddings = self._embed_batch_openai(chunk["Text"].tolist())
                    for review_id, embedding in zip(ids, embeddings):
                        embedding_by_id[review_id] = embedding
            else:
                for _, row in tqdm(missing_df.iterrows(), total=len(missing_df), desc="Generating embeddings", unit="text"):
                    embedding_by_id[int(row["Id"])] = self.embed(row["Text"])

            # Persist updated cache
            np.savez(
                cache_path,
                ids=np.array(list(embedding_by_id.keys())),
                embeddings=np.array(list(embedding_by_id.values())),
            )

        # Attach to dataframe preserving original row order
        df = df.copy()
        tqdm.pandas(desc="Loading embeddings from cache")
        df["embedding"] = df["Id"].progress_apply(lambda i: embedding_by_id[int(i)])

        return df


if __name__ == "__main__":
    embedding_service = EmbeddingService()
    embedding = embedding_service.embed("Hello, world!")
    print(f"Embedding length: {len(embedding)}")