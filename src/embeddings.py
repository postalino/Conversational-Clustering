import os
from typing import Optional, List

from ollama import Client
from openai import OpenAI

from config import load_env
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


class EmbeddingService:
    """Genera embedding usando Ollama locale o API compatibile OpenAI."""

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        load_env()
        self.model = model or os.getenv("EMBEDDING_MODEL", "bge-m3")
        self.base_url = base_url or os.getenv("BASE_URL", "http://localhost:11434")
        self.api_key = api_key or os.getenv("API_KEY") or None

        if self.api_key:
            self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        else:
            self.client = Client(host=self.base_url)

    def embed(self, text: str) -> List[float]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("`text` must be a non-empty string")

        try:
            if self.api_key:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=text,
                )
                return response.data[0].embedding

            response = self.client.embed(
                model=self.model,
                input=text,
            )
            return response.embeddings[0]

        except Exception as e:
            provider = "OpenAI-compatible API" if self.api_key else "Ollama"
            raise RuntimeError(f"{provider} embedding failed: {e}") from e

    def _embed_batch_openai(self, texts: List[str]) -> List[List[float]]:
        """Batch embedding solo per client OpenAI-compatible (es. OpenRouter)."""
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]

    def attach_embeddings_to_dataframe(
        self,
        df: pd.DataFrame,
        cache_path: Path,
        batch_size: int = 100,
    ) -> pd.DataFrame:
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # --- carica cache ---
        if cache_path.exists():
            cache = np.load(cache_path, allow_pickle=True)
            embedding_by_id = {
                int(id_): emb
                for id_, emb in zip(cache["ids"], cache["embeddings"])
            }
        else:
            embedding_by_id = {}

        # --- righe senza embedding in cache ---
        missing_df = df[df["Id"].apply(lambda i: int(i) not in embedding_by_id)]

        if not missing_df.empty:
            if self.api_key:
                batches = range(0, len(missing_df), batch_size)
                for start in tqdm(batches, desc="Generating Embeddings", unit="batch"):
                    chunk = missing_df.iloc[start : start + batch_size]
                    ids = chunk["Id"].astype(int).tolist()
                    texts = chunk["Text"].tolist()

                    embeddings = self._embed_batch_openai(texts)

                    for review_id, embedding in zip(ids, embeddings):
                        embedding_by_id[review_id] = embedding
            else:
                for _, row in tqdm(missing_df.iterrows(), total=len(missing_df), desc="Embedding", unit="testo"):
                    review_id = int(row["Id"])
                    embedding_by_id[review_id] = self.embed(row["Text"])

            # --- aggiorna cache solo se ci sono novità ---
            np.savez(
                cache_path,
                ids=np.array(list(embedding_by_id.keys())),
                embeddings=np.array(list(embedding_by_id.values())),
            )

        # --- assegna al df preservando l'ordine originale ---
        df = df.copy()
        tqdm.pandas(desc="Loading embeddings from cache")
        df["embedding"] = df["Id"].progress_apply(lambda i: embedding_by_id[int(i)])

        return df

if __name__ == "__main__":
    embedding_service = EmbeddingService()
    embedding = embedding_service.embed("Hello, world!")
    print(f"embedding length: {len(embedding)}")