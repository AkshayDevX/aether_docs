import os
import sys
import numpy as np
from typing import List

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

_local_model_cache = None

class EmbeddingGenerator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_local: bool = True):
        self.model_name = model_name
        self.use_local = use_local and HAS_SENTENCE_TRANSFORMERS
        self.is_mock = False
        
        if not self.use_local:
            # We'll use OpenAI embedding API compatibility
            print("Configuring remote API for vector embeddings...", file=sys.stderr)
            self.api_key = os.environ.get("OPENAI_API_KEY")
            self.base_url = os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
            
            if not self.api_key:
                print("WARNING: OPENAI_API_KEY not found in environment variables. "
                      "Remote embedding operations will fall back to mock vectors unless key is provided.", file=sys.stderr)

    def _get_local_model(self):
        global _local_model_cache
        if not self.use_local:
            return None
        if _local_model_cache is None:
            print(f"Loading local SentenceTransformer model '{self.model_name}' for the first time...", file=sys.stderr)
            try:
                _local_model_cache = SentenceTransformer(self.model_name)
                print("Local SentenceTransformer loaded successfully.", file=sys.stderr)
            except Exception as e:
                print(f"Failed to load local model {self.model_name} due to: {e}. Falling back to API embeddings.", file=sys.stderr)
                self.use_local = False
                return None
        return _local_model_cache

    def get_embedding(self, text: str) -> np.ndarray:
        """Generates a 1D float32 numpy array embedding for a given text."""
        if not text.strip():
            # Return empty embedding of size 384 (MiniLM standard) or 1536 (OpenAI standard)
            dim = 384 if self.use_local else 1536
            return np.zeros(dim, dtype=np.float32)

        local_model = self._get_local_model()
        if self.use_local and local_model:
            emb = local_model.encode(text, convert_to_numpy=True)
            return emb.astype(np.float32)
        else:
            # Remote API fallback
            return self._get_remote_embedding(text)

    def get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Generates embeddings for a list of texts in batch."""
        local_model = self._get_local_model()
        if self.use_local and local_model:
            embs = local_model.encode(texts, convert_to_numpy=True)
            return [emb.astype(np.float32) for emb in embs]
        else:
            return [self.get_embedding(text) for text in texts]

    def _get_remote_embedding(self, text: str) -> np.ndarray:
        """Calls the OpenAI API to fetch embeddings."""
        import urllib.request
        import json

        api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        base_url = self.base_url or "https://api.openai.com/v1"
        model = "text-embedding-3-small"
        
        if not api_key:
            print("No OpenAI API key available for remote embedding. Generating stable mock vector...", file=sys.stderr)
            self.is_mock = True
            return self._generate_stable_mock_embedding(text, 384)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        data = {
            "input": text,
            "model": model
        }
        
        req = urllib.request.Request(
            f"{base_url}/embeddings",
            data=json.dumps(data).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        
        try:
            with urllib.request.urlopen(req) as response:
                res_body = json.loads(response.read().decode("utf-8"))
                embedding_vector = res_body["data"][0]["embedding"]
                return np.array(embedding_vector, dtype=np.float32)
        except Exception as e:
            print(f"Error fetching API embedding: {e}. Generating stable mock vector...", file=sys.stderr)
            self.is_mock = True
            return self._generate_stable_mock_embedding(text, 384)

    def _generate_stable_mock_embedding(self, text: str, dimension: int = 384) -> np.ndarray:
        """Generates a stable pseudo-random vector based on text hash (for zero-dependency offline runs)."""
        import hashlib
        self.is_mock = True
        # Hash text to generate a deterministic seed
        hash_seed = int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(hash_seed)
        vec = rng.standard_normal(dimension).astype(np.float32)
        # Normalize the vector
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec
