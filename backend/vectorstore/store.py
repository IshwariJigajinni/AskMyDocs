import faiss
import numpy as np
import os
import pickle
from backend.vectorstore.embedder import generate_embeddings

class VectorStore:
    def __init__(self, index_path="vectorstore/index.faiss", meta_path="vectorstore/meta.pkl"):
        self.index_path = index_path
        self.meta_path = meta_path
        self.index = None
        self.texts = []  # list of {"text": ..., "metadata": ...}
        self.load()      # load existing index and metadata if available

    # ---------------- Add a single text ----------------
    def add_text(self, text, metadata=None):
        vector = np.array(generate_embeddings(text)).astype("float32")
        self.add([vector], [{"text": text, "metadata": metadata}])

    # ---------------- Add multiple vectors ----------------
    def add(self, embeddings, metadatas):
        if not embeddings or not metadatas:
            return
        dim = embeddings[0].shape[0]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings).astype("float32"))
        self.texts.extend(metadatas)
        self.save()  # save after each addition

    # ---------------- Save index + metadata ----------------
    def save(self):
        if self.index:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.texts, f)

    # ---------------- Load index + metadata ----------------
    def load(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "rb") as f:
                self.texts = pickle.load(f)

    # ---------------- Similarity search by query text ----------------
    def similarity_search(self, query, top_k=3):
        vector = np.array(generate_embeddings(query)).astype("float32")
        return self._search_with_vector(vector, top_k)

    # ---------------- Similarity search by embedding ----------------
    def similarity_search_by_vector(self, vector, top_k=3):
        return self._search_with_vector(vector, top_k)

    # ---------------- Internal search ----------------
    def _search_with_vector(self, vector, top_k):
        if self.index is None or self.index.ntotal == 0:
            return []
        D, I = self.index.search(np.array([vector]), top_k * 2)
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < len(self.texts):
                results.append({
                    "text": self.texts[idx]["text"],
                    "metadata": self.texts[idx]["metadata"],
                    "score": float(1 / (1 + dist))  # convert distance to similarity
                })
        # Remove duplicates and limit to top_k
        seen, unique_results = set(), []
        for r in results:
            if r["text"] not in seen:
                unique_results.append(r)
                seen.add(r["text"])
            if len(unique_results) >= top_k:
                break
        return unique_results
