"""
Embedding-based classifier for sticky-constraint extraction -- an alternative/complement
to plain keyword matching.

Two modes:
- With a real OPENAI_API_KEY: uses text-embedding-3-small (multilingual)
- Without a key: falls back to a random vector (keeps the demo runnable, NOT semantically functional)
"""

import os
import numpy as np
from typing import Optional

try:
    from openai import OpenAI
    _openai_available = True
except ImportError:
    _openai_available = False


class EmbeddingProvider:
    def __init__(self, client: Optional["OpenAI"]):
        self.client = client

    def embed(self, texts: list[str]) -> np.ndarray:
        if self.client is not None:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=texts,
            )
            return np.array([d.embedding for d in response.data])

        vectors = []
        for text in texts:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            vectors.append(rng.random(64))
        return np.array(vectors)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# Reference phrases per category, per constraint dimension -- the "few-shot" set you'd
# tune against real usage data.
ORIGIN_REFERENCE = {
    "local": [
        "chci lokální obsah",
        "jen české produkce",
        "domácí tvorba",
        "tuzemský film",
        "chci jen z Česka",
        "něco od nás",
        "nechci zahraniční",
    ],
    "international": [
        "zahraniční filmy",
        "cizí produkce",
        "mezinárodní tituly",
        "chci něco ze zahraničí",
        "něco ze světa",
        "nechci český",
    ],
}

GENRE_REFERENCE = {
    "spy": [
        "špionážní film",
        "agent a tajné služby",
        "studenoválečný thriller",
        "film o špiónech",
    ],
    "comedy": [
        "veselá komedie",
        "chci se zasmát",
        "něco zábavného a lehkého",
        "chci film, u kterého se zasměju",
        "chci se u filmu bavit",
        "chci něco vtipného",
    ],
    "drama": [
        "vážné drama",
        "emotivní příběh",
        "rodinné drama",
    ],
    "horror": [
        "strašlivý film",
        "horrorní příběh",
        "něco děsivého",
        "chci se bát",
        "chci horror",
    ]
}

SORT_REFERENCE = {
    "year_desc": [
        "nejnovější film",
        "chci něco aktuálního",
        "poslední dobou vyšlé tituly",
    ],
}

CAST_REFERENCE = {}


class ConstraintClassifier:
    """
    Holds a centroid per category per dimension (origin, genre, sort_by). A centroid
    is the average embedding of that category's reference phrases, computed once at
    init, not per call.
    """

    def __init__(self, embedder: EmbeddingProvider, threshold: float = 0.6):
        self.embedder = embedder
        self.threshold = threshold

        self.origin_centroids = self._build_centroids(ORIGIN_REFERENCE)
        self.genre_centroids = self._build_centroids(GENRE_REFERENCE)
        self.sort_centroids = self._build_centroids(SORT_REFERENCE)
        self.cast_centroids = self._build_centroids(CAST_REFERENCE)

    def _build_centroids(self, reference: dict[str, list[str]]) -> dict[str, np.ndarray]:
        centroids = {}
        for label, phrases in reference.items():
            vecs = self.embedder.embed(phrases)
            centroids[label] = vecs.mean(axis=0)
        return centroids

    def _classify(self, msg_vec: np.ndarray, centroids: dict[str, np.ndarray]) -> tuple[Optional[str], float]:
        best_label, best_score = None, -1.0
        for label, centroid in centroids.items():
            score = cosine_sim(msg_vec, centroid)
            if score > best_score:
                best_label, best_score = label, score
        if best_score < self.threshold:
            return None, best_score
        return best_label, best_score

    def classify_message(self, user_message: str) -> dict:
        """Classifies across all dimensions at once -- one embedding call per message, not one per dimension."""
        msg_vec = self.embedder.embed([user_message])[0]

        origin_label, origin_score = self._classify(msg_vec, self.origin_centroids)
        genre_label, genre_score = self._classify(msg_vec, self.genre_centroids)
        sort_label, sort_score = self._classify(msg_vec, self.sort_centroids)

        result = {}
        if origin_label:
            result["origin"] = origin_label
        if genre_label:
            result["genre"] = genre_label
        if sort_label:
            result["sort_by"] = sort_label

        return result, {
            "origin_score": origin_score,
            "genre_score": genre_score,
            "sort_score": sort_score,
        }


if __name__ == "__main__":
    api_key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key) if (_openai_available and api_key) else None

    if client is None:
        print("=== FALLBACK režim (bez OPENAI_API_KEY) ===")
        print("POZOR: výsledky níže NEBUDOU sémanticky správné -- fallback je náhodný.")
        print("Se skutečným API klíčem by klasifikace fungovala smysluplně.\n")

    embedder = EmbeddingProvider(client)
    classifier = ConstraintClassifier(embedder, threshold=0.75)

    test_messages = [
        "chci jen tuzemskou tvorbu",
        "něco jako ten starý film s agentem",
        "chci něco aktuálního, co teď vyšlo",
        "nechci nic ze zahraničí",
    ]

    for msg in test_messages:
        constraints, scores = classifier.classify_message(msg)
        print(f"Zpráva: '{msg}'")
        print(f"  Constrainty: {constraints}")
        print(f"  Skóre:       {scores}")
        print()
