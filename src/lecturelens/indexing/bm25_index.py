"""BM25 lexical index (SPEC 7.3) built on rank-bm25 (BM25Okapi with Lucene IDF).

BM25 makes exact terms, acronyms and names ("TF-IDF", "BLEU", "Viterbi") retrieve reliably
where dense embeddings can be fuzzy. The index is built once over all chunks and pickled.
"""

import logging
import math
import pickle
import re
from collections.abc import Callable
from pathlib import Path

from rank_bm25 import BM25Okapi

from lecturelens.config import BM25Cfg
from lecturelens.errors import IndexMismatchError

logger = logging.getLogger(__name__)

_WORD = re.compile(r"[a-z0-9]+")
STEMMER_LANGUAGE = "english"

# Fixed English stopword list (no NLTK download needed).
STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further
    had has have having he her here hers herself him himself his how i if in into is it its
    itself just me more most my myself no nor not now of off on once only or other our ours
    ourselves out over own same she should so some such than that the their theirs them
    themselves then there these they this those through to too under until up very was we were
    what when where which while who whom why will with would you your yours yourself yourselves
    """.split()  # noqa: SIM905 - compact block reads better than 128 literals
)


class LuceneBM25(BM25Okapi):
    """BM25Okapi with the Lucene/BM25+ IDF: ``log(1 + (N - n + 0.5) / (n + 0.5))``.

    The classic Robertson-Sparck Jones IDF used by rank-bm25 is zero for a term in exactly
    half the documents and negative above that; rank-bm25 floors negatives at
    ``epsilon * average_idf``, which is itself ~0 on small corpora. Real matches then score 0.
    Lucene's variant is always positive, so every matching term contributes.
    """

    def _calc_idf(self, nd: dict[str, int]) -> None:
        for word, doc_freq in nd.items():
            self.idf[word] = math.log(1 + (self.corpus_size - doc_freq + 0.5) / (doc_freq + 0.5))
        self.average_idf = sum(self.idf.values()) / len(self.idf) if self.idf else 0.0


def _stemmer(enabled: bool) -> Callable[[str], str] | None:
    if not enabled:
        return None
    import snowballstemmer  # optional dependency, only needed with retrieval.bm25.stemming

    return snowballstemmer.stemmer(STEMMER_LANGUAGE).stemWord


def tokenize(text: str, stem: Callable[[str], str] | None = None) -> list[str]:
    """Lowercase, split into ``[a-z0-9]+`` words, drop stopwords, optionally stem.

    Args:
        text: Input text.
        stem: Optional single-word stemming function.

    Returns:
        Tokens in order.
    """
    tokens = [t for t in _WORD.findall(text.lower()) if t not in STOPWORDS]
    return [stem(t) for t in tokens] if stem else tokens


class BM25Index:
    """BM25Okapi over chunk texts, addressed by chunk id."""

    def __init__(self, cfg: BM25Cfg) -> None:
        """Create an empty index.

        Args:
            cfg: BM25 parameters (k1, b) and the stemming flag.
        """
        self.cfg = cfg
        self._stem = _stemmer(cfg.stemming)
        self._ids: list[str] = []
        self._bm25: LuceneBM25 | None = None

    def __len__(self) -> int:
        return len(self._ids)

    def build(self, ids: list[str], texts: list[str]) -> None:
        """(Re)build the index.

        Args:
            ids: Chunk ids.
            texts: Texts to index, parallel to ``ids`` (the chunker's ``indexed_text``).
        """
        if len(ids) != len(texts):
            raise ValueError(f"{len(ids)} ids but {len(texts)} texts")
        self._ids = list(ids)
        corpus = [tokenize(text, self._stem) for text in texts]
        self._bm25 = LuceneBM25(corpus, k1=self.cfg.k1, b=self.cfg.b) if corpus else None
        logger.info("Built BM25 index over %d chunks", len(ids))

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Return up to ``k`` ``(chunk_id, score)`` pairs with a positive score.

        Sorted by score descending; ties are broken by chunk id so results are deterministic.
        """
        terms = tokenize(query, self._stem)
        if self._bm25 is None or not terms:
            return []
        scores = self._bm25.get_scores(terms)
        ranked = sorted(
            ((chunk_id, float(s)) for chunk_id, s in zip(self._ids, scores, strict=True) if s > 0),
            key=lambda pair: (-pair[1], pair[0]),
        )
        return ranked[:k]

    def save(self, path: Path) -> None:
        """Pickle the index to ``path`` (written atomically)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as fh:
            pickle.dump({"cfg": self.cfg.model_dump(), "ids": self._ids, "bm25": self._bm25}, fh)
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path, cfg: BM25Cfg) -> "BM25Index":
        """Load a pickled index written by :meth:`save` (only load files you created).

        Raises:
            IndexMismatchError: If the index was built with different BM25 settings.
        """
        with path.open("rb") as fh:
            data = pickle.load(fh)  # noqa: S301 - our own index file under data/index
        if data["cfg"] != cfg.model_dump():
            raise IndexMismatchError(
                "BM25 settings changed since the index was built; run `ingest --rebuild`."
            )
        index = cls(cfg)
        index._ids, index._bm25 = data["ids"], data["bm25"]
        return index
