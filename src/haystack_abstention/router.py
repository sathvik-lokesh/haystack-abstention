"""RetrievalSufficiencyRouter - a Haystack component that decides whether the
retrieved context is strong enough to answer, and routes the query accordingly.

The idea (from the gap-aware-rag project): most RAG pipelines answer no matter
what the retriever returned, which is exactly how they hallucinate on
out-of-corpus questions. This component reads the retriever's own coverage
signals and routes to an *abstain* branch when the evidence is too weak -
so the pipeline can say "I don't have enough information" instead of guessing.

Wire it after any retriever that populates `Document.score`:

    retriever -> router.documents
    router.sufficient   -> (prompt builder / generator)
    router.insufficient -> (abstention responder)
"""
from typing import Any, Dict, List

from haystack import Document, component


@component
class RetrievalSufficiencyRouter:
    """Route a query to an answer or abstain branch based on retrieval coverage.

    Signals (all read from the retrieved documents' scores):
      - top1:       best document score - is anything a strong match?
      - score_gap:  top1 - top2 - is the best match clearly separated?
      - n_support:  how many documents clear `min_top_score`

    Verdicts:
      - ANSWERABLE -> `sufficient`   (top1 >= min_top_score and enough support)
      - PARTIAL    -> `insufficient` (some signal, but under the bar) unless
                      `treat_partial_as_sufficient=True`
      - GAP        -> `insufficient` (top1 below the absolute floor)
    """

    def __init__(
        self,
        min_top_score: float = 0.5,
        min_score_gap: float = 0.0,
        min_support: int = 1,
        gap_floor: float = 0.3,
        treat_partial_as_sufficient: bool = False,
    ):
        if not 0.0 <= gap_floor <= min_top_score:
            raise ValueError("gap_floor must be between 0 and min_top_score")
        self.min_top_score = min_top_score
        self.min_score_gap = min_score_gap
        self.min_support = min_support
        self.gap_floor = gap_floor
        self.treat_partial_as_sufficient = treat_partial_as_sufficient

    def _signals(self, documents: List[Document]) -> Dict[str, Any]:
        scores = [d.score for d in documents if d.score is not None]
        scores.sort(reverse=True)
        top1 = scores[0] if scores else 0.0
        top2 = scores[1] if len(scores) > 1 else 0.0
        return {
            "top1": round(top1, 4),
            "score_gap": round(top1 - top2, 4),
            "n_support": sum(1 for s in scores if s >= self.min_top_score),
            "n_docs": len(documents),
        }

    def _verdict(self, sig: Dict[str, Any]) -> str:
        if sig["top1"] < self.gap_floor:
            return "GAP"
        if (
            sig["top1"] >= self.min_top_score
            and sig["score_gap"] >= self.min_score_gap
            and sig["n_support"] >= self.min_support
        ):
            return "ANSWERABLE"
        return "PARTIAL"

    @component.output_types(sufficient=Dict[str, Any], insufficient=Dict[str, Any])
    def run(self, query: str, documents: List[Document]) -> Dict[str, Dict[str, Any]]:
        sig = self._signals(documents)
        verdict = self._verdict(sig)
        payload = {"query": query, "documents": documents, "verdict": verdict, "signals": sig}

        is_sufficient = verdict == "ANSWERABLE" or (
            verdict == "PARTIAL" and self.treat_partial_as_sufficient
        )
        if is_sufficient:
            return {"sufficient": payload}

        reason = (
            f"no document cleared the confidence floor (top score {sig['top1']} < {self.gap_floor})"
            if verdict == "GAP"
            else f"weak/partial match (top score {sig['top1']} below the answer bar {self.min_top_score})"
        )
        return {"insufficient": {**payload, "reason": reason}}
