"""End-to-end Haystack RAG pipeline with the abstention gate, using embedding
retrieval (cosine) so relevance separates cleanly.

Run: pip install -e ".[examples]" && python examples/pipeline_rag.py
(First run downloads a ~90MB sentence-transformers model.)

What to look for:
- In-domain questions (payload, battery) -> ANSWER.
- Clearly out-of-domain questions (capital of France, baking) -> ABSTAIN.
- "Who is the CFO?" is the hard edge case: it's topically in-domain, so it
  scores high and passes the retrieval gate - even though the fact isn't in the
  corpus. Embedding/retrieval signals catch TOPICAL gaps, not fine-grained
  FACTUAL ones; catching that needs an LLM-verification step downstream. This
  gate is the cheap first line, not the whole story.
"""
from haystack import Document, Pipeline, component
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

from haystack_abstention import AbstentionResponder, RetrievalSufficiencyRouter

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CORPUS = [
    "The Aurelia R1 robot arm has a rated payload of 5 kilograms.",
    "The Aurelia R1 operates for up to 6 hours on a full battery charge.",
    "Aurelia Robotics was founded in 2019 and is headquartered in Lyon.",
    "The R1 gripper supports both suction and two-finger parallel modes.",
]


@component
class MockAnswerer:
    """Stand-in for a prompt-builder + generator on the answer branch."""

    @component.output_types(reply=str)
    def run(self, sufficient: dict):
        docs = sufficient.get("documents") or []
        top = docs[0].content if docs else "(none)"
        return {"reply": f"[ANSWER · {sufficient['verdict']} · top={docs[0].score:.2f}] {top}"}


def build() -> Pipeline:
    doc_embedder = SentenceTransformersDocumentEmbedder(model=MODEL, progress_bar=False)
    doc_embedder.warm_up()
    store = InMemoryDocumentStore()
    store.write_documents(doc_embedder.run([Document(content=c) for c in CORPUS])["documents"])

    pipe = Pipeline()
    pipe.add_component("text_embedder", SentenceTransformersTextEmbedder(model=MODEL, progress_bar=False))
    pipe.add_component("retriever", InMemoryEmbeddingRetriever(document_store=store, top_k=4))
    pipe.add_component("router", RetrievalSufficiencyRouter(min_top_score=0.5, gap_floor=0.45))
    pipe.add_component("answerer", MockAnswerer())
    pipe.add_component("abstain", AbstentionResponder())

    pipe.connect("text_embedder.embedding", "retriever.query_embedding")
    pipe.connect("retriever.documents", "router.documents")
    pipe.connect("router.sufficient", "answerer.sufficient")
    pipe.connect("router.insufficient", "abstain.insufficient")
    return pipe


def main() -> None:
    pipe = build()
    questions = [
        "What is the Aurelia R1's payload?",       # in-domain -> answer
        "How long does the R1 run on a charge?",   # in-domain -> answer
        "What is the warranty period on the R1?",  # out-of-corpus -> abstain
        "What is the capital of France?",          # out-of-domain -> abstain
        "Who is the CFO of Aurelia Robotics?",     # topical-but-absent -> known miss
    ]
    for q in questions:
        out = pipe.run({"text_embedder": {"text": q}, "router": {"query": q}})
        node = out.get("answerer") or out.get("abstain")
        print(f"Q: {q}\n   -> {node['reply']}\n")


if __name__ == "__main__":
    main()
