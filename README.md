# haystack-abstention

A **retrieval-sufficiency gate for [Haystack](https://haystack.deepset.ai)**: route a RAG query to an
*abstain* branch when the retrieved evidence is too weak, instead of letting the generator hallucinate
an answer. Two small, composable components, no extra model, no API key.

> Most RAG pipelines answer regardless of what the retriever returned, which is exactly how they
> hallucinate on out-of-corpus questions. This gate reads the retriever's own coverage signals and,
> when they're too weak, routes to a plain "I don't have enough information" instead of guessing.

Built from the [gap-aware-rag](https://github.com/sathvik-lokesh/gap-aware-rag) project, ported to
native Haystack components so it drops into any pipeline.

## Components

**`RetrievalSufficiencyRouter`**: reads `Document.score` from any retriever and emits one of two
outputs so downstream branches fire conditionally:
- `sufficient` → wire to your prompt builder / generator
- `insufficient` → wire to an abstain / clarify branch

It computes three signals, `top1` (best score), `score_gap` (top1 − top2), `n_support` (docs above
the answer bar), and assigns a verdict: **ANSWERABLE**, **PARTIAL**, or **GAP**. PARTIAL and GAP route
to `insufficient` by default (configurable).

**`AbstentionResponder`**: terminal component for the abstain branch; returns a templated
reply that names the gap, plus `meta.abstained=True` for evaluation.

## Install
```bash
pip install haystack-ai
pip install -e .    # from this repo
```

## Use
```python
from haystack_abstention import RetrievalSufficiencyRouter, AbstentionResponder

router = RetrievalSufficiencyRouter(min_top_score=0.55, gap_floor=0.5)
# retriever.documents -> router.documents
# router.sufficient   -> generator branch
# router.insufficient -> AbstentionResponder
```
Full runnable pipeline (embedding retrieval): [`examples/pipeline_rag.py`](examples/pipeline_rag.py)
```
pip install -e ".[examples]"
python examples/pipeline_rag.py
```
Sample output, in-domain answers, out-of-domain abstains:
```
Q: What is the Aurelia R1's payload?      -> [ANSWER · top=0.75] ...5 kilograms.
Q: What is the capital of France?         -> I don't have enough information ... (top score 0.16 < 0.45)
Q: Who is the CFO of Aurelia Robotics?    -> [ANSWER · top=0.71] ...founded in 2019...   <- known miss
```

### Known limitation
Retrieval scores catch **topical** gaps, not fine-grained **factual** ones. "Who is the CFO?" is
topically about Aurelia Robotics, so it scores high and passes the gate, even though the fact isn't
in the corpus. This component is the cheap first line of defense; catching factual gaps needs an
LLM-verification step downstream (see [gap-aware-rag](https://github.com/sathvik-lokesh/gap-aware-rag)).
The signals are exposed on every verdict so this decision is measurable, not hidden.

## Test
```bash
pip install -e ".[test]" && pytest -q
```

## Why this matters (deepset / Haystack context)
Reliability, evaluation, and guardrails for agentic RAG are on the Haystack roadmap. Calibrated
abstention, knowing when a pipeline should *not* answer, is the piece this component adds, with the
sufficiency signals exposed so the decision is measurable, not a black box.

## License
Apache-2.0
