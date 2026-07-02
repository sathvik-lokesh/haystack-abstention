from haystack import Document

from haystack_abstention import AbstentionResponder, RetrievalSufficiencyRouter


def docs(*scores):
    return [Document(content=f"doc {i}", score=s) for i, s in enumerate(scores)]


def test_answerable_routes_to_sufficient():
    r = RetrievalSufficiencyRouter(min_top_score=0.5, gap_floor=0.3)
    out = r.run("q", docs(0.82, 0.4))
    assert "sufficient" in out and "insufficient" not in out
    assert out["sufficient"]["verdict"] == "ANSWERABLE"


def test_gap_routes_to_insufficient():
    r = RetrievalSufficiencyRouter(min_top_score=0.5, gap_floor=0.3)
    out = r.run("q", docs(0.21, 0.18))
    assert "insufficient" in out and "sufficient" not in out
    assert out["insufficient"]["verdict"] == "GAP"
    assert "floor" in out["insufficient"]["reason"]


def test_partial_defaults_to_insufficient_but_flips_when_configured():
    weak = docs(0.42, 0.30)
    assert RetrievalSufficiencyRouter().run("q", weak).get("insufficient")
    flip = RetrievalSufficiencyRouter(treat_partial_as_sufficient=True).run("q", weak)
    assert flip.get("sufficient") and flip["sufficient"]["verdict"] == "PARTIAL"


def test_min_support_requires_enough_strong_docs():
    r = RetrievalSufficiencyRouter(min_top_score=0.5, min_support=2, gap_floor=0.3)
    # only one doc clears 0.5 -> not ANSWERABLE
    assert r.run("q", docs(0.9, 0.2)).get("insufficient")
    assert r.run("q", docs(0.9, 0.7)).get("sufficient")


def test_empty_documents_abstain():
    out = RetrievalSufficiencyRouter().run("q", [])
    assert out.get("insufficient") and out["insufficient"]["signals"]["top1"] == 0.0


def test_responder_marks_abstained():
    payload = {"query": "who?", "reason": "no match", "verdict": "GAP", "signals": {}}
    out = AbstentionResponder().run(payload)
    assert out["meta"]["abstained"] is True
    assert out["meta"]["verdict"] == "GAP"
    assert "who?" in out["reply"]


def test_router_to_responder_end_to_end():
    router = RetrievalSufficiencyRouter(min_top_score=0.5, gap_floor=0.3)
    routed = router.run("who is the CFO?", docs(0.15, 0.1))
    out = AbstentionResponder().run(routed["insufficient"])
    assert out["meta"]["abstained"] and out["meta"]["verdict"] == "GAP"


def test_invalid_config_raises():
    import pytest

    with pytest.raises(ValueError):
        RetrievalSufficiencyRouter(min_top_score=0.5, gap_floor=0.9)
