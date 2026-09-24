"""BM25 retrieval."""

from kb_loader import chunk_documents
from models import KBDoc
from retriever import BM25Index


def _index(texts):
    docs = [KBDoc(id=f"d{i}", title=f"T{i}", text=t) for i, t in enumerate(texts)]
    return BM25Index(chunk_documents(docs))


def test_idf_never_negative_for_common_terms():
    idx = _index(["support one", "support two", "support three", "other"])
    assert idx.idf("support") > 0
    assert idx.idf("missing") > 0


def test_ties_broken_by_kb_order():
    idx = _index(["alpha beta", "alpha beta", "alpha beta"])
    ids = [p.parent_id for p in idx.retrieve(["alpha"], top_k=3)]
    assert ids == ["d0", "d1", "d2"]
    assert ids == [p.parent_id for p in idx.retrieve(["alpha"], top_k=3)]  # deterministic


def test_only_positive_scores_and_top_k():
    idx = _index(["alpha", "beta", "gamma", "alpha alpha"])
    assert [p.parent_id for p in idx.retrieve(["zeta"])] == []
    assert len(idx.retrieve(["alpha", "beta", "gamma"], top_k=2)) == 2


def test_empty_token_docs_do_not_divide_by_zero():
    idx = _index(["the and of", "a an the"])  # only stopwords
    assert idx.avgdl == 1.0
    assert idx.retrieve(["anything"]) == []


def test_sample_retrieval(pipe):
    from text import query_terms

    top = pipe.index.retrieve(query_terms("How do I reset my password?"))
    assert top[0].parent_id == "doc_1"
