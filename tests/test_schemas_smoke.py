"""Smoke tests des invariants metier des schemas.

Pas de pytest pour l'instant : on lance via `uv run python tests/test_schemas_smoke.py`.
Si un assert pete, le contrat est casse.
"""
from pydantic import ValidationError

from rag_souverain_eval.data.schemas import (
    Chunk,
    GoldenQuestion,
    RagAnswer,
    Source,
)


def test_rag_answer_refused_ok_sans_sources():
    a = RagAnswer(
        answer="Je ne peux pas repondre.",
        sources=[],
        confidence="low",
        refused=True,
        refusal_reason="hors scope",
    )
    assert a.refused is True


def test_rag_answer_non_refused_sans_source_doit_planter():
    try:
        RagAnswer(
            answer="reponse bidon",
            sources=[],
            confidence="medium",
            refused=False,
        )
    except (ValidationError, ValueError):
        return
    raise AssertionError("aurait du lever ValidationError (non-refused sans source)")



def test_rag_answer_non_refused_avec_source_ok():
    a = RagAnswer(
        answer="Oui.",
        sources=[Source(chunk_id="c1", title="t", url="https://x")],
        confidence="high",
        refused=False,
    )
    assert len(a.sources) == 1


def test_golden_in_scope_doit_avoir_reference():
    try:
        GoldenQuestion(
            id="q_001",
            theme="travail",
            question_type="simple_factuelle",
            question="?",
            expected_answer_short="reponse",
        )
    except (ValidationError, ValueError):
        return
    raise AssertionError("aurait du lever ValidationError (in-scope sans reference)")



def test_golden_hors_scope_avec_reference_doit_planter():
    try:
        GoldenQuestion(
            id="q_080",
            theme="hors_scope",
            question_type="hors_scope",
            question="?",
            expected_answer_short="refus",
            reference_chunk_ids=["c1"],
            must_refuse=True,
        )
    except (ValidationError, ValueError):
        return
    raise AssertionError("aurait du lever ValidationError (hors_scope avec reference)")



def test_golden_must_refuse_in_scope_doit_planter():
    try:
        GoldenQuestion(
            id="q_x",
            theme="travail",
            question_type="simple_factuelle",
            question="?",
            expected_answer_short="r",
            reference_chunk_ids=["c1"],
            must_refuse=True,
        )
    except (ValidationError, ValueError):
        return
    raise AssertionError("aurait du lever ValidationError (must_refuse + in-scope)")



def test_golden_incoherence_theme_type_doit_planter():
    try:
        GoldenQuestion(
            id="q_x",
            theme="travail",
            question_type="hors_scope",  # incoherent
            question="?",
            expected_answer_short="r",
            reference_chunk_ids=["c1"],
        )
    except (ValidationError, ValueError):
        return
    raise AssertionError("aurait du lever ValidationError (theme/type incoherents)")



def test_chunk_token_count_negatif_doit_planter():
    try:
        Chunk(
            chunk_id="c1",
            doc_id="d1",
            parent_id="d1",
            theme="travail",
            title="t",
            url="https://x",
            text="...",
            token_count=-1,
            chunking_strategy="recursive_512",
        )
    except (ValidationError, ValueError):
        return
    raise AssertionError("aurait du lever ValidationError (token_count negatif)")



if __name__ == "__main__":
    tests = [
        test_rag_answer_refused_ok_sans_sources,
        test_rag_answer_non_refused_sans_source_doit_planter,
        test_rag_answer_non_refused_avec_source_ok,
        test_golden_in_scope_doit_avoir_reference,
        test_golden_hors_scope_avec_reference_doit_planter,
        test_golden_must_refuse_in_scope_doit_planter,
        test_golden_incoherence_theme_type_doit_planter,
        test_chunk_token_count_negatif_doit_planter,
    ]
    for t in tests:
        t()
        print(f"OK {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passes")