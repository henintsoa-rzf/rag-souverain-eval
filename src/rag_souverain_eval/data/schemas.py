"""Schémas Pydantic figés pour P1 RAG-souverain-eval.

Contrat d'interface entre TOUS les modules. Toute modification de ce fichier 
est une décision architecturale, pas un détail d'implémentation.

"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Thèmes du périmètre "vie courante" (cf. config/default.yaml)
ThemeInScope = Literal[
    "famille",
    "travail",
    "logement",
    "papiers_citoyennete",
    "argent_aides_impots",
]

# Thèmes possibles pour le golden set (in-scope + hors-scope)
ThemeGolden = Literal[
    "famille",
    "travail",
    "logement",
    "papiers_citoyennete",
    "argent_aides_impots",
    "hors_scope",
]

QuestionType = Literal[
    "simple_factuelle",
    "multi_hop",
    "conditions_exceptions",
    "ambigue",
    "hors_scope",
]

Confidence = Literal["low", "medium", "high"]

GenerationTrack = Literal["local", "api"]

# =====================================================================
# Corpus
# =====================================================================

class DocSection(BaseModel):
    """Section sémantique d'une fiche service-public (issue d'un h2/h3)."""
    model_config = ConfigDict(extra="forbid")

    section_id: str   # ex: "sp_000123_s01"
    heading: str
    text: str

class CleanDocument(BaseModel):
    """Document nettoyé in-scope. Sortie de src/data/clean_documents.py"""
    model_config = ConfigDict(extra="forbid")

    doc_id: str   # ex: "sp_000123"
    title: str
    theme: ThemeInScope
    url: str
    last_update: str | None = None
    raw_text: str
    sections: list[DocSection] = Field(default_factory=list)
    source: str = "service-public.fr"
    license: str = "Etalab 2.0"

class Chunk(BaseModel):
    """Chunk indexable. SOrtie des modules src/chunking/*."""
    model_config = ConfigDict(extra="forbid")

    chunk_id: str   # ex: "sp_000123_s01_c01"
    doc_id: str     # ex: "sp_000123"
    parent_id: str  # section_id ou doc_id selon stratégie
    theme: ThemeInScope
    title: str
    section_heading: str | None = None
    url: str
    text: str
    token_count: int = Field(ge=0)
    chunking_strategy: str  # ex: "recursive_512" | "section_parent_v1"


# =====================================================================
# Retrieval
# =====================================================================

class RetrievalResult(BaseModel):
    """Unité de sortie de TOUTE stratégie de retrieval
    
    Contrat stable : tout pipeline doit retourner List[RetrievalResult]
    trié par `rank` croissant. Voir src/retrieval/strategies.py::run_strategy
    """
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    score: float # score brut de la stratégie, NON calibré
    rank: int = Field(ge=1) # 1-based, après fusion/rerank éventuels
    text: str  # texte du chunk OU du parent si expand_to_parent
    title: str
    url: str
    doc_id: str
    parent_id: str
    is_parent_expanded: bool = False
    # Latences par étape, scores intermédiaires, etc. Clés libres.
    # Clés conventionnelles : dense_ms, bm25_ms, fusion_ms, rerank_ms
    debug: dict = Field(default_factory=dict)


# =====================================================================
# Génération
# =====================================================================

class Source(BaseModel):
    """Source citée dans une RagAnswer. """
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    title: str
    url: str
    quote: str | None = None 


class RagAnswer(BaseModel):
    """Sortie UNIQUE du RAG (locale OU API). Schéma imposé au LLM.

    Invariants métier (validé en post-parse) :
    - refused=True -> sources peut être vide; answer explique pourquoi
    - refused=False -> len(sources) >= 1 et chaque source vient des chunks réellement fournis au LLM

    La vérification "source chunks fournis" est externe (le schéma ne connaît pas les chunks), faite dans src/generation/rag_chain.py 

    """
    model_config = ConfigDict(extra="forbid")

    answer: str
    sources: list[Source] = Field(default_factory=list)
    confidence: Confidence
    refused: bool = False
    refusal_reason: str | None = None

    @model_validator(mode="after")
    def _check_refusal_invariants(self) -> "RagAnswer":
        if self.refused:
            # refus -> refusal_reason recommandé (pas obligatoire pour rester tolérant aux sorties LLM imparfaites)
            return self
        # !refused -> au moins 1 source
        if len(self.sources) < 1:
            raise ValueError(
                "RagAnswer non-refusée doit citer au moins 1 source. "
                "Si pas de source, mettre refused=True."
            )
        return self

# =====================================================================
# Golden set
# =====================================================================

class GoldenQuestion(BaseModel):
    """Une question du golden set, Sortie de notebooks/02_golden_set_building"""
    model_config = ConfigDict(extra="forbid")

    id: str  # ex: "q_001"
    theme: ThemeGolden
    question_type: QuestionType
    question: str
    expected_answer_short: str
    reference_urls: list[str] = Field(default_factory=list)
    reference_doc_ids: list[str] = Field(default_factory=list)
    reference_chunk_ids: list[str] = Field(default_factory=list)
    must_refuse: bool = False
    notes: str | None = None

    @model_validator(mode="after")
    def _check_hors_scope_consistency(self) -> "GoldenQuestion":
            # Cohérence hors_scope : theme et question_type doivent s'aligner
            is_hors_scope_theme = self.theme == "hors_scope"
            is_hors_scope_type = self.question_type == "hors_scope"
            if is_hors_scope_theme != is_hors_scope_type:
                raise ValueError(
                    f"Incohérence theme/question_type pour {self.id}: "
                    f"theme={self.theme} mais question_type={self.question_type}. "
                    "Les deux doivent être 'hors_scope' ou aucun."
                )
            # Si must_refuse=True, la question DOIT être hors_scope
            # (on n'attend pas de refus sur une question in-scope)
            if self.must_refuse and not is_hors_scope_theme:
                raise ValueError(
                    f"{self.id}: must_refuse=True implique theme=hors_scope."
                )
            # Si hors_scope, pas de référence attendue
            if is_hors_scope_theme:
                if self.reference_chunk_ids or self.reference_doc_ids:
                    raise ValueError(
                        f"{self.id}: question hors_scope ne doit pas avoir "
                        "de reference_chunk_ids/doc_ids."
                    )
            else:
                # In-scope : au moins reference_chunk_ids OU reference_doc_ids
                # doit être renseigné (sinon comment évaluer le recall ?)
                if not self.reference_chunk_ids and not self.reference_doc_ids:
                    raise ValueError(
                        f"{self.id}: question in-scope doit avoir au moins "
                        "reference_chunk_ids ou reference_doc_ids."
                    )
            return self


# =====================================================================
# Résultats d'évaluation (lignes des CSV de bench)
# =====================================================================

class RetrievalEvalRow(BaseModel):
    """Une ligne du benchmark retrieval (1 stratégie x métriques agrégées)."""
    model_config = ConfigDict(extra="forbid")

    strategy: str  # ex: "S2_hybrid_rrf"
    recall_at_3: float = Field(ge=0.0, le=1.0)
    recall_at_5: float = Field(ge=0.0, le=1.0)
    recall_at_10: float = Field(ge=0.0, le=1.0)
    mrr: float = Field(ge=0.0, le=1.0)
    ndcg_at_10: float = Field(ge=0.0, le=1.0)
    dense_ms: float = Field(ge=0.0)
    bm25_ms: float = Field(ge=0.0)
    fusion_ms: float = Field(ge=0.0)
    rerank_ms: float = Field(ge=0.0)
    total_retrieval_ms: float = Field(ge=0.0)


class GenerationEvalRow(BaseModel):
    """Une ligne du benchmark génération (1 track x 1 stratégie x métriques)."""
    model_config = ConfigDict(extra="forbid")

    track: GenerationTrack
    strategy: str
    answer_correctness: float = Field(ge=0.0, le=1.0)
    faithfulness: float = Field(ge=0.0, le=1.0)
    citation_correctness: float = Field(ge=0.0, le=1.0)
    refusal_correctness: float = Field(ge=0.0, le=1.0)
    json_validity: float = Field(ge=0.0, le=1.0)
    latency_s: float = Field(ge=0.0)
    api_cost_eur: float = Field(ge=0.0)

