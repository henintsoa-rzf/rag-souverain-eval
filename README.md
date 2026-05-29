# RAG-Souverain-Eval

> Assistant RAG évalué rigoureusement sur les démarches administratives françaises de vie courante (corpus service-public.fr), avec une variante locale/on-premise.

**Statut :** 🚧 En construction (J1/J10 — setup infrastructure). Mise à jour prévue à chaque jour de développement.

---

## Pourquoi ce projet ?

La majorité des démos RAG publiques sont des notebooks-tutoriels qui branchent un vectorDB sur un PDF, sans jamais mesurer ce qu'elles produisent. Or **un RAG fiable est d'abord un RAG mesuré** — sinon on ne sait pas si on hallucine, si on cite correctement, et si une amélioration de prompt améliore vraiment la qualité.

Ce projet construit :
- Un **protocole d'évaluation reproductible** (golden set + métriques retrieval + métriques génération)
- Sur un **corpus à fort enjeu** (administratif français, ~3 500 fiches service-public.fr)
- Avec une **variante 100% offline** (LLM local Qwen3-8B, embeddings Qwen3-0.6B sur CPU)
- En comparant **plusieurs stratégies** (chunking, retrieval, modèle d'embedding) avec des chiffres défendables

L'objectif n'est PAS de livrer un chatbot administratif. C'est de livrer **un framework d'éval** qui s'applique à n'importe quel domaine où la fiabilité du RAG compte.

## Ce que le projet démontre

- Maîtrise de la pipe RAG complète : ingestion → chunking → indexation → retrieval hybride → reranking → génération structurée
- Évaluation rigoureuse : golden set 80 questions, métriques retrieval (Recall@k, MRR, nDCG) + génération (faithfulness, citation, refus pertinent)
- Comparaisons mesurées : 3 stratégies de chunking × 2 modèles d'embedding × 4 stratégies de retrieval × 2 modèles de génération (local Qwen3-8B vs API GPT-5 mini)
- Souveraineté pragmatique : version offline mesurée

## Stack

| Composant | Choix | Justification |
|---|---|---|
| Vector DB | Qdrant (Docker) | Pattern standard prod, bind-mount stable en LXC |
| Embeddings principal | Qwen3-Embedding-0.6B | Top MTEB 2025, dim 1024 |
| Embeddings baseline | BGE-M3 | Standard 2024, embeddings Etalab fournis gratuits |
| Reranker | Qwen3-Reranker-0.6B | Cohérence d'écosystème |
| LLM local | Qwen3-8B-Instruct Q4_K_M | 8.4 tok/s gen, 107 tok/s prompt (Vulkan iGPU AMD) |
| LLM API | GPT-5 mini | Comparaison + judge éval |
| BM25 | bm25s (FR) | Léger, performant |
| Génération | Ollama (Vulkan) | API HTTP standard, dev confort > perf brute |
| UI démo | Gradio | Vitrine, pas produit |
| Package management | uv | Lockfile reproductible |
| Validation | Pydantic | Schémas figés `src/data/schemas.py` |

## Corpus

[`AgentPublic/service-public`](https://huggingface.co/datasets/AgentPublic/service-public) (Hugging Face, licence Etalab 2.0).

- 34 944 chunks bruts / 3 557 documents uniques
- Pre-chunkés par Etalab (RecursiveCharacterTextSplitter, 1024 tokens BGE-M3, sans overlap)
- Pre-embeddés en BGE-M3 (utilisé tel quel pour la baseline d'ablation)
- 6 thèmes vie courante retenus : `famille`, `travail`, `logement`, `papiers_citoyennete`, `argent_impots`, `social_sante`
- Reconstruction des documents complets via méthode adaptée du [notebook officiel Etalab](https://github.com/etalab-ia/mediatech/blob/main/docs/reconstruct_vector_database.ipynb), avec préservation de la hiérarchie des sections (non gérée par le notebook original)

## Avancement

- [x] **J1** Setup infra : LXC + iGPU + Docker + Qdrant + Ollama + uv + scaffold + schémas Pydantic
- [ ] J2 Ingestion : reconstruction documents + filtrage 6 thèmes + nettoyage
- [ ] J3 Chunking 3 stratégies (C0/C1/C2) + indexes Qdrant
- [ ] J4 Golden set (80 questions, 6 thèmes × types)
- [ ] J5 Benchmark retrieval baseline (S0/S1/S2 × C0/C1/C2)
- [ ] J6 Hybrid RRF consolidé
- [ ] J7 Reranking (S3)
- [ ] J8 Parent-doc retrieval (S4) + structured output
- [ ] J9 Évaluation génération + garde-fous + local vs API
- [ ] J10 Démo Gradio + article + Loom

## Lancer le projet

```bash
# 1. Cloner + setup env
git clone <repo>
cd rag-souverain-eval
cp .env.example .env  # éditer avec vos tokens HF + OpenAI
uv sync

# 2. Démarrer Qdrant (Docker)
make qdrant-up

# 3. (À venir J2-J10) Pipeline complet
# make benchmark-all
```

## Limites assumées

- **Périmètre démonstratif** : pas de conseil juridique ou administratif personnalisé
- **Golden set imparfait** : 80 questions n'épuisent pas un domaine de 3500 docs
- **Score de confiance non calibré** : heuristique opérationnelle basée sur les signaux retrieval, pas une probabilité
- **Freshness du corpus** : pas de mécanisme de mise à jour automatique
- **Indexation longue sur CPU** : ~3h par stratégie de chunking sur 25k chunks (compromis souveraineté)

## ⚠️ Avertissement

Ce projet est une démonstration technique. Il **ne fournit pas de conseil juridique ou administratif personnalisé**. Pour toute démarche réelle, consulter directement [service-public.fr](https://www.service-public.fr/) ou un professionnel qualifié.

## Licences

- Code : MIT
- Corpus dérivé : Etalab 2.0 (héritage de service-public.fr via AgentPublic)
