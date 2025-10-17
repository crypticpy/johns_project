# Changelog
All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and this project adheres to Semantic Versioning.

## [0.1.0] - 2025-10-17
### Added
- Rerank Adapters (integration boundary):
  - Builtin lexical reranker (deterministic, offline) combining Jaccard + weighted-Jaccard and normalized to [0,1].
    - [src/ai/rerank/interface.py](src/ai/rerank/interface.py)
    - [src/ai/rerank/builtin_lexical.py](src/ai/rerank/builtin_lexical.py)
  - Optional Cross-Encoder reranker backed by HuggingFace sentence-transformers, guarded for offline mode.
    - [src/ai/rerank/cross_encoder.py](src/ai/rerank/cross_encoder.py)
  - Reranker factory with env default `APP_RERANK_BACKEND` (defaults to "builtin").
    - [src/ai/rerank/factory.py](src/ai/rerank/factory.py)
- Search API rerank integration:
  - GET `/search/nn` accepts `rerank` (bool) and `rerank_backend` ("builtin" | "cross-encoder").
  - Deterministic reordering of FAISS top-k candidates based on adapter scores; filters preserved.
    - [src/api/routers/search.py](src/api/routers/search.py)
- Offline-friendly integration tests:
  - Builtin rerank changes order when lexical overlap differs; department filter scenario included.
    - [tests/integration/api/test_search_rerank.py](tests/integration/api/test_search_rerank.py)
- Docker packaging (multi-stage) with offline defaults and non-root runtime:
  - [Dockerfile](Dockerfile)
  - [/.dockerignore](.dockerignore)
- Build tooling:
  - Makefile targets for build/test/package/docker.
    - [Makefile](Makefile)
  - Release check script: pre-commit, mypy, pytest, build, twine/pip check.
    - [scripts/release_check.sh](scripts/release_check.sh)

### Changed
- CI workflow updated to enforce offline rerank defaults and validate Docker builds:
  - Set `APP_RERANK_BACKEND=builtin`; added `docker-build` job.
    - [.github/workflows/ci.yml](.github/workflows/ci.yml)
- README updated with rerank usage (API params, env flags) and Docker build/run instructions.
  - [README.md](README.md)

### Fixed
- Deterministic scoring and tiebreaks in rerank ensure stable ordering across runs.

### Security
- Enforced offline-safe defaults to prevent unintended network calls:
  - `APP_EMBED_BACKEND=builtin`, `APP_RERANK_BACKEND=builtin`, `ANALYZER_BACKEND=offline`, `TRANSFORMERS_OFFLINE=1`, `HF_HUB_OFFLINE=1`
- No PII in logs/metrics; outputs sanitized.

### Packaging & Reproducibility
- Version set to `0.1.0` in [pyproject.toml](pyproject.toml).
- Reproducible builds documented and honored via `SOURCE_DATE_EPOCH` in Makefile/CI/Docker runtime.

[0.1.0]: https://example.com/sd-onboarding-analyzer/releases/tag/v0.1.0