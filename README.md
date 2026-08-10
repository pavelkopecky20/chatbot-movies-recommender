---
title: Movie Chatbot
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.61.0"
app_file: app.py
pinned: false
---

# Movie Chatbot

Conversational movie recommendation chatbot over a TMDB catalog (1331 titles). Portfolio demo of a
RAG/agentic pipeline over a streaming catalog.

## What it does

- **Hybrid retrieval** (`brain.py`, `VectorStore`) -- hard filter by genre/origin country, then similarity
  ranking over embeddings (`text-embedding-3-small`).
- **Sticky constraints** -- constraints like genre or origin country are remembered across conversation
  turns until the user explicitly clears them. Extracted two ways: fast keyword rules and semantic
  classification over embeddings (`embedding_classifier.py`, `ConstraintClassifier` -- centroids of
  reference phrases per category).
- **LLM routing** -- simple turns (short constraint change) go to `gpt-4o-mini`, more complex ones to
  `gpt-4o`.
- **Structured output** -- the LLM response is enforced via a pydantic schema (`AgentResponse`), not
  parsed from free text.
- **Guardrails** -- the LLM may only recommend titles from the retrieval candidates; the output is
  validated against real catalog IDs before being returned to the user.

## Note on this deployment

This is a public but unlisted demo instance -- the number of messages per visitor is capped (see the
sidebar) to avoid exhausting API credit. Movie data comes from [TMDB](https://www.themoviedb.org/)
(`fetch_tmdb_catalog.py` -- a one-off script that downloaded the catalog, not part of the running demo).

Note: the chatbot itself converses in Czech (it targets a Czech-language movie catalog and audience) --
type your queries in Czech, e.g. "chci horor" (I want a horror movie) or "film, kde hraje Tom Hanks"
(a movie starring Tom Hanks).
