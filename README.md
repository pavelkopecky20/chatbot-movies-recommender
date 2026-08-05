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

Konverzační doporučovací chatbot nad katalogem filmů (170 titulů z TMDB). Portfolio ukázka RAG/agentního
pipeline pro streamovací katalog.

## Co to umí

- **Hybrid retrieval** (`brain.py`, `VectorStore`) -- tvrdý filtr podle žánru/země původu, pak similarity
  ranking nad embeddingy (`text-embedding-3-small`).
- **Sticky constraints** -- omezení jako žánr nebo země původu se pamatují napříč tahy konverzace, dokud
  je uživatel explicitně nezruší. Extrahují se dvěma způsoby: rychlá keyword pravidla a sémantická
  klasifikace nad embeddingy (`embedding_classifier.py`, `ConstraintClassifier` -- centroidy referenčních
  frází per kategorie).
- **LLM routing** -- jednoduché tahy (krátká změna omezení) jdou na `gpt-4o-mini`, složitější na `gpt-4o`.
- **Structured output** -- odpověď LLM je vynucená přes pydantic schéma (`AgentResponse`), ne parsovaná
  z volného textu.
- **Guardrails** -- LLM smí doporučit jen tituly z retrievalem nalezených kandidátů; výstup se ještě
  validuje proti reálným katalogovým ID před vrácením uživateli.

## Poznámka k tomuhle nasazení

Tohle je veřejná, ale nepropagovaná demo instance -- počet zpráv na návštěvníka je omezený (viz postranní
panel), aby nedošlo k vyčerpání API kreditu. Data o filmech pocházejí z [TMDB](https://www.themoviedb.org/)
(`fetch_tmdb_catalog.py` -- jednorázový skript, který katalog stáhl, není součástí běžícího dema).
