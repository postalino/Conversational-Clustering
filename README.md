# Conversational Clustering: preparazione del set iniziale

GUI (Streamlit) per scaricare un dataset testuale, calcolare gli embedding, generare e confrontare
diversi clustering iniziali e scegliere quello da cui partirà l'agente conversazionale.
La parte agentica non è ancora attiva: per ora c'è solo un chatbot.

## Avvio

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env          # poi inserisci le chiavi
streamlit run app/app.py
```

## Pagine

| Pagina | Cosa fa |
|---|---|
| **Panoramica** | Stato della pipeline, set iniziale scelto, stato di credenziali e modelli |
| **Dataset** | Download da Kaggle, scelta colonne, campione riproducibile (seed, filtri lunghezza, deduplica, solo testi già in cache) |
| **Embedding** | Calcolo con richieste parallele, cache a blocchi per (dataset, modello), ripresa dopo errori, import della vecchia cache |
| **Clustering** | K-Means, HDBSCAN; preprocessing L2 + PCA/UMAP; assegnazioni soft; curva per scegliere k; mappa 2D; parole chiave c-TF-IDF; nomi LLM in parallelo + revisione globale; modifica manuale; salvataggio |
| **Set iniziali** | Set salvati, confronto (ARI, metriche, dimensioni), scelta del set iniziale per l'agente, export CSV |
| **Chatbot** | Chat con il modello scelto, con il set iniziale come contesto (opzionale) |
| **Modelli & API** | Provider (OpenRouter, OpenAI-compatibile, Ollama), elenco modelli con prezzi, test, richieste parallele |
| **Prompt** | Tutti i prompt modificabili, con anteprima su dati reali e prova su un cluster; la versione è salvata in ogni set |
| **Log & Errori** | Errori con traceback e contesto, registro chiamate API (latenza, token, esito), log filtrabile, `logs/app.log` |

## Algoritmi e metriche

- **K-Means**: riferimento classico.
- **HDBSCAN**: basato sulla densità; trova da solo il numero di cluster e marca il rumore.

Metriche mostrate: silhouette (↑), Davies-Bouldin (↓), quota di rumore (HDBSCAN) e quota di punti ambigui
(probabilità massima < 0.5), cioè i casi di confine che l'agente potrà usare per decidere cosa chiedere.

## Nomi dei cluster (LLM)

1. Una chiamata per cluster, **in parallelo**. Il prompt contiene parole chiave distintive, parole chiave dei
   cluster più vicini (per nomi contrastivi), esempi tipici ed esempi dal centro alla periferia. Restituisce
   nome, descrizione, evidenze e un punteggio di **coerenza** 1-5.
2. Una **revisione globale** opzionale che vede tutti i nomi insieme e li rende distinti e coerenti.

L'output strutturato si adatta al modello: schema JSON stretto, poi `json_object`, poi JSON nel prompt.

## Struttura

```
app/            GUI Streamlit (app.py, common.py, explorer.py, pages/)
src/            dataset, embeddings, providers, llm, prompts, clustering, projection,
                labeling, runs (set salvati), chatbot, logging_setup
data/           dataset (+ copia Parquet), cache embedding, set salvati (runs/), prompts.json
logs/app.log    log persistente
tests/          pytest
```
