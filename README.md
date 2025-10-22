# 💰 FinOps Optimisation Engine
Author: Heera Arora 
2nd Year UTS Software Engineering 

An **AI-driven FinOps system** that analyses and optimises Google BigQuery workloads using **Vertex AI (Gemini)**, **heuristic rules**, and **cost intelligence dashboards**.  
The goal is to make cloud spending **visible, explainable, and optimised**.

---

## 🚀 Overview

This project automates the FinOps workflow for BigQuery by:
- Extracting and analysing historical query metadata from `INFORMATION_SCHEMA`.
- Using **Gemini models** to optimise inefficient SQL queries.
- Running **cost heuristics** to flag high-cost patterns (e.g. large scans, unused columns).
- Logging all optimisation recommendations and metrics to **BigQuery**.
- Visualising insights through **Looker Studio dashboards**.

---

## 🧩 Core Components

| Component | Description |
|------------|-------------|
| `optimise_my_sql.py` | Main entry script — fetches queries, optimises via Gemini, validates results. |
| `optimisation_queries.py` | Generates cost heuristics and ranking logic. |
| `config.json` / `config.yaml` | Configuration for project IDs, thresholds, and model settings. |
| `run_demo.py` | Lightweight demo runner for local testing. |
| `logs/` | Stores query results and optimisation summaries. |

---

## 🧠 Features

- **AI-powered SQL rewriting** via Vertex AI Gemini.
- **Automatic cost analysis** using BigQuery metadata.
- **Dry-run validator** to ensure schema equivalence and safety.
- **Heuristic cost baseline** to compare before/after impact.
- **Structured logging** for production deployment.
- **Looker dashboard integration** for cost visibility.

