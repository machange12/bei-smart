# Bei Smart — Kenya Food Price Intelligence System

An end-to-end Kenya food price intelligence system — time series forecasting, market clustering, and a Kiswahili RAG chatbot powered by WFP data.

## Project Overview

Bei Smart is a data science capstone project built by a team of five students at Moringa School, Nairobi. The system analyzes and forecasts food prices across Kenya using 20 years of WFP market price data (2006–2026), and delivers insights through an interactive Streamlit dashboard with a Kiswahili-enabled RAG chatbot.

## Research Question

Can we forecast food commodity prices across Kenyan markets and identify high-risk price volatility regions using spatiotemporal features — and make these insights accessible to everyday Kenyans in Kiswahili?

## Dataset

- Primary: WFP Kenya Food Prices 2006–2026 (HDX) — 27,384 records, 16 features
- Supplementary: KNBS Consumer Price Index, CHIRPS Rainfall Data, World Bank macro indicators
- Source: https://data.humdata.org/dataset/wfp-food-prices-for-kenya
- Google Drive: [add link here]

## Project Structure
bei-smart/
├── data/
│ ├── raw/
│ ├── cleaned/
│ └── exports/
├── notebooks/
│ ├── eda/
│ ├── modeling/
│ └── rag/
├── src/
├── reports/
└── README.md


## Team

| Name | Role | Branch |
|---|---|---|
| Mathews Machanje | Lead, RAG & NLP | eda/mat |
| Millicent Onchari | EDA & Project Management | eda/millicent |
| James Wakhu | Time Series Forecasting | eda/james |
| Esther | Spatial Analysis & Clustering | eda/esther |
| Mohamed Abdirahman | Supplementary Data & Dashboard | eda/mohamed |

## Deliverables

- CRISP-DM Data Report
- Jupyter Notebooks
- GitHub Repository
- Slide Deck
- Trello Board: [add link here]
- Deployment: [add link here]

## Tech Stack

- Python, Pandas, Scikit-learn, Prophet, DBSCAN
- LangChain, ChromaDB, Claude API
- Streamlit, FastAPI
- Supabase, GitHub Actions

## Acknowledgements

- World Food Programme (WFP) Kenya
- Humanitarian Data Exchange (HDX)
- Moringa School Data Science Programme