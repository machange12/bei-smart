# Bei Smart — Kenya Food Price Intelligence System

An end-to-end Kenya food price intelligence system — time series forecasting, market clustering, and a Kiswahili RAG chatbot, built on KAMIS, FEWS NET, WFP and KNBS price data.

Live app: <add streamlit.app URL>

## Project Overview

Bei Smart is a data science capstone project built by a team of five students at Moringa School, Nairobi. The system analyzes and forecasts food prices across Kenya using 20 years of market price data (2006–2026), and delivers insights through an interactive Streamlit dashboard (AgriPulse) with a Kiswahili-enabled RAG chatbot available from the floating "Ask AgriPulse" button on every page. It currently serves around 270 forecast series across about 80 markets in all 8 regions (median test error ~9–10%).

## Research Question

Can we forecast food commodity prices across Kenyan markets and identify high-risk price volatility regions using spatiotemporal features — and make these insights accessible to everyday Kenyans in Kiswahili?

## Dataset

- Prices: KAMIS (Ministry of Agriculture, scraped), FEWS NET, WFP Kenya Food Prices (HDX) + supplementary WFP, KNBS — stitched into one monthly chain (`scripts/build_chained_all.py`)
- Context: ERA5 rainfall, EPRA fuel prices, ONI, world grain futures, KES/USD
- Refreshed automatically on the 5th of each month by GitHub Actions (`.github/workflows/refresh-data.yml`): scrape → rebuild chain, anomalies and forecasts → commit `app/data/`
- Source: https://data.humdata.org/dataset/wfp-food-prices-for-kenya
- Google Drive: [add link here]

## Run locally

```bash
python -m venv venv && venv/Scripts/pip install -r app/requirements.txt
venv/Scripts/python.exe -m streamlit run app/streamlit_app.py   # chat needs GROQ_API_KEY in .env
```

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

| Name | 
|---|
| Mathews Machanje |
| Millicent Onchari 
| James Wakhu 
| Esther 
| Mohamed Abdirahman 
| Marren ojode

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
