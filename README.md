# ✈️ Aviation Delay & Operational Performance Analytics Studio

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Framework: Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Engine: Scikit--Learn](https://img.shields.io/badge/Engine-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)

An end-to-end, production-grade machine learning system and interactive simulation suite designed to quantify network propagation delays, analyze airside congestion, and generate predictive operational risk assessments across domestic aviation corridors.

---

## 📌 Architectural Overview

This system bridges historical exploratory data science with real-time operations. The core architecture integrates an automated feature vectorization pipeline with an ensemble regressor to deliver deterministic arrival risk assessments.

```text
       [ Raw US Flight Ingestion ] 
                    │
                    ▼
     [ Feature Engineering Pipeline ]
   ┌────────────────┬────────────────┐
   ▼                ▼                ▼
[ Runways ]    [ Temporal ]    [ Categorical ]
(Taxi-In/Out) (Day of Month)   (Carrier/Hubs)
   └────────────────┬────────────────┘
                    ▼
       [ Vector Matrix Ingestion ]
                    │
                    ▼
    [ Random Forest Inference Engine ]
                    │
                    ▼
      [ Streamlit BI Simulation ] ───► [ Risk Classification Framework ]
