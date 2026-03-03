# 🚢 AI Underwater Anomaly Detection System

## Overview
This project is a web-based AI system for underwater hull and subsea infrastructure inspection.

It allows:
- Upload image
- Run YOLOv8 detection
- View bounding boxes + confidence
- Generate professional PDF report
- Store results in Supabase
- Admin monitoring dashboard

## Model
YOLOv8 Multi-class Detector

Classes:
- Corrosion
- Marine Growth
- Debris
- Healthy Surface

## Evaluation Metrics
- Precision: 0.886
- Recall: 0.844
- mAP@0.5: 0.882
- mAP@0.5-95: 0.782

## Tech Stack
- Streamlit
- YOLOv8 (Ultralytics)
- Supabase (Database + Storage)
- ReportLab (PDF generation)

## Setup

1. Install dependencies:

pip install -r requirements.txt


2. Add your Supabase credentials inside `supabase_client.py`.

3. Run app:
streamlit run app.py


## Admin Login
Use your Supabase Auth credentials.

---

## Architecture

User → Upload → Detect → Generate PDF → Upload to Supabase → Admin Dashboard