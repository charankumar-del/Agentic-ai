import streamlit as st
from ultralytics import YOLO
from datetime import datetime
import os
import random
import time
import uuid
import agent  # Orchestrating the logic processor from agent.py

from supabase_client import (
    insert_inspection,
    login_user,
    get_all_inspections,
    delete_inspection,
    upload_file
)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Deep-Edge Pipeline OS", layout="wide")

# Model Performance Stats (Citing internal training benchmarks)
MODEL_PRECISION, MODEL_RECALL = 0.886, 0.844
MODEL_MAP50, MODEL_MAP5095 = 0.882, 0.782

# Load YOLO Model
model = YOLO("best.pt")

st.sidebar.title("Navigation")
mode = st.sidebar.radio("Select Mode", ["User", "Admin"])

# =====================================================
# ======================= USER MODE ====================
# =====================================================
if mode == "User":
    st.title("🌊 Pipeline Risk OS: Autonomous Inspector")
    st.markdown("Upload subsea data for AI analysis and compliance reports.")

    conf_thresh = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)
    uploaded_file = st.file_uploader("Upload ROV Image", type=["jpg","png","jpeg"])

    if uploaded_file:
        # Securely saving the local image for processing
        original_filename = f"{uuid.uuid4()}.jpg"
        with open(original_filename, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("🔍 Run Full Inspection"):
            start_time = time.time()
            
            # Step 1: Vision Model Inference (Extraction Phase)
            results = model.predict(source=original_filename, conf=conf_thresh, save=True)
            inference_time = round(time.time() - start_time, 2)
            result = results[0]
            
            # Ensuring the annotated path is correctly captured for display
            annotated_path = os.path.join(result.save_dir, os.path.basename(original_filename))

            col1, col2 = st.columns([2,1])
            with col1:
                st.image(annotated_path, use_container_width=True)

            with col2:
                detected_classes, detection_table, max_conf = [], [], 0
                if len(result.boxes) > 0:
                    for box in result.boxes:
                        # Direct access to class and confidence tensors
                        cls_idx = int(box.cls[0])
                        class_name = result.names[cls_idx]
                        conf = float(box.conf[0])
                        
                        detected_classes.append(class_name)
                        detection_table.append([class_name, round(conf, 2)])
                        max_conf = max(max_conf, conf)
                    
                    st.table(detection_table)
                    # Logic: Risk severity based on model confidence metrics
                    risk = "HIGH" if max_conf > 0.85 else "MEDIUM" if max_conf > 0.60 else "LOW"
                    color = "red" if risk == "HIGH" else "orange" if risk == "MEDIUM" else "green"
                else:
                    risk, color = "SAFE", "green"
                    st.write("No anomalies detected.")

                st.markdown(f"### ⚠ Risk Level: <span style='color:{color}'>{risk}</span>", unsafe_allow_html=True)
                st.write(f"⏱ Inference: {inference_time} sec")

            # Step 2: Agentic Compliance Report (Interpretation Phase)
            st.markdown("---")
            st.subheader("🤖 Agentic Compliance Assessment")
            inspection_id = f"INS-{random.randint(1000,9999)}"
            report_name = f"{inspection_id}_report.txt"

            with st.spinner("Agent consulting DNV regulations..."):
                # Passing technical metadata to the agent for interpretation
                agent_report = agent.generate_compliance_report(detection_table, risk)
                st.write(agent_report)
                
                with open(report_name, "w", encoding="utf-8") as f:
                    f.write(f"ID: {inspection_id} | Date: {datetime.now()}\n\n{agent_report}")

            # Step 3: Supabase Sync (Deployment Phase)
            try:
                # Synchronizing vision data and agent outputs to the cloud
                img_url = upload_file("image_bucket", original_filename)
                ann_url = upload_file("image_bucket", annotated_path)
                rep_url = upload_file("image_bucket", report_name)

                insert_inspection({
                    "inspection_id": inspection_id, 
                    "file_name": uploaded_file.name,
                    "detected_classes": detected_classes, 
                    "highest_confidence": float(max_conf),
                    "risk_level": risk, 
                    "inference_time": float(inference_time),
                    "precision": MODEL_PRECISION, 
                    "recall": MODEL_RECALL,
                    "map50": MODEL_MAP50, 
                    "map5095": MODEL_MAP5095,
                    "image_url": img_url, 
                    "annotated_image_url": ann_url,
                    "pdf_url": rep_url, # Key 'pdf_url' used to store the link to the .txt report
                    "status": "completed"
                })
                st.success("Inspection Synchronized ✅")
            except Exception as e:
                st.error(f"Sync Error: {e}")

            with open(report_name, "rb") as f:
                st.download_button("⬇ Download Report (.txt)", f, file_name=report_name)

# =====================================================
# ======================= ADMIN MODE ===================
# =====================================================
if mode == "Admin":
    st.title("📊 Monitoring Dashboard")
    # ... Your existing Admin authentication and Supabase retrieval logic ...
