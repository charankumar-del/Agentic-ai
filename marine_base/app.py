import streamlit as st
from ultralytics import YOLO
from datetime import datetime
import os
import random
import time
import uuid

# Import our new AI Consulting Agent brain
import agent 

from supabase_client import (
    insert_inspection,
    login_user,
    get_all_inspections,
    delete_inspection,
    upload_file
)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="NautiCAI Marine Inspection", layout="wide")

# ---------------- MODEL METRICS (From Training) ----------------
MODEL_PRECISION = 0.886
MODEL_RECALL = 0.844
MODEL_MAP50 = 0.882
MODEL_MAP5095 = 0.782

# ---------------- LOAD MODEL ----------------
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

# ---------------- SIDEBAR ROLE ----------------
st.sidebar.title("Navigation")
mode = st.sidebar.radio("Select Mode", ["User", "Admin"])

# =====================================================
# ======================= USER MODE ====================
# =====================================================
if mode == "User":

    st.title("🚢 AI Underwater Inspection System")
    st.markdown("Upload subsea infrastructure imagery to receive autonomous risk analysis and compliance reports.")

    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 0.1, 1.0, 0.25, 0.05
    )

    uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if uploaded_file:

        # Save original locally
        original_filename = f"{uuid.uuid4()}.jpg"
        with open(original_filename, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("🔍 Run Full Inspection & Agentic Assessment"):

            start_time = time.time()

            # --- STEP 1: Vision Inference ---
            results = model.predict(
                source=original_filename,
                conf=confidence_threshold,
                save=True
            )

            inference_time = round(time.time() - start_time, 2)
            result = results[0]

            annotated_path = os.path.join(result.save_dir, os.path.basename(original_filename))

            col1, col2 = st.columns([2,1])

            with col1:
                st.image(annotated_path, use_container_width=True)

            with col2:
                detected_classes = []
                detection_table = []
                max_conf = 0

                if len(result.boxes) > 0:
                    boxes = result.boxes
                    classes = boxes.cls.cpu().numpy()
                    confidences = boxes.conf.cpu().numpy()

                    for i in range(len(classes)):
                        class_name = result.names[int(classes[i])]
                        conf = float(confidences[i])

                        detected_classes.append(class_name)
                        detection_table.append([class_name, round(conf,2)])
                        max_conf = max(max_conf, conf)

                    st.table(detection_table)

                    # Dynamic Risk Logic
                    risk = "HIGH" if max_conf > 0.85 else "MEDIUM" if max_conf > 0.60 else "LOW"
                    color = "red" if risk == "HIGH" else "orange" if risk == "MEDIUM" else "green"

                else:
                    risk = "SAFE"
                    color = "green"
                    st.write("No anomalies detected.")

                st.markdown(
                    f"### ⚠ Visual Risk Level: <span style='color:{color}'>{risk}</span>",
                    unsafe_allow_html=True
                )
                st.write("⏱ Inference Time:", inference_time, "sec")

            # --- STEP 2: Agentic Compliance Report ---
            st.markdown("---")
            st.subheader("🤖 Agentic Compliance & Integrity Assessment")
            
            inspection_id = f"INS-{random.randint(1000,9999)}"
            report_name = f"{inspection_id}_compliance_report.txt"

            with st.spinner("Agent is cross-referencing DNV/API maritime regulations..."):
                
                # Fetching the expert assessment from agent.py
                agent_report = agent.generate_compliance_report(detection_table, risk)
                
                # Displaying report in the UI
                st.write(agent_report)

                # Saving local copy for upload
                with open(report_name, "w", encoding="utf-8") as f:
                    f.write(f"INSPECTION ID: {inspection_id}\n")
                    f.write(f"TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"VISUAL RISK LEVEL: {risk}\n")
                    f.write("-" * 30 + "\n\n")
                    f.write(agent_report)

            # --- STEP 3: Sync to Supabase ---
            try:
                # Uploading visual data
                image_url = upload_file("image_bucket", original_filename)
                annotated_url = upload_file("image_bucket", annotated_path)
                
                # Uploading the agent's consulting draft
                report_url = upload_file("image_bucket", report_name)

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
                    "image_url": image_url,
                    "annotated_image_url": annotated_url,
                    "pdf_url": report_url, # Reusing this column for our .txt report
                    "status": "completed"
                })

                st.success("Synchronized with Secure Storage ✅")

            except Exception as e:
                st.error(f"Synchronization Error: {e}")

            # Download Option
            with open(report_name, "rb") as f:
                st.download_button(
                    "⬇ Download Integrity Assessment (.txt)",
                    f,
                    file_name=report_name,
                    mime="text/plain"
                )

# =====================================================
# ======================= ADMIN MODE ===================
# =====================================================
if mode == "Admin":

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:

        st.title("🔐 Admin Login")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            response = login_user(email, password)
            if response is not None:
                st.session_state.authenticated = True
                st.success("Login Successful")
                st.rerun()
            else:
                st.error("Invalid Credentials")

        st.stop()

    st.title("📊 Inspection Monitoring Dashboard")

    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

    response = get_all_inspections()

    if response.data:
        for record in response.data:
            with st.expander(f"{record['inspection_id']} | Risk: {record['risk_level']}"):
                st.write("File:", record["file_name"])
                st.write("Confidence:", record["highest_confidence"])
                st.write("Created At:", record["created_at"])

                if record.get("annotated_image_url"):
                    st.image(record["annotated_image_url"], width=400)

                if record.get("pdf_url"):
                    st.markdown(f"[View Integrity Assessment]({record['pdf_url']})")

                if st.button(f"Delete Record {record['id']}"):
                    delete_inspection(record["id"])
                    st.success("Record Purged")
                    st.rerun()
    else:
        st.info("No records found in current database.")
