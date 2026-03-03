import streamlit as st
from ultralytics import YOLO
from datetime import datetime
import os
import random
import time
import uuid

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

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

# ---------------- LOAD MODEL (with fallback + self-healing) ----------------
PRIMARY_MODEL_PATH = "best.pt"
FALLBACK_MODEL_NAME = os.getenv("YOLO_FALLBACK_MODEL", "yolov8n.pt")

MODEL_LOAD_ERROR = None
MODEL_USING_FALLBACK = False
model = None

try:
    # Try to load the project-specific trained weights first.
    model = YOLO(PRIMARY_MODEL_PATH)
except Exception as primary_err:
    try:
        # First attempt to load a generic YOLOv8 model.
        model = YOLO(FALLBACK_MODEL_NAME)
        MODEL_USING_FALLBACK = True
    except Exception as fallback_err:
        retry_err = None

        # If the fallback error indicates a corrupted local .pt file, delete it
        # and retry once so Ultralytics can re-download a fresh copy.
        if (
            "PytorchStreamReader failed reading zip archive" in str(fallback_err)
            and os.path.exists(FALLBACK_MODEL_NAME)
        ):
            try:
                os.remove(FALLBACK_MODEL_NAME)
                model = YOLO(FALLBACK_MODEL_NAME)
                MODEL_USING_FALLBACK = True
                fallback_err = None  # Clear error since retry succeeded
            except Exception as e2:
                retry_err = e2

        if fallback_err is not None:
            # Both primary and fallback ultimately failed
            base_msg = (
                f"Primary model '{PRIMARY_MODEL_PATH}' failed with: {primary_err}. "
                f"Fallback model '{FALLBACK_MODEL_NAME}' also failed with: {fallback_err}."
            )
            if retry_err is not None:
                base_msg += f" Retry after deleting local fallback file also failed with: {retry_err}."
            MODEL_LOAD_ERROR = base_msg

# ---------------- SIDEBAR ROLE ----------------
st.sidebar.title("Navigation")
mode = st.sidebar.radio("Select Mode", ["User", "Admin"])

# =====================================================
# ======================= USER MODE ====================
# =====================================================
if mode == "User":

    st.title("🚢 AI Underwater Inspection System")
    st.markdown("Upload subsea infrastructure imagery to receive autonomous risk analysis and compliance reports.")

    # If we are running with a generic YOLO fallback model, let the user know.
    if MODEL_USING_FALLBACK and not MODEL_LOAD_ERROR:
        st.info(
            f"Using fallback vision model '{FALLBACK_MODEL_NAME}' because "
            f"'{PRIMARY_MODEL_PATH}' could not be loaded."
        )

    # If the YOLO model failed to load (e.g., corrupted or missing weights),
    # show a clear message instead of crashing the whole app.
    if MODEL_LOAD_ERROR:
        st.error(
            "Vision model failed to load, and no fallback model is available. "
            "Please ensure you have a valid Ultralytics/YOLOv8 model file "
            "accessible to the app.\n\n"
            f"Details: {MODEL_LOAD_ERROR}"
        )
        st.stop()

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

            # --- STEP 2: Agentic Compliance Report (PDF) ---
            st.markdown("---")
            st.subheader("🤖 Agentic Compliance & Integrity Assessment")
            
            inspection_id = f"INS-{random.randint(1000,9999)}"
            pdf_name = f"{inspection_id}_compliance_report.pdf"

            with st.spinner("Agent is cross-referencing DNV/API maritime regulations..."):
                # Fetching the expert assessment from agent.py
                agent_report = agent.generate_compliance_report(detection_table, risk)

                # Displaying report in the UI (text)
                st.write(agent_report)

                # Generate a PDF version of the report
                width, height = A4
                c = canvas.Canvas(pdf_name, pagesize=A4)
                text_obj = c.beginText(50, height - 50)
                text_obj.setFont("Helvetica", 11)

                header_lines = [
                    f"INSPECTION ID: {inspection_id}",
                    f"TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    f"VISUAL RISK LEVEL: {risk}",
                    "-" * 60,
                    "",
                ]

                for line in header_lines + agent_report.splitlines():
                    text_obj.textLine(line)

                c.drawText(text_obj)
                c.showPage()
                c.save()

            # --- STEP 3: Sync to Supabase ---
            try:
                # Uploading visual data
                image_url = upload_file("image_bucket", original_filename)
                annotated_url = upload_file("image_bucket", annotated_path)
                
                # Uploading the agent's PDF report
                pdf_url = upload_file("image_bucket", pdf_name)

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
                    "pdf_url": pdf_url,
                    "status": "completed"
                })

                st.success("Synchronized with Secure Storage ✅")

            except Exception as e:
                msg = str(e)
                if "Supabase is not configured" in msg:
                    st.info("Cloud synchronization is disabled because Supabase is not configured. PDF download is still available below.")
                else:
                    st.error(f"Synchronization Error: {e}")
            
            # Download Option (PDF)
            with open(pdf_name, "rb") as f:
                st.download_button(
                    "⬇ Download Integrity Assessment (PDF)",
                    f,
                    file_name=pdf_name,
                    mime="application/pdf"
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
