import streamlit as st
from ultralytics import YOLO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from datetime import datetime
import tempfile
import os
import random
import time
import uuid

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
    st.markdown("Upload inspection image and receive AI analysis.")

    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 0.1, 1.0, 0.25, 0.05
    )

    uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if uploaded_file:

        # Save original locally
        original_filename = f"{uuid.uuid4()}.jpg"
        with open(original_filename, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("🔍 Run Inspection"):

            start_time = time.time()

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
                st.image(annotated_path, use_column_width=True)

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

                    if max_conf > 0.85:
                        risk = "HIGH"
                        color = "red"
                    elif max_conf > 0.60:
                        risk = "MEDIUM"
                        color = "orange"
                    else:
                        risk = "LOW"
                        color = "green"

                else:
                    risk = "SAFE"
                    color = "green"
                    st.write("No anomalies detected.")

                st.markdown(
                    f"### ⚠ Risk Level: <span style='color:{color}'>{risk}</span>",
                    unsafe_allow_html=True
                )

                st.write("⏱ Inference Time:", inference_time, "sec")

            # -------- PDF GENERATION --------
            inspection_id = f"INS-{random.randint(1000,9999)}"
            report_name = f"{inspection_id}.pdf"

            doc = SimpleDocTemplate(report_name, pagesize=A4)
            elements = []
            styles = getSampleStyleSheet()

            elements.append(Paragraph("<b>Underwater Inspection Report</b>", styles["Title"]))
            elements.append(Spacer(1, 12))

            summary_data = [
                ["Inspection ID", inspection_id],
                ["Date", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                ["Risk Level", risk],
                ["Inference Time", f"{inference_time} sec"],
                ["Precision", MODEL_PRECISION],
                ["Recall", MODEL_RECALL],
                ["mAP@0.5", MODEL_MAP50],
                ["mAP@0.5-95", MODEL_MAP5095]
            ]

            table = Table(summary_data, colWidths=[2.5*inch, 3*inch])
            table.setStyle(TableStyle([
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                ('BACKGROUND', (0,0), (-1,0), colors.lightgrey)
            ]))

            elements.append(table)
            elements.append(Spacer(1, 20))

            if len(detection_table) > 0:
                pdf_table = Table(
                    [["Class","Confidence"]] + detection_table,
                    colWidths=[3*inch,2*inch]
                )
                elements.append(pdf_table)
                elements.append(Spacer(1,20))

            elements.append(RLImage(annotated_path, width=5*inch, height=4*inch))

            doc.build(elements)

            # -------- UPLOAD TO SUPABASE STORAGE --------
            try:
                image_url = upload_file("image_bucket", original_filename)
                annotated_url = upload_file("image_bucket", annotated_path)
                pdf_url = upload_file("image_bucket", report_name)

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

                st.success("Inspection saved to database & storage ✅")

            except Exception as e:
                st.error(f"Upload or DB Error: {e}")

            # Download PDF
            with open(report_name, "rb") as f:
                st.download_button(
                    "⬇ Download Inspection Report",
                    f,
                    file_name=report_name
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
                st.write("Inference Time:", record["inference_time"])
                st.write("Precision:", record["precision"])
                st.write("Recall:", record["recall"])
                st.write("mAP@0.5:", record["map50"])
                st.write("mAP@0.5-95:", record["map5095"])
                st.write("Created At:", record["created_at"])

                if record.get("annotated_image_url"):
                    st.image(record["annotated_image_url"], width=400)

                if record.get("pdf_url"):
                    st.markdown(f"[Download Report]({record['pdf_url']})")

                if st.button(f"Delete {record['id']}"):
                    delete_inspection(record["id"])
                    st.success("Deleted")
                    st.rerun()

    else:
        st.info("No inspection records found.")
