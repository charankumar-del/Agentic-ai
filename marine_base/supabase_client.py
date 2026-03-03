from supabase import create_client
import uuid
import os

import streamlit as st


# Try to read Supabase credentials from environment first, then from Streamlit secrets.
# Guard access to st.secrets so Streamlit doesn't crash if no secrets.toml exists.
secrets_url = None
secrets_key = None
try:
    # Accessing st.secrets may raise StreamlitSecretNotFoundError when no secrets are defined.
    secrets_url = st.secrets.get("SUPABASE_URL", None)  # type: ignore[attr-defined]
    secrets_key = st.secrets.get("SUPABASE_KEY", None)  # type: ignore[attr-defined]
except Exception:
    secrets_url = None
    secrets_key = None

SUPABASE_URL = os.getenv("SUPABASE_URL") or secrets_url
SUPABASE_KEY = os.getenv("SUPABASE_KEY") or secrets_key

supabase = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def _require_supabase():
    """
    Ensure Supabase is configured before calling any DB/storage functions.
    Raises a clear error message instead of failing at import time.
    """
    if supabase is None:
        raise RuntimeError(
            "Supabase is not configured. Set SUPABASE_URL and SUPABASE_KEY "
            "in your environment (.env) or in Streamlit secrets."
        )
    return supabase


# ---------------- DATABASE FUNCTIONS ----------------

def insert_inspection(data):
    client = _require_supabase()
    return client.table("inspections").insert(data).execute()


def get_all_inspections():
    client = _require_supabase()
    return client.table("inspections").select("*").order("created_at", desc=True).execute()


def delete_inspection(record_id):
    client = _require_supabase()
    return client.table("inspections").delete().eq("id", record_id).execute()


# ---------------- AUTH ----------------

def login_user(email, password):
    client = _require_supabase()
    response = client.auth.sign_in_with_password({
        "email": email,
        "password": password
    })

    if response.user is None:
        return None

    return response


# ---------------- STORAGE UPLOAD ----------------

def upload_file(bucket_name, file_path):
    """
    Upload file to Supabase Storage and return public URL.
    """
    client = _require_supabase()

    file_extension = os.path.splitext(file_path)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"

    with open(file_path, "rb") as f:
        client.storage.from_(bucket_name).upload(unique_filename, f)

    public_url = client.storage.from_(bucket_name).get_public_url(unique_filename)

    return public_url
