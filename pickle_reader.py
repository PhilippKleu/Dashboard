# pickle_from_github_app.py

import streamlit as st
import pickle
import pandas as pd
import requests
import io

st.set_page_config(page_title="📦 GitHub Pickle Viewer", layout="wide")
st.title("📦 Load Pickle File from GitHub")

# Input-Feld für URL
url = st.text_input("🔗 Enter raw GitHub URL to the pickle file:")

if url:
    try:
        st.info("Downloading file from GitHub...")
        response = requests.get(url)
        response.raise_for_status()

        data = pickle.load(io.BytesIO(response.content))

        st.success(f"✅ File loaded successfully! Type: `{type(data).__name__}`")

        # Anzeige wie vorher
        if isinstance(data, pd.DataFrame):
            st.subheader("📊 Detected DataFrame")
            st.dataframe(data)
            st.write("Shape:", data.shape)
            st.write("Columns:", data.columns.tolist())

        elif isinstance(data, dict):
            st.subheader("🗂️ Detected Dictionary")
            st.write(f"Length: {len(data)}")
            selected_key = st.selectbox("Select key to view:", list(data.keys()))
            st.write(f"Type: {type(data[selected_key])}")
            st.write(data[selected_key])

        elif isinstance(data, list):
            st.subheader("📋 Detected List")
            st.write(f"Length: {len(data)}")
            st.write("First 5 items:")
            st.write(data[:5])

        else:
            st.subheader("🔍 Raw Object")
            st.write(data)

    except Exception as e:
        st.error(f"❌ Failed to load or parse pickle: {e}")
