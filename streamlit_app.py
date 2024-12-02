# |--------------------------------|
# | STREAMLIT APP FOR SEGMENTATION |
# |--------------------------------|

import streamlit as st
import requests
import os

from app.constants import TMP_PATH
from app import APP_HOST, APP_PORT

API_URL = f"http://localhost:{APP_PORT}/segmentation"

st.title("SegMNet: A Kidney Tumor Segmentation Service")

uploaded_file = st.file_uploader("Upload your file", type=["txt", "jpg", "png"])
if uploaded_file is not None:
    with st.spinner("Uploading and processing..."):
        # Upload file to FastAPI
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(f"{API_URL}/upload/", files=files)
        result = response.json()
        st.success("File uploaded and segmentation started!")
        st.write(f"Result will be saved to: {result['result_path']}")

st.markdown("---")

st.header("Download Processed Files")
if os.path.exists(TMP_PATH):
    files = os.listdir(TMP_PATH)
    for file in files:
        st.markdown(f"- [{file}](file://{os.path.join(TMP_PATH, file)})")
