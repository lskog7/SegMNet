import streamlit as st
import requests
from pathlib import Path
import os

from app.main import APP_PORT, APP_HOST
from app.constants import TMP_PATH

# Define FastAPI endpoints
API_UPLOAD_URL = f"http://localhost:{APP_PORT}/segmentation/upload/"
API_PROCESS_URL = f"http://localhost:{APP_PORT}/segmentation/process/"
API_DOWNLOAD_URL = f"http://localhost:{APP_PORT}/segmentation/download/"
TMP_PATH = TMP_PATH

# Title and description
st.title("SegMNet: A Kidney Tumor Segmentation Service")
st.markdown(
    """
    This application allows users to upload **NIFTI files** for kidney tumor segmentation.
    The segmentation process is handled by a FastAPI backend.
    """
)

# File upload section
uploaded_file = st.file_uploader("Upload your NIFTI file (.nii, .nii.gz)", type=[".nii", ".nii.gz"])

if uploaded_file:
    st.subheader("File Upload")
    with st.spinner("Uploading file..."):
        try:
            # Send file to the API for upload
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/octet-stream")}
            response = requests.post(API_UPLOAD_URL, files=files)
            response_data = response.json()

            if response.status_code == 200:
                st.success("File uploaded successfully!")
                uploaded_filename = uploaded_file.name

                # Start the segmentation process
                st.subheader("Start Segmentation")
                if st.button("Start Segmentation"):
                    with st.spinner("Processing file..."):
                        process_response = requests.post(
                            f"{API_PROCESS_URL}?filename={uploaded_filename}"
                        )
                        process_data = process_response.json()

                        if process_response.status_code == 200:
                            st.success(f"Segmentation started! Result will be saved to: {process_data['result_path']}")
                        else:
                            st.error(f"Error: {process_data.get('detail', 'Segmentation failed.')}")
            else:
                st.error(f"Error: {response_data.get('detail', 'File upload failed.')}")
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to segmentation service: {e}")

st.markdown("---")

# Processed files section
st.header("Download Segmentation Results")
if os.path.exists(TMP_PATH):
    processed_files = list(Path(TMP_PATH).glob("result_*.nii.gz"))
    if processed_files:
        for file in processed_files:
            file_name = file.name
            download_url = f"{API_DOWNLOAD_URL}{file_name}"
            st.markdown(f"- [Download {file_name}]({download_url})")
    else:
        st.write("No processed files available yet.")
else:
    st.write("Processing directory not found. Please check your setup.")
