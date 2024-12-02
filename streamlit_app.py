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
st.markdown("""
    This application allows users to upload **NIFTI files** for kidney tumor segmentation.
    The segmentation process is handled by a FastAPI backend.
""")

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

# TODO: Add pretty interface:
# 1. Upload button.
# 2. Uploaded file list (list of files in TMP_DIR).
# 3. Segmentation button.
# 4. List of result files.
# 5. Window with result files and ability to download them.
# 6. Possibility to choose one of the downloaded images to segment from the list.
# 7. When image to segment is selected - window with metadata of this image. (size, name, etc.)

# Display uploaded files
st.header("Uploaded Files")
if os.path.exists(TMP_PATH):
    uploaded_files = list(Path(TMP_PATH).glob("*.nii.gz"))
    if uploaded_files:
        selected_file = st.selectbox("Select a file to view metadata:", [file.name for file in uploaded_files])
        if selected_file:
            file_path = Path(TMP_PATH) / selected_file
            file_info = file_path.stat()
            st.write(f"**File Name:** {selected_file}")
            st.write(f"**File Size:** {file_info.st_size / (1024 * 1024):.2f} MB")
            st.write(f"**Last Modified:** {file_info.st_mtime}")
    else:
        st.write("No uploaded files available.")
else:
    st.write("Upload directory not found. Please check your setup.")

# Segmentation button
if st.button("Segment Selected File"):
    if selected_file:
        with st.spinner("Segmenting selected file..."):
            process_response = requests.post(
                f"{API_PROCESS_URL}?filename={selected_file}"
            )
            process_data = process_response.json()

            if process_response.status_code == 200:
                st.success(f"Segmentation completed! Result saved to: {process_data['result_path']}")
            else:
                st.error(f"Error: {process_data.get('detail', 'Segmentation failed.')}")
    else:
        st.error("Please select a file to segment.")
