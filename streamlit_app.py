import streamlit as st
import requests
from pathlib import Path

API_URL = "http://localhost:8000/segmentation/upload/"
TMP_PATH = "./tmp"

st.title("SegMNet: A Kidney Tumor Segmentation Service")

# Форма загрузки файла
uploaded_file = st.file_uploader("Upload your NIFTI file (.nii, .nii.gz)", type=[".nii", ".nii.gz"])
if uploaded_file:
    with st.spinner("Uploading and processing..."):
        try:
            # Загрузка файла в FastAPI
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/octet-stream")}
            response = requests.post(API_URL, files=files)
            response_data = response.json()

            if response.status_code == 200:
                st.success("File uploaded successfully!")
                st.write(f"Segmentation result will be saved to: `{response_data['result_path']}`")
            else:
                st.error(f"Error: {response_data.get('error', 'Unknown error occurred')}")
        except Exception as e:
            st.error(f"Failed to connect to segmentation service: {e}")

st.markdown("---")

# Список доступных обработанных файлов
st.header("Processed Files")
processed_files = list(Path(TMP_PATH).glob("segmented_*"))
if processed_files:
    for file in processed_files:
        st.markdown(f"- [{file.name}](file://{file.resolve()})")
else:
    st.write("No processed files available.")
