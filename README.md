# **SegMNet v0.1.0**

**SegMNet** is a Python-based web application designed for processing and segmenting kidney CT scans. Users can upload CT studies in `.nii.gz` format, which are processed by a segmentation model to generate 3D segmentations. While future versions aim to support interactive online visualization, the application currently enables downloading processed results for further use.

---

## **Table of Contents**

1. [Features](#features)  
2. [Technologies Used](#technologies-used)  
3. [Project Structure](#project-structure)  
4. [Setup Instructions](#setup-instructions)  
5. [Usage Guide](#usage-guide)  
6. [API Endpoints](#api-endpoints)  
7. [Contributing](#contributing)  
8. [License](#license)  

---

## **Features**

- **File Upload**:  
  Supports all popular images formats (`.png`, `.jpg`, `.jpeg`, `.webp`), numpy-arrays, torch-tensors and `.nii.gz` files for 3D CT scan uploads.
- **Segmentation**:  
  Leverages a trained segmentation model for efficient predictions. Predict both kidney and tumor, if present.
- **Visualization**:  
  Interactive visualization using Gradio functions.
- **Download Results**:  
  Export segmentation outputs in `.npy` or `.png`.

---

## **Technologies Used**

| Component              | Technology/Library | Purpose                           |  
|------------------------|--------------------|-----------------------------------|  
| **Backend Framework**  | Gradio             | API management and web server     |
| **Segmentation Model** | PyTorch            | 3D segmentation model             |  
| **Visualization**      | Gradio             | Interactive visualization         |
| **Deployment**         | Poetry             | Containerization and serving      |  

---

## **Setup Instructions**

### Prerequisites

- Python 3.11+  
- PyTorch 2.4+
- Poetry
- CUDA-enabled GPU (optional, for faster inference)  

### Installation Steps

1. **Clone the Repository**  
   ```bash  
   git clone https://github.com/lskog7/SegMNet.git  
   cd SegMNet  
   ```  

2. **Create virtual environment and install dependencies**  
   ```bash  
   poetry config virtualenvs.in-project true
   poetry install
   poetry shell
   ```
   
3. **Run the Server**  
   ```bash  
   gradio app.py
   ```  

---

## **Usage Guide**

1. **Upload CT Scans**: Use simple Gradio interface to upload an image.
2. **Process Scans**: The segmentation model processes uploads to generate results.  
3. **Review and Download Results**: Seen the model output and export processed files in desired formats.  

---

## **Contributing**

We welcome contributions! To contribute:  

1. Fork the repository.  
2. Create a feature branch.  
3. Submit a pull request with detailed changes.  

---

## **License**

This project is licensed under the [MIT License](LICENSE).  

---  

### **Note**  
Features and APIs marked as "Planned" are not yet implemented. Updates will be rolled out incrementally.