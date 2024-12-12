# **SegMNet v0.1.0**

**SegMNet** is a Python-based web application designed for processing and segmenting 3D CT kidney scans. Users can upload CT studies in `.nii.gz` format, which are processed by a segmentation model to generate 3D segmentations. While future versions aim to support interactive online visualization, the application currently enables downloading processed results for further use.

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

- **User Authentication**:  
  Secure user registration and login with JWT-based authentication.
- **File Upload**:  
  Supports `.nii.gz` files for 3D CT scan uploads.
- **3D Segmentation**:  
  Leverages a trained segmentation model for efficient predictions.
- **Visualization**:  
  (Planned) Interactive visualization using tools like Streamlit.
- **Download Results**:  
  Export segmentation outputs in NIfTI or compatible formats.
- **User Settings**:  
  Manage account settings, including email and password updates.

---

## **Technologies Used**

| Component              | Technology/Library  | Purpose                           |  
|------------------------|---------------------|-----------------------------------|  
| **Backend Framework**  | FastAPI             | API management and web server     |  
| **Database**           | SQLite + SQLAlchemy | Data storage and ORM              |  
| **Authentication**     | Authlib + bcrypt    | User authentication and security  |  
| **Segmentation Model** | PyTorch             | 3D segmentation model             |  
| **Visualization**      | Streamlit           | Interactive visualization         |  
| **File Handling**      | nibabel             | File extraction and preprocessing |  
| **Deployment**         | Poetry + Docker     | Containerization and serving      |  

---

## **Project Structure**

```
SegMNet/
│  
├── LICENSE                     # License file  
├── README.md                   # Documentation  
├── alembic.ini                 # Alembic configuration  
├── app/                        # Core application logic  
│   ├── main.py                 # FastAPI entry point  
│   ├── config.py               # Configuration settings  
│   ├── routers/                # API route handlers  
│   ├── models/                 # Data models and segmentation logic  
│   ├── schemas/                # Pydantic schemas  
│   ├── services/               # Business logic utilities  
│   ├── static/                 # Static resources (e.g., visualization)  
│   ├── templates/              # HTML templates (if applicable)  
│   └── utils/                  # Helper utilities  
├── data/                       # Data storage and migrations  
├── docker/                     # Docker configuration  
├── requirements.txt            # Python dependencies  
├── pyproject.toml              # Poetry configuration  
├── streamlit_app.py            # Visualization interface  
├── tests/                      # Unit and integration tests  
└── scripts/                    # Helper scripts  
```

---

## **Setup Instructions**

### Prerequisites

- Python 3.11+  
- PyTorch 2.4+
- Poetry  
- SQLite  
- CUDA-enabled GPU (optional, for faster inference)  

### Installation Steps

1. **Install pipx**  
   Follow the [official pipx documentation](https://pipx.pypa.io/stable/installation/).  

   **MacOS**:  
   ```bash  
   brew install pipx  
   pipx ensurepath  
   ```  

   **Ubuntu**:  
   ```bash  
   sudo apt update  
   sudo apt install pipx  
   pipx ensurepath  
   ```  

2. **Install Poetry**  
   ```bash  
   pipx install poetry  
   ```  

3. **Clone the Repository**  
   ```bash  
   git clone https://github.com/lskog7/SegMNet.git  
   cd SegMNet  
   ```  

4. **Install Dependencies**  
   ```bash  
   poetry install  
   ```  

5. **Set Up Environment Variables**  
   Create a `.env` file based on `.env.template`.  
   ```  
   DB_HOST=1.1.1.1  
   DB_PORT=1111  
   DB_NAME=segmnet_db  
   DB_USER=your_user  
   DB_PASSWORD=your_password  
   ```  

6. **Run the Server**  
   ```bash  
   uvicorn app.main:app --reload --host 0.0.0.0 --port 1234  
   ```  

---

## **Usage Guide**

1. **Authenticate**: Register and log in to the application.  
2. **Upload CT Scans**: Use the `/upload` endpoint to upload `.nii.gz` files.  
3. **Process Scans**: The segmentation model processes uploads to generate results.  
4. **Download Results**: Export processed files in desired formats.  

---

## **API Endpoints**

| Method | Endpoint         | Description               |  
|--------|------------------|---------------------------|  
| POST   | `/auth/register` | Register a new user       |  
| POST   | `/auth/login`    | Log in to obtain a token  |  
| POST   | `/upload`        | Upload CT scans           |  
| GET    | `/result/view`   | (Planned) View results    |  
| GET    | `/result/download` | Download results        |  

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