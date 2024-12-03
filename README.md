# **SegMNet v0.1**  

This project is a Python-based web application for processing and segmenting 3D CT scans. Users can upload CT studies in `.tar.gz` format, which are processed by a segmentation model to generate 3D segmentations. Results can be viewed online and downloaded for further use.  

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
  Secure user registration and login using JWT tokens.  
- **File Upload**:  
  Upload CT studies in `.tar.gz` format.  
- **3D Segmentation**:  
  Integration with a trained segmentation model to generate predictions.  
- **Visualization**:  
  View segmentation results in an interactive viewer (e.g., Gradio or Streamlit).  
- **Download Results**:  
  Download processed segmentations in NIfTI or other formats.  
- **User Settings**:  
  Modify account details, such as email and password.  

---

## **Technologies Used**  

| Component            | Technology/Library           | Purpose                          |  
|-----------------------|------------------------------|----------------------------------|  
| **Backend Framework** | FastAPI                     | Web server and API management    |  
| **Database**          | PostgreSQL + SQLAlchemy     | Data storage and ORM             |  
| **Authentication**    | Authlib + bcrypt            | User authentication and security |  
| **Segmentation Model**| PyTorch/TensorFlow          | 3D segmentation                  |  
| **Visualization**     | Gradio/Streamlit            | Interactive result visualization |  
| **File Handling**     | tarfile + SimpleITK         | File extraction and preprocessing|  
| **Deployment**        | Docker + Nginx              | Containerization and serving     |  

---

## **Project Structure**

```  
project_root/  
│  
├── app/                          # Core application logic  
│   ├── main.py                   # FastAPI entry point  
│   ├── config.py                 # Configuration settings  
│   ├── models/                   # ORM models and segmentation logic  
│   │   ├── user.py  
│   │   ├── segmentation.py  
│   │   └── segmentation_model/   # Segmentation model files  
│   │       ├── model.pth         # Trained model weights  
│   │       ├── preprocess.py     # Data preprocessing logic  
│   │       └── inference.py      # Model inference logic  
│   ├── routers/                  # API routes  
│   ├── schemas/                  # Pydantic schemas  
│   ├── services/                 # Business logic and utilities  
│   └── static/                   # Static files and visualization tools  
│  
├── data/                         # Local data storage  
│   ├── app.db                    # SQLite database file (for development)  
│   └── migrations/               # Alembic migration files  
│  
├── docker/                       # Docker-related configuration  
├── tests/                        # Unit and integration tests  
├── scripts/                      # Helper scripts  
├── requirements.txt              # Python dependencies  
├── README.md                     # Project documentation  
└── .env                          # Environment variables  
```  

---

## **Setup Instructions**  

### Prerequisites  

- Python 3.8 or higher  
- Docker (recommended for deployment)  
- PostgreSQL database (if not using Docker Compose)  

### Installation Steps  

1. **Clone the Repository**:  

   ```bash  
   git clone https://github.com/your-repo-name.git  
   cd your-repo-name  
   ```  

2. **Install Dependencies**:  

   ```bash  
   pip install -r requirements.txt  
   ```  

3. **Set Up Environment Variables**:
   Create a `.env` file based on the `.env.template`:  

   ```  
   DATABASE_URL=postgresql://username:password@localhost/dbname  
   SECRET_KEY=your_secret_key  
   ```  

4. **Run Database Migrations**:  

   ```bash  
   alembic upgrade head  
   ```  

5. **Start the Server**:  

   ```bash  
   uvicorn app.main:app --reload  
   ```  

6. **Access the Application**:  
   Open [http://localhost:8000/docs](http://localhost:8000/docs) to explore the API.  

---

## **Usage Guide**  

### 1. **User Registration and Login**  

- Register with your email and password.  
- Log in to obtain a JWT token for authorization.  

### 2. **Upload CT Study**  

- Upload a `.tar.gz` archive containing the CT scans.  

### 3. **Run Segmentation**  

- The application processes the upload and returns a segmented 3D model.  

### 4. **View and Download Results**  

- Results are visualized in an interactive viewer.  
- Download the segmented file in your preferred format.  

---

## **API Endpoints**  

### Authentication  

| Method | Endpoint          | Description         |  
|--------|--------------------|---------------------|  
| POST   | `/auth/register`   | Register a new user |  
| POST   | `/auth/login`      | Log in to the app   |  

### CT Segmentation  

| Method | Endpoint            | Description            |  
|--------|----------------------|------------------------|  
| POST   | `/segmentation/upload` | Upload CT study        |  
| GET    | `/segmentation/view` | View segmentation result |  
| GET    | `/segmentation/download` | Download result        |  

### User Settings  

| Method | Endpoint          | Description              |  
|--------|--------------------|--------------------------|  
| GET    | `/user/settings`   | View user settings       |  
| PUT    | `/user/settings`   | Update user settings     |  

---

## **Contributing**  

We welcome contributions! Please fork the repository and create a pull request with your changes.  

---

## **License**  

This project is licensed under the [MIT License](LICENSE).  

---

For additional details, templates, or updates, please modify the placeholders in this README after completing the corresponding project sections.
