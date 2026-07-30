# Week-11-Team-13: Adverse Drug Event (ADE) Detection & Explainability

## Project Overview
This repository contains the end-to-end development of an AI-driven system for detecting Adverse Drug Events (ADEs) from clinical text. The project covers data preprocessing, model experimentation (including Deep Learning and Transformers), Explainable AI (XAI) analysis, and the deployment of a full-stack web application named **SafetyNet**.

## Key Features
*   **Multi-Model Approach:** Implementation of BiLSTM with Attention, BioBERT fine-tuning, and SVM baselines.
*   **Explainable AI (XAI):** Integration of interpretability modules to visualize model decision-making.
*   **Cross-Domain Analysis:** Experiments to test model robustness across different datasets.
*   **Full-Stack Application:** A web interface for real-time ADE prediction and data management.
*   **Automated Case Retrieval:** Scripts for searching and retrieving relevant clinical cases.

---

## Repository Structure

```text
├── Modules/                # Core Python scripts and utility functions
│   ├── ade_agent.py        # Main agent for ADE processing
│   ├── biobert_finetune.py # Scripts for fine-tuning BioBERT
│   ├── explainability.py   # XAI logic and methods
│   ├── eval_utils.py       # Performance metrics and evaluation helpers
│   └── ... (see folder for full list)
│
├── Notebooks/              # Experimental phase and model development
│   ├── Preprocessing_and_EDA.ipynb  # Data cleaning and exploratory analysis
│   ├── biobert_model.ipynb          # Transformer model experiments
│   ├── bilstm_model.ipynb           # Recurrent Neural Network experiments
│   ├── explainability.ipynb         # XAI visualization experiments
│   └── ... 
│
├── Project Apps/           # Web Application source code
│   ├── frontend/           # UI components
│   ├── backend/            # API and server-side logic
│   ├── app.py              # Main application entry point
│   ├── requirements.txt    # Python dependencies for the app
│   ├── schema.sql          # Database structure
│   └── safetynet.db        # Local SQLite database
│
├── Project Report/         # Documentation
│   └── Group13_Final_Project_Report.pdf
│
├── LICENSE                 # MIT License
└── README.md
```

---

## Getting Started

### Prerequisites
*   Python 3.8+
*   Virtual environment (recommended)

### Installation
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Professional-Team-Project-Group-13/Week-11-Team-13.git
    cd Week-11-Team-13
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r "Project Apps/requirements.txt"
    ```

3.  **Setup the Database:**
    ```bash
    # If using the provided schema to initialize
    sqlite3 safetynet.db < "Project Apps/schema.sql"
    ```

---

## Usage

### Running the Web Application
Navigate to the `Project Apps` directory and run the main application file:
```bash
cd "Project Apps"
python app.py
```
The application will typically be available at `http://127.0.0.1:5000`.

### Model Training & Evaluation
For detailed insights into the model training process, refer to the Jupyter Notebooks in the `/Notebooks` directory. These contain step-by-step documentation of:
*   Data preprocessing and tokenization.
*   Hyperparameter tuning for BioBERT and BiLSTM.
*   Cross-domain evaluation results.

---

## Tech Stack
*   **Models:** BioBERT, BiLSTM + Attention, SVM
*   **NLP Libraries:** Hugging Face Transformers, NLTK, SpaCy
*   **Backend:** Python, Flask/FastAPI (as indicated by `app.py`)
*   **Database:** SQLite / PostgreSQL
*   **XAI:** SHAP/LIME or custom interpretability scripts (via `xai_visuals.py`)

## Contributors
*   **francischisom** (Primary Maintainer)
*   *Team 13 Members*

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
