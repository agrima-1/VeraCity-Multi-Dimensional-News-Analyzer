# VeraCity-Multi-Dimensional-News-Analyzer
NLP-focused academic project
An NLP-powered tool designed to detect news reliability and linguistic bias using SVM classification and linguistic-based heuristics.

## Project Components
- **`Streamlit/`**: Core application files (`main.py`, `VeraCity_engine.py`).
- **`NLP_Prototype_VeraCity_Agrima.ipynb`**: Complete notebook including EDA and model training.
- **`Paper_Agrima_Jain.pdf`**: Final technical project report.
- **`requirements.txt`**: List of dependencies for automated environment setup.

## ⚠️ Mandatory Model Setup (Action Required)
Due to GitHub's file size limits for web uploads, the machine learning model is stored in a compressed format. **The app will not run until these steps are followed:**

1. Navigate to the `Streamlit/` folder.
2. **Unzip/Extract** the file `VeraCity_model - Copy_pickle.zip`.
3. **Rename** the extracted `.pkl` file to exactly: `VeraCity_model.pkl`
4. Ensure `VeraCity_model.pkl` remains inside the `Streamlit/` folder.

## 🚀 How to Run Locally
1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt

2. **Launch the Application:**
   ```bash
   streamlit run Streamlit/main.py
