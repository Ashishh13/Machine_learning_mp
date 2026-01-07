# End-to-End Machine Learning Credit Prediction System

## 📌 Project Overview

This project demonstrates an end-to-end machine learning workflow focused on predicting credit-related outcomes using structured tabular data. The goal of the project is not only to train a model, but also to package it properly and make it usable through a simple interactive application.

The project follows real-world ML engineering practices including data preprocessing, model training, model persistence, and deployment using a lightweight web interface.

---

## 🧠 Problem Statement

Financial institutions often rely on historical data to assess credit risk and make informed decisions. This project uses a structured dataset to train a machine learning model that can generate predictions based on user-provided inputs.

---

## ⚙️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Streamlit
* Joblib

---

## 🏗️ Project Structure

```
Machine_learning_mp-main/
│
├── app_streamlit.py              # Streamlit application for predictions
├── train_model.py                # Model training script
├── generate_metadata_from_csv.py # Metadata generation from dataset
├── requirements.txt              # Project dependencies
│
├── data/
│   └── credit.csv                # Dataset used for training
│
├── models/
│   ├── model.pkl                 # Trained ML model
│   └── metadata.json             # Encoders and feature metadata
│
├── backend/
│   └── app.py                    # Backend inference logic
│
├── frontend/                     # Frontend-related files (if any)
└── README.md                     # Project documentation
```

---

## 🔄 Workflow

1. Load and explore the dataset
2. Perform preprocessing and encoding
3. Train the machine learning model
4. Save the trained model using Joblib
5. Generate metadata for consistent inference
6. Build a Streamlit app for user interaction

---

## ▶️ How to Run the Project

### Step 1: Clone the Repository

```bash
git clone <your-github-repo-link>
cd Machine_learning_mp-main
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Train the Model (Optional)

```bash
python train_model.py
```

### Step 4: Run the Streamlit App

```bash
streamlit run app_streamlit.py
```

---

## 📊 Output

* Interactive Streamlit web interface
* Real-time predictions based on user input

---

## 📈 Key Learnings

* Understanding complete ML pipelines beyond notebooks
* Handling preprocessing consistency using metadata
* Model persistence and reuse
* Basic ML application deployment using Streamlit

---

## 🚀 Future Improvements

* Hyperparameter tuning
* Model performance evaluation metrics
* UI enhancements
* Deployment on cloud platforms

---

## 👤 Author

**Ashish Singh**

---

## 📜 License

This project is for educational purposes.

