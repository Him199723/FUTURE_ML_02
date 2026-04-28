 Customer Churn Prediction Engine
Project Overview
This project builds a production-ready Machine Learning pipeline to predict customer attrition (churn). The goal is to identify high-risk customers using behavioral data, allowing businesses to take proactive retention measures.
Key Features & Engineering Highlights
Imbalanced Data Handling: Implemented SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance, significantly improving minority class recall.
Modular Architecture: Designed with a clear separation of concerns—data preprocessing, feature scaling, and model training are handled in decoupled stages.
Model Ensemble: Evaluated multiple architectures including XGBoost, Random Forest, and Logistic Regression to find the optimal bias-variance tradeoff.
Serialization: Integrated Pickle for model persistence, enabling the model to be loaded instantly for real-time inference in production.
Tech Stack
Core: Python 3.x
Libraries: Pandas, NumPy, Scikit-learn, XGBoost, Imbalanced-learn
Visualization: Matplotlib, Seaborn
Deployment Ready: Serialized models via Pickle
How to Run
Clone the repository.
Install dependencies: pip install -r requirements.txt
Run the training script: python train.py
The serialized model will be saved in the /models directory.
Project Structure
text
├── data/               # Raw and processed datasets
├── models/             # Serialized .pkl files
├── notebooks/          # Exploratory Data Analysis (EDA)
├── src/                # Modular Python scripts (OOP)
└── requirements.txt    # Project dependencies
