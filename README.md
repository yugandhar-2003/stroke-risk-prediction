❤️ ML-Based Heart Disease Risk Prediction
📌 Project Overview
This project builds a Machine Learning model to predict the likelihood of a person having heart disease based on medical data.
We use the heart.csv dataset from Kaggle, which contains patient health records such as age, cholesterol, blood pressure, chest pain type, and more.

The main goal is to help healthcare professionals identify high-risk patients early, enabling timely diagnosis and treatment.

📊 Dataset
Source: Heart Disease Dataset on Kaggle

Key Features in Dataset:

age – Age of the patient

sex – Gender (1 = male, 0 = female)

cp – Chest pain type (categorical: 0–3)

trestbps – Resting blood pressure (mm Hg)

chol – Serum cholesterol (mg/dl)

fbs – Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)

restecg – Resting ECG results (0–2)

thalach – Maximum heart rate achieved

exang – Exercise-induced angina (1 = yes, 0 = no)

oldpeak – ST depression induced by exercise

slope – Slope of the ST segment (0–2)

ca – Number of major vessels colored by fluoroscopy (0–3)

thal – Thalassemia (categorical: 0–3)

target – Target variable (1 = disease, 0 = no disease)

🛠 Tech Stack
Python

Pandas – Data manipulation

NumPy – Numerical operations

Scikit-learn – ML modeling

Matplotlib & Seaborn – Data visualization

🔄 Workflow
Import & Explore Dataset

Handle Missing Data (if any)

Encode Categorical Variables

Split Data into Train & Test Sets

Apply Preprocessing Pipelines

Train ML Model (Logistic Regression / Random Forest / XGBoost)

Evaluate Model Accuracy

Predict on New Data

🚀 Installation & Usage
bash

# Clone the repository
git clone https://github.com/yugandhar-2003/heart-disease-prediction.git

# Install dependencies
pip install -r requirements.txt

# Run the script
python heart_disease_prediction.py
📈 Model Performance
The model was evaluated using accuracy, precision, recall, and F1-score.
Example accuracy: ~85% (varies depending on train-test split and model choice).


