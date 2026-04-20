import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

DATA_PATH = r"C:\Users\jb124\Desktop\diabetes\diabetes.csv"

def load_model():

    # قراءة dataset الحقيقي
    data = pd.read_csv(DATA_PATH)

    # اختيار features المهمة
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # موديل قوي (Ensemble)
    model = VotingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=100)),
            ("gb", GradientBoostingClassifier()),
            ("lr", LogisticRegression(max_iter=1000))
        ],
        voting="soft"
    )

    model.fit(X_scaled, y)

    return model, scaler


def predict(model, scaler, features):

    X = scaler.transform([features])

    prob = model.predict_proba(X)[0][1]
    result = int(prob > 0.5)

    score = prob * 100

    explanation = {
        "Glucose": "High impact" if features[0] > 140 else "Normal",
        "Blood Pressure": "Check" if features[1] > 90 else "Normal",
        "BMI": "Risk" if features[2] > 30 else "Good",
        "Age": "Moderate factor",
        "Insulin": "Metabolic indicator"
    }

    return result, score, explanation