import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier # دي اللي المفروض تكون موجودة
import pickle
import os
import streamlit as st  # السطر ده اللي ناقص يا هندسة

# دالة التدريب
def train_and_save_model():
    data = pd.read_csv('diabetes.csv')
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # الموديل المطور
    rf = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    
    final_model = VotingClassifier(
        estimators=[('rf', rf), ('xgb', xgb)],
        voting='soft'
    )
    final_model.fit(X_train_scaled, y_train)
    
    with open('model.pkl', 'wb') as f:
        pickle.dump(final_model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

# دالة التحميل (تأكد إنها كدة ومافيهاش استيراد من model)
@st.cache_resource
def load_model():
    if not os.path.exists('model.pkl') or not os.path.exists('scaler.pkl'):
        train_and_save_model()
        
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

# دالة التوقع
def predict(model, scaler, features):
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    confidence = probability[1] * 100 if prediction == 1 else probability[0] * 100
    return prediction, confidence, "Explanation placeholder"
