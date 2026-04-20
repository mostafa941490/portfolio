import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier # موديل جبار لرفع الدقة
import pickle

def train_and_save_model():
    # 1. تحميل البيانات (تأكد من مسار الملف عندك)
    data = pd.read_csv('diabetes.csv')
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']

    # 2. تقسيم البيانات ومعالجتها
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # 3. إعداد الـ Hyperparameters (البحث عن أفضل إعدادات)
    # هنركز على الـ Random Forest كبداية قوية
    rf_params = {
        'n_estimators': [100, 300, 500],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }

    print("🚀 Starting Grid Search to find best parameters...")
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train)
    
    best_rf = grid_search.best_estimator_
    print(f"✅ Best Settings Found! Accuracy: {grid_search.best_score_:.4f}")

    # 4. دمج XGBoost مع أفضل Random Forest لزيادة الثقة (Ensemble)
    xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    
    # الـ VotingClassifier بيخلي الموديلات "تتشاور" قبل القرار النهائي
    final_model = VotingClassifier(
        estimators=[('rf', best_rf), ('xgb', xgb_model)],
        voting='soft' # عشان يدينا نسبة ثقة (Probability)
    )
    
    final_model.fit(X_train_scaled, y_train)

    # 5. حفظ الموديل والـ scaler
    with open('model.pkl', 'wb') as f:
        pickle.dump(final_model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("✨ Model Optimized and Saved Successfully!")

def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

def predict(model, scaler, features):
    # معالجة المدخلات
    features_scaled = scaler.transform([features])
    
    # التوقع ونسبة الثقة
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    confidence = probability[1] * 100 if prediction == 1 else probability[0] * 100
    
    # تفسير بسيط لأهم العوامل (بناءً على ترتيب الداتا)
    explanation = {
        "Glucose Impact": "High" if features[1] > 140 else "Normal",
        "BMI Status": "High" if features[5] > 30 else "Normal",
        "Age Factor": "Risk Group" if features[7] > 45 else "Low Risk"
    }
    
    return prediction, confidence, explanation

if __name__ == "__main__":
    train_and_save_model()
