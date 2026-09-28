import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

def train_and_save_model():
    print("⏳ Loading training dataset 'Mental disorder symptoms (1).xlsx'...")
    
    file_path = 'Mental disorder symptoms (1).xlsx'
    
    try:
        # Load Excel file
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"\n❌ FAILED TO LOAD FILE: {e}")
        print("💡 Tip: Make sure 'openpyxl' is installed by running: pip install openpyxl\n")
        return

    # Drop ID column if present
    if 'ag+1:629e' in df.columns:
        df = df.drop(columns=['ag+1:629e'])
        
    # Extract Features (27 symptoms) and Target Label ('Disorder')
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    print(f"✅ Loaded {X.shape[1]} symptom features for {len(y)} patient records.")

    # Encode Disorder Target Labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Train Random Forest Classifier
    print("🌲 Training Random Forest Classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate Model
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Model Training Complete! Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # Save artifacts
    joblib.dump(clf, 'disorder_model.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')
    print("💾 Saved artifacts successfully: 'disorder_model.pkl' and 'label_encoder.pkl'.\n")

if __name__ == '__main__':
    train_and_save_model()