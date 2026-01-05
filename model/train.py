import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from model.preprocess import preprocess_text

# Paths
DATA_CANCELLED = "data/cancelled.csv"
DATA_NONCANCELLED = "data/noncancelled.csv"
ART_DIR = "artifacts"

def load_data():
    # Load datasets
    cancelled = pd.read_csv(DATA_CANCELLED)
    noncancelled = pd.read_csv(DATA_NONCANCELLED)

    # Add labels
    cancelled['Status'] = 'Invalid'
    noncancelled['Status'] = 'Valid'

    # Merge and clean
    df = pd.concat([cancelled, noncancelled], axis=0)
    df = df.drop_duplicates(keep='first')

    # Ensure Summary exists
    if 'Summary' not in df.columns:
        raise ValueError("Dataset must contain a 'Summary' column.")

    df['Summary'] = df['Summary'].astype(str)
    return df

def split_data(df):
    return train_test_split(df, test_size=0.2, random_state=42, stratify=df["Status"])

def train_and_save(train_df, test_df):
    os.makedirs(ART_DIR, exist_ok=True)

    # Preprocess
    train_df["clean"] = train_df["Summary"].apply(preprocess_text)
    test_df["clean"] = test_df["Summary"].apply(preprocess_text)

    # Labels
    le = LabelEncoder()
    y_train = le.fit_transform(train_df["Status"])
    y_test = le.transform(test_df["Status"])

    # Vectorize
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2), min_df=1)
    X_train_tfidf = tfidf.fit_transform(train_df["clean"])
    X_test_tfidf = tfidf.transform(test_df["clean"])

    # Dimensionality reduction
    n_comp = max(1, min(300, X_train_tfidf.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    X_train = svd.fit_transform(X_train_tfidf)
    X_test = svd.transform(X_test_tfidf)

    # Classifier
    base = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    clf = CalibratedClassifierCV(estimator=base, cv=3, method="sigmoid")
    clf.fit(X_train, y_train)

    # Evaluate
    probs = clf.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    print("\nConfusion matrix:\n", confusion_matrix(y_test, preds))
    print("\nReport:\n", classification_report(y_test, preds, target_names=le.classes_))
    try:
        print(f"ROC-AUC: {roc_auc_score(y_test, probs):.4f}")
    except Exception as e:
        print("ROC-AUC could not be computed:", e)

    # Save artifacts
    joblib.dump(tfidf, f"{ART_DIR}/tfidf.pkl")
    joblib.dump(svd, f"{ART_DIR}/svd.pkl")
    joblib.dump(le, f"{ART_DIR}/label_encoder.pkl")
    joblib.dump(clf, f"{ART_DIR}/model.pkl")
    print("\nSaved artifacts to 'artifacts/'.")

if __name__ == "__main__":
    df = load_data()
    train_df, test_df = split_data(df)
    train_and_save(train_df, test_df)