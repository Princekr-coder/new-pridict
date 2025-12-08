import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)
from model_utils2 import compute_rule_columns


def compute_fee_weight(months_unpaid):
    if months_unpaid >= 3:
        return 1.0
    elif months_unpaid == 2:
        return 0.7
    elif months_unpaid == 1:
        return 0.4
    return 0.0


def load_merge(att_path, cgpa_path, fees_path):
    att = pd.read_csv(att_path)
    cgpa = pd.read_csv(cgpa_path)
    fees = pd.read_csv(fees_path)

    fees["fee_weight"] = fees["outstanding_months"].apply(compute_fee_weight)

    df = att.merge(cgpa, on="student_id").merge(fees, on="student_id")

    # avg cgpa
    if "avg_cgpa" not in df.columns:
        df["avg_cgpa"] = df[["cgpa_sem1", "cgpa_sem2", "cgpa_sem3"]].mean(axis=1)

    return df


def create_labels(df):
    df = compute_rule_columns(df)
    df["dropout_label"] = ((df["rule_score"] >= 0.6) | (df["rule_flag"] == 1)).astype(int)
    return df


def visualize(df, model_path="dropout_model_xgb.pkl"):
    features = ["attendance", "avg_cgpa", "fee_weight", "rule_score"]
    X = df[features].fillna(0)
    y = df["dropout_label"]

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.25, random_state=42
    )

    # load trained model
    model = joblib.load(model_path)

    # predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print("\n=== Classification Report (Test Set) ===")
    print(classification_report(y_test, y_test_pred))

    # ------------------------------------
    # GRAPH 1 — Train vs Test Accuracy
    # ------------------------------------
    plt.figure(figsize=(7, 5))
    plt.bar(["Train Accuracy", "Test Accuracy"], [train_acc, test_acc])
    plt.title("Train vs Test Accuracy")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.show()

    # ------------------------------------
    # GRAPH 2 — Confusion Matrix (Test)
    # ------------------------------------
    cm = confusion_matrix(y_test, y_test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - Test Set")
    plt.show()


if __name__ == "__main__":
    df = load_merge("attendance_test.csv", "cgpatest.csv", "fee.csv")
    df = create_labels(df)
    visualize(df)
