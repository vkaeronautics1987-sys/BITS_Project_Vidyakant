import os
import warnings
import joblib
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef
)

# ============================
# Create model folder
# ============================

os.makedirs("model", exist_ok=True)

# ============================
# Read Dataset
# ============================

print("Loading Dataset...")

df = pd.read_csv(
    "E:\\WILP_BITS_Assignment\\ML\\Assignment-2\\bank-full.csv",
    sep=";"
)

print("Dataset Shape:", df.shape)

# ============================
# Features and Target
# ============================

X = df.drop("y", axis=1)
y = df["y"]

# ============================
# Train Test Split
# ============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ============================
# Categorical Columns
# ============================

cat_cols = X.select_dtypes(include="object").columns

# ============================
# Preprocessing
# ============================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=True),  # Added sparse_output=True
            cat_cols
        )
    ],
    remainder="passthrough"
)

# ============================
# Models
# ============================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(
        probability=True,
        random_state=42
    )
}

results = []

print("\nTraining Started...\n")

# ============================
# Training Loop
# ============================

for name, model in models.items():
    print("=" * 60)
    print(f"Training: {name}")

    if name == "Naive Bayes":
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            (
                "dense",
                FunctionTransformer(
                    lambda x: x.toarray(),
                    accept_sparse=True
                )
            ),
            ("classifier", model)
        ])
    else:
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        pos_label="yes"
    )

    recall = recall_score(
        y_test,
        predictions,
        pos_label="yes"
    )

    f1 = f1_score(
        y_test,
        predictions,
        pos_label="yes"
    )

    auc = roc_auc_score(
        (y_test == "yes").astype(int),
        probabilities[:, 1]
    )

    mcc = matthews_corrcoef(
        y_test,
        predictions
    )

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("AUC :", round(auc, 4))
    print("MCC :", round(mcc, 4))

    results.append([
        name,
        accuracy,
        auc,
        precision,
        recall,
        f1,
        mcc
    ])

    file_name = (
        name.lower()
        .replace(" ", "_")
        + ".pkl"
    )

    joblib.dump(
        pipeline,
        f"model/{file_name}"
    )

# ============================
# Save Results
# ============================

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1",
        "MCC"
    ]
)

results_df.to_csv(
    "model_results.csv",
    index=False
)

# ============================
# Save Test Data
# ============================

test_df = X_test.copy()
test_df["y"] = y_test

test_df.to_csv(
    "test_data.csv",
    index=False
)

print("\n")
print("=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print(results_df)

print("\nFiles Generated:")
print("✔ model_results.csv")
print("✔ test_data.csv")
print("✔ model/logistic_regression.pkl")
print("✔ model/decision_tree.pkl")
print("✔ model/knn.pkl")
print("✔ model/naive_bayes.pkl")
print("✔ model/svm.pkl")
