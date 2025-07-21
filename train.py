# train.py — Train a RandomForest model and log with MLflow

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from mlflow.models.signature import infer_signature

# Load the dataset from DVC-tracked file
df = pd.read_csv("data/iris.csv")

# Encode species as numeric values
le = LabelEncoder()
df["species"] = le.fit_transform(df["species"])

# Split into features and labels
X = df.drop("species", axis=1)
y = df["species"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate accuracy
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

# Create example and model signature for MLflow UI
input_example = X_train.iloc[:1]
signature = infer_signature(X_train, model.predict(X_train))

# Start MLflow logging
with mlflow.start_run():
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(
        sk_model=model,
        name="iris_classifier",
        input_example=input_example,
        signature=signature
    )
