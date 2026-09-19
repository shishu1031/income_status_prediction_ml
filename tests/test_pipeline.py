import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.features import make_linear_preprocessor


def test_linear_pipeline_handles_unknown_categories():
    X = pd.DataFrame({
        "age": [25, 35, 45, 30, 28, 52, 41, 22],
        "education": ["Bachelors", "Masters", "HS-grad", "PhD", "Bachelors", "HS-grad", "Masters", "HS-grad"],
        "workclass": ["Private", "State-gov", "Private", "Federal-gov", "Private", "Self-emp", "State-gov", "Private"],
    })
    y = [0, 1, 0, 1, 0, 1, 1, 0]
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    pipe = Pipeline([
        ("preprocessor", make_linear_preprocessor(X_train)),
        ("model", LogisticRegression(max_iter=500)),
    ])
    pipe.fit(X_train, y_train)
    X_test = X_test.copy()
    X_test["workclass"] = "Previously unseen category"
    pred = pipe.predict(X_test)
    assert len(pred) == len(X_test)
