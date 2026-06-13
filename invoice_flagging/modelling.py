from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, f1_score
import joblib
import os


def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "criterion": ["gini"]
    }

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring="f1_weighted",
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print("\nBest Parameters:", grid_search.best_params_)
    print("Best F1 Score:", round(grid_search.best_score_, 4))

    return grid_search.best_estimator_


def evaluate_classifier(model, X_test, y_test, model_name="Model"):
    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")

    print(f"\n{model_name} Performance")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, preds))


def save_model(model, path="models/random_forest.pkl"):
    full_path = os.path.join(os.path.dirname(__file__), path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    joblib.dump(model, full_path)
    print(f"\nModel saved at {full_path}")