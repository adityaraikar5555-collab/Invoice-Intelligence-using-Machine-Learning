import joblib
from pathlib import Path

from data_preprocessing import load_vendor_invoice_data, prepare_features, split_data
from modeling import (
    train_linear_regression,
    train_decision_tree,
    train_random_forest,
    evaluate_model
)


def train_all_models(X_train, y_train):
    return {
        "Linear Regression": train_linear_regression(X_train, y_train),
        "Decision Tree Regression": train_decision_tree(X_train, y_train),
        "Random Forest Regression": train_random_forest(X_train, y_train)
    }


def evaluate_all_models(models, X_test, y_test):
    results = []
    for name, model in models.items():
        result = evaluate_model(model, X_test, y_test, name)
        results.append(result)
    return results


def get_best_model(models, results):
    best = min(results, key=lambda x: x["mae"])
    return best["model_name"], models[best["model_name"]]


def save_model(model, path):
    joblib.dump(model, path)


def main():
    db_path = "Data_Project/inventory.db"
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    # Load and prepare data
    df = load_vendor_invoice_data(db_path)
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train models
    models = train_all_models(X_train, y_train)

    # Evaluate models
    results = evaluate_all_models(models, X_test, y_test)

    # Get best model
    best_name, best_model = get_best_model(models, results)

    # Save model
    model_path = model_dir / "predict_freight_model.pkl"
    save_model(best_model, model_path)

    print(f"\nBest Model: {best_name}")
    print(f"Saved at: {model_path}")


if __name__ == "__main__":
    main()