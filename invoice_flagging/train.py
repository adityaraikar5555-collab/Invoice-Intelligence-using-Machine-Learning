from data_processing import preprocess_pipeline   # <-- FIXED
from modelling import train_random_forest, evaluate_classifier, save_model


FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "days_po_to_invoice",
    "days_to_pay",
    "total_brands",
    "total_item_quantity",
    "total_item_dollars",
    "avg_receiving_delay"
]

TARGET = "flag_invoice"


def main():
    df, X_train, X_test, y_train, y_test = preprocess_pipeline(FEATURES, TARGET)

    model = train_random_forest(X_train, y_train)

    evaluate_classifier(model, X_test, y_test, "Random Forest Classifier")

    save_model(model, "models/predict_flag_invoice.pkl")


if __name__ == "__main__":
    main()