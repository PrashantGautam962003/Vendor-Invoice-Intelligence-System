from data_preprocessing import load_invoice_data, split_data, scale_features, apply_labels
from modelling_evaluation import evaluate_classifier, train_random_forest
import joblib
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars",
]

TARGET = "flag_invoice"


def main():
    # Load data
    df = load_invoice_data()
    df = apply_labels(df)

    # Prepare data
    x_train, x_test, y_train, y_test = split_data(df, FEATURES, TARGET)
    x_train_scaled, x_test_scaled = scale_features(
        x_train, x_test, "models/scaler.pkl"
    )

    # Train and evaluate models
    grid_search = train_random_forest(x_train_scaled, y_train)

    evaluate_classifier(
        grid_search.best_estimator_,
        x_test_scaled,
        y_test,
        "Random Forest Classifier",
    )

    # Save best model
    joblib.dump(
        grid_search.best_estimator_, "models/predict_flagged_invoice.pkl"
    )


if __name__ == "__main__":
    main()

