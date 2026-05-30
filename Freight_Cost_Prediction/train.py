import joblib
from pathlib import Path

# Importing functions from your project modules
from data_preprocessing import load_vendor_invoice_data, prepare_features, split_data
from Model_Evaluation import (
    train_linear_regression,
    train_decision_tree,
    train_random_forest,
    evaluate_model
)


def main():
    # 1. Locate the script's current folder
    current_dir = Path(__file__).resolve().parent

    # 2. Smart Path Resolution: Look for 'data' locally or one level up
    if (current_dir / "data").exists():
        project_root = current_dir
    elif (current_dir.parent / "data").exists():
        project_root = current_dir.parent
    else:
        # Fallback to current folder if not found anywhere yet
        project_root = current_dir

    # 3. Set absolute paths based on the project root
    db_path = project_root / "data" / "inventory.db"
    model_dir = project_root / "models"
    model_dir.mkdir(exist_ok=True)

    # Safety check: Print exactly where it's looking if it still fails
    if not db_path.exists():
        raise FileNotFoundError(
            f"\n❌ Database file not found!\n"
            f"Looked in: {db_path.resolve()}\n"
            f"Please ensure the 'data' folder and 'inventory.db' share the same workspace."
        )

    print(f"📂 Successfully located database at: {db_path.resolve()}")

    # Load data (converting Path object to string for sqlite3 compatibility)
    df = load_vendor_invoice_data(str(db_path))

    # Prepare data (Using lowercase x to avoid strict linter warnings)
    x, y = prepare_features(df)
    x_train, x_test, y_train, y_test = split_data(x, y)

    # Train models
    lr_model = train_linear_regression(x_train, y_train)
    dt_model = train_decision_tree(x_train, y_train)
    rf_model = train_random_forest(x_train, y_train)

    # Evaluate models using a clean list literal (Removes the multi-step warning)
    results = [
        evaluate_model(lr_model, x_test, y_test, "Linear Regression"),
        evaluate_model(dt_model, x_test, y_test, "Decision Tree Regression"),
        evaluate_model(rf_model, x_test, y_test, "Random Forest Regression")
    ]

    # Select best model (lowest MAE)
    best_model_info = min(results, key=lambda item: item["mae"])
    best_model_name = best_model_info["model_name"]

    # Fixed: Dictionary keys now perfectly match the evaluation strings above
    best_model = {
        "Linear Regression": lr_model,
        "Decision Tree Regression": dt_model,
        "Random Forest Regression": rf_model
    }[best_model_name]

    # Save best model
    model_path = model_dir / "predict_freight_model.pkl"
    joblib.dump(best_model, model_path)

    print(f"🚀 Success! Best model saved: {best_model_name}")
    print(f"📦 Model saved to: {model_path.resolve()}")


if __name__ == "__main__":
    main()