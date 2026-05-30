from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV


def train_random_forest(x_train, y_train):
    # Initialize the base Random Forest model configured for multi-core parallelism
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    # Define hyperparameter values to evaluate during tuning grid search
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 4, 5, 6],
        "min_samples_split": [2, 3, 5],
        "min_samples_leaf": [1, 2, 5],
        "criterion": ["gini", "entropy"],
    }

    # Set up F1-score optimization strategy
    scorer = make_scorer(f1_score)

    # Configure grid search parameters with cross-validation engine
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring=scorer,
        cv=5,
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(x_train, y_train)
    return grid_search


def evaluate_classifier(model, x_test, y_test, model_name):
    preds= model.predict(x_test)

    accuracy = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)

    print(f"\n{model_name} Performance")
    print(f"Accuracy: {accuracy:.2f}")
    print(report)

