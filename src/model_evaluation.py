from sklearn.metrics import confusion_matrix, classification_report

# def evaluate_model(model, X_test, y_test):
#     """Evaluate the trained model."""
#     y_pred = model.predict(X_test)
#     print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
#     print("\nClassification Report:\n", classification_report(y_test, y_pred))


# def evaluate_model(model, X_test, y_test):
#     """Evaluate the trained model."""
#     model_name = type(model).__name__  # Get the model's name
#     y_pred = model.predict(X_test)
    
#     # Display evaluation results
#     print(f"\n=== Evaluation of {model_name} ===")
#     print("\n--- Confusion Matrix ---\n")
#     print(confusion_matrix(y_test, y_pred))
#     print("\n--- Classification Report ---\n")
#     print(classification_report(y_test, y_pred))
#     print("=" * 30 + "\n")
def evaluate_models(lr_model, rf_model, X_train, X_test, y_train, y_test):
    """Evaluate both logistic regression and random forest models."""
    
    # Evaluation for Logistic Regression
    print("\n=== Logistic Regression Evaluation ===")
    print(f"Training Accuracy: {lr_model.score(X_train, y_train):.2f}")
    print(f"Test Accuracy: {lr_model.score(X_test, y_test):.2f}")

    predictions_lr = lr_model.predict(X_test)
    conf_matrix_lr = confusion_matrix(y_test, predictions_lr)
    print("\nConfusion Matrix:\n", conf_matrix_lr)
    
    # True Negatives, False Positives, False Negatives, True Positives
    TN, FP, FN, TP = conf_matrix_lr.ravel()
    
    # Calculate accuracy for low-scoring recipes
    accuracy_low_scores = TN / (TN + FN) if (TN + FN) != 0 else 0
    print(f"Accuracy on all low scoring recipes: {accuracy_low_scores:.2%}")
    
    print("\n--- Classification Report ---\n")
    print(classification_report(y_test, predictions_lr))

    print("=" * 50)

    # Evaluation for Random Forest
    print("\n=== Random Forest Evaluation ===")
    print(f"Training Accuracy: {rf_model.score(X_train, y_train):.2f}")
    print(f"Test Accuracy: {rf_model.score(X_test, y_test):.2f}")

    predictions_rf = rf_model.predict(X_test)
    conf_matrix_rf = confusion_matrix(y_test, predictions_rf)
    print("\nConfusion Matrix:\n", conf_matrix_rf)
    print("\n--- Classification Report ---\n")
    print(classification_report(y_test, predictions_rf))
    
    print("=" * 50)