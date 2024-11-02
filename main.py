import logging
from src.data_preprocessing import load_data, preprocess_data
from src.visualization import plot_correlation_heatmap, plot_high_score_distribution
from src.model_training import train_models
from src.model_evaluation import evaluate_models

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_preprocess_data(file_path):
    """Load and preprocess the dataset."""
    try:
        df = load_data(file_path)
        logging.info("Data loading completed successfully.")
        df = preprocess_data(df)
        logging.info("Data preprocessing completed successfully.")
        return df
    except Exception as e:
        logging.error(f"Error in data loading/preprocessing: {e}")
        raise

def visualize_data(df):
    """Generate visualizations for data analysis."""
    try:
        # Uncomment these lines to enable visualizations
        # plot_correlation_heatmap(df)
        # plot_high_score_distribution(df)
        logging.info("Data visualization completed.")
    except Exception as e:
        logging.warning(f"Data visualization skipped or encountered an error: {e}")

def train_and_evaluate_models(df):
    """Train and evaluate both logistic regression and random forest models."""
    try:
        lr_model, rf_model, X_train, X_test, y_train, y_test = train_models(df)
        logging.info("Model training completed successfully.")
        evaluate_models(lr_model, rf_model, X_train, X_test, y_train, y_test)
        logging.info("Model evaluation completed successfully.")
    except Exception as e:
        logging.error(f"Error in model training/evaluation: {e}")
        raise

def main():
    data_file_path = "data/recipes.csv"

    # Step 1: Load and preprocess data
    df = load_and_preprocess_data(data_file_path)

    # Step 2: Visualize data (optional)
    visualize_data(df)

    # Step 3: Train and evaluate models
    train_and_evaluate_models(df)

if __name__ == "__main__":
    main()
