from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler

# def train_model(df):
#     """Train a model to predict high scores."""
#     X = df.drop(columns=['HighScore', 'Name', 'RecipeId', 'RecipeCategory'])
#     y = df['HighScore']

#     # Handle class imbalance
#     ros = RandomOverSampler()
#     X_resampled, y_resampled = ros.fit_resample(X, y)

#     X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

#     model = RandomForestClassifier(random_state=42)
#     model.fit(X_train, y_train)

#     return model, X_test, y_test
def train_models(df):
    """Train and return both logistic regression and random forest models."""
    X = df.drop(columns=['HighScore', 'Name', 'RecipeId', 'RecipeCategory'])
    y = df['HighScore']

    # Handle class imbalance
    ros = RandomOverSampler()
    X_resampled, y_resampled = ros.fit_resample(X, y)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Logistic Regression model
    lr_model = LogisticRegression(random_state=300)
    lr_model.fit(X_train, y_train)

    # Random Forest model
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)

    return lr_model, rf_model, X_train, X_test, y_train, y_test