from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def train_random_forest(X_train, y_train, params):
    """This function trains a Random Forest model with given parameters."""
    clf = RandomForestClassifier(**params, random_state=42)
    clf.fit(X_train, y_train)
    return clf

def train_logistic_regression(X_train, y_train, params):
    """This function Trains a Logistic Regression model with given parameters."""
    clf = LogisticRegression(**params, random_state=42, max_iter=2000) 
    clf.fit(X_train, y_train)
    return clf

 