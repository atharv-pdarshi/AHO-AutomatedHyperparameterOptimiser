from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def objective(params, model_name, X_train, y_train):
    """This function evaluates model performance with given hyperparameters"""
    if model_name == 'RandomForest':
        clf = RandomForestClassifier(**params, random_state=42)
    elif model_name == 'LogisticRegression':
        clf = LogisticRegression(**params, random_state=42, max_iter=2000) 
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if len(set(y_train)) > 2:  
        score = cross_val_score(clf, X_train, y_train, cv=3, scoring='accuracy').mean()
    else:
        score = cross_val_score(clf, X_train, y_train, cv=3, scoring='roc_auc').mean()

    return score