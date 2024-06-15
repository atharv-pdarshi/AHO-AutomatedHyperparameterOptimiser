import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score
from sklearn.preprocessing import label_binarize
from itertools import cycle 
import numpy as np 

def evaluate_model(model, X_test, y_test, model_name):
    """This function evaluates a trained model using ROC AUC and accuracy"""
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)

    results = {}

    if y_pred_prob.shape[1] == 2:  # Binary classification
        y_pred_prob = y_pred_prob[:, 1]  
        results['roc_auc'] = roc_auc_score(y_test, y_pred_prob)
        results['accuracy'] = accuracy_score(y_test, y_pred)

    else:
        # Multi-class
        results['roc_auc'] = roc_auc_score(y_test, y_pred_prob, multi_class='ovo')
        results['accuracy'] = accuracy_score(y_test, y_pred)

    return results