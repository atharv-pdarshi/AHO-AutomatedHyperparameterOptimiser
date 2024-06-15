import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

def load_and_preprocess_data(file_path):
    """This function loads data, splits into train/test sets, preprocesses for multi-class"""
    data = pd.read_csv(file_path)
    X = data.drop('target', axis=1)
    y = data['target']

    # Ensuring y is suitable for multi-class even if only 2 classes in data
    if len(set(y)) == 2: 
        y = label_binarize(y, classes=list(set(y))).ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test