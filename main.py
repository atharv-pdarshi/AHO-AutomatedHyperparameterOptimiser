from src.data_preparation import load_and_preprocess_data
from src.hyperparameter_optimization import random_search, bayesian_optimization
from src.model_evaluation import evaluate_model
from src.visualization import compare_models
from src.model_selection import train_random_forest, train_logistic_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials  
import random  # Import random
from src.objective_function import objective  

# Initializing configurations
MODEL_NAME = 'RandomForest' #'LogisticRegression'  #Here, we can switch the model for the project  
N_ITER_RANDOM = 50  
N_ITER_BAYESIAN = 20 

# Defining hyperparameter distributions for random search 
PARAM_DISTRIBUTIONS_RANDOM = { 
    'RandomForest': {
        'n_estimators': (10, 200, 1), 
        'max_depth': [2, 4, 6, 8, 10, 12, None],
        'min_samples_split': (2, 14, 1),
        'min_samples_leaf': (1, 14, 1),
    },
    'LogisticRegression': {
        'C': (1e-5, 1e5, 0.1), 
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']   
    }
}

# Defining hyperparameter bounds for Bayesian Optimization
PARAM_BOUNDS_BAYESIAN = {
    'RandomForest': {
        'n_estimators': (10, 200), 
        'max_depth': (2, 12),  
        'min_samples_split': (2, 14),
        'min_samples_leaf': (1, 14),
    },
    'LogisticRegression': {
        'C': (1e-5, 1e5),
        'penalty': ['l1', 'l2'], 
        'solver': ['liblinear', 'saga', 'lbfgs'] 
    }
}

# Loading the data from the dataset named 'dataset.csv'
X_train, X_test, y_train, y_test = load_and_preprocess_data('data/dataset.csv')

# Code for Random Search (for comparison purpose only)
print("-" * 30)
print("Running Random Search...")
best_result_random = random_search(
    PARAM_DISTRIBUTIONS_RANDOM[MODEL_NAME], MODEL_NAME, X_train, y_train, n_iter=N_ITER_RANDOM
)
best_params_random = best_result_random['best_params']
best_score_random = best_result_random['best_score']
print(f"Best Hyperparameters (Random): {best_params_random}")
print(f"Best Score (Random): {best_score_random:.4f}")

# Code for Bayesian Optimization 
print("-" * 30)
print("Running Bayesian Optimization...")
best_result_bayesian = bayesian_optimization(
    PARAM_BOUNDS_BAYESIAN[MODEL_NAME], MODEL_NAME, X_train, y_train, n_iter=N_ITER_BAYESIAN
)
best_params_bayesian = best_result_bayesian['best_params']
best_score_bayesian = best_result_bayesian['best_score']
print(f"Best Hyperparameters (Bayesian): {best_params_bayesian}")
print(f"Best Score (Bayesian): {best_score_bayesian:.4f}")

# Training and Evaluating the models
# Training models using the best parameters from Bayesian Optimization got in line 70 
if MODEL_NAME == 'RandomForest':
    optimized_model = train_random_forest(X_train, y_train, best_params_bayesian)
elif MODEL_NAME == 'LogisticRegression':
    optimized_model = train_logistic_regression(X_train, y_train, best_params_bayesian)
else:
    raise ValueError(f"Unsupported model: {MODEL_NAME}")

default_model = RandomForestClassifier(random_state=42)
default_model.fit(X_train, y_train)

dummy_model = DummyClassifier(strategy='most_frequent')
dummy_model.fit(X_train, y_train)

# Evaluating the models
optimized_results = evaluate_model(optimized_model, X_test, y_test, f"Optimized {MODEL_NAME}")
default_results = evaluate_model(default_model, X_test, y_test, "Default RandomForest")
dummy_results = evaluate_model(dummy_model, X_test, y_test, "Dummy Classifier (Most Frequent)")

print("-" * 30)
print(f"Optimized {MODEL_NAME} Results:", optimized_results)
print("-" * 30)
print(f"Default RandomForest Results:", default_results)
print("-" * 30)
print(f"Dummy Classifier Results:", dummy_results)


# Hyperopt (for comparison only)
print("-" * 30)
print("Running Hyperopt...")

def objective_hyperopt(params):
    """This function is the Objective function for Hyperopt"""
    if MODEL_NAME == 'LogisticRegression':  
        if params['penalty'] == 'l1':
            params['solver'] = random.choice(['liblinear', 'saga']) 
        else:
            params['solver'] = 'lbfgs'

    
    if 'n_estimators' in params:
        params['n_estimators'] = int(params['n_estimators'])   
    if 'max_depth' in params:
        params['max_depth'] = int(params['max_depth'])
    if 'min_samples_split' in params:
        params['min_samples_split'] = int(params['min_samples_split'])
    if 'min_samples_leaf' in params:
        params['min_samples_leaf'] = int(params['min_samples_leaf'])

    score = objective(params, MODEL_NAME, X_train, y_train)
    return {'loss': -score, 'status': STATUS_OK}  

if MODEL_NAME == 'RandomForest':
    space = {
        'n_estimators': hp.quniform('n_estimators', 10, 200, 1),
        'max_depth': hp.quniform('max_depth', 2, 12, 1),
        'min_samples_split': hp.quniform('min_samples_split', 2, 14, 1),
        'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 14, 1),
    }
elif MODEL_NAME == 'LogisticRegression':
    space = {
        'C': hp.uniform('C', 1e-5, 1e5),
        'penalty': hp.choice('penalty', ['l1', 'l2']),
        'solver': hp.choice('solver', ['liblinear', 'saga', 'lbfgs']),
    } 

trials = Trials()
best_hyperopt = fmin(
    fn=objective_hyperopt,
    space=space,
    algo=tpe.suggest,
    max_evals=N_ITER_BAYESIAN,
    trials=trials
)

best_params_hyperopt = best_hyperopt
best_score_hyperopt = -min(trials.losses())  
print(f"Best Hyperparameters (Hyperopt): {best_params_hyperopt}")
print(f"Best Score (Hyperopt): {best_score_hyperopt:.4f}")

# Visualizing the models for Comparison
compare_models(
    [
        (dummy_model, "Dummy Classifier (Most Frequent)"),
        (default_model, "Default RandomForest"),
        (optimized_model, f"Optimized {MODEL_NAME} (Bayesian)"), 
    ],
    X_test,
    y_test
)
plt.show()