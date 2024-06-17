# Automated Hyperparameter Optimization for Machine Learning Models

This project demonstrates automated hyperparameter optimization (HPO) for Random Forest and Logistic Regression models in Python.

## Running the Project

1. **Prerequisites:**
   - Make sure you have Python installed on your system.
   - Install the required libraries: `pip install -r requirements.txt`

2. **Configuration:**
   - Open the `main.py` file.
   - Set the `MODEL_NAME` variable to either `'RandomForest'` or `'LogisticRegression'`.
   - You can adjust the number of iterations for Random Search and Bayesian Optimization using the `N_ITER_RANDOM` and `N_ITER_BAYESIAN` variables, respectively.
   - The hyperparameter search space for each model is defined in the `PARAM_DISTRIBUTIONS_RANDOM` and `PARAM_BOUNDS_BAYESIAN` dictionaries.

3. **Run the Script:**
   - Execute the main script: `python main.py`

## Output:

- The script will print the best hyperparameters and score found by Random Search, Bayesian Optimization, and Hyperopt for the selected model.
- It will also generate ROC curves to visually compare the performance of the optimized model, a default Random Forest model, and a dummy classifier.
