import numpy as np
import random
from src.objective_function import objective
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.optimize import minimize
from scipy.stats import norm

def random_search(param_distributions, model_name, X_train, y_train, n_iter=50):
    """This function performs random search (for comparison with Bayesian Optimization)."""
    best_params = None
    best_score = float('-inf')

    for i in range(n_iter):
        params = {}
        for param_name, param_dist in param_distributions.items():
            if isinstance(param_dist, list):
                params[param_name] = random.choice(param_dist)
            elif isinstance(param_dist, tuple):
                if len(param_dist) == 3:
                    if isinstance(param_dist[0], int) and isinstance(param_dist[1], int):
                        params[param_name] = random.randrange(*param_dist)
                    else:
                        params[param_name] = random.uniform(param_dist[0], param_dist[1])
                else:
                    raise ValueError("Invalid parameter distribution. Use a list or a 3-tuple.")
            else:
                raise ValueError("Invalid parameter distribution. Use a list or a tuple.")

        score = objective(params, model_name, X_train, y_train)

        if score > best_score:
            best_score = score
            best_params = params.copy()

    return {'best_params': best_params, 'best_score': best_score}

def bayesian_optimization(param_bounds, model_name, X_train, y_train, n_iter=20, init_points=5):
    """ This function performs Bayesian Optimization using a Gaussian Process """
    # Defining Kernel and Gaussian Process Model
    kernel = RBF(length_scale_bounds=(1e-6, 1e6))  
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, random_state=42)

    # Initializing with Random Points
    X_sample = []
    y_sample = []
    for _ in range(init_points):
        
        params = sample_random_params(param_bounds, model_name)

        if 'n_estimators' in params:
            params['n_estimators'] = int(params['n_estimators'])
        if 'max_depth' in params:
            params['max_depth'] = int(params['max_depth'])
        if 'min_samples_split' in params:
            params['min_samples_split'] = int(params['min_samples_split'])
        if 'min_samples_leaf' in params:
            params['min_samples_leaf'] = int(params['min_samples_leaf'])

        score = objective(params, model_name, X_train, y_train)
        X_sample.append(encode_params(params, param_bounds))
        y_sample.append(score)

    # Initializing Bayesian Optimization Loop
    for i in range(n_iter):
        # Fitting the Gaussian Process
        gp.fit(X_sample, y_sample)

        # Finding the next point to sample using Expected Improvement
        best_new_params, best_ei = optimize_acquisition_function(gp, X_sample, y_sample, param_bounds, model_name, X_train, y_train)

        if 'n_estimators' in best_new_params:
            best_new_params['n_estimators'] = int(best_new_params['n_estimators'])
        if 'max_depth' in best_new_params:
            best_new_params['max_depth'] = int(best_new_params['max_depth'])
        if 'min_samples_split' in best_new_params:
            best_new_params['min_samples_split'] = int(best_new_params['min_samples_split'])
        if 'min_samples_leaf' in best_new_params:
            if best_new_params['min_samples_leaf'] <= 1:
                best_new_params['min_samples_leaf'] = round(best_new_params['min_samples_leaf'], 2)
            else:
                best_new_params['min_samples_leaf'] = int(best_new_params['min_samples_leaf'])  

        # Evaluating the objective at the new point
        new_score = objective(best_new_params, model_name, X_train, y_train)

        # Updating the sample data
        X_sample.append(encode_params(best_new_params, param_bounds))
        y_sample.append(new_score)

        print(f"Iteration {i+1}/{n_iter}, Best EI: {best_ei:.4f}, New Score: {new_score:.4f}")

    # At the end returning the Best Parameters
    best_idx = np.argmax(y_sample)
    best_params = decode_params(X_sample[best_idx], param_bounds)

    if 'n_estimators' in best_params:
        best_params['n_estimators'] = int(best_params['n_estimators'])
    if 'max_depth' in best_params:
        best_params['max_depth'] = int(best_params['max_depth'])
    if 'min_samples_split' in best_params:
        best_params['min_samples_split'] = int(best_params['min_samples_split'])
    if 'min_samples_leaf' in best_params:
        best_params['min_samples_leaf'] = int(best_params['min_samples_leaf'])

    return {'best_params': best_params, 'best_score': y_sample[best_idx]}

def sample_random_params(param_bounds, model_name):
    """This function samples random parameters within the specified bounds."""
    params = {}
    for param_name, bounds in param_bounds.items():
        if isinstance(bounds, tuple):
            params[param_name] = random.uniform(bounds[0], bounds[1])
        elif isinstance(bounds, list):
            params[param_name] = random.choice(bounds)

    if model_name == 'LogisticRegression':
        if params.get('penalty') == 'l1':
            params['solver'] = random.choice(['liblinear', 'saga'])
        else:
            params['solver'] = 'lbfgs'
    return params

def encode_params(params, param_bounds):
    """This function encodes hyperparameters for the Gaussian Process."""
    encoded = []
    for param_name, value in params.items():
        if param_name == 'solver':
            encoded.append(param_bounds[param_name].index(value))
        else:
            bounds = param_bounds[param_name]
            if isinstance(bounds, tuple):
                encoded.append((value - bounds[0]) / (bounds[1] - bounds[0]))
            elif isinstance(bounds, list):
                encoded.append(bounds.index(value))
    return np.array(encoded)

def decode_params(encoded_params, param_bounds):
    """This function decodes hyperparameters from the encoded representation."""
    params = {}
    for i, (param_name, bounds) in enumerate(param_bounds.items()):
        if isinstance(bounds, tuple):
            params[param_name] = encoded_params[i] * (bounds[1] - bounds[0]) + bounds[0]
            if param_name == 'max_depth':
                params[param_name] = int(params[param_name])
            if param_name == 'min_samples_leaf':
                if params[param_name] > 1:
                    params[param_name] = int(params[param_name])
                else:
                    params[param_name] = round(params[param_name], 2)
            if param_name == 'min_samples_split':
                params[param_name] = int(params[param_name])
        elif isinstance(bounds, list):
            params[param_name] = bounds[int(encoded_params[i])]
            if param_name == 'min_samples_leaf' and params[param_name] == 1.0:
                params[param_name] = 1 
    return params

def expected_improvement(x, gp, X_sample, y_sample, xi=0.01):
    """This function calculates the Expected Improvement acquisition function."""
    mu, sigma = gp.predict(x, return_std=True)
    mu_sample_opt = np.max(y_sample)
    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
    return ei[0]

def optimize_acquisition_function(gp, X_sample, y_sample, param_bounds, model_name, X_train, y_train):
    """This function finds the parameters that maximize the acquisition function."""
    dim = len(param_bounds)
    best_x = None
    best_ei = float('-inf')

    def neg_expected_improvement(x):
        #returning the negative Expected Improvement for minimization.
        return -expected_improvement(x.reshape(1, dim), gp, X_sample, y_sample)  

    for _ in range(10):
        x0 = np.random.uniform(0, 1, size=dim)
        res = minimize(
            neg_expected_improvement,
            x0,
            bounds=[(0, 1)] * dim,
            method='L-BFGS-B' 
        )
        if -res.fun > best_ei:
            best_ei = -res.fun
            best_x = res.x

    best_params = decode_params(best_x, param_bounds)

    if 'min_samples_leaf' in best_params:
        if best_params['min_samples_leaf'] > 1:
            best_params['min_samples_leaf'] = int(best_params['min_samples_leaf'])
        else:
            best_params['min_samples_leaf'] = round(best_params['min_samples_leaf'], 2)

    return best_params, best_ei