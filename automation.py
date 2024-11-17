import pandas as pd
import cryptpandas as crp
import numpy as np
from scipy.optimize import minimize 
import matplotlib.pyplot as plt
import json
import time

# timestamp,user_id,filename,password

passwords = pd.read_csv('passwords.csv')

def decrypt_latest_n(n):
    res = []
    for filename, password in zip(passwords.iloc[-n:]['filename'], passwords.iloc[-n:]['password']):
        res.append(crp.read_encrypted(path = f"data_files/{filename}", password = password))
        # print(f"Decrypted {filename}")
    return res

def process_df(df_list, decay_factor):
    res = None
    for i, df in enumerate(reversed(df_list)):
        df = df.apply(lambda x: x * np.exp(-(i + 1) * decay_factor))
        if res is None:
            res = df
        else:
            res = pd.concat([df, res])
        # print(f"Processed df {i}")
    return res

def clean_df(df, standard_deviations = 3):
    count_nan = 0
    count_outlier = 0

    def replace_outlier(x, mu, sigma):
        nonlocal count_nan, count_outlier
        if x < mu - 3*sigma or x > mu + 3*sigma:
            count_outlier += 1
            x = np.random.normal(mu, sigma)
        elif np.isnan(x):
            count_nan += 1
            x = np.random.normal(mu, sigma)
        return x

    for column in df.columns:
        df_slice = df[column].replace([-np.inf, np.inf], np.nan).dropna()
        mu = df_slice.mean()
        sigma = df_slice.std()
        df[column] = df[column].apply(replace_outlier, args = (mu, sigma))

    x, y = df.shape
    # print(f"replace {count_outlier} outliers and {count_nan} NaNs out of {x * y} values, {(count_nan + count_outlier) / (x * y) * 100:.2f}% loss of data")
    return df

def full_clean_df(n = 5, decay_factor = 0.5):
    df_list = decrypt_latest_n(n)
    for df in df_list:
        df = clean_df(df)
    return process_df(df_list, decay_factor)
    

# Optimise, basically our slowest function so maybe can optimise 
# with generators or cut the optimisation setup when it 
# reaches an acceptable threshold

def maximize_sharpe_ratio(cleaned_df):
    cov_matrix = cleaned_df.cov()
    avg_returns = [cleaned_df[col].mean() for col in cleaned_df.columns]

    # print(f"cov_matrix has shape {cov_matrix.shape}")
    # print(f"avg_returns has shape {len(avg_returns)}")

    num_assets = len(avg_returns) 

    # Define the negative Sharpe ratio function
    def negative_sharpe(weights):
        # Ensure weights is a numpy array
        weights = np.array(weights)

        # Portfolio return
        portfolio_return = np.dot(weights, avg_returns)

        # Portfolio volatility
        portfolio_volatility = np.sqrt(np.dot(np.transpose(weights), np.dot(cov_matrix, weights))) + 0.0001
 
        # Sharpe Ratio
        sharpe_ratio = portfolio_return / portfolio_volatility # hits 0.0 at some point
        return -sharpe_ratio  # Negate because we are minimizing
    
    def optimizer_callback(weights):
        global best_weights, best_obj_value
        obj_value = negative_sharpe(weights)
        if obj_value < best_obj_value:
            best_obj_value = obj_value
            best_weights = weights.copy()

    # def smooth_abs_approx(x, delta = 0.001):
    #     return np.sqrt(x**2 + delta)

    # def constraint_func(x):
    #     f = np.vectorize(smooth_abs_approx)
    #     return sum(f(x)) - 1
    
    # def constraint_func_alt(x):
    #     return sum(np.abs(x)) - 1
    
    # def constraint_jacobian(x, delta = 0.0001):
    #     return x / smooth_abs_approx(x, delta) # done on paper needed because better than numerical approximation + sqrt derivative is funky

    # Constraints: Weights sum to 1
    constraints = [{'type': 'eq', 'fun': lambda x: sum(np.abs(x)) - 1}] # non differentiable at x = 0 so cant use SLSQP method

    # constraints = [
    #     {'type': 'ineq', 'fun': lambda w: np.sum(np.abs(w)) - 1},
    #     {'type': 'ineq', 'fun': lambda w: -np.sum(np.abs(w)) + 1}
    # ]

    bounds = [(-0.1, 0.1) for _ in range(num_assets)] 

    initial_weights = np.random.uniform(-0.1, 0.1, num_assets)
    # initial_weights = np.array([1] + [0] * (num_assets - 1))

    # Minimize the negative Sharpe ratio
    result = minimize(negative_sharpe, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints, callback = optimizer_callback, options = {'maxiter' : 1000, 'disp' : True})

    if not result.success:
        # return best_weights
        raise ValueError("Optimization failed:", result.message)

    
    return result.x  # Optimized weights

def get_positions(pos_dict):
    pos = pd.Series(pos_dict)
    pos = pos.replace([np.inf, -np.inf], np.nan)
    pos = pos.dropna()
    pos = pos / pos.abs().sum()
    pos = pos.clip(-0.1,0.1)
    if pos.abs().max() / pos.abs().sum() > 0.1:
        raise ValueError(f"Portfolio too concentrated {pos.abs().max()=} / {pos.abs().sum()=}")
    return pos

def get_submission_dict(
    pos_dict,
    your_team_name: str = "The Big O Squad",
    your_team_passcode: str = "We have Sharpe minds",
):
    
    return {
        **get_positions(pos_dict).to_dict(),
        **{
            "team_name": your_team_name,
            "passcode": your_team_passcode,
        },
    }

decay_factor = 0.5
best_weights, best_obj_value = None, float('inf') # callback code but tbh we dgaf

# fuck it we just force until it works

while True:
    try:
        cleaned_df = full_clean_df(n = 5, decay_factor = decay_factor)
        portfolio_weights = maximize_sharpe_ratio(cleaned_df)
        break
    except ValueError as e:
        print(f"Optimization failed with error: {e}. Retrying in 1 second... ")
        time.sleep(1)

strategy_weights = {strat: weight for strat, weight in zip(cleaned_df.columns, portfolio_weights)}
submission_weights = get_submission_dict(strategy_weights)

data = get_submission_dict(strategy_weights)
with open('submission.json', 'w') as f:
    json.dump(data, f)