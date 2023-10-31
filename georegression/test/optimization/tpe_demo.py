from hyperopt import hp, fmin, tpe, Trials


# 1. Setting up the Custom Search Space
def custom_prior(name):
    def sample_and_clip():
        # 80% chance of choosing the normal distribution
        chosen_distr = hp.choice(name + "_choice", [0, 1])
        if chosen_distr == 0:
            val = hp.normal(name + "_normal", 0.25, 0.05)
            return max(0, min(1, val))  # Clip the value to the [0, 1] range
        else:
            return hp.uniform(name + "_uniform", 0, 1)

    return sample_and_clip()


space = {"param_name": custom_prior("param_name")}


# 2. Defining the Objective Function
def objective(params):
    # Clipping the parameter value to ensure it's in the [0, 1] range
    params["param_name"] = max(0, min(1, params["param_name"]))

    # Hypothetical objective function (replace with your actual function)
    loss = (params["param_name"] - 0.3) ** 2  # Minimized when param_name is 0.3

    return loss


# 3. Optimization
trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials)

print("Best hyperparameter value:", best)
