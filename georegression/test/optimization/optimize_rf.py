import hyperopt
from hyperopt import fmin, tpe, hp, Trials
import numpy as np


# Define the search space
def custom_prior(name):
    return hp.choice(
        name,
        [hp.normal(name + "_normal", 0.25, 0.05), hp.uniform(name + "_uniform", 0, 1)],
    )


space = {"param_name": custom_prior("param_name")}


# Define the objective function
def objective(params):

    if params["param_name"] < 0.25:
        return {"status": hyperopt.STATUS_FAIL}

    # Ensure the parameter is within [0, 1]
    params["param_name"] = max(0, min(1, params["param_name"]))

    # Fictional objective function: (x-0.25)^2, the minimum is at x=0.25
    loss = (params["param_name"] - 0.25) ** 2

    return {"loss": loss, "status": hyperopt.STATUS_OK}


# Optimization
trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=1000, trials=trials)

print(f"Best hyperparameter value: {best['param_name']}")
