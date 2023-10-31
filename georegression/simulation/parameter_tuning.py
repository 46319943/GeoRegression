import hyperopt
from georegression.stacking_model import StackingWeightModel
import hyperopt
from hyperopt import hp

space = {
    'n_estimators': hp.choice('n_estimators', range(10, 101)),
    'max_depth': hp.choice('max_depth', range(1, 21)),
    'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2']),
    'criterion': hp.choice('criterion', ['gini', 'entropy'])
}

def objective(params):
    clf = StackingWeightModel(**params)
    return {'loss': -clf.llocv_stacking_, 'status': hyperopt.STATUS_OK}


trials = hyperopt.Trials()
best = hyperopt.fmin(
    fn=objective,
    space=space,
    algo=hyperopt.tpe.suggest,
    max_evals=100,
    trials=trials
)

print("Best Hyperparameters:", best)