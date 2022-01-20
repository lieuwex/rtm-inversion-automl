from shared import calc_errors, calc_err, calc_err_single, biomass_from_lai_and_lma

import autokeras as ak
import numpy as np

def get_regressor(max_trials=5, name="manual"):
    return ak.StructuredDataRegressor(seed=0, max_trials=max_trials, loss="mean_absolute_percentage_error", project_name=name)

import time

def run(i, tag, max_trials, x_train, y_train, x_test, y_test):
    name="ak_cross_test_{}_{}_{}".format(tag, max_trials, i)

    start = time.process_time()
    reg = get_regressor(max_trials, name)
    reg.fit(x=x_train, y=y_train)
    taken = time.process_time() - start
    train_err = calc_errors(calc_err, y_train, reg.predict(x_train))
    epochs_used = reg.tuner._get_best_trial_epochs()

    yhat = reg.predict(x_test)
    test_err = calc_errors(calc_err, y_test, yhat)

    res = { 'param': max_trials,
            'taken': taken,
            'method_specific': {
                'epochs': None,
                'tuner': None,
                'epochs_used': epochs_used,
            },
            'train_err': train_err,
            'test_err': test_err,
            'x_test': x_test,
            'y_test': y_test,
            'yhat': yhat
    }
    return res

def rerun(i, tag, max_trials, x_test, y_test):
    name = "ak_cross_test_{}_{}_{}".format(tag, max_trials, i)

    reg = get_regressor(max_trials, name)

    start = time.process_time()
    yhat = reg.predict(x_test)
    taken = time.process_time() - start

    predicted = np.array([ biomass_from_lai_and_lma(row[0], row[1]) for row in yhat])
    test_err = calc_errors(calc_err_single, y_test, predicted)

    res = { 'param': max_trials,
            'taken': taken,
            'method_specific': {},
            'test_err': test_err,
            'x_test': x_test,
            'y_test': y_test,
            'yhat': yhat
    }
    return res
