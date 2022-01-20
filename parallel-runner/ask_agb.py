from shared import calc_errors, calc_err, calc_err_single, biomass_from_lai_and_lma

import autosklearn.regression
import numpy as np

def get_regressor(time):
    return autosklearn.regression.AutoSklearnRegressor(time_left_for_this_task=time, memory_limit=None)

import time
import pickle

def run(i, tag, max_time, x_train, _y_train, x_test, _y_test):
    name = "ask_agb_cross_{}_{}_{}".format(tag, max_time, i)
    y_train = np.array([ [ biomass_from_lai_and_lma(row[0], row[1]) ] for row in _y_train ])
    y_test = np.array([ biomass_from_lai_and_lma(row[0], row[1]) for row in _y_test ])

    start = time.process_time()
    reg = get_regressor(max_time)
    reg.fit(x_train, y_train)
    taken = time.process_time() - start
    pickle.dump(reg, open('./{}.pickle'.format(name), 'wb'))
    train_err = calc_errors(calc_err_single, y_train, reg.predict(x_train))

    yhat = reg.predict(x_test)
    test_err = calc_errors(calc_err_single, y_test, yhat)

    res = { 'param': max_time,
            'taken': taken,
            'method_specific': {},
            'train_err': train_err,
            'test_err': test_err,
            'x_test': x_test,
            'y_test': y_test,
            'yhat': yhat
    }
    return res

def rerun(i, tag, max_time, x_test, y_test):
    name = "ask_agb_cross_{}_{}_{}".format(tag, max_time, i)

    reg = pickle.load(open('./{}.pickle'.format(name), 'rb'))

    start = time.process_time()
    yhat = reg.predict(x_test)
    taken = time.process_time() - start

    predicted = yhat
    test_err = calc_errors(calc_err_single, y_test, predicted)

    res = { 'param': max_time,
            'taken': taken,
            'method_specific': {},
            'test_err': test_err,
            'x_test': x_test,
            'y_test': y_test,
            'yhat': yhat
    }
    return res
