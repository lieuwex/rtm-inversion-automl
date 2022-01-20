from shared import calc_errors, calc_err, calc_err_single, biomass_from_lai_and_lma

import numpy as np

from sklearn.ensemble import RandomForestRegressor
import skopt
from skopt.space.space import Integer

def make_cls(params, verbose=0):
    cls = RandomForestRegressor(n_estimators=100, max_depth=params[0], min_samples_split=params[1], min_samples_leaf=params[2], random_state=0, verbose=verbose)
    #cls = sklearn.ensemble.RandomForestRegressor(n_estimators=params[0], verbose=verbose, n_jobs=-1)
    return cls

dimensions = [
    Integer(low=5, high=50, name='max_depth'),
    Integer(low=2, high=11, name='min_samples_split'),
    Integer(low=1, high=11, name='min_samples_leaf'),
]

def create_and_fit(max_time, X, Y):
    start = time.process_time()
    space = skopt.Space(dimensions)

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.25, random_state=0)

    def search_fn(params):
        cls = make_cls(params)
        cls.fit(x_train, y_train)
        return calc_errors(calc_err_single, y_test, cls.predict(x_test))['rmse']['product'] * 1e10

    best = { 'x': None, 'y': np.inf, 'iterations': 0 }

    while (time.process_time() - start) < max_time:
        point = space.rvs(n_samples=1, random_state=best['iterations'])[0]
        score = search_fn(point)
        if score < best['y']:
            best['x'] = point
            best['y'] = score
        best['iterations'] += 1


    gpr = make_cls(best['x'], verbose=3)
    gpr.fit(x_train, y_train)
    return [gpr, best]

import time
import pickle

def run(i, tag, max_time, x_train, _y_train, x_test, _y_test):
    name = "rf_agb_cross_{}_{}_{}".format(tag, max_time, i)
    y_train = np.array([ [ biomass_from_lai_and_lma(row[0], row[1]) ] for row in _y_train ])
    y_test = np.array([ biomass_from_lai_and_lma(row[0], row[1]) for row in _y_test ])

    start = time.process_time()
    [reg, res] = create_and_fit(max_time, x_train, y_train)
    taken = time.process_time() - start
    pickle.dump(reg, open('./{}.pickle'.format(name), 'wb'))
    train_err = calc_errors(calc_err_single, y_train, reg.predict(x_train))

    yhat = reg.predict(x_test)
    test_err = calc_errors(calc_err_single, y_test, yhat)

    res = { 'param': max_time,
            'taken': taken,
            'method_specific': {
                'x': res['x'],
                'feature_importances': reg.feature_importances_,
                'iterations': res['iterations']
            },
            'train_err': train_err,
            'test_err': test_err,
            'x_test': x_test,
            'y_test': y_test,
            'yhat': yhat
    }
    return res

def rerun(i, tag, max_time, x_test, y_test):
    name = "rf_agb_cross_{}_{}_{}".format(tag, max_time, i)

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
