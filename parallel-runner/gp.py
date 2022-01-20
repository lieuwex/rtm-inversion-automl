from shared import calc_errors, calc_err, calc_err_single, biomass_from_lai_and_lma

import numpy as np

def get_regressor(kernel_name, fixed, n_restarts=10, alpha=0.05, scale_x=True, normalize_y=True):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import DotProduct, Matern, RBF, RationalQuadratic

    k = None
    if not fixed:
        if kernel_name == "RBF":
            k = RBF()
        elif kernel_name == "RationalQuadratic":
            k = RationalQuadratic()
        elif kernel_name == "Matern":
            k = Matern()
        elif kernel_name == "DotProduct":
            k = DotProduct()
    else:
        if kernel_name == "RBF":
            k = RBF(length_scale_bounds='fixed')
        elif kernel_name == "RationalQuadratic":
            k = RationalQuadratic(length_scale_bounds='fixed', alpha_bounds='fixed')
        elif kernel_name == "Matern":
            k = Matern(length_scale_bounds='fixed')
        elif kernel_name == "DotProduct":
            k = DotProduct(sigma_0_bounds='fixed')

    gpr = GaussianProcessRegressor(kernel=k, random_state=0, normalize_y=normalize_y, n_restarts_optimizer=n_restarts, alpha=alpha)

    if scale_x:
        return make_pipeline(MinMaxScaler(), gpr)
    else:
        return make_pipeline(gpr)

import skopt
from skopt.space.space import Integer, Real, Categorical
import time

dimensions = [
    Categorical(["RBF", "RationalQuadratic", "Matern", "DotProduct"], name='kernel_name'),
    #Categorical([False, True], name='fixed'),
    Categorical([True], name='fixed'),
    Integer(low=10, high=100, name='n_restarts'),
    Real(low=1e-10, high=1, name='alpha'),
    Categorical([False, True], name='scale_x'),
    Categorical([False, True], name='normalize_y')
]

def create_and_fit(max_time, X, Y):
    start = time.process_time()
    space = skopt.Space(dimensions)

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.25, random_state=0)

    def search_fn(params):
        print("using", params)
        gpr = get_regressor(*params)
        try:
            gpr.fit(x_train, y_train)
            return calc_errors(calc_err, y_test, gpr.predict(x_test))['rmse']['product'] * 1e10
        except:
            import sys
            print("Unexpected error:", sys.exc_info()[0])
            return np.inf

    best = { 'x': None, 'y': np.inf, 'iterations': 0 }

    while (time.process_time() - start) < max_time:
        point = space.rvs(n_samples=1, random_state=best['iterations'])[0]
        score = search_fn(point)
        if score < best['y']:
            best['x'] = point
            best['y'] = score
        best['iterations'] += 1

        print("got", score, "best is now", best['y'])

    gpr = get_regressor(*best['x'])
    gpr.fit(x_train, y_train)
    return [gpr, best]

import time
import pickle

def run(i, tag, max_time, x_train, y_train, x_test, y_test):
    name = "gp_cross_{}_{}_{}".format(tag, max_time, i)
    start = time.process_time()
    [reg, res] = create_and_fit(max_time, x_train, y_train)
    taken = time.process_time() - start
    pickle.dump(reg, open('./{}.pickle'.format(name), 'wb'))
    train_err = calc_errors(calc_err, y_train, reg.predict(x_train))

    yhat = reg.predict(x_test)
    test_err = calc_errors(calc_err, y_test, yhat)

    res = { 'param': max_time,
            'taken': taken,
            'method_specific': {
                'x': res['x'],
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
    name = "gp_cross_{}_{}_{}".format(tag, max_time, i)

    reg = pickle.load(open('./{}.pickle'.format(name), 'rb'))

    start = time.process_time()
    yhat = reg.predict(x_test)
    taken = time.process_time() - start

    predicted = np.array([ biomass_from_lai_and_lma(row[0], row[1]) for row in yhat])
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
