import numpy as np
import sklearn.metrics

def biomass_from_height(height):
    #return (14.706*height - 12.094) / 10000
    return -0.0120942 + 0.0147056 * height

def biomass_from_lai_and_lma(lai, lma):
    return lai * lma

def calc_err(fn, y, yhat):
    lai = fn(y[:,0], yhat[:,0])
    cm = fn(y[:,1], yhat[:,1])
    product = fn(y[:,0]*y[:,1], yhat[:,0]*yhat[:,1])

    return {
        'lai': lai,
        'cm': cm,
        'product': product,

        'lai_norm': lai / (np.max(y[:,0]) - np.min(y[:,0])),
        'cm_norm': cm / (np.max(y[:,1]) - np.min(y[:,1])),
        'product_norm': product / (np.max(y[:,0]*y[:,1]) - np.min(y[:,0]*y[:,1]))
    }

def calc_err_single(fn, y, yhat):
    product = fn(y, yhat)

    return {
        'product': product,
        'product_norm': product / (np.max(y) - np.min(y))
    }

def calc_errors(fn, y, yhat):
    mape = fn(sklearn.metrics.mean_absolute_percentage_error, y, yhat)
    rmse = fn(sklearn.metrics.mean_squared_error, y, yhat)
    mae = fn(sklearn.metrics.mean_absolute_error, y, yhat)
    r2 = fn(sklearn.metrics.r2_score, y, yhat)

    return {
        'mape': mape,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
    }

def get_x_y(name, df):
    calculated = df.to_numpy()
    X = calculated[:, :9]
    Y = calculated[:, 10:]

    print(name, X[0], '->', Y[0])

    return (X, Y)

def get_field_x_y(name, df):
    X_bands = df.filter(regex='B[123457]$').to_numpy()
    X_angles = df[['tts', 'tto', 'phi']].to_numpy()
    X = np.column_stack((X_angles, X_bands))

    Y = np.array([ [ biomass_from_height(x[0]) ] for x in df[['h_grass']].to_numpy() ])

    print(name, X[0], '->', Y[0])

    return (X, Y)
