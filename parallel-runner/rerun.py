#!/usr/bin/env python3

import sys
import sqlite3
import uuid
from shared import get_field_x_y
import json

import numpy as np
import pandas as pd

[ tag, model, csv_name, n_splits, test_size, param, split ] = sys.argv[1:]

run_id = ':'.join(sys.argv[1:7])

n_splits = int(n_splits)
test_size = float(test_size)
split = int(split)
param = json.loads(param)

assert split >= 0
assert split < n_splits

con = sqlite3.connect('db.db', timeout=3600)

#row = con.execute('SELECT * FROM runs WHERE run_id = ? LIMIT 1', (run_id)).fetchone()

df = pd.read_csv('./data_with_angles_etm_grouped.csv', index_col=0)
x, y = get_field_x_y('data', df)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

i = split

if model == 'gp':
    from gp import rerun
elif model == 'rf':
    from rf import rerun
elif model == 'ak':
    from ak import rerun
elif model == 'ask':
    from ask import rerun
elif model == 'gp_agb':
    from gp_agb import rerun
elif model == 'rf_agb':
    from rf_agb import rerun
elif model == 'ak_agb':
    from ak_agb import rerun
elif model == 'ask_agb':
    from ask_agb import rerun

row = rerun(i, tag, param, x, y)
con.execute('''INSERT INTO rerun_rows(time, run_id, iter, param, res) VALUES(strftime('%s', 'now'), ?, ?, ?, ?)''', (run_id, i, param, json.dumps(row, cls=NpEncoder)))
con.commit()

con.close()
