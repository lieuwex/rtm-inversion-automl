#!/usr/bin/env python3

import sys
import sqlite3
import uuid
from shared import get_x_y
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit

[ tag, model, csv_name, n_splits, test_size, param, split ] = sys.argv[1:]

run_id = ':'.join(sys.argv[1:7])

n_splits = int(n_splits)
test_size = float(test_size)
split = int(split)
param = json.loads(param)

assert split >= 0
assert split < n_splits

con = sqlite3.connect('db.db', timeout=3600)

con.execute('''INSERT OR IGNORE INTO runs(run_id, model, start_time, csv, n_splits, test_size, params, tag) VALUES(?, ?, strftime('%s', 'now'), ?, ?, ?, ?, ?)''', (run_id, model, csv_name, n_splits, test_size, param, tag))
con.commit()

count = con.execute('SELECT COUNT(*) FROM rows WHERE run_id = ? AND param = ? AND iter = ?', (run_id, param, split)).fetchone()[0]
if count == 1:
    print('already done for run {}, skipping'.format(run_id))
    sys.exit(0)

df = pd.read_csv(csv_name, index_col=0)
x_cross, y_cross = get_x_y('data', df)

kf = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=0)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

(train, test) = list(kf.split(x_cross, y_cross))[split]
i = split

if model == 'gp':
    from gp import run
elif model == 'rf':
    from rf import run
elif model == 'ak':
    from ak import run
elif model == 'ask':
    from ask import run
elif model == 'gp_agb':
    from gp_agb import run
elif model == 'rf_agb':
    from rf_agb import run
elif model == 'ak_agb':
    from ak_agb import run
elif model == 'ask_agb':
    from ask_agb import run

row = run(i, tag, param, x_cross[train], y_cross[train], x_cross[test], y_cross[test])
print(row)
con.execute('''INSERT INTO rows(time, run_id, iter, param, res) VALUES(strftime('%s', 'now'), ?, ?, ?, ?)''', (run_id, i, param, json.dumps(row, cls=NpEncoder)))
con.commit()

con.close()
