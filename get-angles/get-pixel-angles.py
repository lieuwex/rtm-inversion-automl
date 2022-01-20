import numpy as np
import math
import pandas
from tqdm import tqdm

def read_img(wrs_pathrow, date, obj, band):
    import rasterio
    path = "{0}.{1}/angle_{2}_B{3:02d}.img".format(wrs_pathrow, date, obj, band)
    return rasterio.open(path)

def read_img_row(row, obj, band):
    pathrow = str(int(row['PathRow']))
    date = str(int(row['RS_DateNR']))
    x = row['X_coord']
    y = row['Y_coord']

    img = read_img(pathrow, date, obj, band)
    items = img.sample(xy=[(x, y)], indexes=[1,2])
    return next(items)

df = pandas.read_csv('./file.csv')
df = df[df['SensorType'] == 'ETM+']
info = df[['PathRow', 'RS_DateNR', 'X_coord', 'Y_coord']]

for obj in tqdm([ 'solar', 'sensor' ]):
    for band in tqdm([1,2,3,4,5,6,7,10], leave=False):
        (azimuth, zenith) = ([], [])
        idxs = []

        for (index, row) in tqdm(info.iterrows(), leave=False):
            idxs.append(index)
            try:
                angles = read_img_row(row, obj, band)
                zenith.append(angles[0] / 100)
                azimuth.append(angles[1] / 100)
            except Exception:
                zenith.append(math.inf)
                azimuth.append(math.inf)

        d = {}
        d['B{}_{}_azimuth'.format(band, obj)] = azimuth
        d['B{}_{}_zenith'.format(band, obj)] = zenith
        df = pandas.concat([df, pandas.DataFrame(d, index=idxs)], axis=1)

print(df)
df.to_csv('./output.csv')
