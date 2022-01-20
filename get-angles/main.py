import datetime
import os

import ee
import pandas

ee.Authenticate()
ee.Initialize()

fmtstr = '%Y-%m-%d'
def add_day(s):
    d = datetime.datetime.strptime(s, fmtstr)
    d += datetime.timedelta(days=1)
    return d.strftime(fmtstr)

def get_url(wrs_path, wrs_row, picture_id):
    return "https://storage.googleapis.com/gcp-public-data-landsat/LE07/01/{0:03d}/{1:03d}/{2}/{2}_ANG.txt".format(wrs_path, wrs_row, picture_id)

def get_images(wrs_path, wrs_row, date):
    return ee.ImageCollection('LANDSAT/LE07/C01/T1_SR') \
            .filterMetadata('WRS_PATH', 'equals', wrs_path) \
            .filterMetadata('WRS_ROW', 'equals', wrs_row) \
            .filterDate(date, add_day(date))

df = pandas.read_csv('./file.csv')

df = df[df['SensorType'] == 'ETM+']

info = df[['PathRow', 'RS_DateNR']]

colldict = {}

for row in info.values:
    pathrow = str(row[0])
    datestr = str(row[1])

    [p, r] = [int(pathrow[:3]), int(pathrow[3:])]

    d = "{}-{}-{}".format(datestr[:4], datestr[4:6], datestr[6:])

    key = "{}.{}".format(pathrow, datestr)
    if key not in colldict:
        coll = get_images(p, r, d)
        colldict[key] = coll

for key in colldict:
    coll = colldict[key]
    image = coll.first()

    id = image.get('LANDSAT_ID').getInfo()
    path = image.get('WRS_PATH').getInfo()
    row = image.get('WRS_ROW').getInfo()

    url = get_url(path, row, id)

    d = "./{}".format(key)
    os.makedirs(d)
    os.chdir(d)
    os.system(f"""wget -O angles.txt '{url}'""")
    os.system('./LANDSAT_ANGLES_15_3_0/landsat_angles/landsat_angles angles.txt')
    os.chdir('..')
