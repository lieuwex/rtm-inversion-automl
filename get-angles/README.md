1. Download, extract here and `make`: https://landsat.usgs.gov/sites/default/files/documents/LANDSAT\_ANGLES\_15\_3\_0.tgz
2. Run `main.py` to download the angle coefficient files from Google Cloud and
   run the the LANDSAT angle tool to convert these coefficient files to angle
   values.
   Which images to download is determined from the rows in `file.csv`.
3. Run `get-pixel-angles.py` to extract the angles from the generated `.img`
   files per row in the `file.csv` (the x and y values are used from this csv
   file), generating an `output.csv` which is identical to `file.csv` augmented
   with the solar and sensor angles for every band.
