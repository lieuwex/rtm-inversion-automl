# Evaluating AutoML methods on hybrid inversion of PROSAIL RTM on Landsat-7 data for AGB estimation

This repository contains files used in my bachelors thesis [Evaluating AutoML
methods on hybrid inversion of PROSAIL RTM on Landsat-7 data for AGB
estimation.](https://liacs.leidenuniv.nl/~s2012820/bsc-thesis/thesis.pdf).

In `files/` are the input data that is required to run the scripts.
In `get-angles/` is the script that gathers the angles from the Google Cloud
Engine.
In `parallel-runner` is the code that was used to train and evaluate the models.
In `notebooks/` are the notebooks that are used to generate the plots

You can download the sqlite3 database with the results
[directly](https://liacs.leidenuniv.nl/~s2012820/bsc-thesis/db.db.zst).
The database is compressed using [Zstandard](https://facebook.github.io/zstd/)
compression, you can decompress the downloaded file using `unzstd bsc-thesis.db.zst`.
This will generate a file `bsc-thesis.db` (**note**: this file is 13GiB).
Also note that decompressing the file will require around 133MiB RAM, and will
take a little while depending on your CPU speed.
Some more information about the database can be found
[here](https://liacs.leidenuniv.nl/~s2012820/bsc-thesis/).
