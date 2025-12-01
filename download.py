"""
Data download utility for CSDI datasets.

Downloads and extracts the PhysioNet Challenge 2012 healthcare dataset
and the PM2.5 air quality dataset.
"""

import os
import pickle
import sys
import tarfile
import zipfile

import pandas as pd
import requests
import wget


os.makedirs("data/", exist_ok=True)

if len(sys.argv) > 1:
    if sys.argv[1] == "physio":
        # Download PhysioNet Challenge 2012 dataset
        url = "https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz?download"
        wget.download(url, out="data")
        with tarfile.open("data/set-a.tar.gz", "r:gz") as t:
            t.extractall(path="data/physio")
        print("\nPhysio dataset downloaded successfully!")

    elif sys.argv[1] == "pm25":
        # Download PM2.5 air quality dataset
        url = "https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/STMVL-Release.zip"
        urlData = requests.get(url).content
        filename = "data/STMVL-Release.zip"
        with open(filename, mode="wb") as f:
            f.write(urlData)
        with zipfile.ZipFile(filename) as z:
            z.extractall("data/pm25")

        def create_normalizer_pm25():
            """Create mean and std normalization parameters for PM2.5 dataset."""
            df = pd.read_csv(
                "./data/pm25/Code/STMVL/SampleData/pm25_ground.txt",
                index_col="datetime",
                parse_dates=True,
            )
            test_month = [3, 6, 9, 12]
            for i in test_month:
                df = df[df.index.month != i]
            mean = df.describe().loc["mean"].values
            std = df.describe().loc["std"].values
            path = "./data/pm25/pm25_meanstd.pk"
            with open(path, "wb") as f:
                pickle.dump([mean, std], f)

        create_normalizer_pm25()
        print("\nPM2.5 dataset downloaded successfully!")
    else:
        print(f"Unknown dataset: {sys.argv[1]}")
        print("Usage: python download.py [physio|pm25]")
else:
    print("Usage: python download.py [physio|pm25]")
    print("  physio - Download PhysioNet Challenge 2012 healthcare dataset")
    print("  pm25   - Download PM2.5 air quality dataset")
