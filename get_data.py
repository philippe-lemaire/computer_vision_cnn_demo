import os
import pandas as pd
import zipfile


def get_training_data():
    """Download, unzip data and return a dataframe"""

    file_url = "https://www.tuba-lyon.com/wp-content/uploads/digit-recognizer.zip"

    data_folder = "data"
    os.makedirs(data_folder, exist_ok=True)
    os.chdir(data_folder)

    if "digit-recognizer.zip" not in os.listdir():
        os.system(f"wget -q {file_url}")
    else:
        print("Fichier déjà présent")

    with zipfile.ZipFile("digit-recognizer.zip", "r") as zip_obj:
        zip_obj.extract("train.csv")
    X = pd.read_csv("train.csv")
    y = X.label
    X = X.drop(columns="label")
    ## data reshaping to 28 x 28
    X = X.to_numpy().reshape(X.shape[0], 28, 28, 1)
    return X, y
