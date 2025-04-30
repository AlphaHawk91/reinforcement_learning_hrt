import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler

import plotly.io as pio
pio.renderers.default = "png"

import json

from downloader import download_data
from preprocessor import preprocess_data


if __name__ == "__main__" : 

    print("Initializing the script...")

    # Load the configuration file
    with open('./config.json') as json_file:
        config = json.load(json_file)

    print("Configuration loaded.")

    for data_type in ["training", "validation"]:
        if data_type in config:

            # Download the data
            filename = download_data(config, data_type)
            print("Data downloaded.")            

            # Preprocess the data
            cleaned_file_path = preprocess_data(config, data_type=data_type)
            print(f"Data preprocessed and saved to {cleaned_file_path}.")
