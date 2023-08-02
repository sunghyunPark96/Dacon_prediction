import pandas as pd
import numpy as np
import os
import warnings

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from matplotlib import font_manager,rc
from matplotlib import dates
from matplotlib.ticker import MultipleLocator, IndexLocator, FuncFormatter
from matplotlib.dates import MonthLocator, DateFormatter,DayLocator

from pycaret.regression import *

def train_preprocess(data):
    y_data = data["전력소비량(kWh)"]
    
    data.drop("전력소비량(kWh)",axis=1,inplace=True)
    if "시" in data.columns:
        data["시"] = cyclical_transform(data["시"])
    x_one_hot_data = pd.get_dummies(data)
    scaler = MinMaxScaler()
    x_scaled_data = scaler.fit_transform(x_one_hot_data)
    x_scaled_data = pd.DataFrame(x_scaled_data,index=x_one_hot_data.index,columns=x_one_hot_data.columns)
    x_scaled_data["y"] = y_data

    valid = x_one_hot_data.iloc[:500]
    return x_scaled_data

    # x_one_hot_data["y"] = y_data
    # return x_one_hot_data


def cyclical_transform(df):
    hour_in_day = 24
    sin_time = np.sin(2*np.pi*df/hour_in_day)

    return sin_time

def get_path(dir):
    paths = []
    for (directory, _, filenames) in os.walk(dir):
        for filename in filenames:
            if ".csv" in filename:
                file_path = os.path.join(directory, filename)
                paths.append(file_path)
    return paths

if __name__ == "__main__":
    dir = "C:/Users/MOBIS/Desktop/딥러닝 공부 자료/딥러닝 사용량 예측/select_column/"
    data_dir = get_path(dir)
    warnings.filterwarnings('ignore')

    for dir in data_dir:
        data = pd.read_csv(dir,encoding="cp949")
        total_data = train_preprocess(data)
        name = dir.split("/")[7]
        print(f"###########################{name}#########################")
        sup = setup(total_data,target='y',train_size=0.8,use_gpu=True)
        top5 = compare_models(n_select=5,sort="MAPE")
        plot_model(top5[0])
        plot_model(top5[0], plot='error')
