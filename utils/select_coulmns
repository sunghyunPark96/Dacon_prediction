import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import matplotlib

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder

from matplotlib import font_manager,rc
from matplotlib import dates
from matplotlib.ticker import MultipleLocator, IndexLocator, FuncFormatter
from matplotlib.dates import MonthLocator, DateFormatter,DayLocator


if __name__ == "__main__":
    font_path = "C:/Windows/Fonts/HMKMMAG.TTF"
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    matplotlib.rc('font',family=font_name)

    # pd.set_option('display.max_rows', None)
    data_dir = "C:/Users/MOBIS/Desktop/딥러닝 공부 자료/딥러닝 사용량 예측/pre_train_data.csv"
    data = pd.read_csv(data_dir,encoding="cp949")

    # 건물 유형 별 평균 사용 전력량 (6월~8월)
    building_type = list(pd.Series(data.groupby(["건물유형"])["전력소비량(kWh)"].mean()).index)

    heatmap_datas = data.drop(columns=["num_date_time","date","건물번호"],axis=1)
    encoder1 = LabelEncoder()
    encoder2 = LabelEncoder()

    output_label1 = encoder1.fit_transform(data["요일"])
    output_label2 = encoder2.fit_transform(data["holiday"])

    heatmap_datas["요일"] = output_label1
    heatmap_datas["holiday"] = output_label2

    for i,building in enumerate(building_type):
        heatmap_data = heatmap_datas.loc[heatmap_datas["건물유형"] == building]
        heatmap_data = heatmap_data.drop("건물유형",axis=1)

        scaler = MinMaxScaler()
        heatmap_data_scaled = scaler.fit_transform(heatmap_data)
        heatmap_data_scaled = pd.DataFrame(heatmap_data_scaled,index=heatmap_data.index,columns=heatmap_data.columns)

        plt.figure(figsize=(15,15))
        sns.heatmap(data = heatmap_data_scaled.corr(), annot=True, fmt = '.2f', linewidths=.5, cmap='Blues')
        plt.title(f'Heatmap of {building}', fontsize=20)
        plt.show()
        index_ranking = abs(heatmap_data_scaled.corr()["전력소비량(kWh)"]).sort_values(ascending=False).index[:6]
        
        for i,column in enumerate(heatmap_data.columns):
            if column in index_ranking:
                pass
            elif column not in index_ranking:
                heatmap_data.drop(column,axis=1,inplace=True)
                heatmap_data.to_csv(f'./select_column/{building}_select_columns.csv',index=False,encoding="cp949")
