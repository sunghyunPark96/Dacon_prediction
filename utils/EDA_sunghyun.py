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

"""
EDA
"""

if __name__ == "__main__":
    font_path = "C:/Windows/Fonts/HMKMMAG.TTF"
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    matplotlib.rc('font',family=font_name)

    # pd.set_option('display.max_rows', None)
    data_dir = "C:/Users/MOBIS/Desktop/딥러닝 공부 자료/딥러닝 사용량 예측/pre_train_data.csv"
    data = pd.read_csv(data_dir,encoding="cp949")

    # 건물 유형 별 평균 사용 전력량 (6월~8월)
    building_type = list(pd.Series(data.groupby(["건물유형"])["전력소비량(kWh)"].mean()).index)
    value = pd.Series(data.groupby(["건물유형"])["전력소비량(kWh)"].mean())
    plt.figure(figsize=(15,5))
    plt.title("건물유형별 전력사용량 평균(6~8)")
    plt.bar(building_type,value)
    plt.show()

    # 
    value1 = pd.Series(data.groupby(["건물유형","date"])["전력소비량(kWh)"].mean())

    for i,building in enumerate(building_type):
        value2 = value1.loc[building]
        time = pd.to_datetime(pd.Series(value2.index))

        plt.figure(i)
        plt.title(building)
        plt.plot(time,value2.values)
        plt.ylim((0,10000))
        plt.xticks(rotation=90)
        ax = plt.gca()
        ax.xaxis.set_major_locator(DayLocator(interval=5))
        plt.show()

    value1 = pd.Series(data.groupby(["건물유형","요일"])["전력소비량(kWh)"].mean())
    bar_width = 0.06
    fig, ax = plt.subplots(figsize=(12,6))
    color_list = ["black", "silver", "rosybrown", "red", "sienna", "bisque", "gold", "darkkhaki", "slategray", "chartreuse","blue","palevioletred"]

    x = np.arange(bar_width, len(set(data["요일"])) + bar_width, 1)

    for i,building in enumerate(building_type):
        value2 = value1.loc[building]
        value2 = copy.deepcopy(value2[[1,5,6,4,0,2,3]])

        # 그림 사이즈, 바 굵기 조정
        plt.bar(x+bar_width*i,value2,width=bar_width,label=building,color = color_list[i])
        plt.plot(x+bar_width*i, value2, linestyle='--', color = color_list[i]) ## 선 그래프 출력


    # x축의 텍스트를 year 정보와 매칭
    plt.xticks(x,["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

    # x축, y축 이름 및 범례 설정
    plt.xlabel('요일', size = 13)
    plt.ylabel('전력사용량', size = 13)
    plt.legend(labels=building_type,bbox_to_anchor = (1,1))
    plt.show()

    # heatmap_datas = data.drop(columns=["num_date_time","date","월","일","건물번호"],axis=1)
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
