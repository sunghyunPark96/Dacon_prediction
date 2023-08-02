import copy
import numpy as np
import pandas as pd
from tqdm import tqdm


def holiday_check(data):
    holidays = []
    holiday_2022 = ["2022-06-01","2022-06-06","2022-08-15"]

    for i in tqdm(range(len(data)),desc="holiday check"):
        if data.iloc[i]["요일"] == "Saturday" or data.iloc[i]["요일"] == "Sunday":
            holiday = True

        else:
            holiday = False

        single_date = str(data.iloc[i]["date"])[:10]
    
        if single_date in holiday_2022:
            holiday = True
            
        else:
            pass

        holidays.append(holiday)

    data["holiday"] = holidays

    return data
    
def preprocess(**kwargs):
    print(" ##start! preprocess ")
    print("\n##preprocess list \n1. combine location & main data \n2. manage NaN(mean or 0) \n3. drop the some column \n4. add some column \n5. change encoding type(utf-8 --> cp949)")

    location_data = pd.read_csv(kwargs["location_data"])
    location_data = location_data.replace("-",0)

    data = pd.read_csv(kwargs["data"])
    data["강수량(mm)"].fillna(0.0,inplace=True)
    data["풍속(m/s)"].fillna(round(data["풍속(m/s)"].mean()),inplace=True)
    data["습도(%)"].fillna(round(data["습도(%)"].mean()),inplace=True)
    
    data["date"] = pd.to_datetime(data["일시"])

    data["월"] = data["date"].dt.month
    data["일"] = data["date"].dt.day
    data["시"] = data["date"].dt.hour
    data["요일"] = data["date"].dt.day_name()

    data = holiday_check(data)

    # data.drop("date",axis=1,inplace=True)
    data.drop("일시",axis=1,inplace=True)

    if "일조(hr)" in list(data.keys()) or "일사(MJ/m2)" in list(data.keys()):
        data.drop("일조(hr)",axis=1,inplace=True)
        data.drop("일사(MJ/m2)",axis=1,inplace=True)

    types_building = []
    areas_building = []
    cooling_areas_building = []
    sun_powers = []

    for i in tqdm(data.values,desc="combine location & main data"):
        building_number = i[1]
        filter_building = location_data[location_data["건물번호"] == building_number]
        type_building, area_building, cooling_area_building,sun_power = filter_building.values[0][1:5]
        types_building.append(type_building)
        areas_building.append(area_building)
        cooling_areas_building.append(cooling_area_building)
        sun_powers.append(sun_power)

    data["건물유형"] = types_building
    data["건물면적"] = areas_building
    data["냉방면적"] = cooling_areas_building
    data["태양광"] = sun_powers


    if "전력소비량(kWh)" in data.keys():
        data = data[['num_date_time','date','월', '일','요일','시',"holiday",'건물유형',"건물번호", '건물면적', '냉방면적', '태양광', '기온(C)', '강수량(mm)', '풍속(m/s)', '습도(%)',"전력소비량(kWh)"
       ]]

    elif "전력소비량(kWh)" not in data.keys():
        data = data[['num_date_time','date','월', '일', '요일','시',"holiday", '건물유형',"건물번호",'건물면적', '냉방면적', '태양광','기온(C)', '강수량(mm)', '풍속(m/s)', '습도(%)']]


    data.to_csv(kwargs["save_dir"],index=False,encoding="cp949")
    print("\ncomplete the preprocess!! ")

if __name__ == "__main__":
    dir1 = "C:/Users/MOBIS/Desktop/딥러닝 공부 자료/딥러닝 사용량 예측/building_info.csv"
    dir2 = "C:/Users/MOBIS/Desktop/딥러닝 공부 자료/딥러닝 사용량 예측/test.csv"
    save_dir = "C:/Users/MOBIS/Desktop/딥러닝 공부 자료/딥러닝 사용량 예측/pre_test_data.csv"

    preprocess(location_data=dir1,data=dir2,save_dir=save_dir)
