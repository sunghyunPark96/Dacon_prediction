import pandas as pd
import numpy as np
from tqdm import tqdm

def preprocess(deep_type,**kwargs):
    print(" ##start! preprocess ")
    print("\n##preprocess list \n1. add location information \n2. manage NaN(mean) \n3. drop the some column \n4. change encoding type(cp949)\n")

    location_data = pd.read_csv(kwargs["location_data"])
    location_data = location_data.replace("-",0)

    data = pd.read_csv(kwargs["data"])
    data["강수량(mm)"].fillna(0.0,inplace=True)
    data["풍속(m/s)"].fillna(round(data["풍속(m/s)"].mean()),inplace=True)
    data["습도(%)"].fillna(round(data["습도(%)"].mean()),inplace=True)

    if deep_type == "yt":
        data["월일"] = data["일시"].apply(lambda x : float(x[4:8]))
        data["시각"] = data["일시"].apply(lambda x : float(x[9:]))

    elif deep_type == "nt":
        data["시각"] = data["일시"].apply(lambda x : float(x[9:]))

    if "일조(hr)" in list(data.keys()) or "일사(MJ/m2)" in list(data.keys()):
        data.drop("일조(hr)",axis=1,inplace=True)
        data.drop("일사(MJ/m2)",axis=1,inplace=True)

    types_building = []
    areas_building = []
    cooling_areas_building = []
    sun_powers = []

    for i in tqdm(data.values):
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

    data.drop(["건물번호","일시"],axis=1,inplace=True)

    data.to_csv(kwargs["save_dir"],index=False,encoding="cp949")
    print("\ncomplete the preprocess!! ")

if __name__ == "__main__":
    deep_type = str(input(" nt or yt (yt is timeseries) "))
    dir1 = "C:/Users/MOBIS/Desktop/딥러닝 공부 자료/딥러닝 사용량 예측/building_info.csv"
    dir2 = "C:/Users/MOBIS/Desktop/딥러닝 공부 자료/딥러닝 사용량 예측/train.csv"
    save_dir = "C:/Users/MOBIS/Desktop/딥러닝 공부 자료/딥러닝 사용량 예측/pre_train_data.csv"

    preprocess(deep_type=deep_type,location_data=dir1,data=dir2,save_dir=save_dir)
