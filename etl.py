#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def get_angle(vec1, vec2):
    unit_vec1 = unit_vector(vec1)
    unit_vec2 = unit_vector(vec2)
    return np.arccos(np.clip(np.dot(unit_vec1, unit_vec2), -1.0, 1.0))


class ETL:

    def __init__(self, data_path):
        self.DATA_PATH = data_path
        self.cima = {}
        self.angles = {
            "V1": ["upper_chest", "nose", "right_wrist"],
            "V2": ["upper_chest", "nose", "left_wrist"],
            "V3": ["upper_chest", "hip_center", "right_wrist"],
            "V4": ["upper_chest", "hip_center", "left_wrist"],
            "V5": ["hip_center", "upper_chest", "right_ankle"],
            "V6": ["hip_center", "upper_chest", "left_ankle"],
        }

    def get_cima(self):
        return self.cima

    def load_metadata(self, dataset):
        meta_path = os.path.join(self.DATA_PATH, dataset)
        if os.path.exists(os.path.join(meta_path, "metadata.xls")):
            path = os.path.join(meta_path, "metadata.xls")
            metadata = pd.read_excel(path)
            metadata = metadata.drop(["CP score", "Unnamed: 5", "Unnamed: 6", "Unnamed: 7", "Unnamed: 8"], axis=1)
            metadata.rename(columns={"ID nummer": "id", "CP score": "score", "CP utkom": "cp"}, inplace=True)

            # Rename video ids to remove suffix
            # Either III or AAA_III where A is alphabetic and I is integers
            metadata["id"] = metadata["id"].apply(lambda name: name[:3] if name[0].isnumeric() else name[:7])
            self.metadata = metadata
        else:
            path = os.path.join(meta_path, "metadata.csv")
            self.metadata = pd.read_csv(path)


    def load(self, dataset, tiny=False):
        cima_files = []
        cima_path = os.path.join(self.DATA_PATH, dataset)

        self.load_metadata(dataset)

        cima_path = os.path.join(cima_path, "data") if os.path.exists(os.path.join(cima_path, "data")) else cima_path

        for root, dirs, files in os.walk(cima_path):
            for filename in files:
                if filename[-4:] == ".csv":
                    cima_files.append(os.path.join(root, filename))

        if tiny:
            cima_files = cima_files[:5]

        print("\n\n----------------")
        print(" Loading CIMA ")
        print("----------------\n")

        for file in tqdm(cima_files):
            file_name = file.split(os.sep)[-1].split(".")[0]
            file_id = file_name[:3] if file_name[0].isnumeric() else file_name[:7]
            meta_row = self.metadata.loc[self.metadata["id"] == file_id]
            if meta_row.empty:
                continue
            data = pd.read_csv(file)
            data = data.drop(columns=["Unnamed: 0"], errors="ignore")
            self.cima[file_id] = {"data": data, "label": meta_row.iloc[0]["cp"]}


    def create_angles(self):
        cima_angles = {}
        print("\n\n----------------")
        print(" Creating angles ")
        print("----------------\n")
        for key, item in tqdm(self.cima.items()):
            data = item["data"]
            angles = {key: [] for key in self.angles.keys()}
            for row in data.iterrows():
                index = row[0]
                row_data = row[1]
                for angle_key, points in self.angles.items():
                    p0 = [row_data[points[0] + "_x"], row_data[points[0] + "_y"]]
                    p1 = [row_data[points[1] + "_x"], row_data[points[1] + "_y"]]
                    p2 = [row_data[points[2] + "_x"], row_data[points[2] + "_y"]]
                    vec1 = np.array(p0) - np.array(p1)
                    vec2 = np.array(p2) - np.array(p1)
                    angle = get_angle(vec1, vec2)
                    angles[angle_key].append(angle)
            for new_key, angles_list in angles.items():
                data[new_key] = pd.Series(angles_list)
            self.cima[key]["data"] = data

    def save(self, name="CIMA_Transformed"):
        save_path = os.path.join(self.DATA_PATH, name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        metadata_path = os.path.join(save_path, "metadata.csv")
        self.metadata.to_csv(metadata_path)
        save_data_path = os.path.join(save_path, "data")
        if not os.path.exists(save_data_path):
            os.makedirs(save_data_path)
        for key, data in self.cima.items():
            path = os.path.join(save_data_path, key + ".csv")
            data["data"].to_csv(path)


if __name__ == "__main__":
    etl = ETL("/home/login/Dataset/")
    etl.load("CIMA_Transformed")
    # etl.create_angles()
    cima = etl.get_cima()
    keys = list(cima.keys())
    x = 0
    for key, item in cima.items():
        print(x)
        x += 1
        data = item["data"]
        data = data.drop(columns=["nose_x",    "nose_y",  "upper_chest_x",  "upper_chest_y",  "right_wrist_x",  "right_wrist_y",  "left_wrist_x",  "left_wrist_y",  "hip_center_x",  "hip_center_y",  "right_ankle_x",  "right_ankle_y",  "left_ankle_x",  "left_ankle_y"])
        print(data.describe())
    # etl.save()
