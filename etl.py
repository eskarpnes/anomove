#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import pandas as pd


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def get_angle(vec1, vec2):
    unit_vec1 = unit_vector(vec1)
    unit_vec2 = unit_vector(vec2)
    return np.arccos(np.clip(np.dot(unit_vec1, unit_vec2), -1.0, 1.0))


class ETL:

    def __init__(self, data_path):
        self.DATA_PATH = data_path
        self.META_PATH = os.path.join(self.DATA_PATH, "metadata.xls")

        metadata = pd.read_excel(self.META_PATH)
        metadata = metadata.drop(["CP score", "Unnamed: 5", "Unnamed: 6", "Unnamed: 7", "Unnamed: 8"], axis=1)
        metadata.rename(columns={"ID nummer": "id", "CP score": "score", "CP utkom": "cp"}, inplace=True)

        # Rename video ids to remove suffix
        # Either III or AAA_III where A is alphabetic and I is integers
        metadata["id"] = metadata["id"].apply(lambda name: name[:3] if name[0].isnumeric() else name[:7])
        self.metadata = metadata
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

    def load(self):
        cima_files = []
        for root, dirs, files in os.walk(self.DATA_PATH):
            for filename in files:
                if filename[-4:] == ".csv":
                    cima_files.append(os.path.join(root, filename))

        for file in cima_files:
            file_name = file.split(os.sep)[-1].split(".")[0]
            file_id = file_name[:3] if file_name[0].isnumeric() else file_name[:7]
            meta_row = self.metadata.loc[self.metadata["id"] == file_id]
            if meta_row.empty:
                continue
            self.cima[file_id] = {"data": pd.read_csv(file), "label": meta_row.iloc[0]["cp"]}
