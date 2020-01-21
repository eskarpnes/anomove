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

    def __init__(self, data_path, window_sizes=[128, 256, 512]):
        self.DATA_PATH = data_path
        self.cima = {}
        self.window_sizes = window_sizes
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
        meta_path = os.path.join(self.DATA_PATH, dataset, "metadata.csv")
        self.metadata = pd.read_csv(meta_path)


    def load(self, dataset, tiny=False):
        cima_files = []
        missing_metadata = []
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
            meta_row = self.metadata.loc[self.metadata["ID"] == file_id]
            if meta_row.empty:
                missing_metadata.append(file_id)
                continue
            data = pd.read_csv(file)
            # data = data.drop(columns=["Unnamed: 0"], errors="ignore")
            self.cima[file_id] = {"data": data, "label": meta_row.iloc[0]["CP"], "fps": meta_row.iloc[0]["FPS"]}

    def create_angles(self):
        cima_angles = {}
        print("\n\n----------------")
        print(" Creating angles ")
        print("----------------\n")
        for key, item in tqdm(self.cima.items()):
            data = item["data"]
            angles = {key: [] for key in self.angles.keys()}
            for row in data.iterrows():
                row_data = row[1]
                for angle_key, points in self.angles.items():
                    p0 = [row_data[points[0] + "_x"], row_data[points[0] + "_y"]]
                    p1 = [row_data[points[1] + "_x"], row_data[points[1] + "_y"]]
                    p2 = [row_data[points[2] + "_x"], row_data[points[2] + "_y"]]
                    vec1 = np.array(p0) - np.array(p1)
                    vec2 = np.array(p2) - np.array(p1)
                    angle = np.abs(np.math.atan2(np.linalg.det([vec1,vec2]),np.dot(vec1,vec2)))
                    angles[angle_key].append(angle)
            for new_key, angles_list in angles.items():
                data[new_key] = angles_list
            self.cima[key]["data"] = data

    # Resample to 30 fps by interpolation.
    def resample(self, target_framerate=30):
        for key, item in tqdm(self.cima.items()):
            data = item["data"]
            time = (data["frame"]-1) * 1/item["fps"]
            data["time"] = time
            data = data.set_index("time")

            if item["fps"] == target_framerate:
                item["data"] = data
                continue

            end_time = max(time)
            interpolated_length = int(end_time / (1/target_framerate))
            interpolated_frames = pd.Series(range(0, interpolated_length))
            interpolated_time = interpolated_frames * 1/target_framerate

            time = time.append(interpolated_time, ignore_index=True).drop_duplicates().sort_values()
            data = data.reindex(time).interpolate(method="slinear")
            resampled_data = data.filter(items=interpolated_time, axis=0)
            resampled_data["frame"] = list(interpolated_frames)
            item["data"] = resampled_data
            item["fps"] = target_framerate

    def generate_fourier_dataset(self):
        for window_size in self.window_sizes:
            self.generate_fourier_all_angles(window_size)

    def generate_fourier_all_angles(self, window_size):
        for angle in tqdm(self.angles.keys()):
            self.generate_fourier_data(window_size, angle)

    def generate_fourier_data(self, window_size, angle):
        dataset = pd.DataFrame(columns=["label", "data"])
        for key, item in self.cima.items():
            data = item["data"]
            data = data.filter(items=[angle])
            for i in range(0, len(data), window_size):
                window = data.loc[i:i+window_size-1, : ]
                if len(window) < window_size:
                    continue
                angle_data = window[angle]
                angle_data = angle_data - angle_data.mean()
                fourier_data = np.abs(np.fft.fft(angle_data))
                dataset = dataset.append({"label": item["label"], "data": list(fourier_data[1:window_size//2])}, ignore_index=True)
        self.save_fourier_dataset(window_size, angle, dataset)

    def save_fourier_dataset(self, window_size, angle, data):
        save_path = os.path.join(self.DATA_PATH, str(window_size))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        data.to_csv(os.path.join(save_path, angle + ".csv"))

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
    etl = ETL("/home/erlend/datasets/", window_sizes=[128, 256, 512, 1024])
    etl.load("CIMA_angles_resampled", tiny=False)
    # etl.resample()
    # etl.create_angles()
    # etl.save(name="CIMA_angles_resampled")
    etl.generate_fourier_dataset()
