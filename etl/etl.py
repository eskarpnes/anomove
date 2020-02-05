import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from multiprocessing import Pool, cpu_count

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

vector_lengths = []


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def get_angle(vec1, vec2):
    unit_vec1 = unit_vector(vec1)
    unit_vec2 = unit_vector(vec2)
    return np.arccos(np.clip(np.dot(unit_vec1, unit_vec2), -1.0, 1.0))


def get_vectors(points, row_data):
    p0 = [row_data[points[0] + "_x"], row_data[points[0] + "_y"]]
    p1 = [row_data[points[1] + "_x"], row_data[points[1] + "_y"]]
    p2 = [row_data[points[2] + "_x"], row_data[points[2] + "_y"]]
    vec1 = np.array(p0) - np.array(p1)
    vector_length(vec1)
    vec2 = np.array(p2) - np.array(p1)
    vector_length(vec2)
    return vec1, vec2


def vector_length(vec):
    length = np.sqrt(vec[0] ** 2 + vec[1] ** 2)
    return length


class ETL:

    def __init__(self, data_path, window_sizes=[128, 256, 512], bandwidth=5, pooling="mean", noise_reduction="movement"):
        self.DATA_PATH = data_path
        # Minimal difference in x or y axis in one frame to count as movement
        all_noise_reduction = ["movement", "short_vector"]
        self.noise_reduction = all_noise_reduction if noise_reduction is "all" else [noise_reduction]
        self.bandwidth = bandwidth
        self.pooling = pooling
        self.MINIMAL_MOVEMENT = 0.02
        self.MINIMAL_VECTOR_LENGTH = 0.1
        self.cima = {}
        self.invalid_frames = {}
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

        pickle_path = os.path.join(cima_path, "invalid_frames.pkl")
        if os.path.exists(pickle_path):
            self.invalid_frames = pickle.load(open(pickle_path, "rb"))

        cima_path = os.path.join(cima_path, "data") if os.path.exists(os.path.join(cima_path, "data")) else cima_path

        for root, dirs, files in os.walk(cima_path):
            for filename in files:
                if filename[-4:] == ".csv":
                    cima_files.append(os.path.join(root, filename))

        if tiny:
            cima_files = cima_files[:5]

        # print("\n\n----------------")
        # print(" Loading CIMA ")
        # print("----------------\n")

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
                    vec1, vec2 = get_vectors(points, row_data)
                    if any([vector_length(vec) < self.MINIMAL_VECTOR_LENGTH for vec in [vec1, vec2]]):
                        frame = int(row_data["frame"])
                        if not key in self.invalid_frames.keys():
                            self.invalid_frames[key] = {}
                            if not angle_key in self.invalid_frames[key]:
                                self.invalid_frames[key][angle_key] = set()
                            else:
                                self.invalid_frames[key][angle_key].add(frame)
                        else:
                            if not angle_key in self.invalid_frames[key]:
                                self.invalid_frames[key][angle_key] = set()
                            else:
                                self.invalid_frames[key][angle_key].add(frame)
                    angle = np.abs(np.math.atan2(np.linalg.det([vec1, vec2]), np.dot(vec1, vec2)))
                    angles[angle_key].append(angle)
            for new_key, angles_list in angles.items():
                data[new_key] = angles_list
            self.cima[key]["data"] = data

    # Resample to 30 fps by interpolation.
    def resample(self, target_framerate=30):
        for key, item in tqdm(self.cima.items()):
            data = item["data"]
            time = (data["frame"] - 1) * 1 / item["fps"]
            data["time"] = time
            data = data.set_index("time")

            if item["fps"] == target_framerate:
                item["data"] = data
                continue

            end_time = max(time)
            interpolated_length = int(end_time / (1 / target_framerate))
            interpolated_frames = pd.Series(range(0, interpolated_length))
            interpolated_time = interpolated_frames * 1 / target_framerate

            time = time.append(interpolated_time, ignore_index=True).drop_duplicates().sort_values()
            data = data.reindex(time).interpolate(method="slinear")
            resampled_data = data.filter(items=interpolated_time, axis=0)
            resampled_data["frame"] = list(interpolated_frames)
            item["data"] = resampled_data
            item["fps"] = target_framerate

    def remove_outliers(self):
        pass

    def smooth_sma(self, window_size):
        pass

    def detect_movement(self, window, angle):
        points = []
        for point in self.angles[angle]:
            points.append(point + "_x")
            points.append(point + "_y")
        window = window.filter(items=points)
        differences = [window[column].max() - window[column].min() for column in window]
        return any([difference > self.MINIMAL_MOVEMENT for difference in differences])

    def generate_fourier_dataset(self):
        num_processes = len(self.window_sizes) * len(self.angles.keys())
        if cpu_count() > 12:
            pool = Pool(num_processes)
        pbar = tqdm(total=num_processes)

        def update_progress(*a):
            pbar.update()

        for window_size in self.window_sizes:
            if cpu_count() <= 12:
                pool = Pool(6)
            for angle in self.angles.keys():
                pool.apply_async(self.generate_fourier_data, args=(window_size, angle,), callback=update_progress)
            if cpu_count() <= 12:
                pool.close()
                pool.join()
        if cpu_count() > 12:
            pool.close()
            pool.join()

    def generate_fourier_all_angles(self, window_size):
        for angle in self.angles.keys():
            self.generate_fourier_data(window_size, angle)

    def generate_fourier_data(self, window_size, angle):
        dataset = pd.DataFrame(columns=["label", "data"])
        for key, item in self.cima.items():
            data = item["data"]
            data = data.set_index("frame")
            for i in range(0, len(data), window_size):
                window = data.loc[i:i + window_size - 1, :]
                if len(window) < window_size:
                    continue
                if "movement" in self.noise_reduction and not self.detect_movement(window, angle):
                    continue
                if "short_vector" in self.noise_reduction:
                    if key in self.invalid_frames.keys():
                        if angle in self.invalid_frames[key].keys():
                            frames = set(window.index.values)
                            if len(frames.intersection(self.invalid_frames[key][angle])) > 0:
                                continue
                angle_data = window[angle]
                angle_data = angle_data - angle_data.mean()
                fourier_data = np.abs(np.fft.fft(angle_data))
                dataset = dataset.append({"label": item["label"], "data": list(fourier_data[1:window_size // 2])},
                                         ignore_index=True)
        if self.bandwidth is not None:
            dataset = self.generate_frequency_bands(dataset)
        self.save_fourier_dataset(window_size, angle, dataset)

    def generate_frequency_bands(self, dataset):
        # Will not check if last window is of size band_width
        for idx, row in dataset.iterrows():
            data_series = pd.Series(row["data"])
            means = []
            for i in range(0, len(data_series), self.bandwidth):
                window = data_series.loc[i:i + self.bandwidth - 1]
                # Can be either max or mean or another measure
                if self.pooling == "mean":
                    means.append(window.mean())
                if self.pooling == "max":
                    means.append(window.max())
            dataset["data"][idx] = means
        return dataset

    def save_fourier_dataset(self, window_size, angle, data):
        save_path = os.path.join(self.DATA_PATH, str(window_size))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        data.to_json(os.path.join(save_path, angle + ".json"))

    def save(self, name="CIMA_Transformed"):
        save_path = os.path.join(self.DATA_PATH, name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        metadata_path = os.path.join(save_path, "metadata.csv")
        self.metadata.to_csv(metadata_path)

        pickle_path = os.path.join(save_path, "invalid_frames.pkl")
        pickle.dump(self.invalid_frames, open(pickle_path, "wb"))

        save_data_path = os.path.join(save_path, "data")
        if not os.path.exists(save_data_path):
            os.makedirs(save_data_path)
        for key, data in self.cima.items():
            path = os.path.join(save_data_path, key + ".csv")
            data["data"].to_csv(path)


if __name__ == "__main__":
    etl = ETL("/home/login/Dataset/", window_sizes=[128, 256, 512, 1024])
    etl.load("CIMA", tiny=True)
    etl.remove_outliers()
    etl.generate_fourier_dataset()
