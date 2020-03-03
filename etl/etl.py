import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from multiprocessing import Pool, Manager, cpu_count
from scipy.signal import find_peaks
import shutil


def unit_vector(vector):
    # TODO delete
    return vector / np.linalg.norm(vector)


def get_angle(vec1, vec2):
    # TODO delete
    unit_vec1 = unit_vector(vec1)
    unit_vec2 = unit_vector(vec2)
    return np.arccos(np.clip(np.dot(unit_vec1, unit_vec2), -1.0, 1.0))


def get_vectors(points, row_data, z_data):
    p0 = [row_data[points[0] + "_x"], row_data[points[0] + "_y"]]
    p1 = [row_data[points[1] + "_x"], row_data[points[1] + "_y"]]
    p2 = [row_data[points[2] + "_x"], row_data[points[2] + "_y"]]
    vec1 = np.array(p0) - np.array(p1)
    vec1 = np.append(vec1, z_data[points[1] + "_z"])
    vec2 = np.array(p2) - np.array(p1)
    vec2 = np.append(vec2, z_data[points[2] + "_z"])
    return vec1, vec2


def vector_length(vec):
    length = np.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)
    return length


class Node:
    def __init__(self, name, parent):
        self.name = name
        self.parent = parent
        self.length = 0

    def get_z_value(self, row):
        dx = row[self.name + "_x"] - row[self.parent + "_x"]
        dy = row[self.name + "_y"] - row[self.parent + "_y"]
        dz = np.sqrt(np.abs(self.length ** 2 - dx ** 2 - dy ** 2))
        if np.isnan(dz):
            dz = 0
        return dz


class ETL:
    def __init__(
            self,
            data_path,
            window_sizes,
            size=0,
            bandwidth=5,
            pooling="mean",
            noise_reduction=["movement"],
            minimal_movement=0.02,
            random_seed=np.random.randint(1000),
            sma_window=5
    ):

        self.DATA_PATH = data_path
        self.cima = {}
        self.window_sizes = window_sizes
        self.bandwidth = bandwidth
        self.pooling = pooling
        self.noise_reduction = noise_reduction
        self.MINIMAL_MOVEMENT = minimal_movement
        self.random_seed = random_seed
        self.size = size
        self.sma_window = sma_window
        self.angles = {
            "right_elbow": ["right_shoulder", "right_elbow", "right_wrist"],
            "left_elbow": ["left_shoulder", "left_elbow", "left_wrist"],
            "right_shoulder": ["thorax", "right_shoulder", "right_elbow"],
            "left_shoulder": ["thorax", "left_shoulder", "left_elbow"],
            "right_hip": ["pelvis", "right_hip", "right_knee"],
            "left_hip": ["pelvis", "left_hip", "left_knee"],
            "right_knee": ["right_hip", "right_knee", "right_ankle"],
            "left_knee": ["left_hip", "left_knee", "left_ankle"]
        }

    def load(self, dataset):

        cima_id = f"{self.size}_{self.sma_window}_{self.MINIMAL_MOVEMENT}"
        save_path = os.path.join("cache", cima_id)

        if os.path.exists(save_path):
            with open(save_path, "rb") as f:
                self.cima = pickle.load(f)
                return

        cima_files = []
        missing_metadata = []
        cima_path = os.path.join(self.DATA_PATH, dataset)

        self.load_metadata(dataset)

        cima_path = os.path.join(cima_path, "data") if os.path.exists(os.path.join(cima_path, "data")) else cima_path

        for root, dirs, files in os.walk(cima_path):
            for filename in files:
                if filename[-4:] == ".csv":
                    cima_files.append(os.path.join(root, filename))

        self.check_metadata(cima_files, cima_path)

        if self.size != 0:
            healthy = self.metadata.loc[self.metadata["CP"] == 0]
            impaired = self.metadata.loc[self.metadata["CP"] == 1]

            healthy_ids = list(healthy.sample(self.size-self.size//10, random_state=self.random_seed)["ID"])
            impaired_ids = list(impaired.sample(self.size//10, random_state=self.random_seed)["ID"])

            all_ids = []
            all_ids.extend(healthy_ids)
            all_ids.extend(impaired_ids)

            cima_files = [os.path.join(cima_path, infant_id + ".csv") for infant_id in all_ids]

        for file in cima_files:
            file_name = file.split(os.sep)[-1].split(".")[0]
            file_id = file_name[:3] if file_name[0].isnumeric() else file_name[:7]
            meta_row = self.metadata.loc[self.metadata["ID"] == file_id]
            if meta_row.empty:
                missing_metadata.append(file_id)
                continue
            data = pd.read_csv(file)
            self.cima[file_id] = {"data": data, "label": meta_row.iloc[0]["CP"], "fps": meta_row.iloc[0]["FPS"]}

    def check_metadata(self, cima_files, cima_path):
        missing_files = []
        for i, row in self.metadata.iterrows():
            path = os.path.join(cima_path, row["ID"] + ".csv")
            if not path in cima_files:
                missing_files.append(i)
        self.metadata = self.metadata.drop(missing_files)

    def load_metadata(self, dataset):
        meta_path = os.path.join(self.DATA_PATH, dataset, "metadata.csv")
        self.metadata = pd.read_csv(meta_path)

    def save(self, name="CIMA_Transformed"):
        # TODO delete
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

    def key_chunks(self, dictionary, n):
        keys = list(dictionary.keys())
        for i in range(0, len(keys), n):
            yield keys[i:i+n]

    def preprocess_pooled(self, batch_size=cpu_count()):

        cima_id = f"{self.size}_{self.sma_window}_{self.MINIMAL_MOVEMENT}"
        save_path = os.path.join("cache", cima_id)

        if os.path.exists(save_path):
            print("Using cached preprocessed dataset")
            return

        # pbar = tqdm(total=int(np.ceil(len(self.cima) / batch_size)))

        def update_progress(*a):
           #pbar.update()
            pass

        if os.path.exists("tmp"):
            shutil.rmtree("tmp")

        os.mkdir("tmp")

        batches = self.key_chunks(self.cima, batch_size)
        for batch in batches:
            pool = Pool(batch_size)
            for key in batch:
                item = self.cima[key]
                pool.apply_async(self.preprocess_item, args=(key, item,))
            pool.close()
            pool.join()
            update_progress()

        # pbar.close()

        for key in os.listdir("tmp"):
            path = os.path.join("tmp", key)
            with open(path, "rb") as f:
                self.cima[key] = pickle.load(f)

        shutil.rmtree("tmp")

        if not os.path.exists("cache"):
            os.mkdir("cache")

        with open(save_path, "wb") as f:
            pickle.dump(self.cima, f)




    def preprocess_item(self, key, item):
        # Last 10-20 frames are often corrupted, remove last 100 to be sure
        item["data"] = item["data"].iloc[:-100, :]

        item = self.resample(item)
        data = item["data"]
        data = self.remove_outliers(data, 0.1)
        data = self.smooth_sma(data, self.sma_window)
        # data = self.create_angles(data)
        z_data = self.extrapolate_z_axis(data)
        angle_data = self.create_angles(data, z_data)
        item["data"] = data
        item["angles"] = angle_data
        item["z_interpolation"] = z_data
        # print(f"Writing to key {key}")
        save_path = os.path.join("tmp", key)
        with open(save_path, "wb") as f:
            pickle.dump(item, f)

    def resample(self, item, target_framerate=30):
        if item["fps"] == target_framerate:
            return item
        data = item["data"]
        time = (data["frame"] - 1) * 1 / item["fps"]
        data["time"] = time
        data = data.set_index("time")

        end_time = max(time)
        interpolated_length = int(end_time / (1 / target_framerate))
        interpolated_frames = pd.Series(range(0, interpolated_length))
        interpolated_time = interpolated_frames * 1 / target_framerate

        time = time.append(interpolated_time, ignore_index=True).drop_duplicates().sort_values()
        data = data.reindex(time).interpolate(method="slinear")
        resampled_data = data.filter(items=interpolated_time, axis=0)
        resampled_data["frame"] = list(interpolated_frames)
        resampled_data = resampled_data.set_index("frame")

        item["data"] = resampled_data
        item["fps"] = target_framerate

        return item

    def remove_outliers(self, data, threshold):
        columns = data.columns
        for column_index in range(1, len(data.columns)):
            slice = np.array(data.iloc[:, column_index])
            neg_slice = [-x for x in slice]
            peaks, _ = find_peaks(slice, threshold=threshold)
            neg_peaks, _ = find_peaks(neg_slice, threshold=threshold)

            indices = np.append(peaks, neg_peaks)

            for index in indices:
                slice[index] = (slice[index - 1] + slice[index + 1]) / 2

            data[columns[column_index]] = slice
        return data

    def smooth_sma(self, data, window_size):
        columns = data.columns
        for column_index in range(1, len(data.columns)):
            slice = data.iloc[:, column_index]
            slice = slice.rolling(window_size, center=True).mean()
            slice = slice.fillna(method="ffill")
            slice = slice.fillna(method="bfill")
            data[columns[column_index]] = slice
        return data

    def extrapolate_z_axis(self, data):
        nodes = [
            Node("right_shoulder", "thorax"),
            Node("right_elbow", "right_shoulder"),
            Node("right_wrist", "right_elbow"),
            Node("left_shoulder", "thorax"),
            Node("left_elbow", "left_shoulder"),
            Node("left_wrist", "left_elbow"),
            Node("pelvis", "thorax"),
            Node("right_hip", "pelvis"),
            Node("right_knee", "right_hip"),
            Node("right_ankle", "right_knee"),
            Node("left_hip", "pelvis"),
            Node("left_knee", "left_hip"),
            Node("left_ankle", "left_knee")
        ]

        self.set_max_length(data, nodes)

        z_dataframe = pd.DataFrame(columns=[node.name + "_z" for node in nodes])

        for node in nodes:
            for other_node in nodes:
                if node.name.split("_")[-1] == other_node.name.split("_")[-1]:
                    node.length = max(node.length, other_node.length)

        for i, row in data.iterrows():
            vals = {}
            for node in nodes:
                z = node.get_z_value(row)
                vals[node.name + "_z"] = z
            z_dataframe = z_dataframe.append(vals, ignore_index=True)

        return z_dataframe

    def set_max_length(self, data, nodes):
        for _, row in data.iterrows():
            for i, node in enumerate(nodes):
                p0 = [row[node.name + "_x"], row[node.name + "_y"]]
                p1 = [row[node.parent + "_x"], row[node.parent + "_y"]]
                vec = np.array(p1) - np.array(p0)
                length = np.sqrt(vec[0] ** 2 + vec[1] ** 2)
                node.length = length if length > node.length else node.length

    def detect_movement(self, window, angle):
        points = []
        for point in self.angles[angle]:
            points.append(point + "_x")
            points.append(point + "_y")
            points.append(point + "_z")
        window = window.filter(items=points)
        differences = [window[column].max() - window[column].min() for column in window]
        return any([difference > self.MINIMAL_MOVEMENT for difference in differences])

    def create_angles(self, data, z_data):
        angles = {key: [] for key in self.angles.keys()}
        for i, row in data.iterrows():
            for angle_key, points in self.angles.items():
                vec1, vec2 = get_vectors(points, row, z_data.iloc[i, :])
                dot_product = np.dot(vec1, vec2)
                angle = np.math.acos(dot_product / (vector_length(vec1) * vector_length(vec2)))
                angles[angle_key].append(angle)
        angle_dataframe = pd.DataFrame(angles)
        return angle_dataframe

    def generate_fourier_dataset(self, window_overlap=1):
        num_processes = len(self.window_sizes) * len(self.angles.keys())
        if cpu_count() > 12:
            pool = Pool(num_processes)
        # pbar = tqdm(total=num_processes)

        def update_progress(*a):
            # pbar.update()
            pass

        for window_size in self.window_sizes:
            if cpu_count() <= 12:
                pool = Pool(6)
            for angle in self.angles.keys():
                pool.apply_async(self.generate_fourier_data, args=(window_size, angle, window_size//window_overlap,), callback=update_progress)
            if cpu_count() <= 12:
                pool.close()
                pool.join()
        if cpu_count() > 12:
            pool.close()
            pool.join()

        # pbar.close()

    def generate_fourier_all_angles(self, window_size):
        # TODO delete
        for angle in self.angles.keys():
            self.generate_fourier_data(window_size, angle)

    def generate_fourier_data(self, window_size, angle, window_step):
        dataset = pd.DataFrame(columns=["label", "data"])
        for key, item in self.cima.items():
            angles = item["angles"]
            data = item["data"]
            z_data = item["z_interpolation"]
            for i in range(0, len(data), window_step):
                window = data.loc[i:i + window_size - 1, :].join(z_data.loc[i:i + window_size - 1, :])
                if len(window) < window_size:
                    continue
                if "movement" in self.noise_reduction and not self.detect_movement(window, angle):
                    continue
                angle_window = angles.loc[i:i + window_size - 1, angle]
                angle_window = angle_window - angle_window.mean()
                fourier_data = np.abs(np.fft.fft(angle_window))
                dataset = dataset.append(
                    {"id": key, "label": item["label"], "data": list(fourier_data[1:window_size // 2])},
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


if __name__ == "__main__":
    etl = ETL("/home/login/datasets", [128, 256, 512, 1024], size=16)
    etl.load("CIMA")
    etl.preprocess_pooled()
    etl.generate_fourier_dataset(window_overlap=4)
