from PIL import Image, ImageDraw, ImageFont
import cv2
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

vectors = [
    ("head_top", "nose"),
    ("nose", "right_ear"),
    ("nose", "left_ear"),
    ("nose", "upper_neck"),
    ("upper_neck", "thorax"),
    ("thorax", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("thorax", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("thorax", "pelvis"),
    ("pelvis", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("pelvis", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle")
]

z_parents = {
    "thorax": None,
    "right_shoulder": "thorax",
    "right_elbow": "right_shoulder",
    "right_wrist": "right_elbow",
    "left_shoulder": "thorax",
    "left_elbow": "left_shoulder",
    "left_wrist": "left_elbow",
    "pelvis": None,
    "right_hip": "pelvis",
    "right_knee": "right_hip",
    "right_ankle": "right_knee",
    "left_hip": "pelvis",
    "left_knee": "left_hip",
    "left_ankle": "left_knee"
}

# 960 x 540
# 1280 x 720

width = 1280
height = 720

ORANGE = (51, 153, 255)


def random_color():
    return (np.random.randint(256), np.random.randint(256), np.random.randint(256))


def lenght_color(length):
    green_val = length * 3 if length < 85 else 255
    red_val = 255 - length * 3 if length < 85 else 0
    return (0, green_val, red_val)

def result_color(result):
    if result == -1:
        return (200, 200, 200)
    elif result == 0:
        return (60, 220, 60)
    elif result == 1:
        return (60, 60, 220)
    else:
        return ORANGE

def draw_skeleton(frame_number, row, result=None):
    frame = np.full((height, width, 3), 220, np.uint8)
    # draw = ImageDraw.Draw(frame)

    origin = scale_to_pixels(0, 0)
    axis = [
        ((0, 1), (80, 255, 80)),
        ((1, 0), (255, 80, 80))
    ]

    for ax in axis:
        coords = scale_to_pixels(*ax[0])
        frame = cv2.line(frame, origin, coords, ax[1], 2)

    for i in range(0, 10):
        i = i/10
        x_start = (i, 0)
        y_start = (i, 1)
        start = scale_to_pixels(*x_start)
        end = scale_to_pixels(*y_start)
        frame = cv2.line(frame, start, end, (255, 80, 80), 1)
        x_start = (0, i)
        y_start = (1, i)
        start = scale_to_pixels(*x_start)
        end = scale_to_pixels(*y_start)
        frame = cv2.line(frame, start, end, (80, 255, 80), 1)

    for vector in vectors:
        start_x = row[vector[0] + "_x"]
        start_y = row[vector[0] + "_y"]
        start_coordinates = scale_to_pixels(start_x, start_y)
        end_x = row[vector[1] + "_x"]
        end_y = row[vector[1] + "_y"]
        end_coordinates = scale_to_pixels(end_x, end_y)
        length = np.sqrt((start_x - end_x) ** 2 + (start_y - end_y) ** 2)
        try:
            start_color = result_color(result[vector[0] + "_pred"])
            end_color = result_color(result[vector[1] + "_pred"])
            frame = cv2.circle(frame, start_coordinates, 6, start_color, 6)
            frame = cv2.circle(frame, end_coordinates, 6, end_color, 6)
        except KeyError:
            # Expected, pass
            pass
        frame = cv2.line(frame, start_coordinates, end_coordinates, (0,0,0), 3)

    frame_string = str(frame_number)

    font = cv2.FONT_HERSHEY_SIMPLEX
    frame = cv2.putText(frame, frame_string, (10, height - 10), font, 1, (0, 0, 0), 2, cv2.LINE_AA)

    return frame


def isometric_projection(x, y, z):

    # Rotation around the y axis
    alpha = np.radians(15)
    # Rotation around the z axis
    beta = np.radians(45+180)

    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.math.cos(alpha), np.math.sin(alpha)],
        [0, -np.math.sin(alpha), np.math.cos(alpha)]
    ]) @ np.array([
        [np.math.cos(beta), 0, -np.math.sin(beta)],
        [0, 1, 0],
        [np.math.sin(beta), 0, np.math.cos(beta)]
    ])
    c = rotation_matrix @ np.array([y,z,x])

    projection = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]]) @ c

    return projection[0], projection[1]


def get_z_val(row, part):
    z_val = 0
    try:
        while z_parents[part] != None:
            if "knee" in z_parents[part]:
                z_val -= row[part + "_z"]
            else:
                z_val += row[part + "_z"]
            part = z_parents[part]
        return z_val
    except KeyError:
        return z_val


def scale_to_pixels_3d(x, y):
    return int(x * width) + width//2, int(1-y * height) + height//2

def scale_to_pixels(x, y):
    return int(x * width), int(y * height)

def draw_skeleton_3d(frame_number, row, result=None):
    frame = np.full((height, width, 3), 220, np.uint8)

    origin = scale_to_pixels_3d(*isometric_projection(0, 0, 0))
    axis = [
        (isometric_projection(0, 0, 0.5), (80, 80, 255)),
        (isometric_projection(0, 1, 0), (80, 255, 80)),
        (isometric_projection(1, 0, 0), (255, 80, 80))
    ]

    for ax in axis:
        coords = scale_to_pixels_3d(*ax[0])
        frame = cv2.line(frame, origin, coords, ax[1], 2)

    for i in range(0, 10):
        i = i/10
        x_start = (i, 0, 0)
        y_start = (i, 1, 0)
        start = scale_to_pixels_3d(*isometric_projection(*x_start))
        end = scale_to_pixels_3d(*isometric_projection(*y_start))
        frame = cv2.line(frame, start, end, (255, 80, 80), 1)
        x_start = (0, i, 0)
        y_start = (1, i, 0)
        start = scale_to_pixels_3d(*isometric_projection(*x_start))
        end = scale_to_pixels_3d(*isometric_projection(*y_start))
        frame = cv2.line(frame, start, end, (80, 255, 80), 1)


    for vector in vectors:
        start_x = row[vector[0] + "_x"]
        start_y = row[vector[0] + "_y"]
        start_z = get_z_val(row, vector[0])
        start_coordinates = isometric_projection(start_x, start_y, start_z)
        end_x = row[vector[1] + "_x"]
        end_y = row[vector[1] + "_y"]
        end_z = get_z_val(row, vector[1])
        end_coordinates = isometric_projection(end_x, end_y, end_z)
        start_coordinates = scale_to_pixels_3d(*start_coordinates)
        end_coordinates = scale_to_pixels_3d(*end_coordinates)
        try:
            start_color = result_color(result[vector[0] + "_pred"])
            end_color = result_color(result[vector[1] + "_pred"])
            frame = cv2.circle(frame, start_coordinates, 6, start_color, 6)
            frame = cv2.circle(frame, end_coordinates, 6, end_color, 6)
        except KeyError:
            # Expected, pass
            pass
        frame = cv2.line(frame, start_coordinates, end_coordinates, (0, 0, 0), 3)

    return frame


def animate(dataframe, data_path, video_name):
    out_path = os.path.join(data_path, video_name + ".avi")

    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'XVID'), 30.0, (width, height))

    pbar = tqdm(total=len(dataframe))

    for i, row in dataframe.iterrows():
        frame = draw_skeleton(i, row)
        out.write(frame)
        pbar.update()

    out.release()
    pbar.close()


def animate_3d(dataframe, data_path, video_name, result=None):
    out_path = os.path.join(data_path, video_name + ".avi")

    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'XVID'), 30.0, (width, height*2))

    pbar = tqdm(total=len(dataframe))

    for i, row in dataframe.iterrows():
        if result is not None:
            result_row = result.iloc[i, :]
        frame_3d = draw_skeleton_3d(i, row, result=result_row)
        frame_2d = draw_skeleton(i, row, result=result_row)
        frame = np.append(frame_3d, frame_2d, 0)
        out.write(frame)
        pbar.update()

    out.release()
    pbar.close()


if __name__ == "__main__":
    import pickle

    with open("/home/login/projects/anomove/etl/cache/16_5", "rb") as f:
        data = pickle.load(f)
    infant = data["087"]
    coord_data = infant["data"]
    z_data = infant["z_interpolation"]
    all_data = coord_data.join(z_data)
    all_data = all_data.iloc[:500, :]
    animate_3d(all_data, "", "3dboi")
