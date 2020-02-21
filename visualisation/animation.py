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

# 960 x 540
# 1280 x 720

width = 1280
height = 720

ORANGE = (51, 153, 255)

def random_color():
    return (np.random.randint(256), np.random.randint(256), np.random.randint(256))

def lenght_color(length):
    green_val = length*3 if length < 85 else 255
    red_val = 255 - length*3 if length < 85 else 0
    return (0, green_val, red_val)

def draw_skeleton(row):
    frame = np.full((height, width, 3), 220, np.uint8)
    # draw = ImageDraw.Draw(frame)

    for vector in vectors:
        start_x = int(row[vector[0] + "_x"] * width)
        start_y = int(row[vector[0] + "_y"] * height)
        end_x = int(row[vector[1] + "_x"] * width)
        end_y = int(row[vector[1] + "_y"] * height)
        length = np.sqrt((start_x-end_x)**2 + (start_y-end_y)**2)
        frame = cv2.line(frame, (start_x, start_y), (end_x, end_y), lenght_color(length), 3)

    for i in range(1, len(row), 2):
        radius = 3
        point = (int(row[i] * width), int(row[i+1]*height))
        frame = cv2.circle(frame, point, 3, ORANGE, 3)

    frame_string = str(int(row["frame"]))

    font = cv2.FONT_HERSHEY_SIMPLEX
    frame = cv2.putText(frame, frame_string, (10, height-10), font, 1, (0, 0, 0), 2, cv2.LINE_AA)

    return frame

def animate(dataframe, data_path, video_name):
    out_path = os.path.join(data_path, video_name + ".avi")

    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'XVID'), 30.0, (width, height))

    for _, row in tqdm(dataframe.iterrows()):
        frame = draw_skeleton(row)
        out.write(frame)

    out.release()