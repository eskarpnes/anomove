from PIL import Image, ImageDraw, ImageFont
import cv2
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

data_path = "/home/login/datasets/"
dataframe = pd.read_csv(os.path.join(data_path, "CIMA_short.csv"))

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

fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 16)
width = 960
height = 540


def draw_skeleton(row):
    frame = Image.new("RGB", (width, height), color="gray")
    draw = ImageDraw.Draw(frame)

    for i in range(1, len(row), 2):
        radius = 3
        point = (int(row[i] * width), int(row[i+1]*height))
        draw.ellipse([point[0]-radius, point[1]-radius, point[0]+radius, point[1]+radius], fill="red")

    for vector in vectors:
        start_x = int(row[vector[0] + "_x"] * width)
        start_y = int(row[vector[0] + "_y"] * height)
        end_x = int(row[vector[1] + "_x"] * width)
        end_y = int(row[vector[1] + "_y"] * height)
        draw.line([start_x, start_y, end_x, end_y], fill="red", width=3)


    draw.text((10, height - 22), str(int(row["frame"])), font=fnt, fill="black")

    savepath = os.path.join(data_path, "temp", str(int(row["frame"])) + ".png")
    if not os.path.exists(os.path.join(data_path, "temp")):
        os.makedirs(os.path.join(data_path, "temp"))

    frame.save(savepath)

def draw_skeleton_cv2(row):
    frame = np.zeros((height, width, 3), np.uint8)
    # draw = ImageDraw.Draw(frame)


    for vector in vectors:
        start_x = int(row[vector[0] + "_x"] * width)
        start_y = int(row[vector[0] + "_y"] * height)
        end_x = int(row[vector[1] + "_x"] * width)
        end_y = int(row[vector[1] + "_y"] * height)
        frame = cv2.line(frame, (start_x, start_y), (end_x, end_y), (255,0,0), 3)

    for i in range(1, len(row), 2):
        radius = 3
        point = (int(row[i] * width), int(row[i+1]*height))
        frame = cv2.circle(frame, point, 3, (0,0,255))

    # draw.text((10, height - 22), str(int(row["frame"])), font=fnt, fill="black")

    frame_string = str(int(row["frame"]))

    font = cv2.FONT_HERSHEY_SIMPLEX
    frame = cv2.putText(frame, frame_string, (10, height-10), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return frame

out_path = os.path.join(data_path, "skeleton.avi")

out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'XVID'), 30.0, (width, height))

for _, row in tqdm(dataframe.iterrows()):
    frame = draw_skeleton_cv2(row)
    out.write(frame)

out.release()
