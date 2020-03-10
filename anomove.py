import os
import pandas as pd
from etl.etl import ETL
from models import predictor
from visualisation import animation


def load_validation_set(data_path):
    etl = ETL(data_path, [128, 256, 512, 1024])
    etl.load("CIMA", validation=True)
    etl.preprocess_pooled()
    return etl.cima


def load_infant(data_path):
    etl = ETL(data_path, [128, 256, 512, 1024])


def evaluate_infant(infant):
    # Get a prediction from an infant on a per frame basis
    infant["predictions"] = predictor.predict(infant["angles"])
    visualise_result(infant)


def visualise_result(infant, output_path="", video_name="result"):
    animation.animate_3d(
        infant["data"].join(infant["z_interpolation"]),
        output_path,
        video_name,
        result=infant["predictions"]
    )


cima = load_validation_set("/home/login/datasets")

for key, infant in cima.items():
    evaluate_infant(infant)
    break
