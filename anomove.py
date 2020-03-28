import os
import pandas as pd
from etl.etl import ETL
from models.predictor import Predictor
from visualisation import animation


def load_validation_set(data_path):
    etl = ETL(data_path, [128, 256, 512, 1024])
    etl.load("CIMA", validation=True)
    etl.preprocess_pooled()
    return etl.cima


def load_infant(data_path):
    window_sizes = [128, 256, 512, 1024]
    etl = ETL(data_path, window_sizes)


def evaluate_infant(infant_id):
    # Get a prediction from an infant on a per frame basis
    predictor = Predictor()
    predictor.load_model("ensemble_model")
    infant = predictor.predict("/home/erlend/datasets/CIMA/data", infant_id)
    visualise_result(infant)


def visualise_result(infant, output_path="", video_name="result"):
    animation.animate_3d(
        infant["data"].join(infant["z_interpolation"]),
        output_path,
        video_name,
        result=infant["predictions"]
    )


# cima = load_validation_set("/home/erlend/datasets")

evaluate_infant("001")
