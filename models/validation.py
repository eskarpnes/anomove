import pandas as pd
import os
from etl.etl import ETL
from predictor import Predictor

VALIDATION_PATH = "/home/erlend/datasets/CIMA/validation"


def load_predictor(model_name):
    predictor = Predictor()
    predictor.load_model(model_name)
    return predictor


def predict_infant(predictor, infant_id):
    infant, score = predictor.predict(infant_id)
    return score


def predict_validation_set(model_name):

    predictor = load_predictor(model_name)
    results = dict()
    for filename in os.listdir(VALIDATION_PATH):
        infant_id = filename.split(".")[0]
        infant, score = predictor.predict(VALIDATION_PATH, infant_id)
        results[str(infant_id)] = score
        print(f"{infant_id} - {score}")
    results_df = pd.DataFrame.from_dict(results, orient="index")
    results_df.to_csv("results/validation_results_mean.csv")


def sensitivity_and_specificity_score(meta_path, result_path, ):
    metadata = pd.read_csv(meta_path)
    results = pd.read_csv(result_path)
    results = results.rename(columns={"Unnamed: 0": "ID", "0": "score"})
    results = pd.merge(metadata, results, on="ID").drop(columns=["FPS"])
    print("ueh")

|
if __name__ == "__main__":
    # predict_validation_set("xgbod")
    sensitivity_and_specificity_score("/home/erlend/datasets/CIMA/metadata.csv", "results/validation_results.csv")
