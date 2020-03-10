from combo.models.detector_comb import SimpleDetectorAggregator
from pyod.models.abod import ABOD
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM


def create_models():
    models = [KNN, LOF, ABOD, OCSVM]
    methods = ["mean", "largest"]
    ensemble_combinations = ["average", "maximization"]
    pca = 10

    model_list = []

    for model in models:
        if model is KNN:
            for method in methods:
                for i in range(1, pca + 1):
                    element = {
                        "model": model,
                        "supervised": False,
                        "parameters": {
                            "n_neighbors": i,
                            "method": method,
                        }
                    }
                    model_list.append(element)
        if model is LOF:
            for i in range(1, pca + 1):
                element = {
                    "model": model,
                    "supervised": False,
                    "parameters": {
                        "n_neighbors": i,
                    }
                }
                model_list.append(element)
        if model is ABOD:
            element = {
                "model": model,
                "supervised": False,
                "parameters": {}
            }
            model_list.append(element)
        if model is OCSVM:
            element = {
                "model": model,
                "supervised": False,
                "parameters": {}
            }
            model_list.append(element)

    for ensemble_combination in ensemble_combinations:
        for i in range(1, pca + 1):
            for j in range(1, pca + 1):
                for method in methods:
                    element = {
                        "model": SimpleDetectorAggregator,
                        "supervised": False,
                        "parameters": {
                            "method": ensemble_combination,
                            "base_estimators": [
                                KNN(n_neighbors=i, method=method),
                                LOF(n_neighbors=j),
                                ABOD(),
                                OCSVM()
                            ],
                        }
                    }
                    model_list.append(element)

    return model_list

