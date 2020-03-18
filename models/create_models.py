from combo.models.detector_comb import SimpleDetectorAggregator
from pyod.models.abod import ABOD
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM


def get_models(ensemble=False, knn_methods=None, ensemble_combinations=None, pca=10, only_LOF=False):

    if knn_methods is None:
        knn_methods = ["mean", "largest"]
    if ensemble_combinations is None:
        ensemble_combinations = ["average", "maximization"]

    if not ensemble:
        return create_models(knn_methods, pca)
    if ensemble:
        if only_LOF:
            return create_ensemble_LOF(ensemble_combinations, pca)
        else:
            return create_ensemble_models(knn_methods, ensemble_combinations, pca)


def create_ensemble_models(knn_methods, ensemble_combinations, pca):
    model_list = []

    for ensemble_combination in ensemble_combinations:
        for i in range(1, pca + 1):
            for j in range(1, pca + 1):
                for method in knn_methods:
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


def create_models(methods, pca):
    models = [KNN, LOF, ABOD, OCSVM]
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

    return model_list


def create_ensemble_LOF(ensemble_combinations, pca):
    model_list = []
    for ensemble_combination in ensemble_combinations:
        for i in range(3, pca + 1):
            for j in range(3, pca + 1):
                for k in range(3, pca + 1):
                    if i == j or i == k or j == k:
                        continue
                    else:
                        element = {
                            "model": SimpleDetectorAggregator,
                            "supervised": False,
                            "parameters": {
                                "method": ensemble_combination,
                                "base_estimators": [
                                    LOF(n_neighbors=i),
                                    LOF(n_neighbors=j),
                                    LOF(n_neighbors=k),
                                ],
                            }
                        }
                        model_list.append(element)

    return model_list
