from combo.models.detector_comb import SimpleDetectorAggregator
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.hbos import HBOS
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from sklearn.neighbors import LocalOutlierFactor
import itertools
from pprint import pprint


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
            for j in range(i + 1, pca + 1):
                for k in range(j + 1, pca + 1):
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
    print(len(model_list))
    return model_list


def create_tunable_ensemble(knn_neighbors, lof_neighbors, abod_neighbors):
    model_list = []
    for knn_neighbor in knn_neighbors:
        for lof_neighbor in lof_neighbors:
            for abod_neighbor in abod_neighbors:
                element = {
                    "model": SimpleDetectorAggregator,
                    "supervised": False,
                    "parameters": {
                        "method": "average",
                        "base_estimators": [
                            KNN(n_neighbors=knn_neighbor),
                            LOF(n_neighbors=lof_neighbor),
                            ABOD(n_neighbors=abod_neighbor),
                            OCSVM()
                        ],
                    }
                }
                model_list.append(element)

    return model_list


def create_abod(min_neighbours, max_neighbours):
    model_list = []
    for i in range(min_neighbours, max_neighbours + 1):
        model_list.append({
            "model": ABOD,
            "supervised": False,
            "parameters": {
                "n_neighbors": i
            }
        })
    return model_list


def create_base_models(models, pca):
    model_list = []
    for model in models:
        if model is OCSVM:
            element = {
                "model": model,
                "supervised": False,
                "parameters": {}
            }
            model_list.append(element)
        else:
            if model is CBLOF:
                parameter = "n_clusters"
                minimum = 5
            elif model is HBOS:
                parameter = "n_bins"
                minimum = 2
            elif model is ABOD:
                parameter = "n_neighbors"
                minimum = 2
            else:
                parameter = "n_neighbors"
                minimum = 1

            for i in range(minimum, pca + 1):
                element = {
                    "model": model,
                    "supervised": False,
                    "parameters": {
                        parameter: i,
                    }
                }
                model_list.append(element)
    return model_list


def create_novelty_models(min_neighbours, max_neighbours):
    model_list = []
    for i in range(min_neighbours, max_neighbours + 1):
        model_list.append({
            "model": LocalOutlierFactor,
            "supervised": False,
            "parameters": {
                "n_neighbors": i,
                "novelty": True
            }
        })
    return model_list


# Ha antall av hver model og antall n
# knn = [2,3,4] her er det 3 knn modeller med naboer 2, 3, og 4. Lag dette for alle modeller.
def create_ensemble_with_n_models(models):
    base_estimators = []
    for key, value in models.items():
        model = key
        parameter_numbers = value
        if model is OCSVM:
            base_estimators.append(OCSVM())
        else:
            for parameter_number in parameter_numbers:
                if model is ABOD:
                    base_estimators.append(ABOD(n_neighbors=parameter_number))
                elif model is KNN:
                    base_estimators.append(KNN(n_neighbors=parameter_number))
                elif model is LOF:
                    base_estimators.append(LOF(n_neighbors=parameter_number))
                elif model is CBLOF:
                    base_estimators.append(CBLOF(n_clusters=parameter_number))
                elif model is HBOS:
                    base_estimators.append(HBOS(n_bins=parameter_number))

    return [add_ensemble(base_estimators)]


def create_all_ensemble_methods():
    knn = []
    knn_neighbors = [5, 9, 10]
    lof = []
    lof_neighbors = [6, 7, 8, 9, 10]
    abod = []
    abod_neighbors = [3, 4, 5, 6]
    hbos = []
    hbos_bins = [3, 5, 7, 8, 9, 10]

    ensemble_methods = []
    for knn_neighbor in knn_neighbors:
        knn.append(knn_neighbor)
        for lof_neighbor in lof_neighbors:
            lof.append(lof_neighbor)
            for abod_neighbor in abod_neighbors:
                abod.append(abod_neighbor)
                for hbos_bin in hbos_bins:
                    hbos.append(hbos_bin)

                    ensemble_methods.append(create_ensemble_with_n_models({
                        KNN: knn,
                        LOF: lof,
                        ABOD: abod,
                        HBOS: hbos,
                        OCSVM: None
                    }))
                hbos = []
            abod = []
        lof = []

    return ensemble_methods


def add_ensemble(base_estimators):
    return {
        "model": SimpleDetectorAggregator,
        "supervised": False,
        "parameters": {
            "method": "maximum",
            "base_estimators": base_estimators,
        }
        }

