import pandas as pd
import re
import os


def get_data(filename):
    return pd.read_csv(filename, index_col=0)


def drop_rows(df, rows):
    for row in rows:
        df.drop(df[df[row[0]] == row[1]].index, inplace=True)
    return df


def drop_columns(df, columns):
    return df.drop(columns=columns)


def column_filter(df, filters):
    for filter in filters:
        column_name = filter[0]
        column_value = filter[1]
        df = df.loc[df[column_name] == column_value]
    return df


def print_results(filename, sort_by=["sensitivity"], number_of_rows=25, filters=None, drop_row=None, drop_column=None):
    df = get_data(filename)

    if filters is not None:
        df = column_filter(df, filters)

    if drop_row is not None:
        df = drop_rows(df, drop_row)

    if drop_column is not None:
        df = drop_columns(df, drop_column)

    with pd.option_context('display.max_rows', None, "display.expand_frame_repr", False, "display.max_colwidth", 30):
        print('\nPrinting the results from file: ' + str(filename))
        print(f"Sorted by {sort_by}, filtered on {filters}, dropped rows: {drop_row}, and dropped columns: {drop_column} \n")

        df = df.sort_values(by=sort_by, ascending=False)
        print(df.iloc[0:number_of_rows, :])


def print_angles(filename, sort_by=["sensitivity"], number_of_rows=25, filters=None, drop_row=None, drop_column=None):
    angles = ["shoulder", "hip", "knee", "elbow"]
    for angle in angles:
        print_results(filename,
                      sort_by=sort_by,
                      number_of_rows=number_of_rows,
                      filters=[["angle", angle]],
                      drop_row=drop_row,
                      drop_column=drop_column)


def print_windows(filename, sort_by=["sensitivity"], number_of_rows=25, filters=None, drop_row=None, drop_column=None):
    window_size = [128, 256, 512, 1024]
    for window in window_size:
        print_results(filename,
                      sort_by=sort_by,
                      number_of_rows=number_of_rows,
                      filters=[["window_size", window]],
                      drop_row=drop_row,
                      drop_column=drop_column)


def print_base_estimators_by_id(filename, ids):
    df = get_data(filename)

    for id in ids:
        print(f"\n\nPrinting parameters for id: {id}.")
        print("-"*100)
        print('\n')
        row = df.iloc[id, :]
        model_parameters = row["model_parameter"]
        model_parameters = re.sub(r'\s+', '', model_parameters)
        model_parameters = model_parameters.replace(",", ",\n").replace("),", ")\n").replace("[", "[\n")
        print(model_parameters)


if __name__ == '__main__':
    models = {
        "KNN": "<class 'pyod.models.knn.KNN'>",
        "ABOD": "<class 'pyod.models.abod.ABOD'>",
        "CBLOF": "<class 'pyod.models.cblof.CBLOF'>",
        "LOF": "<class 'pyod.models.lof.LOF'>",
        "HBOS": "<class 'pyod.models.hbos.HBOS'>",
        "OCSVM": "<class 'pyod.models.ocsvm.OCSVM'>"
    }

    minimal_movement = [0.02, 0.04, 0.1]
    pooling = ["max, mean"]
    pca = [0, 5, 10]
    window_sizes = [128, 256, 512, 1024]
    angles = ["shoulder", "hip", "knee", "elbow"]
    drop_parameters = [
        "minimal_movement",
        "noise_reduction",
        "bandwidth",
        "pooling",
        "sma",
        "window_overlap",
        "pca",
        "Unnamed: 0.1"
    ]

    file_path_basis = "results//model_search_kfold.csv"
    file_path_basis_groupBy = "results//model_search_kfold_groupBy.csv"
    file_path_ensemble = "results//ensemble_Search_kfold.csv"
    file_path_ensemble_groupBy = "results//ensemble_Search_kfold_groupBy.csv"

    file_path_tuned = "results//tuned_ensemble_search_kfold.csv"
    file_path_tuned_groupBy = "results//tuned_ensemble_search_kfold_groupBy.csv"

    file_path_parameter_search = "results//model_search_results_large_param_search.csv"

    file_path_novelty = "results//novelty_search.csv"
    file_path_novelty_groupBy = "results//novelty_search_groupBy.csv"

    file_path_alle_base_models_kfold = "results//alle_base_models_k_fold_groupBy.csv"

    for model in models:
        for angle in angles:
            for window_size in window_sizes:
                print_results(file_path_alle_base_models_kfold,
                              number_of_rows=5,
                              filters=[
                                  ["model", models[model]],
                                  ["angle", angle],
                                  ["window_size", window_size],
                                  ["minimal_movement", 0.1],
                                  ["bandwidth", 5],
                                  ["pooling", "mean"],
                                  ["sma", 3],
                                  ["window_overlap", 1],
                                  ["pca", 10]
                              ],
                              drop_column=drop_parameters)

    # print_angles(file_path_ensemble_groupBy, number_of_rows=10, drop_column=drop_parameters)

    # print_results(file_path_novelty_groupBy,
    #               number_of_rows=5,
    #               drop_column=drop_parameters)

    # for angle in angles:
    #     for window_size in window_sizes:
    #         print_results(file_path_novelty_groupBy,
    #                       number_of_rows=5,
    #                       filters=[["window_size", window_size], ["angle", angle]],
    #                       drop_column=drop_parameters)

            # print_results(file_path_parameter_search,
            #               number_of_rows=1,
            #               filters=[
            #                   ["minimal_movement", 0.1],
            #                   ["bandwidth", 5],
            #                   ["pooling", "mean"],
            #                   ["sma", 3],
            #                   ["window_overlap", 1],
            #                   ["pca", 10],
            #                   ["window_size", window_size],
            #                   ["angle", angle]
            #               ],
            #               drop_column=drop_parameters[0:len(drop_parameters)-1]
            #               )

    # print_base_estimators_by_id(file_path_ensemble_groupBy, [15])
    # print_angles(file_path_ensemble_groupBy, number_of_rows=5, drop_column=drop_parameters, sort_by=["model", "sensitivity"])

    # print_base_estimators_by_id(file_path_ensemble_groupBy, [15, 1615, 3, 1613, 13, 1609, 10, 1610, 3202, 4800, 3200, 8])



