import pandas as pd


def get_data(filename):
    return pd.read_csv(filename, index_col=0)


def create_start(column_name, ensemble=False):
    number = len(column_name)
    begin = ""
    columns = ""
    if ensemble:
        begin= "m{4cm}|m{1.5cm}|m{2cm}|m{2cm}|m{2cm}|"
    else:
        for i in range(0, number):
            begin += "c|"

    for name in column_name:
        columns += "\\textbf{" + name + "} & "

    start = "\\begin{table}[H] \n" \
             "\t\\centering\n" \
             "\t\\rowcolors{1}{lavender2}{white}\n"
    begin = "\t\\begin{tabular}{" + begin[0:-1] + "}\n"
    columns = "\t\t" + columns[0:-2] + "\\\\\\hline\n"

    return start + begin + columns


def create_end(caption, label):
    footer = "\t\\end{tabular}\n" \
             "\t\\caption{" + caption + "}\n" \
                                        "\t\\label{tab:" + label + "}\n" \
                                                                   "\\end{table}"
    return footer


def end_line():
    return "\\\\\n"


def to_txt(filename, string):
    with open("files//" + filename + ".txt", "w") as text_file:
        text_file.write(string)


def fix_parameter(parameter, ensemble=False):
    methods = ["KNN", "ABOD", "LOF", "OCSVM"]
    if ensemble:
        parameter_list = parameter.split(",")
        parameter = ""
        for element in parameter_list:
            for method in methods:
                if method in element:
                    parameter += method + ": "
            if "n_neighbors" in element:
                parameter_neighbor = [int(s) for s in element.split("=")[1] if s.isdigit()]
                parameter += "neighbors = " + str(parameter_neighbor[0]) + " \\newline \n\t\t"
    else:
        parameter = parameter.replace("{", "")
        parameter = parameter.replace("}", "")
        parameter = parameter.replace("'", "")
        parameter = parameter.replace("n_", "")
    return parameter


# Collects the best results for each angle and each window
def create_table_data(filename, model, angles, window_sizes, ensemble=False):
    table_data = []
    decimals = 3

    for angle in angles:
        for window_size in window_sizes:
            df = get_data(filename)
            df = df.loc[df["model"] == model]
            df = df.loc[df["angle"] == angle]
            df = df.loc[df["window_size"] == window_size]
            df = df.sort_values(by="sensitivity", ascending=False)
            row = df.iloc[0]
            sensitivity = round(row["sensitivity"], decimals)
            specificity = round(row["specificity"], decimals)
            model_parameter = row["model_parameter"]

            if model_parameter == "{}":
                model_parameter = "Default"
            else:
                model_parameter = fix_parameter(model_parameter, ensemble)
            table_data.append({
                "angle": angle,
                "window_size": window_size,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "model_parameter": model_parameter
            })
    table_data = pd.DataFrame(table_data)
    return table_data


def convert_table_data_to_table(filename, model, angles, window_sizes, parameter=False, ensemble=False):
    df = create_table_data(filename, model, angles, window_sizes, ensemble)
    string = ""
    for idx, row in df.iterrows():
        angle = row["angle"]
        window = row["window_size"]
        sensitivity = row["sensitivity"]
        specificity = row["specificity"]
        model_parameter = row["model_parameter"]
        if parameter:
            string += "\t\t" + model_parameter + " & " + str(angle.capitalize()) + " & " + str(window) + " & " + str(sensitivity) + " & " + str(
                specificity) + end_line()
        else:
            string += "\t\t" + str(angle.capitalize()) + " & " + str(window) + " & " + str(sensitivity) + " & " + str(specificity) + end_line()
    return string


def create_ensemble_model_tables(filename, models, angles, window_sizes):
    columns = ["Parameter", "Angle", "Window size", "Sensitivity", "Specificity"]
    table = ""

    for model in models:
        for angle in angles:
            caption = "Result from Ensemble for " + angle + "."
            label = "results_from_ensemble_" + angle
            table += create_start(columns, ensemble=True) + \
                    convert_table_data_to_table(filename, models[model], angles=[angle], window_sizes=window_sizes, parameter=True, ensemble=True) + \
                    create_end(caption, label) + "\n\n"
        to_txt("results_from_ensemble", table)


def create_base_model_tables(filename, models, angles, window_sizes, parameter):
    if parameter:
        columns = ["Parameter", "Angle", "Window size", "Sensitivity", "Specificity"]
    else:
        columns = ["Angle", "Window size", "Sensitivity", "Specificity"]

    if "kfold" in filename:
        suffix = "_kfold"
        pre_caption = " with kfold = 5."
    else:
        suffix = ""
        pre_caption = "."

    for model in models:
        caption = f"Result from {model}" + pre_caption
        label = "results_from_" + model + suffix
        table = create_start(columns) + convert_table_data_to_table(filename, models[model], angles, window_sizes, parameter=parameter, ensemble=False) + create_end(caption, label)
        to_txt(f"results_from_{model}" + suffix, table)


def create_table(filename, models, angles, window_sizes, parameter=False, ensemble=False):
    if ensemble:
        create_ensemble_model_tables(filename, models, angles, window_sizes)
    else:
        create_base_model_tables(filename, models, angles, window_sizes, parameter)


if __name__ == '__main__':
    angles = ["shoulder", "hip", "knee", "elbow"]
    window_sizes = [128, 256, 512, 1024]

    ensemble_model = {
        "SDA": "<class 'combo.models.detector_comb.SimpleDetectorAggregator'>"
    }
    novelty_models = {
        "LOF_novelty": "<class 'sklearn.neighbors._lof.LocalOutlierFactor'>"
    }
    outlier_models = {
        "all":{
            "KNN": "<class 'pyod.models.knn.KNN'>",
            "ABOD": "<class 'pyod.models.abod.ABOD'>",
            "CBLOF": "<class 'pyod.models.cblof.CBLOF'>",
            "LOF": "<class 'pyod.models.lof.LOF'>",
            "HBOS": "<class 'pyod.models.hbos.HBOS'>",
            "OCSVM": "<class 'pyod.models.ocsvm.OCSVM'>"
        },
        "choosen":{
            "KNN": "<class 'pyod.models.knn.KNN'>",
            "ABOD": "<class 'pyod.models.abod.ABOD'>",
            "LOF": "<class 'pyod.models.lof.LOF'>",
            "OCSVM": "<class 'pyod.models.ocsvm.OCSVM'>"
        }
    }

    # path_model_search = "C://Users//hloek//Desktop//Master//Kode//anomove//models//results//model_search_results_large_param_search.csv"
    # create_table(path_model_search, parameter=False, ensemble=False)
    #
    # path_base_models = "C://Users//hloek//Desktop//Master//Kode//anomove//models//results//model_search_kfold_groupBy.csv"
    # create_table(path_base_models, parameter=True, ensemble=False)

    # path_ensemble = "C://Users//hloek//Desktop//Master//Kode//anomove//models//results//ensemble_search_kfold_groupBy.csv"
    # create_table(path_ensemble, ensemble_model, ensemble=True)

    path_novelty = "..//models//results//novelty_search_groupBy.csv"
    create_table(path_novelty, novelty_models, angles, window_sizes, parameter=True)
