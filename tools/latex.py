import pandas as pd


def get_data(filename):
    return pd.read_csv(filename, index_col=0)


def create_start(column_name):
    number = len(column_name)
    begin = ""
    columns = ""
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


def fix_parameter(parameter):
    parameter = parameter.replace("{", "")
    parameter = parameter.replace("}", "")
    parameter = parameter.replace("'", "")
    parameter = parameter.replace("n_", "")
    return parameter


# Collects the best results for each angle and each window
def create_table_data(filename, model):
    angles = ["shoulder", "hip", "knee", "elbow"]
    window_sizes = [128, 256, 512, 1024]
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
                model_parameter = fix_parameter(model_parameter)
            table_data.append({
                "angle": angle,
                "window_size": window_size,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "model_parameter": model_parameter
            })
    table_data = pd.DataFrame(table_data)
    return table_data


def convert_table_data_to_table(filename, model, parameter):
    df = create_table_data(filename, model)
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


def create_table(filename, base_models=False, parameter=False, ensemble=False):
    if ensemble:
        create_ensemble_model_tables()
    else:
        create_base_model_tables(filename, base_models, parameter)


def create_ensemble_model_tables():
    pass


def create_base_model_tables(filename, base_models, parameter):
    if base_models:
        models = {
            "KNN": "<class 'pyod.models.knn.KNN'>",
            "ABOD": "<class 'pyod.models.abod.ABOD'>",
            "LOF": "<class 'pyod.models.lof.LOF'>",
            "OCSVM": "<class 'pyod.models.ocsvm.OCSVM'>"
        }

    else:
        models = {
            "KNN": "<class 'pyod.models.knn.KNN'>",
            "ABOD": "<class 'pyod.models.abod.ABOD'>",
            "CBLOF": "<class 'pyod.models.cblof.CBLOF'>",
            "LOF": "<class 'pyod.models.lof.LOF'>",
            "HBOS": "<class 'pyod.models.hbos.HBOS'>",
            "OCSVM": "<class 'pyod.models.ocsvm.OCSVM'>"
        }

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
        table = create_start(columns) + convert_table_data_to_table(filename, models[model], parameter) + create_end(caption, label)
        to_txt(f"results_from_{model}" + suffix, table)


path_model_search = "C://Users//hloek//Desktop//Master//Kode//anomove//models//results//model_search_results_large_param_search.csv"
create_table(path_model_search, base_models=False, parameter=False, ensemble=False)

path_base_models = "C://Users//hloek//Desktop//Master//Kode//anomove//models//results//model_search_kfold_groupBy.csv"
create_table(path_base_models, base_models=True, parameter=True, ensemble=False)


