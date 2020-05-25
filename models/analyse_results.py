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

def average_results(file):
    results = pd.read_csv(file, index_col=0, engine="python")
    results = results.fillna(-1)
    results = results.groupby([
        "window_method",
        "window_sizes",
        "angle_method",
        "body_part_method",
        "threshold"
    ]).mean().reset_index()
    results.to_csv(file[0:-4] + "_groupBy.csv")

if __name__ == '__main__':
    average_results("results/report/xgbod.csv")
    average_results("results/report/lscp.csv")
    average_results("results/report/simple-max.csv")
    average_results("results/report/simple-mean.csv")
    print_results("results/report/xgbod_groupBy.csv", sort_by=["roc_auc"], number_of_rows=10)
    print_results("results/report/lscp_groupBy.csv", sort_by=["roc_auc"], number_of_rows=10)
    print_results("results/report/simple-max_groupBy.csv", sort_by=["roc_auc"], number_of_rows=10)
    print_results("results/report/simple-mean_groupBy.csv", sort_by=["roc_auc"], number_of_rows=10)



