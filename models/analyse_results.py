import pandas as pd
import numpy as np


def get_data(filename):
    DATA_PATH = "C://Users//haavalo//Desktop//Master//anomove//models//" + filename
    return pd.read_csv(DATA_PATH)


def print_results(filename):
    df = get_data(filename)

    # Sort by sensitivity and print
    with pd.option_context('display.max_rows', None, "display.expand_frame_repr", False):
        print('\n Printing the results from file: ' + str(filename) + "\n")

        print("\n Sorted by sensitivity and then specificity \n")
        print(df.sort_values(by=["sensitivity", "specificity"]))

        # Sort by specificity and print
        print("\n Sorted by specificity and then sensitivity \n")
        print(df.sort_values(by=["specificity", "sensitivity"]))
