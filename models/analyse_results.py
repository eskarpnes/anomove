import pandas as pd
import os

def get_data(filename):
    return pd.read_csv(filename)


def print_results(filename):
    df = get_data(filename)
    df = df.drop(columns=["Unnamed: 0"])
    # Sort by sensitivity and print
    with pd.option_context('display.max_rows', None, "display.expand_frame_repr", False):
        print('\n Printing the results from file: ' + str(filename) + "\n")

        print("\n Sorted by sensitivity and then specificity \n")
        print(df.sort_values(by=["sensitivity", "specificity"], ascending=False))

        # Sort by specificity and print
        print("\n Sorted by specificity and then sensitivity \n")
        print(df.sort_values(by=["specificity", "sensitivity"], ascending=False))
