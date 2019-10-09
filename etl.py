#!/usr/bin/env python
# coding: utf-8
import os
import pandas as pd


class ETL:

    def __init__(self, data_path):
        self.DATA_PATH = data_path
        self.META_PATH = os.path.join(self.DATA_PATH, "metadata.xls")

        metadata = pd.read_excel(self.META_PATH)
        metadata = metadata.drop(["CP score", "Unnamed: 5", "Unnamed: 6", "Unnamed: 7", "Unnamed: 8"], axis=1)
        metadata.rename(columns={"ID nummer": "id", "CP score": "score", "CP utkom": "cp"}, inplace=True)

        # Rename video ids to remove suffix
        # Either III or AAA_III where A is alphabetic and I is integers
        metadata["id"] = metadata["id"].apply(lambda name: name[:3] if name[0].isnumeric() else name[:7])
        self.metadata = metadata

    def get_cima(self):
        cima_files = []
        for root, dirs, files in os.walk(self.DATA_PATH):
            for filename in files:
                if filename[-4:] == ".csv":
                    cima_files.append(os.path.join(root, filename))

        cima = {}
        for file in cima_files:
            file_name = file.split(os.sep)[-1].split(".")[0]
            file_id = file_name[:3] if file_name[0].isnumeric() else file_name[:7]
            meta_row = self.metadata.loc[self.metadata["id"] == file_id]
            if meta_row.empty:
                continue
            cima[file_id] = {"data": pd.read_csv(file), "cp": meta_row.iloc[0]["cp"]}

        return cima
