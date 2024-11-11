import os.path

import pandas as pd


def txt_to_csv(path):
    with open(path, 'r') as file:
        row = {}
        index = 0
        for line in file:
            line = line.strip()
            # Split only at the first ':'
            parts = line.split(":", 1)
            if len(parts) > 1:
                column_name = parts[0].strip()
                value = parts[1].strip()
                row[column_name] = value
            else:
                df = pd.DataFrame(data=row, index=[index])
                index += 1
                row = {}
                new_path = path.replace(".txt", ".csv")
                header = False if os.path.exists(new_path) else True
                df.to_csv(new_path, mode='a', index=False, header=header)


txt_to_csv("BeerAdvocate/ratings.txt")
