import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Normalizing function; values between 0 and 1.
def normalize(x, min, max):
    return (x - min) / (max - min)


# Reading file
df = pd.read_csv('M3C.csv')

# Only data of Category FINANCE; indexes: 331 - 388
finance_df = df.loc[df['Category'] == "FINANCE     "]

# Subsetting only the timeseries; dropping every other column
finance_df = finance_df.iloc[:, list(range(6,53))]

# Iterating through every row, applying the normalizing function
for index, row in finance_df.iterrows():
    max_value = max(row)
    min_value = min(row)
    normalized_row = row.apply(normalize, min=min_value, max=max_value)

    # PLotting one normalized row; just to visualize the data
    normalized_row.plot()
    plt.show()
    break
