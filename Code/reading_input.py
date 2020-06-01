import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


# Normalizing function; values between 0 and 1.
def normalize(x, min, max):
    return (x - min) / (max - min)


# Function to create some random noise between start 0 and the 1 - mean to not get a value bigger than 1
# Needs work, different method
def random_noise():
    noise = random.uniform(-0.10, 0.20)
    return noise


# Reading file
df = pd.read_csv('M3C.csv')

# Only data of Category FINANCE; indexes: 331 - 388
finance_df = df.loc[df['Category'] == "FINANCE     "]

# Subsetting only the timeseries; dropping every other column
finance_df = finance_df.iloc[:, list(range(6, 53))]

final_df = pd.DataFrame()

# Iterating through every row, applying the normalizing function
for index, row in finance_df.iterrows():
    max_value = max(row)
    min_value = min(row)

    # Normalizing the row
    normalized_row = row.apply(normalize, min=min_value, max=max_value)

    # Replacing NaN's with the mean with added random noise
    normalized_row = normalized_row.fillna(normalized_row.mean() + random_noise())

    # Adding the new normalized, filled row to the final dataframe
    final_df = final_df.append(normalized_row, ignore_index=True)

    # Plotting one normalized row; just to visualize the data
    # normalized_row.plot()
    # plt.show()

print(final_df)
