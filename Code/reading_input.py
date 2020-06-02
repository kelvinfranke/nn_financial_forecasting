
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


# Normalizing function; values between 0 and 1.
def normalize(x, min, max):
    return (x - min) / (max - min)


# Function to create some random noise between start 0 and the 1 - mean to not get a value bigger than 1
# Maybe changes this: - probability curve of the slope of the series. Uit deze distributie halen we waardes
def random_noise():
    noise = random.uniform(-0.10, 0.10)
    return noise


def NaN_replacements(last_value, mean_slope):
    replacement = last_value + mean_slope + random_noise()
    return replacement

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

    final_series = row

    # NaN vervangen met de vorige waarde + mean_Slope + random_noise
    if row.hasnans:
        number_of_NaNs = row.isna().sum()
        # Index van de eerste NaN (column)
        column_last_value = len(row) - number_of_NaNs
        last_value = row.at[str(column_last_value)]

        new_row = row.dropna()

        filling_values = np.array([])
        # Mean slope of the series
        mean_slope = new_row.diff().mean()
        while number_of_NaNs > 0:
            replacement = NaN_replacements(last_value,mean_slope)
            filling_values = np.append(filling_values,replacement)
            number_of_NaNs -= 1
            last_value = replacement
        filling_series = pd.Series(filling_values)

        final_series = pd.concat([new_row,filling_series], ignore_index=True)

    # Normalizing the row
    normalized_row = final_series.apply(normalize, min=min(final_series), max=max(final_series))
    print(normalized_row)
    # Replacing NaN's with the mean with added random noise
    # normalized_row = normalized_row.fillna(normalized_row.mean() + random_noise())

    # Adding the new normalized, filled row to the final dataframe
    final_df = final_df.append(normalized_row, ignore_index=True)

    # Plotting one normalized row; just to visualize the data
    # normalized_row.plot()
    # plt.show()

print(final_df)
