import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


class ReadingInput:
    # Normalizing function; values between 0 and 1.
    def normalize(self, x, min, max):
        return (x - min) / (max - min)

    def reverse_normalize(self, x, min, max):
        return (x * (max - min)) + min

    # Function to create some random noise between start 0 and the 1 - mean to not get a value bigger than 1
    # Maybe changes this: - probability curve of the slope of the series. Uit deze distributie halen we waardes
    def random_noise(self):
        noise = random.uniform(-0.10, 0.10)
        return noise

    def NaN_replacements(self, last_value, mean_slope):
        trend = random.uniform(-2, 2)
        replacement = last_value + trend * mean_slope + self.random_noise()
        if replacement > 1:
            return 1
        if replacement < -1:
            return -1
        return replacement
    
    # Exponential moving averager
    def averager(self, row):
        days = 14
        smoothing = 2.1
        averaged_row = np.array([])
        new_value = 0
        for value in row:
            new_value = (value * (smoothing/(1+days))) + new_value*(1-(smoothing/(1+days)))
            averaged_row = np.append(averaged_row, [new_value])
        return averaged_row

    def process_data(self):
        # Reading file
        df = pd.read_csv('M3C.csv')

        # Only data of Category FINANCE; indexes: 331 - 388
        # finance_df = df.loc[df['Category'] == "FINANCE     "]
        finance_df = df.loc[df['Category'] == "MICRO       "]

        # Subsetting only the ti  print(np.shape(final_df))meseries; dropping every other column
        # finance_df = finance_df.iloc[:, list(range(6, 53))]
        finance_df = finance_df.iloc[:, list(range(6, 27))]

        # final_df = np.zeros([59, 46])
        final_df = np.zeros([147, 20])
        timeseries = 0
        # Iterating through every row, applying the normalizing function
        for index, row in finance_df.iterrows():
            timeseries += 1
            final_series = row
            # Length = 20
            final_series = final_series.to_numpy()
            # Detrending
            detrended_row = np.diff(final_series)
            # Length = 19

            # Averaging
            averaged_row = self.averager(detrended_row)
            # Length = 19

            normalized_row = self.normalize(detrended_row, min(detrended_row), max(detrended_row))
            normalized_averaged_row = self.normalize(averaged_row, min(averaged_row), max(averaged_row))

            final_df[timeseries, :] = normalized_row
            #self.plot(final_df, timeseries)
        return(final_df)

    def plot(self, final_df, timeseries):
        # Plotting one normalized row; just to visualize the data
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(np.linspace(0, len(final_df[timeseries, :]), len(final_df[timeseries, :])), final_df[timeseries, :])
        plt.show()


if __name__ == "__main__":
    reading_input = ReadingInput()
    reading_input.process_data()
