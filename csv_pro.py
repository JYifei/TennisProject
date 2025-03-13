import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Read the CSV file and create a copy to avoid modifying the original data


def csv_processing(df):
    filled_df = df.copy()

    # List of coordinate columns to fill missing values (do not modify 'Frame')
    cols = ["Person1_X", "Person1_Y", "Person2_X", "Person2_Y", "Ball_Frame_X", "Ball_Frame_Y"]

    # For each coordinate column, fill in only the missing values using cubic interpolation
    for col in cols:
        mask = filled_df[col].notna()
        if mask.sum() >= 2:  # Perform interpolation only if there are at least 2 non-missing data points
            f_interp = interp1d(
                filled_df.loc[mask, 'Frame'],
                filled_df.loc[mask, col],
                kind='cubic',
                fill_value="extrapolate"
            )
            missing = filled_df[col].isna()
            filled_df.loc[missing, col] = f_interp(filled_df.loc[missing, 'Frame'])

    return filled_df  # Return the processed DataFrame instead of saving


# Generate plots to visualize the interpolation results

# # Plot for Person 1 (X and Y)
# plt.figure(figsize=(12, 6))
# plt.subplot(2, 1, 1)
# plt.plot(df['Frame'], df['Person1_X'], 'o', label='Original Person1_X')
# plt.plot(filled_df['Frame'], filled_df['Person1_X'], '-', label='Filled Person1_X')
# plt.xlabel('Frame')
# plt.ylabel('Person1_X')
# plt.title('Person1_X vs. Frame (Missing Values Filled)')
# plt.legend()

# plt.subplot(2, 1, 2)
# plt.plot(df['Frame'], df['Person1_Y'], 'o', label='Original Person1_Y')
# plt.plot(filled_df['Frame'], filled_df['Person1_Y'], '-', label='Filled Person1_Y')
# plt.xlabel('Frame')
# plt.ylabel('Person1_Y')
# plt.title('Person1_Y vs. Frame (Missing Values Filled)')
# plt.legend()

# plt.tight_layout()
# plt.show()

# # Plot for Person 2 (X and Y)
# plt.figure(figsize=(12, 6))
# plt.subplot(2, 1, 1)
# plt.plot(df['Frame'], df['Person2_X'], 'o', label='Original Person2_X')
# plt.plot(filled_df['Frame'], filled_df['Person2_X'], '-', label='Filled Person2_X')
# plt.xlabel('Frame')
# plt.ylabel('Person2_X')
# plt.title('Person2_X vs. Frame (Missing Values Filled)')
# plt.legend()

# plt.subplot(2, 1, 2)
# plt.plot(df['Frame'], df['Person2_Y'], 'o', label='Original Person2_Y')
# plt.plot(filled_df['Frame'], filled_df['Person2_Y'], '-', label='Filled Person2_Y')
# plt.xlabel('Frame')
# plt.ylabel('Person2_Y')
# plt.title('Person2_Y vs. Frame (Missing Values Filled)')
# plt.legend()

# plt.tight_layout()
# plt.show()

# # Plot for Ball (X and Y)
# plt.figure(figsize=(12, 6))
# plt.subplot(2, 1, 1)
# plt.plot(df['Frame'], df['Ball_Frame_X'], 'o', label='Original Ball_Frame_X')
# plt.plot(filled_df['Frame'], filled_df['Ball_Frame_X'], '-', label='Filled Ball_Frame_X')
# plt.xlabel('Frame')
# plt.ylabel('Ball_Frame_X')
# plt.title('Ball_Frame_X vs. Frame (Missing Values Filled)')
# plt.legend()

# plt.subplot(2, 1, 2)
# plt.plot(df['Frame'], df['Ball_Frame_Y'], 'o', label='Original Ball_Frame_Y')
# plt.plot(filled_df['Frame'], filled_df['Ball_Frame_Y'], '-', label='Filled Ball_Frame_Y')
# plt.xlabel('Frame')
# plt.ylabel('Ball_Frame_Y')
# plt.title('Ball_Frame_Y vs. Frame (Missing Values Filled)')
# plt.legend()

# plt.tight_layout()
# plt.show()
