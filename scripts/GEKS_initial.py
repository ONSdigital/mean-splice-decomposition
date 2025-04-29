import numpy as np
import pandas as pd
from GEKS_decomp_functions import *

# determine first period from input data
# specify folder with inputs and desired start_period (YYYY-MM-DD format)
data_in, first_date = read_dominicks_data("your\path\Dominicks\Oatmeal\woat", start_period="1992-01-01")

window_length = 25

# partial GEKS-T threshold
threshold = 1

# Calculate the end date of the first window
first_win_end = first_date + pd.DateOffset(months=window_length - 1)

# Filter the data based on the periods in the data
filtered_data = data_in[
    (data_in['period'] >= first_date) & (data_in['period'] < first_win_end)
]

# grouping columns
grouping_cols = ["region_code", "retailer", "consumption_segment_code"]

# make timetables
time_table = pd.DataFrame(data_in["period"]).drop_duplicates().sort_values(by="period")
time_table["time_count"] = np.arange(len(time_table))

# read date, format, filter out relavent times make time counter
# have to reset the index after filter to allow selection by vector position
month_vector = time_table.period.reset_index(drop=True)
time_vector = time_table.time_count.reset_index(drop=True)

# merge ttimetable with data
data_in = pd.merge(data_in, time_table, how="left", on="period")

# loop over first window to find contribs in first window
master_contrib = init_window_decomp(data_in, month_vector, window_length, grouping_cols, threshold)

# some column edits to for required schemas etc.
master_contrib = master_contrib.rename(columns={"time_count": "period"})
master_contrib["period"] = first_date + master_contrib["period"].apply(lambda x: pd.DateOffset(months=x))

# write output
# set append to False to overwrite file
master_contrib_file = write_master_contributions(master_contrib, "your\path\Dominicks\Oatmeal", append=False)
