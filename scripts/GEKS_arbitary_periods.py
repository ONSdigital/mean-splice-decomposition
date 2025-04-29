import pandas as pd
import numpy as np
from GEKS_decomp_functions import *

# read in input file
# update input file path
data_file, first_date = read_dominicks_data("your\path\Dominicks\Oatmeal\woat", start_period="1992-01-01")
master_contrib_file = read_master_contributions("your\path\Dominicks\Oatmeal")

# determine last month calcalcuted from previous GEKS_initial run
master_contrib_file["period"] = pd.to_datetime(master_contrib_file["period"])
last_date = master_contrib_file['period'].max().strftime('%Y-%m-%d')

print("last period calculated=" + last_date)

# finds the first period of the current range
current_period = pd.to_datetime(last_date).tz_localize('UTC') + pd.DateOffset(months=1)
format = '%Y-%m-%d'
current_period_str = current_period.strftime(format)
print("next period to calculate=" + current_period_str)

# settings
window_length = 25
threshold = 1

# Calculate contrib_upper_limit
contrib_upper_limit = current_period + pd.DateOffset(months=window_length - 1)
contrib_upper_limit = contrib_upper_limit.tz_convert('UTC')
print(contrib_upper_limit)

contrib_range = (current_period, contrib_upper_limit)

# grouping columns
grouping_cols = ["region_code", "retailer", "consumption_segment_code"]

# make global timetable for segment
data_end = pd.to_datetime(contrib_range[1], utc=True)

# the first period we need to load is the beginning of the window that ends on the first contrib selected
data_start = pd.to_datetime(contrib_range[0], utc=True)-pd.DateOffset(months=window_length-1)

# make global table of times for relavent ranges
time_table_global = pd.DataFrame({"period": pd.date_range(data_start, data_end, freq="MS")})
time_table_global["period"] = pd.to_datetime(time_table_global["period"]).dt.tz_convert('UTC')
time_table_global["time_count"] = np.arange(len(time_table_global))

# make vectors of time and month
month_vector_global = time_table_global.period.reset_index(drop=True)
time_vector_global = time_table_global.time_count.reset_index(drop=True)

# contributions we need to loop over
time_table_global["period"] = pd.to_datetime(time_table_global["period"]).dt.tz_convert('UTC')
contrib_range = (pd.to_datetime(contrib_range[0]).tz_convert('UTC'), pd.to_datetime(contrib_range[1]).tz_convert('UTC'))
contrib_time_table = time_table_global[time_table_global.period.between(contrib_range[0], contrib_range[1])]

contrib_months = contrib_time_table.period.reset_index(drop=True)

# read in the new data_in file appropiate for all times we are calculating
data_in = data_file[(data_file['period'] >= data_start) & (data_file['period'] <= data_end)]

# similarly, get all contrib data at once
contrib_data_end_date = pd.to_datetime(last_date, utc=True)
master_contrib = master_contrib_file[(master_contrib_file['period'] >= data_start) & (master_contrib_file['period'] <= contrib_data_end_date)]

# merge on the time counter
data_in = pd.merge(data_in, time_table_global, how="left",on="period")
master_contrib = pd.merge(master_contrib, time_table_global, how="left", on="period")

# loop over

for m in contrib_months:
    print(m)
    # find end date and time counter, this is just the datetime being looped over in contrib_months
    end_time = month_vector_global[month_vector_global == m].index[0]
    start_time = end_time-window_length+1

    contrib_end_time = end_time-1
    contrib_end_date = month_vector_global[contrib_end_time]

    # use recursive contributions function to find the new contribution, then append onto master contributions file
    append_contrib = recursive_contrib(data_in, master_contrib, end_time, window_length, month_vector_global, grouping_cols, threshold)
    master_contrib = pd.concat([master_contrib, append_contrib])


# drop time count column, filter out the new times in the contributions ranges, append
master_contrib = master_contrib.drop(["time_count"], axis=1)
master_contrib = master_contrib[master_contrib.period.between(pd.to_datetime(contrib_range[0], utc=True), data_end)]

# write output
master_contrib_file = write_master_contributions(master_contrib, "your\path\Dominicks\Oatmeal", append=True)
