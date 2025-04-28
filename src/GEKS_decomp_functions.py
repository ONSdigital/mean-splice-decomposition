import numpy as np
import pandas as pd
import os


def contrib_table(data_cut, start, flag_start, grouping_cols):
    """Calculates contributions for tornqvists pairs that make up GEKS-T. Uses inner join to cycle through required
    combinations of time periods (tornqvist) without using a loop.  
    Just take the start and end periods, inner join products + aggregation groups onto entire data window
    Inner join ensures start and end tornqivsts are evaluated using matched pairs only
    If we group by the time column of the right part of the join, we can calculate the weights for each item
    for each period base we have to cycle through simultaneously

    Calc average weight for tornqvist, evaluate (p(k)/p(start))^weight avg for each product
    flag_start parameter = end flips the price fraction to create (p(end)/p(k))^weight avg for each product for the inverted "end" tornqvist,
    used in each tornqvist pair in GEKS-T

    Parameters
    ----------
    data_cut: dataframe
    The input dataframe. Requires, price and quantity data, grouping variables to define aggregation groupings, period and time counter.
    Requires only times in the window be included in data inputs (assumes non-window time periods have been filtered out) 
    start: integer
    The tornqvist "start" time in time counter terms (a sequential numerical index representing time). When flag_start is false, wants to be
    the end time of the second tornqvist pair. 
    flag_start: True/False
    True used for when calculating first tornqvists in pair, False for end tornqvist in pairs
    grouping_cols: list of strings
    list of index grouping variables (aggregation groupings)

    Returns
    --------
    data_start: dataframe
    A dataframe of partially calculated contribution components (still needs to have geomean applied over window) for each product component of each index. 
    """

    # set up filtering groups

    unique_id_cols = grouping_cols + ["product_id"]
    grouping_cols_time = grouping_cols + ["period_y", "time_count_y"]
    output_colnames = grouping_cols_time + ["product_id", "contrib_start"]

    # join start period to all other periods to cycle through different weight bases, much faster than a loop!
    # inner join so that matched products only are incorporated. Periods with no matches don't show up
  
    data_start = data_cut.loc[data_cut.time_count == start].copy()
    data_start = pd.merge(data_start, data_cut, how="inner", on=unique_id_cols)

    # calculate turnovers and average weights, get calculations using price relatives + weights
  
    data_start["turnover_x"] = data_start.quantity_x * data_start.price_x
    data_start["turnover_y"] = data_start.quantity_y * data_start.price_y
    data_start["tot_turnover_x"] = data_start.groupby(grouping_cols_time)[["turnover_x"]].transform("sum")
    data_start["tot_turnover_y"] = data_start.groupby(grouping_cols_time)[["turnover_y"]].transform("sum")
    data_start["weight"] = ((data_start.turnover_x / data_start.tot_turnover_x) + (data_start.turnover_y / data_start.tot_turnover_y))/2
    data_start["contrib_start"] = np.power((data_start["price_y"] / data_start["price_x"]), data_start["weight"])
    data_start = data_start[output_colnames]
    
    # invert the contribution bit if function used for "end" tornqvist, second tornqvist is 
    # cycled weight base to targed end of index
    
    if flag_start == False:
        data_start = data_start.rename(columns={"contrib_start": "contrib_end"})
        data_start["contrib_end"] = 1 / data_start["contrib_end"]
    return (data_start)


def GEKS_T_contrib(data_in, window, start_end, threshold, grouping_cols):
    """Code that calculates a contribution for products into GEKS-T indices that are defined by aggregation groupings. 
    Works for a single window GEKS-T between a start and end value for time counter. 
    Does multiple indices (aggregation groupings) simultaneously.
    Requires a time window to be specified in date/period form, and a vector of start/end times in time counter form.
    Expects a time count column (a sequential numerical index representing time) to be attached to the input dataframe.
    Implements a partial GEKS-T proctocol for cases where tornqvists in the GEKS-T with no matched products exist.
    In that case, that pair is thrown out of the geometric mean of tornqvist pair and a geometric mean is taken over the remaining pairs
    with matched products. If there are less GEKS-T pairs than the specified threshold, returns NaN as the contribution (to match pipeline behvaiour)
    Calls previously defined contrib function to calculate start and end components of GEKS-T pairs.

    Parameters
    ----------
    data_in: dataframe
    Input dataframe with time counter, period, grouping variables + product ID, price and quantity data.
    window: a size 2 list of periods (datetime format)
    Start and end times of the timke window in datetime format (needs to match format of period variable in input).
    start_end: a size 2 vector of integers
    This has the reference period and target end period of the index in time counter format.
    threshold: integer
    This is the number of valid matched tornqvist pairs required for index to be deemed acceptable. 
    grouping_cols: list of strings
    list of index grouping variables (aggregation groupings)

    Returns
    --------
    all_contrib: dataframe
    Dataframe of contributions of products into aggregation groupings for the index between start and end times with input data window.
    """
    # read date, format, filter
    # expects a global time count column (starting at 0, ending at however many periods there are -1)
    data_in["period"] = pd.to_datetime(data_in["period"])
    data_cut = data_in[data_in.period.between(window[0], window[1])]
  
    time_table = data_cut[["period", "time_count"]].drop_duplicates().sort_values(by="time_count")

    # calls code that calculates contribution table (defined previously)       
    data_start = contrib_table(data_cut, start_end[0], True, grouping_cols)
    data_end = contrib_table(data_cut, start_end[1], False, grouping_cols)
    
    # check if we have periods with no matched products
    # find all times for all groups, need to a table of every agg group matched with every time period in window
    window_length = len(time_table)
    all_times = data_cut[grouping_cols].drop_duplicates()
    group_count = len(all_times)
    all_times = pd.DataFrame(np.repeat(all_times.values, window_length, axis = 0), columns = all_times.columns)
    all_times["time_count_y"] = np.tile(time_table["time_count"], group_count)
    
    grouping_cols_time = grouping_cols + ["period_y", "time_count_y"]

    # find all times in contrib tables with matches in "start" and "end"

    actual_times_start = data_start[grouping_cols_time].drop(["period_y"], axis=1).drop_duplicates().assign(matched=True)
    actual_times_end = data_end[grouping_cols_time].drop(["period_y"], axis=1).drop_duplicates().assign(matched=True)
    
    # merge times with matches, assign unmatched status if the time is missing in start or end
    # calculate "actual" window length, the number of periods where both tornqvist in pair have matched products
    # i.e. allow excecution of partial GEKS-T 
    
    time_check_cols_y = grouping_cols + ["time_count_y"]
    all_times = pd.merge(all_times, actual_times_start, how="left", on = time_check_cols_y)
    all_times = pd.merge(all_times, actual_times_end, how="left", on=time_check_cols_y)
    all_times["matched"] = np.where((all_times.matched_x.isna()) | (all_times.matched_y.isna()), False, True)
    all_times = all_times.drop(["matched_x", "matched_y"], axis=1)
    all_times["actual_window_length"] = all_times.groupby(grouping_cols)[["matched"]].transform("sum")
    
    # total contribution is a product of all bits due to individual products from tonrqvists at start and end
    # join start and end tables onto each other, make product and apply geomean
    # outer join to avoid throwing away products that are never matched in the "start" table, same for end table
    # those products have some matched pairs in the "start" or "end" tornqivsts, but not both,. 
    # when absent, they have zero weight have contribution 1 for that tornvqvist
    # if at least valid one matched pair exists there will be a contribution
    
    unique_id_time = grouping_cols + ["period_y", "time_count_y", "product_id"]
    unique_id_cols = grouping_cols + ["product_id"]
    final_cols = unique_id_time + ["contrib", "actual_window_length"]
    
    # outer join first to include contributions for products that are in one of the start or end tornqvists only
    # then join onto all possibe times and agg indices to see what pairs are matched

    data_start = pd.merge(data_start, data_end, how="outer", on=unique_id_time)
    data_start = pd.merge(all_times, data_start, how="left", on=time_check_cols_y)

    # drop all those where matched is false - these will be where only one of the start or end 
    # tornqvists is unmatched, we want to drop these weight bases entirely
    # those where both the start end tornqvists are unmatched won't even show up, so don't need to be considered
    # set NaN contribs to 1 (see above)
    # NaN contributions (GEKS-T threshold failure) are effectively imputed as 1, rest of contributions are still comparable if we multiply
    # all by a constant

    data_start = data_start.loc[data_start.matched == True]
    data_start["contrib_start"] = np.where(data_start.contrib_start.isna(), 1, data_start.contrib_start)
    data_start["contrib_end"] = np.where(data_start.contrib_end.isna(), 1, data_start.contrib_end)

    # combine and apply geomean factor

    data_start["contrib"] = data_start["contrib_start"] * data_start["contrib_end"]
    data_start = data_start[final_cols]
    data_start = data_start.rename(columns = {"period_y":"period", "time_count_y":"time_count"})

    # final calculation
    # data_start["contrib"]=data_start.contrib**(1/data_start.actual_window_length)
    # threshold for partial GEKS-T?
    # we want to function spit out a NaN when the threshold fails rather than 1, set to nan all products
    # for groupings that fail partial geks-t threshold

    data_start["contrib"] = np.where(data_start.actual_window_length < threshold, np.nan, data_start.contrib ** (1/data_start.actual_window_length))
    all_contrib = data_start.groupby(unique_id_cols)["contrib"].prod(min_count=1).reset_index()

    # need to have the correct column names

    all_contrib = all_contrib.rename(columns={"contrib": "contributions"})
    
    return all_contrib


def recursive_contrib(data_in, master_contrib, contrib_time, window_length, month_vector, grouping_cols, threshold):
    """Function that recursively calls the base GEKS-T window index contributions function to recursively calculate the contribution by
    product into mean spliced on publsihed GEKS-T index as defined by grouping variables. Takes as input previous contributions history 
    needed to apply the mean splice formula, and calculates the required GEKS-T window indices needed to perform the splice for the period 
    by calling the window GEKS-T contributions function repeatedly. Applies the mean splice formula to the contribution to get the contribution 
    by product to a mean spliced on published GEKS-T as defined by the grouping variables. 
    Excecutes for multiple indices and products simultaneously.

    Parameters
    ----------
    data_in: dataframe
    Input dataframe with time counter, period, grouping variables + product ID, price and quantity data.
    master_contrib: dataframe
    A dataframe that contains the contributions to the published index for the previous times in the most recent window.
    window_length: integer
    Length of data/index window. 
    contrib_time: integer
    The time counter value for the current period we are calculating the contribution for in the dataframe. Needs to lineup with month vector
    used below (time count = 0 is first period in that series, etc.)
    month_vector: series
    A series of datetimes for all periods from the start of the overlap window up to at least the current period we are calculating the contribution for
    Probably just use series of all datetimes in source data.
    grouping_cols: list of strings
    list of index grouping variables (aggregation groupings)
    threshold: integer
    This is the number of valid matched tornqvist pairs required for index to be deemed acceptable. 

    Returns
    --------
    temp_contrib: dataframe
    Dataframe of contributions of products into aggregation groupings for the mean spliced on published index at contrib_time and 
    contribution history and price/quantity data used as inputs
    """
    # overlap vector, arange is weird and doesn't include the end of the range?

    overlap = pd.Series(np.arange(contrib_time - window_length + 1, contrib_time))
    
    for o in overlap:
        print(o)
        contrib = GEKS_T_contrib(data_in, [month_vector[overlap[0]], month_vector[contrib_time]], (o, contrib_time), threshold, grouping_cols)
        contrib["time_count"] = o
        if o == overlap[0]:
            temp_contrib = contrib.copy()
        else:
            temp_contrib = pd.concat([temp_contrib, contrib])

    # group by to find product of contribs by agg group/product

    unique_id = grouping_cols + ["product_id"]
    temp_contrib = temp_contrib.groupby(unique_id)["contributions"].prod().reset_index()
    
    # contributions for missing GEKS-T need to be filled in here - fill in with 1s? Might not need if will fill
    # with 1s
    # temp_contrib["contrib"]=np.where(temp_contrib.contrib.isna(),1,temp_contrib.contrib)
    
    # get relavent periods from previous contributions table, group by and product

    pub_contrib = master_contrib.loc[master_contrib.time_count.isin(overlap)]
    pub_contrib = pub_contrib.groupby(unique_id)["contributions"].prod().reset_index()
    
    # multiply the two together by row and take average over overlap period, 
    # merge onto old master contrib file
    # need outer merge to stop items that exist in published contributions only from being dropped

    temp_contrib = pd.merge(temp_contrib, pub_contrib, how="outer", on=unique_id)

    # set nan to 1 allow multiplication to work
    # if both pub and new spliced contributions are NaN then we want to keep one NaN to make operation return a NaN
    # otherwise set 1 to NaN
    # we can just keep contrib_x as NaN if both are NaN for some reason, will make overall calculation be a NaN

    temp_contrib["contributions_x"] = np.where(
        ((temp_contrib.contributions_x.isna()) & (temp_contrib.contributions_y.isna())) |  (temp_contrib.contributions_x.notna()),
        temp_contrib.contributions_x,
        1
        )
    temp_contrib["contributions_y"] = np.where(temp_contrib.contributions_y.isna(), 1, temp_contrib.contributions_y)
    temp_contrib["contributions"] = (temp_contrib.contributions_x * temp_contrib.contributions_y) ** (1 / (window_length - 1))
    
    # some column name fiddling
    temp_contrib = temp_contrib.drop(["contributions_x", "contributions_y"], axis=1)
    temp_contrib["period"] = month_vector[contrib_time]
    temp_contrib["time_count"] = contrib_time
            
    return temp_contrib

def init_window_decomp(data_in, month_vector, window_length, grouping_cols, threshold):

    # find initial window
    init_window = time_vector[time_vector <= (window_length-1)]

    # loop over the first window
    for t in init_window:
        print(t)
        contrib = GEKS_T_contrib(data_in, [month_vector[0], month_vector[window_length-1]],(0, t), threshold, grouping_cols)
        contrib["time_count"]=time_vector[t]
        if t==0:
            master_contrib = contrib.copy()
        else:
            master_contrib = pd.concat([master_contrib, contrib])

    return master_contrib



def read_dominicks_data(data_folder, start_period=None):
    """
    Code that reads and processes Dominick's data from multiple CSV files, merges them, and prepares the data for further analysis.
    The function loads data from the specified folder, merges it based on common columns, and performs necessary transformations
    to align with the processed data schema.

    Parameters
    ----------
    data_folder: str
        Path to the folder containing the CSV files.
    start_period: str, optional
        The starting period to use for the data. If None, the earliest date in the data will be used.

    Returns
    -------
    dominicks_data: dataframe
        Dataframe containing processed Dominick's data with columns for product ID, price, consumption segment code, retailer,
        region code, period, quantity, run_id, and month.
    first_date: datetime
        The starting period used for the data.
    """

    # Load Dominick's data
    dominicks_movement = os.path.join(data_folder, 'WOAT.csv')
    dominicks_upc = os.path.join(data_folder, 'upcoat.csv')
    dominicks_stores = os.path.join(data_folder, 'dominicks_stores.csv')
    dominicks_weeks = os.path.join(data_folder, 'dominicks_weeks.csv')

    dominicks_movement = pd.read_csv(dominicks_movement)
    dominicks_upc = pd.read_csv(dominicks_upc)
    dominicks_stores = pd.read_csv(dominicks_stores)
    dominicks_weeks = pd.read_csv(dominicks_weeks)

    # Drop rows where MOVE = 0
    dominicks_movement = dominicks_movement[dominicks_movement['MOVE'] != 0]

    # Merge Dominick's data to give conseg and region equivalents
    # for conseg, COM_CODE from dominicks_upc input used
    # for region, zone is used from dominicks_stores input
    dominicks_merged = pd.merge(dominicks_movement, dominicks_upc, on='UPC')
    dominicks_merged = pd.merge(dominicks_merged, dominicks_stores, on='STORE')
    dominicks_merged = pd.merge(dominicks_merged, dominicks_weeks, on='WEEK')
    
    # Calculate price (PRICE/QTY)
    dominicks_merged['price'] = dominicks_merged['PRICE'] / dominicks_merged['QTY']

    # Rename columns to match processed data
    dominicks_merged = dominicks_merged.rename(columns={
        'NITEM': 'product_id',
        'COM_CODE': 'consumption_segment_code',
        'STORE': 'retailer',
        'ZONE': 'region_code',
        'START': 'period',
        'MOVE': 'quantity'
    })
    
    # drop NA cases in region code and retailer
    # could also set to some filler value
    dominicks_merged.dropna(axis = 0, how = 'any', inplace = True, subset = ['region_code','retailer'])

    # Align with processed schema by adding a run_id column and dropping columns
    dominicks_merged['run_id'] = 1

    dominicks_data = dominicks_merged[[
        'product_id', 'price',
        'consumption_segment_code', 'retailer',
        'region_code', 'period', 'quantity', 'run_id'
    ]]
    
    
    # Convert period to datetime64[ns, UTC]
    dominicks_data['period'] = pd.to_datetime(dominicks_data['period'], format='%m/%d/%Y').dt.tz_localize('UTC')
    
    # Add a month column in YYYY/MM/DD format
    #dominicks_data['month'] = dominicks_data['period'].dt.to_period('M').dt.to_timestamp().dt.strftime('%Y/%m/01')
    dominicks_data.loc[:,'month'] = (dominicks_data.loc[:,'period'].dt.floor('d') + 
                               pd.offsets.MonthEnd(0) - pd.offsets.MonthBegin(1))
    
    # need to sum quantities and average prices across each month
    dominicks_data.loc[:,'exp']  = dominicks_data.loc[:,'price'] * dominicks_data.loc[:,'quantity']
    agg_functions = {'exp':'sum','quantity':'sum'}
    dominicks_data = dominicks_data.groupby(['month',
                                             'region_code',
                                             'retailer',
                                             'product_id',
                                             'consumption_segment_code',
                                             'run_id']).aggregate(agg_functions)
    dominicks_data = dominicks_data.reset_index()
    dominicks_data.loc[:,'price'] = dominicks_data.loc[:,'exp'] /  dominicks_data.loc[:,'quantity']
    dominicks_data.loc[:,'period'] = dominicks_data.loc[:,'month']
    
    dominicks_data = dominicks_data[[
        'product_id', 'price',
        'consumption_segment_code', 'retailer',
        'region_code', 'period', 'month', 'quantity', 'run_id'
    ]]
    
    # dtypes to match processed schema
    dominicks_data = dominicks_data.astype({
        'consumption_segment_code': 'str',
        'region_code': 'str',
        'retailer': 'str',
        'product_id': 'str',
        'price': 'float',
        'quantity': 'float',
        'run_id': 'Int64'
    })

    # Determine the starting period and drop data prior to that
    if start_period:
        first_date = pd.to_datetime(start_period).tz_localize('UTC')
        dominicks_data = dominicks_data[dominicks_data['period']>=first_date]
        
    else:
        first_date = pd.to_datetime(dominicks_data['month'].min())


    return dominicks_data, first_date


def write_master_contributions(df, output_folder, append=False):
    """
    Code that writes the contributions DataFrame to a CSV file in the specified folder. 
    The function can either append to an existing file or overwrite it based on the append flag.

    Parameters
    ----------
    df: dataframe
        DataFrame containing the contributions data to be written to the CSV file.
    output_folder: str
        Path to the folder where the CSV file will be saved.
    append: boolean, optional (default=False)
        If True, appends the data to an existing file. If False, overwrites the existing file.

    Returns
    -------
    None
    """
    # Define file path
    file_path = os.path.join(output_folder, 'master_contributions.csv')

    # Write the DataFrame to a CSV file
    if append and os.path.exists(file_path):
        df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        df.to_csv(file_path, index=False)

    print(f"File saved to {file_path}")


def read_master_contributions(input_folder):
    """
    Code that reads the contributions DataFrame from a CSV file in the specified folder.
    The function loads the data and ensures the correct data types for each column.

    Parameters
    ----------
    input_folder: str
        Path to the folder containing the CSV file.

    Returns
    -------
    df: dataframe
        DataFrame containing the contributions data read from the CSV file.
    """
    # Define file path
    file_path = os.path.join(input_folder, 'master_contributions.csv')

    # Read the DataFrame
    df = pd.read_csv(file_path, dtype={
        'region_code': 'object',
        'retailer': 'object',
        'consumption_segment_code': 'object',
        'product_id': 'object',
        'contributions': 'float64',
    })

    return df


