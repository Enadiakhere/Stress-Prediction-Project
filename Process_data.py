import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# Constants
start = 1
end = 2
base_url = "https://raw.githubusercontent.com/italha-d/Stress-Predict-Dataset/main/Raw_data/"

# Function to add datetime to the dataframe based on its frequency
def add_datetime_to_df(df, column_name):
    df["Datetime"] = datetime.utcfromtimestamp(df.loc[0, column_name])
    frequency = df.loc[1, column_name]
    df = df.reset_index(drop=True)
    for k, _ in df.iterrows():
        df["Datetime"].iloc[k] += timedelta(seconds=k * (1 / frequency))
    return df

# Reading the 35 datasets into variables
for i in range(start, end):
    # Construct the URL based on the loop variable
    url_format = f"{base_url}S{'0' if i < 10 else ''}{i}/{{}}.csv"
    
    sensors = {
        "acc": ["x", "y", "z"],
        "bvp": ["bvp"],
        "eda": ["eda"],
        "temp": ["temp"]
    }
    
    for sensor, columns in sensors.items():
        url = url_format.format(sensor.upper())
        df = pd.read_csv(url, header=None, names=columns)
        globals()[f'{sensor}{i}'] = add_datetime_to_df(df, columns[0])



log = pd.read_excel('https://github.com/italha-d/Stress-Predict-Dataset/raw/main/Processed_data/Time_logs.xlsx', sheet_name=0)


# Function to adjust the time
def adjust_time(time_str):
    # Split the time string into hours, minutes, and potentially seconds
    parts = time_str.split(':')
    
    # Check if the entry has the expected time format
    if len(parts) < 2:
        return time_str

    hours, minutes = int(parts[0]), int(parts[1])
    
    # Checking if hours are between 1 and 5 and adjusting
    if 1 <= hours <= 5:
        hours += 12
    
    # Returning the adjusted time string without seconds
    return f"{hours}:{minutes:02d}"

# Iterate through columns 2 to 24 (excluding the first and the last two columns)
for col in log.columns[1:-2]:
    # Applying the adjust_time function to rows 3 to 37 for each of those columns
    log.loc[1:36, col] = log[col][1:37].apply(lambda x: adjust_time(str(x)) if pd.notna(x) else x)



    from datetime import datetime

def get_time_log(col, id):
    """
    Fetch the date and time from a log based on an ID and combine them into a single datetime.
    
    Parameters:
    - col: Name of the column in the log dataframe containing the time.
    - id: ID of the row in the log dataframe from which to fetch the date and time.
    
    Returns:
    - A datetime object representing the combined date and time.
    """
    time_str = str(log.loc[log["S. ID."]==id, col].iloc[0])
    date_str = log.loc[log["S. ID."]==id, "Date"].dt.date.astype(str).iloc[0]
    
    return pd.to_datetime(f"{date_str} {time_str}", format="%Y-%m-%d %H:%M:%S.%f")


def time_utc_timestamp_value(timval):
    """
    Convert a datetime object to a timestamp value.
    
    Parameters:
    - timval: The datetime object to be converted.
    
    Returns:
    - A float representing the timestamp value of the datetime object.
    """
    return timval.timestamp()


def utc_value(tim):
    """
    Convert a string representation of a datetime to a timestamp value.
    
    Parameters:
    - tim: A string representing a datetime in the format '%Y-%m-%d %H:%M:%S.%f'.
    
    Returns:
    - A float representing the timestamp value of the datetime.
    """
    return datetime.strptime(tim, '%Y-%m-%d %H:%M:%S.%f').timestamp()


def use_time_to_frame(dataframe, start_time, end_time):
    """
    Filter rows of a dataframe based on a time range.
    
    Parameters:
    - dataframe: The input dataframe. Assumes there's a 'Datetime' column of datetime objects.
    - start_time: The starting timestamp value of the time range.
    - end_time: The ending timestamp value of the time range.
    
    Returns:
    - A dataframe containing only rows where the 'Datetime' value falls within the specified time range.
    """
    mask = dataframe['Datetime'].apply(lambda x: start_time <= x.timestamp() < end_time)
    
    return dataframe[mask].copy()


def label_data_with_resampling(result_df, person=None, col_name=None):
    """
    Process a dataframe by setting the index to 'Datetime', dropping the 'Datetime' column,
    resampling data at 0.25 second intervals, and optionally adding a 'Person' column.
    
    Parameters:
    - result_df: Input dataframe with a 'Datetime' column.
    - person (optional): If provided, a new 'Person' column will be added to the dataframe.
    - col_name (optional): Name of the column to be resampled.
                          If not provided, it's assumed the dataframe contains 'x', 'y', and 'z' columns.
                          
    Returns:
    - A processed dataframe.
    """
    result_df.set_index('Datetime', inplace=True)
    
    if col_name:  # If a column name is given, resample only that column.
        result_df[col_name] = result_df[col_name].resample('0.25S').mean()
    else:  # If no column name is given, resample the 'x', 'y', and 'z' columns.
        for axis in ['x', 'y', 'z']:
            result_df[axis] = result_df[axis].resample('0.25S').mean()
    
    result_df.dropna(inplace=True)
    
    if person:  # If a person ID is provided, add it to the dataframe.
        result_df['Person'] = person

    return result_df

def Label_Acc(result_df, person, label):
    """
    Function for labeling accelerometer data.
    
    Note: The 'label' argument is currently not being used inside this function 
    based on the original code you provided, but it's added here to match the required function signature.
    If you intend to use it inside the function, additional code can be added.
    """
    return label_data_with_resampling(result_df, person)

def Label_BVP_EDA_TEMP(result_dff, col_name, label):
    """
    Function for labeling other types of data.
    
    Note: The 'label' argument is currently not being used inside this function 
    based on the original code you provided, but it's added here to match the required function signature.
    If you intend to use it inside the function, additional code can be added.
    """
    return label_data_with_resampling(result_dff, col_name=col_name)



def process_data(event_name, unnamed_col_name, data_suffix, label):
    # Loop over a predefined range of indices.
    for i in range(start, end):
        
        # Generate subject ID, e.g., "S01", "S02", ..., "S10", ...
        s_id = f"S{'0' + str(i) if i < 10 else i}"
        
        # Calculate the start and end timestamps based on provided event names and subject ID.
        start_timestamp = time_utc_timestamp_value(get_time_log(event_name, s_id))
        end_timestamp = time_utc_timestamp_value(get_time_log(unnamed_col_name, s_id))
        
        # Dictionary to hold processed dataframes for different data types.
        dataframes = {}
        
        # Iterate over the list of data types ['acc', 'bvp', 'eda', 'temp'].
        for dtype in ['acc', 'bvp', 'eda', 'temp']:
            # Extract relevant time frame from global dataframe based on data type and timestamp range.
            frame = use_time_to_frame(globals()[f'{dtype}{i}'], start_timestamp, end_timestamp)
            
            # Label the data frame; if data type is 'acc', use Label_Acc function, else use Label_BVP_EDA_TEMP.
            if dtype == 'acc':
                dataframes[dtype] = Label_Acc(frame, s_id, label)
            else:
                dataframes[dtype] = Label_BVP_EDA_TEMP(frame, dtype, label)
        
        # Concatenate the processed dataframes column-wise.
        final_df = pd.concat([dataframes['acc'], dataframes['bvp'], dataframes['eda'], dataframes['temp']], axis=1)
        
        # Assign label to the final dataframe.
        final_df['Label'] = label
        
        # Save the final dataframe to a global variable with a dynamic name based on the data_suffix and loop index.
        globals()[f'{data_suffix}_df_{i}'] = final_df



# List of events. Each tuple contains the event name, unnamed column name, data suffix, and label.
events = [
    ("Stroop Test", "Unnamed: 9", "str", "Stress"),
    ("Interview", "Unnamed: 13", "int", "Stress"),
    ("Hyperventilation", "Unnamed: 17", "hyp", "Stress"),
    ("Relax", "Unnamed: 11", "rlx1", "Normal"),
    ("Relax.1", "Unnamed: 15", "rlx2", "Normal"),
    ("Relax.2", "Unnamed: 19", "rlx3", "Normal"),
    ("Questionniare", "Unnamed: 21", "quest", "Normal"),
    ("Relax/Baseline", "Unnamed: 23", "rlxBase", "Normal")
]


# Loop through each event and process the data using the process_data function.
for event in events:
    process_data(*event)


# Concatenate the dataframes generated for each individual in the range.
# Then reset their indices and drop any rows with missing data.
for i in range(start, end):
    # Concatenate the dataframes column-wise for each data type and individual.
    concatenated_df = pd.concat([
        globals().get(f'str_df_{i}', pd.DataFrame()), 
        globals().get(f'rlx1_df_{i}', pd.DataFrame()), 
        globals().get(f'int_df_{i}', pd.DataFrame()), 
        globals().get(f'rlx2_df_{i}', pd.DataFrame()), 
        globals().get(f'hyp_df_{i}', pd.DataFrame()), 
        globals().get(f'rlx3_df_{i}', pd.DataFrame()), 
        globals().get(f'quest_df_{i}', pd.DataFrame()), 
        globals().get(f'rlxBase_df_{i}', pd.DataFrame())
    ], axis=0)
    
    # Reset the index of the concatenated dataframe.
    concatenated_df.reset_index(drop=True, inplace=True)
    
    # Drop rows with any missing values.
    concatenated_df.dropna(inplace=True)
    
    # Save the concatenated dataframe to a global variable with a dynamic name based on the loop index.
    globals()[f'df{i}'] = concatenated_df


