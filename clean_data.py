
from glob import glob
import logging
import os
import sys

import click
import numpy as np
import pandas as pd

def get_log_level(level):
    if level == 'INFO':
        return logging.INFO
    elif level == 'WARNING':
        return logging.WARNING
    elif level == 'ERROR':
        return logging.ERROR
    else:
        logging.critical(f'Logging level {level} Unsupported.')
        sys.exit()


ADULT_FLY_PRESENCE = 'Presence of Adult Fly'
BOOK_KEEPING = ['Row Number', 'Filename']
ACCEPTED_COLUMNS = BOOK_KEEPING + ['Latitude', 'Longitude', ADULT_FLY_PRESENCE]


@click.command()
@click.option('--in_dir', type=click.Path(),
              help='Directory containing field data CSVs to clean.')
@click.option('--out_dir', default='.', type=click.Path(),
              help='Directory to save cleaned field data. (Default: Current Directory)')
@click.option('--out_file_prefix', default='usf_pipeline', type=str,
              help='Prefix of output files (Default: usf_pipeline)')
@click.option('--log_level', default='INFO', type=str,
              help='INFO, WARNING, ERROR (Default: INFO)')
def clean_data(in_dir, out_dir, out_file_prefix, log_level):
    logging.basicConfig(level=get_log_level(log_level))
    # Create a list to store all of the accepted rows
    accepted = []
    # Create a dictionary to store the rejected columns from each format
    rejected = {'FORMAT_ONE': [], 'FORMAT_TWO': [], 'FORMAT_THREE': [], 'FORMAT_FOUR': []}
    for file in glob(os.path.join(in_dir, '*.csv')):
        logging.info(f'Processing {file}')
        df = pd.read_csv(file, dtype=str)
        # Remove any white space from beginning or end of column names
        df = df.rename(columns={col: col.strip() for col in df.columns})
        # Save the filename as a column in the dataframe for book keeping
        df['Filename'] = os.path.basename(file)
        # Assign a 'Row Number' column for book keeping
        df['Row Number'] = df.index + 1
        # If all of the columns for a known format are present, process accordingly
        if all([col in df.columns for col in FORMAT_ONE_COLS]):
            accept, reject = clean_known_format_one(df)
            rejected['FORMAT_ONE'].append(reject)
        elif all([col in df.columns for col in FORMAT_TWO_COLS]):
            accept, reject = clean_known_format_two(df)
            rejected['FORMAT_TWO'].append(reject)
        elif all([col in df.columns for col in FORMAT_THREE_COLS]):
            accept, reject = clean_known_format_three(df)
            rejected['FORMAT_THREE'].append(reject)
        elif len([col for col in df.columns if col.startswith('n_flies_total_')]) == NUM_N_COLS:
            accept, reject = clean_known_format_four(df)
            rejected['FORMAT_FOUR'].append(reject)
        else:
            logging.warning(f'Unknown Format: {file}')
            continue
        accepted.append(accept[ACCEPTED_COLUMNS])
        if len(df) != len(accept) + len(reject):
            logging.critical(f'Number of accepted and rejected rows do not add to total rows: {file}')
            sys.exit()
    # if there are accepted dataframes
    if accepted:
        # Turn list of dataframes into single dataframe
        accepted_df = pd.concat(accepted)
        # Save dataframe to CSV file
        accepted_df.to_csv(os.path.join(out_dir, out_file_prefix + '_accepted.csv'), index=False)
    # For each dataframe of rejected rows
    for known_format, rejected_dfs in rejected.items():
        if rejected_dfs:
            reject_csv_name = out_file_prefix + '_' + known_format.lower() + '_rejected.csv'
            reject_path = os.path.join(out_dir, reject_csv_name)
            rejected_df = pd.concat(rejected_dfs)
            if not rejected_df.empty:
                rejected_df.to_csv(reject_path, index=False)


def is_valid_geo(series):
    """check if value has greater that three digits after the decima
       check that value is not a range (e.g. 5.6372 - 5.64652)
    
    Args:
        series: Pandas Series to check for valid geodetic values
    
    Returns:
        list of True/False values with True where geodetic values are valid
    """
    # Regular Expression which matches a digit, then a space (zero or more),
    # a dash, then a space (zero or more), then a digit.
    range_regex = r'\d\s*-\s*\d'
    return series.str.split('.').apply(lambda x: len(x[-1]) > 3) &\
           ~series.str.contains(range_regex, regex=True)


# KNOWN FORMAT 1. When CSV contains these column names we know how to process it.
CATCH_LAT = 'Latitude of catching point'
CATCH_LON = 'Longitude of catching point'
VILLAGE_LAT = 'Latitude of the village (if exist)'
VILLAGE_LON = 'Longitude of the village (if exist)'
ADULT_FLY = 'Presence of adult flies/Yes or No'
FORMAT_ONE_COLS = [CATCH_LAT, CATCH_LON, VILLAGE_LAT, VILLAGE_LON, ADULT_FLY]
def clean_known_format_one(df):
    # List of True only where either VILLAGE_LAT or VILLAGE_LON is blank
    village_nulls = df[VILLAGE_LAT].isnull() | df[VILLAGE_LON].isnull()
    # List of True only where either CATCH_LAT or CATCH_LON is blank
    catch_nulls = df[CATCH_LAT].isnull() | df[CATCH_LON].isnull()
    # Save rows with no valid lat/lons to reject dataframe
    reject = df[(village_nulls & catch_nulls)]
    # Remove rows where at least one of village lat/lon or catch lat/lon is present
    df = df[(~village_nulls | ~catch_nulls)]
    # New 'Latitude' column with CATCH_LAT if filled in, otherwise use VILLAGE_LAT
    df = df.assign(Latitude=df[CATCH_LAT].combine_first(df[VILLAGE_LAT]))
    # New 'Longitude' column with CATCH_LAT if filled in, otherwise use VILLAGE_LON
    df = df.assign(Longitude=df[CATCH_LON].combine_first(df[VILLAGE_LON]))
    # List of True where geodetic values are valid
    valid_geos = is_valid_geo(df['Latitude']) & is_valid_geo(df['Longitude'])
    # Save rows with invalid geodetic values, concatenated to previous rejections
    reject = pd.concat([reject, df[~valid_geos]])
    # Remove rows with invalid geodetic values
    df = df[valid_geos]
    # Set ADULT_FLY_PRESENCE to ADULT_FLY
    df[ADULT_FLY_PRESENCE] = df[ADULT_FLY]
    # Reformat ADULT_FLY_PRESENCE column values to have first letter capatilized
    df[ADULT_FLY_PRESENCE] = df[ADULT_FLY_PRESENCE].str.title()
    # Fill NaN with Not Available
    df[ADULT_FLY_PRESENCE] = df[ADULT_FLY_PRESENCE].fillna('NA')
    return df, reject[BOOK_KEEPING + FORMAT_ONE_COLS]


EVIDENCE_OF_ADULT = 'r_EvidenceOfAdult'
GPS_RIVER_BASIN = 'r_GPS_river_basin'
FORMAT_TWO_COLS = [EVIDENCE_OF_ADULT, GPS_RIVER_BASIN]
# KNOWN FORMAT 2. When CSV contains these column names we know how to process it.
def clean_known_format_two(df):
    # Replace any '---' value in the dataframe with not a number
    df = df.replace('---', np.nan)
    # Expand the 'r_GPS_river_basin' column into four new_columns
    new_columns = ['Latitude', 'Longitude', GPS_RIVER_BASIN + '_3', GPS_RIVER_BASIN + '_4']
    df[new_columns] = df[GPS_RIVER_BASIN].str.split(' ', expand=True)
    # List of True only where either Latitude or Longitude is blank
    geo_nulls = df['Latitude'].isnull() | df['Longitude'].isnull()
    # Save rows with nulls in reject dataframe
    reject = df[geo_nulls]
    # Remove rows where village or catch lat/lon have nulls
    df = df[~geo_nulls]
    # List of True where geodetic values are valid
    valid_geos = is_valid_geo(df['Latitude']) & is_valid_geo(df['Longitude'])
    # Save rows with invalid geodetic values concatenated to previous rejections
    reject = pd.concat([reject, df[~valid_geos]])
    # Remove rows with invalid geodetic values
    df = df[valid_geos]
    # Set ADULT_FLY_PRESENCE to EVIDENCE_OF_ADULT
    df[ADULT_FLY_PRESENCE] = df[EVIDENCE_OF_ADULT]
    # Reformat ADULT_FLY_PRESENCE column values to have frst letter capatilized
    df[ADULT_FLY_PRESENCE] = df[ADULT_FLY_PRESENCE].str.title()
    # Fill NaN with Not Available
    df[ADULT_FLY_PRESENCE] = df[ADULT_FLY_PRESENCE].fillna('NA')
    return df, reject[BOOK_KEEPING + FORMAT_TWO_COLS]


FORMAT_THREE_COLS = ['lat', 'lon', 'fly_stage']
def clean_known_format_three(df):
    df = df.replace('NA', np.nan)
    # Set Latitude and Longitude columns
    df[['Latitude', 'Longitude']] = df[['lat', 'lon']]
    # List of True only where either Latitude or Longitude is blank
    geo_nulls = df['Latitude'].isnull() | df['Longitude'].isnull()
    # Save rows with nulls in reject dataframe
    reject = df[geo_nulls]
    # Remove rows where village or catch lat/lon have nulls
    df = df[~geo_nulls]
    # List of True where geodetic values are valid
    valid_geos = is_valid_geo(df['Latitude']) & is_valid_geo(df['Longitude'])
    # Save rows with invalid geodetic values concatenated to previous rejections
    reject = pd.concat([reject, df[~valid_geos]])
    # Remove rows with invalid geodetic values
    df = df[valid_geos]
    # List of True only where adult is in fly_stage, ignore case sensitivity
    adults = df.fly_stage.str.contains('adult', case=False).fillna(False)
    # Copy fly_stage column to ADULT_FLY_PRESENCE
    df[ADULT_FLY_PRESENCE] = df['fly_stage']
    # Replace NaN with Not Available
    df[ADULT_FLY_PRESENCE] = df[ADULT_FLY_PRESENCE].fillna('NA')
    # Everywhere there's adults, fill in yes
    df.loc[adults, ADULT_FLY_PRESENCE] = 'Yes'
    # Everywhere there's not adults, fill in No
    df.loc[~adults, ADULT_FLY_PRESENCE] = 'No'
    return df, reject[BOOK_KEEPING + FORMAT_THREE_COLS]


NUM_N_COLS = 12
FORMAT_FOUR_COLS = ['Latitude', 'Longitude']
def clean_known_format_four(df):
    # List of True only where either Latitude or Longitude is blank
    geo_nulls = df['Latitude'].isnull() | df['Longitude'].isnull()
    # Save rows with nulls in reject dataframe
    reject = df[geo_nulls]
    # Remove rows where village or catch lat/lon have nulls
    df = df[~geo_nulls]
    # List of True where geodetic values are valid
    valid_geos = is_valid_geo(df['Latitude']) & is_valid_geo(df['Longitude'])
    # Save rows with invalid geodetic values concatenated to previous rejections
    reject = pd.concat([reject, df[~valid_geos]])
    # Remove rows with invalid geodetic values
    df = df[valid_geos]
    # Get a list of every column that starts with n_flies_total_
    n_flies_total_cols = [col for col in df.columns if col.startswith('n_flies_total_')]
    # Sum all the n_flies_total_ rows
    n_flies_total_col_sums = df[n_flies_total_cols].fillna(0).astype(int).sum(axis=1)
    # Initialize new column ADULT_FLY_PRESENCE to 'No'
    df[ADULT_FLY_PRESENCE] = 'No'
    # Set 'Yes' where 1 or more flies counted.
    df.loc[n_flies_total_col_sums > 0, ADULT_FLY_PRESENCE] = 'Yes'
    return df, reject[BOOK_KEEPING + FORMAT_FOUR_COLS + n_flies_total_cols]


if __name__ == '__main__':
    clean_data()
