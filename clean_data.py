
from glob import glob
import logging
import os
import sys

import click
import numpy as np
import pandas as pd


ADULT_FLY_PRESENCE = 'Presence of Adult Fly'
BOOK_KEEPING = ['Row Number', 'File Name']
ACCEPTED_COLUMNS = BOOK_KEEPING + ['Year', 'Month', 'Latitude', 'Longitude', ADULT_FLY_PRESENCE]

# FORMAT 1
# e.g. 2022-09-20-ento_sites_information_fct_ogun_oyo_states_cbm_juliana-amanyi-enegela.csv
CATCH_LAT = 'Latitude of catching point'
CATCH_LON = 'Longitude of catching point'
VILLAGE_LAT = 'Latitude of the village (if exist)'
VILLAGE_LON = 'Longitude of the village (if exist)'
ADULT_FLY = 'Presence of adult flies/yes or no'
FORMAT_ONE_COLS = [CATCH_LAT, CATCH_LON, VILLAGE_LAT, VILLAGE_LON, ADULT_FLY]

# FORMAT 2
# e.g. forms.csv
EVIDENCE_OF_ADULT = 'R_evidenceofadult'
GPS_RIVER_BASIN = 'R_gps_river_basin'
FORMAT_TWO_COLS = [EVIDENCE_OF_ADULT, GPS_RIVER_BASIN]

# FORMAT 3
# e.g. 22_5.4_nga_lit_extr.csv
FORMAT_THREE_COLS = ['Lat', 'Lon', 'Fly_stage']

# FORMAT 4
# e.g. Nigeria_ento_TCCstates_14Nov22_monthly.csv
FORMAT_FOUR_COLS = ['Latitude', 'Longitude', 'N_flies_total']


def get_log_level(level):
    """ Set the logging level.

    Args:
        str: INFO, WARNING or ERROR
    """
    if level == 'INFO':
        return logging.INFO
    elif level == 'WARNING':
        return logging.WARNING
    elif level == 'ERROR':
        return logging.ERROR
    else:
        logging.critical(f'Logging level {level} Unsupported.')
        sys.exit()


def determine_accepted_and_rejected_rows(df):
    """Clean data frame based on which format is detected

    Args:
        df: data frame of data to clean

    Returns:
        Truple of (pd.DataFrame of accepted rows, pd.DataFrame of rejected rows, Boolean of if format is unknown)
    """
    formats = [{'cols': FORMAT_ONE_COLS, 'processor': clean_format_one},
               {'cols': FORMAT_TWO_COLS, 'processor': clean_format_two},
               {'cols': FORMAT_THREE_COLS, 'processor': clean_format_three},
               {'cols': FORMAT_FOUR_COLS, 'processor': clean_format_four}]
    accept, reject, unknown = pd.DataFrame(), pd.DataFrame(), True
    for fmt in formats:
        if all([col in df.columns for col in fmt['cols']]):
            accept, reject = fmt['processor'](df)
            unknown = False
    return accept, reject, unknown


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
    # Create a list to store the rejected columns for each input file
    rejected = []
    for file in glob(os.path.join(in_dir, '*.csv')):
        logging.info(f'Processing {file}')
        df = pd.read_csv(file, dtype=str)
        # Drop empty rows
        df = df.dropna(how='all')
        # Remove any white space from beginning or end of column names
        df = df.rename(columns={col: col.strip().capitalize() for col in df.columns})
        # Save the file name as a column in the dataframe for book keeping
        df['File Name'] = os.path.basename(file)
        # Assign a 'Row Number' column for book keeping
        df['Row Number'] = df.index + 1
        # Clean data
        accept, reject, unknown = determine_accepted_and_rejected_rows(df)
        if unknown:
            logging.warning(f'Unknown Format: {file}')
            continue
        accept = accept.fillna('NA')
        for column in ACCEPTED_COLUMNS:
            if column not in accept:
                accept[column] = 'NA'
        accepted.append(accept[ACCEPTED_COLUMNS])
        rejected.append(reject)
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
    for rejected_df in rejected:
        if not rejected_df.empty:
            reject_csv_name = out_file_prefix + '_rejected_for_' + rejected_df['File Name'].unique()[0]
            reject_path = os.path.join(out_dir, reject_csv_name)
            rejected_df = rejected_df.sort_values(by=['Row Number'])
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
    not_a_geodetic_range = ~series.str.contains(range_regex, regex=True)
    more_than_three_decimals = series.str.split('.').apply(lambda x: len(x[-1]) > 3)
    return not_a_geodetic_range & more_than_three_decimals


def common_processing(df):
    """ Common validation across all data formats.

    Args:
        df: Dataframe of data to validate

    Returns:
        Tuple (DataFrame of valid rows, DataFrame of rejected rows)
    """
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
    return df, reject


def clean_format_one(df):
    # Set Year since this format's Year column is not useful
    df['Year'] = 'NA'
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
    # Common processing
    df, rejected_geo = common_processing(df)
    # Concatenate previously rejected and rejected geodetic DataFrames
    reject = pd.concat([reject, rejected_geo])
    # Set ADULT_FLY_PRESENCE to ADULT_FLY
    df[ADULT_FLY_PRESENCE] = df[ADULT_FLY]
    # Reformat ADULT_FLY_PRESENCE column values to have first letter capatilized
    df[ADULT_FLY_PRESENCE] = df[ADULT_FLY_PRESENCE].str.title()
    return df, reject[BOOK_KEEPING + FORMAT_ONE_COLS]


def clean_format_two(df):
    # Replace any '---' value in the dataframe with not a number
    df = df.replace('---', np.nan)
    # Extract year to new column
    df['Year'] = df['R_date'].apply(lambda x: x if type(x) is float else x.split('-')[0])
    # Extract month to new column
    df['Month'] = df['R_date'].apply(lambda x: x if type(x) is float else x.split('-')[1])
    # Expand the 'r_GPS_river_basin' column into four new_columns
    new_columns = ['Latitude', 'Longitude', GPS_RIVER_BASIN + '_3', GPS_RIVER_BASIN + '_4']
    df[new_columns] = df[GPS_RIVER_BASIN].str.split(' ', expand=True)
    # Common processing
    df, reject = common_processing(df)
    # Set ADULT_FLY_PRESENCE to EVIDENCE_OF_ADULT
    df[ADULT_FLY_PRESENCE] = df[EVIDENCE_OF_ADULT]
    # Reformat ADULT_FLY_PRESENCE column values to have frst letter capatilized
    df[ADULT_FLY_PRESENCE] = df[ADULT_FLY_PRESENCE].str.title()
    return df, reject[BOOK_KEEPING + FORMAT_TWO_COLS]


def clean_format_three(df):
    # Any cell with 'NA", replace with np.nan
    df = df.replace('NA', np.nan)
    # Set Latitude and Longitude columns
    df[['Latitude', 'Longitude']] = df[['Lat', 'Lon']]
    # Common processing
    df, reject = common_processing(df)
    # List of True only where adult is in Fly_stage, ignore case sensitivity
    adults = df.Fly_stage.str.contains('adult', case=False).fillna(False)
    # Copy Fly_stage column to ADULT_FLY_PRESENCE
    df[ADULT_FLY_PRESENCE] = df['Fly_stage']
    # Everywhere there's adults, fill in yes
    df.loc[adults, ADULT_FLY_PRESENCE] = 'Yes'
    # Everywhere there's not adults, fill in No
    df.loc[~adults, ADULT_FLY_PRESENCE] = 'No'
    return df, reject[BOOK_KEEPING + FORMAT_THREE_COLS]


def clean_format_four(df):
    # Common processing
    df, reject = common_processing(df)
    # Ensure Month has leading zeros
    df['Month'] = df['Month'].str.zfill(2)
    # Initialize new column ADULT_FLY_PRESENCE to 'No'
    df[ADULT_FLY_PRESENCE] = 'No'
    # Convert N_flies_total column data type from string to int
    df['N_flies_total'] = df['N_flies_total'].astype(int)
    # Set 'Yes' where 1 or more flies counted.
    df.loc[df['N_flies_total'] > 0, ADULT_FLY_PRESENCE] = 'Yes'
    return df, reject[BOOK_KEEPING + FORMAT_FOUR_COLS]


if __name__ == '__main__':
    clean_data()
