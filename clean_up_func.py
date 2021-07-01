import numpy as np
import pandas as pd
import math


def test_negative_values(df, tag):
    """
    find rows in containg negative values.

    Parameters
    ----------
    df : data frame
        data frame containing data from eeg experiment devided by trail.
    tag: str
        type of data frame: ad df or BDM df

    """
    utag = tag.upper()
    if utag == 'AD':
        columns_to_check = ['sub_id', 'ad_id',
                            'label', 'repitition', 'item_id', 'liking']
    elif utag == 'BDM':
        columns_to_check = ['sub_id', 'label', 'repitition']
    for column in columns_to_check:
        query_str = column + ' < 0'
        problematic_trails = df.query(query_str)
    if len(problematic_trails) > 0:
        raise Exception('some subjects contain negative values')
    else:
        print('the data does not contain negative values')

        
def hunt_eeg_nans(df):
    """
    find rows in which there are NaN eeg values.

    Parameters
    ----------
    df : data frame
        data frame containing data from eeg experiment devided by trail.

    Returns
    -------
    Ndarray
        and array containing indeces of rows containg NaN values
    """
    nan_ind = []
    for i in range(len(df)):
        nan_count = 0
        for j in range(len(df['eeg'][i])):
            if math.isnan(np.sum(df.eeg[i][j])):
                nan_count += 1
        nan_ind.append(nan_count)
    nan_eeg_entries = np.argwhere(nan_ind).flatten()
    return nan_eeg_entries


def hunt_column_nans(df, col):
    """
    find rows in which there are NaN values.

    Parameters
    ----------
    df : data frame
        data frame containing data from eeg experiment devided by trail.

    col: str
        the name of the column to check: 'Liking' \ 'label'

    Returns
    -------
    Ndarray
        and array containing indeces of rows containg NaN values
    """
    l_col = df[col]
    b = []
    for index in range(0, len(l_col)):
        b.append(l_col[index])
    l_col = np.where(np.isnan(b))
    nan_col_enteries = np.asarray(l_col)
    return nan_col_enteries


def hunt_duplicates(df):
    """
    find duplicate rows:
        rows in which there are the same subject id and trail id.

    Parameters
    ----------
    df : data frame
        data frame containing data from eeg experiment devided by trail.

    Returns
    -------
    Ndarray
        and array containing indeces of duplicate rows.
    """
    # create a list containg all the ad id
    ad_id = df['ad_id']
    ad_id_list = []
    for id in ad_id:
        ad_id_list.append(float(id))
    # create a list of all subjects
    sub_id = df['sub_id']
    sub_id_list = []
    # create a dataframe containig only the user id and ad id
    for id in sub_id:
        sub_id_list.append(float(id))
    duplicate_dict = {'sub': sub_id_list, 'ad': ad_id_list}
    duplicate_test = pd.DataFrame(data=duplicate_dict)
    duplicates = duplicate_test.duplicated()
    duplicate_rows = []
    # create a list of duplicate rows indeces
    for index in range(0, len(duplicates)):
        if duplicates[index] == True:
            duplicate_rows.append(index)
    return duplicate_rows


def clean_data(df, tag):
    """
    cleans the dataframe:
        removes rows containing duplictes and NaN values.

    Parameters
    ----------
    df : data frame
        data frame containing data from eeg experiment devided by trail.
    tag: str
        type of data frame: ad df or BDM df

    Returns
    -------
    df
        clean dataframe where each row contains a trail.
    """
    utag = tag.upper()
    if utag == 'AD':
        nan_eeg_entries = hunt_eeg_nans(df)
        liking_nans = hunt_column_nans(df, 'liking')
        label_nans = hunt_column_nans(df, 'label')
        duplicate_rows = hunt_duplicates(df)
        nans = np.concatenate(
            (liking_nans, nan_eeg_entries, duplicate_rows, label_nans), axis=None)
    elif utag == 'BDM':
        label_nans = hunt_column_nans(df, 'label')
        duplicate_rows = hunt_duplicates(df)
        nans = np.concatenate(
            (label_nans, duplicate_rows), axis=None)

    nans_set = set(nans)
    # Remove nan eeg and likings entries from dataframe
    data = df.drop(nans_set)
    return data
