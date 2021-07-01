import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt


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


def test_liking_dis(df, l_tresh=2, h_tresh=5):
    liking = df.liking
    liking_list = []
    for index in range(0, len(liking)):
        liking_list.append(float(liking[index]))
    high_tresh = 'liking > ' + str(h_tresh)
    low_tresh = 'liking < ' + str(l_tresh)
    print('number of liking above ' + str(h_tresh) +
          str(len(df.query(high_tresh))))
    print('number of liking below 2: ' + str(len(df.query(low_tresh))))
    fig, ax = plt.subplots()
    N, bins, patches = ax.hist(liking_list, bins=np.arange(
        0, 7, 0.1), edgecolor='white', linewidth=1)
    plt.title('distribution of liking scores')
    plt.xlabel('liking score')
    plt.ylabel('frequency')

    for i in range(0, 10*l_tresh):
        patches[i].set_facecolor('darkorange')
    for i in range(10*l_tresh, 10*h_tresh):
        patches[i].set_facecolor('gainsboro')
    for i in range(10*h_tresh, len(patches)):
        patches[i].set_facecolor('darkorange')

    plt.show()
