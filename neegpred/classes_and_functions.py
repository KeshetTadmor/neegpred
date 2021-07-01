import cleanup_func
from tests import *
import pandas as pd
import numpy as np
import mat73
from matplotlib import pyplot as plt
import math
import scipy.stats as stats
import tqdm
import seaborn as sns
from sklearn.model_selection import train_test_split

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

'''
This script defines classes and functions for the EEG analysis pipeline built for prof. Dino Levy's lab
'''


class MatlabDataImporter:
    """
    Class for importing a .mat file based dataset with the specific fields specified in README.md
    import, parse and package data into two pandas DataFrames
    """
    def __init__(self, path):
        """
            initialize a new instance with a path to a .mat file with the following fields

            Parameters
            ----------
            path : pathlib Path
                Input data path

        """
        self.path = path
        self.data_dict = None
        self.lv1_key = None
        self.lv2_keys = None
        self.bdm = None
        self.ad = None
        self.triggers = None
        self.subid = None
        self.bdm_df = None
        self.ad_df = None

        print(f'The matlab file is ready to be read! \n Use the "read_data" method to load it')

    def read_data(self):
        """
            Reads the .mat file via the mat73 module

            Returns
            ----------
            data_dict : dict
                Dict with the different keys present in the matlab structure
        """
        print(f'Reading data into a python dict\n'
              f'resulting keys will be corroborated with expected fields for the\n'
              f'analysis to continue smoothly')
        self.data_dict = mat73.loadmat(self.path)
        self.lv1_key = str(list(self.data_dict.keys())[0])
        self.lv2_keys = list(self.data_dict[self.lv1_key].keys())
        expected_keys = ['AdEEG', 'BDMEEG', 'SubID', 'Triggers']
        assert self.lv2_keys == expected_keys,\
            'The fields of the structure do not match the required format, see README'
        print(f'fields are checked - we\'re good to go!\n'
              f'use the "parse_data" method next')
        return self.data_dict

    def parse_data(self):
        """
            Parses the different fields of the dict into different entities

            Returns
            ----------
            ad_df : pandas DataFrame
                dataframe with relevant fields for the advertisement section of the experiment

            bdm_df : pandas DataFrame
                dataframe with relevant fields for the BDM section of the experiment
        """
        print('Parsing the data from the dict into DataFrames...\n')
        self.bdm = self.data_dict[self.lv1_key].BDMEEG
        self.ad = self.data_dict[self.lv1_key].AdEEG
        self.subid = self.data_dict[self.lv1_key].SubID
        self.triggers = self.data_dict[self.lv1_key].Triggers

        # Parse BDM data:
        bdm_eeg = []
        bdm_cat = []
        bdm_item = []
        bdm_label = []
        bdm_repitition = []
        bdm_sub_id = []
        for i in range(len(self.bdm)):
            sub_id = self.subid[i]
            for j in range(50):
                bdm_eeg.append(self.bdm[i]['EEG'][j])
                bdm_cat.append(self.bdm[i]['Category'][j])
                bdm_item.append(self.bdm[i]['Item'][j])
                bdm_label.append(self.bdm[i]['Label'][j])
                bdm_repitition.append(self.bdm[i]['Repitition'][j])
                bdm_sub_id.append(sub_id)
        # into dataframe:
        colnames = ['sub_id', 'eeg', 'category', 'label', 'repitition', 'item']
        self.bdm_df = pd.DataFrame(data=[bdm_sub_id, bdm_eeg, bdm_cat, bdm_label, bdm_repitition, bdm_item],
                                   index=colnames).transpose()
        # Parse Ad data
        ad_eeg = []
        ad_id = []
        ad_cat = []
        ad_item = []
        ad_label = []
        ad_repitition = []
        ad_item_id = []
        ad_liking = []
        ad_sub_id = []
        for i in range(len(self.ad)):
            sub_id = self.subid[i]
            for j in range(len(self.ad[i]['EEG'])):
                ad_eeg.append(self.ad[i]['EEG'][j])
                ad_cat.append(self.ad[i]['Category'][j])
                ad_item.append(self.ad[i]['Item'][j])
                ad_label.append(self.ad[i]['Label'][j])
                ad_repitition.append(self.ad[i]['Repitition'][j])
                ad_id.append(self.ad[i]['AdID'][j])
                ad_item_id.append(self.ad[i]['ItemID'][j])
                ad_liking.append(self.ad[i]['Liking'][j])
                ad_sub_id.append(sub_id)
        colnames = ['sub_id', 'eeg', 'ad_id', 'category', 'item', 'label', 'repitition', 'item_id', 'liking']
        self.ad_df = pd.DataFrame(data=[ad_sub_id, ad_eeg, ad_id, ad_cat, ad_item,
                                        ad_label, ad_repitition, ad_item_id, ad_liking],
                                  index=colnames).transpose()
        print('Done! The data is organized in DataFrames.\n'
              'The next phase of the process is cleaning the data with the "clean_dfs" method')
        return self.ad_df, self.bdm_df

    def clean_dfs(self):
        """
        find and remove duplicate trials, trials with bad eeg (where relevant) and nan value scores
        for specifics see cleanup func documentation

        """
        self.ad_df = cleanup_func.clean_data(self.ad_df, 'ad')
        self.bdm_df = cleanup_func.clean_data(self.bdm_df, 'bdm')
        print('DataFrames are clean, The next stage of the process is testing, with "test_dfs" (method)')

    def test_dfs(self, thr_low=10, thr_high=10):
        test_negative_values(self.ad_df, 'ad')
        test_negative_values(self.bdm_df, 'bdm')
        if thr_low != 10 & thr_high != 10:
            test_liking_dis(self.ad_df, thr_low, thr_high)
        else:
            test_liking_dis(self.ad_df)
        print('tests concluded, now we can really begin! \n'
              'continue with the analysis and enrichment of the EEG with the "enrich_eeg_data" function')


# EEG Feature Extraction

def mean_half_diff(sample):
    """
        Splits the sample into two halves and returns the difference between the mean of the two halves

        Parameters
        ----------
        sample : np.array , float
            sample is a list representing a time series with (length in seconds)*500hz sample rate amplitude datapoints

        Returns
        ----------
        half_diff : float
            difference of the mean amplitude of the second and first halves of the sample
        """
    h1 = np.mean(sample[0:len(sample)//2])
    h2 = np.mean(sample[len(sample)//2:])
    return h2-h1


def mean_com_diff(sample, com_start_t=1000):
    """
        Splits the sample into two halves and returns the difference between the mean of the two halves

        Parameters
        ----------
        sample : np.array , float
            sample is a list representing a time series with (length in seconds) * (500hz sample-rate)
            amplitude data points

        com_start_t: time of commercial start (in samples)

        Returns
        ----------
        com_diff : float
            difference of the mean amplitude of the second and first halves of the sample
        """
    before_com = np.mean(sample[0:com_start_t])
    during_com = np.mean(sample[com_start_t+1:])
    return during_com - before_com


def mean_quart(sample):
    """
        Splits the sample into quartiles and returns the means and differences
        between the means of each pair of quartiles

        Parameters
        ----------
        sample : np.array , float
            sample is a list representing a time series with (length in seconds) * (500hz sample-rate)
            amplitude data points

        Returns
        ----------
        mean_list : list, float
            a list containing the mean for each quarter:
            [mean_q1, mean_q2, mean_q3, mean_q4]
        diff_list : list, float
            a list containing the difference for each pair of quarters:
            [q1-q2, q1-q3, q1-q4, q2-q3, q2-q4, q3-q4]
        """
    q1 = sample[:len(sample) // 4]
    q2 = sample[len(sample) // 4:2 * (len(sample) // 4)]
    q3 = sample[2 * (len(sample) // 4):3 * (len(sample) // 4)]
    q4 = sample[3 * (len(sample) // 4):]

    q_list = [q1, q2, q3, q4]

    mean_list = []
    for q in q_list:
        mean_list.append(np.mean(q))

    diff_q1_q2 = mean_list[0] - mean_list[1]
    diff_q1_q3 = mean_list[0] - mean_list[2]
    diff_q1_q4 = mean_list[0] - mean_list[3]
    diff_q2_q3 = mean_list[1] - mean_list[2]
    diff_q2_q4 = mean_list[1] - mean_list[3]
    diff_q3_q4 = mean_list[2] - mean_list[3]
    diff_list = [diff_q1_q2, diff_q1_q3, diff_q1_q4, diff_q2_q3, diff_q2_q4, diff_q3_q4]

    # titles : ['mean_q1' , 'mean_q2' , 'mean_q3', 'mean_q4'] ,
    #  ['diff_q1_q2', 'diff_q1_q3', 'diff_q1_q4', 'diff_q2_q3', 'diff_q2_q4', 'diff_q3_q4']
    return mean_list[:], diff_list[:]


def std_half_diff(sample):
    """
        Splits the sample into two halves and returns the difference between the standard deviation of the two halves

        Parameters
        ----------
        sample : np.array , float
            sample is a list representing a time series with (length in seconds) * (500hz sample-rate)
            amplitude data points


        Returns
        ----------
        half_diff : float
            difference of the standard deviation of the second and first halves of the sample
        """
    h1 = np.std(sample[0:len(sample)//2])
    h2 = np.std(sample[len(sample)//2:])
    return h2-h1


def std_com_diff(sample, com_start_t=1000):
    """
    Splits the sample to before and during the commercial and returns the difference
    between the standard deviation of the two halves

    Parameters
    ----------
    sample : np.array , float
        sample is a list representing a time series with 15000 amplitude datapoints
        corresponding to a 3 seconds segment of iEEG recording at 5000 Hz.

    com_start_t: time of commercial start (in samples)

    Returns
    ----------
    half_diff : float
        difference of the standard deviation of the second and first halves of the sample
    """
    h1 = np.std(sample[0:com_start_t])
    h2 = np.std(sample[com_start_t+1:])
    return h2-h1


def min_com_diff(sample):
    """
        Calculate the difference between the minimum value of the sample before the commercial and during the commercial

        Parameters
        ----------
        sample : np.array , float
            sample is a list representing a time series with (length in seconds) * (500hz sample-rate)
            amplitude data points

        Returns
        ----------
         com_diff : float
            difference of the minimum amplitude of the sample before and during the commercial
        """
    h1 = np.min(sample[0:1000])
    h2 = np.min(sample[1001:])
    return h2-h1


def max_com_diff(sample, com_start_t=1000):
    """
        Calculate the difference between the maximum value of the sample before the commercial and during the commercial

        Parameters
        ----------
        sample : np.array , float
            sample is a list representing a time series with (length in seconds) * (500hz sample-rate)
            amplitude data points

        com_start_t: time of commercial start (in samples)

        Returns
        ----------
         com_diff : float
            difference of the maximum amplitude of the sample before and during the commercial
        """
    h1 = np.max(sample[0:com_start_t])
    h2 = np.max(sample[com_start_t+1:])
    return h2-h1


def fft_feature(sample):
    """
    Performs fast Fourier Transform on sample and returns the frequencies, amplitudes, standard deviation
    of amplitudes and the frequnecy with the highest amplitude

    Parameters
    ----------
    sample : np.array , float
        sample is a list representing a time series with (length in seconds) * (500hz sample-rate)
        amplitude data points

    Returns
    ----------
     fft_freqs, amplitudes : list , float
        list of frequencies and corresponding amplitudes extracted with fast Fourier Transform

     fft_std : float
        standard deviation of the fast Fourier Transform amplitudes

     peak_freq : float
        the frequency with the highest amplitude in the sample
    """
    # find the closest power of 2 larger than the length of sample for padding fft
    n_samples = 2 ** (math.ceil(math.log2(len(sample))))
    sample_fft = np.fft.rfft(sample, n=n_samples)
    amplitudes = (2 / n_samples) * np.abs(sample_fft)
    fft_freqs = np.fft.rfftfreq(n_samples, d=1. / 500)

    fft_std = np.std(amplitudes)
    peak_freq_ind = amplitudes.argsort()[-1]
    peak_freq = fft_freqs[peak_freq_ind]

    return fft_freqs, amplitudes, fft_std, peak_freq


def frequency_band_analysis(fft_freqs, amplitudes, band_range):
    """
       Receives output from fast Fourier Transform (frequencies and amplitudes) and a frequency band,
       and returns the band root mean square, maximum amplitude and standard deviation

       Parameters
       ----------
       fft_freqs, amplitudes : list , float
           frequencies and amplitudes from fast Fourier Transform

       band_range : list , float
           list of bottom and top frequencies defining a frequency band to calculate parameters on

       Returns
       ----------
       fft_band_rms, fft_band_max, fft_band_std : float
           root mean square, maximum amplitude and standard deviation of the frequncy band
       """
    # using fft instead of spectogram
    band_ind = get_frequncy_indices(fft_freqs, band_range[0], band_range[1])
    fft_band_rms = np.sqrt(np.mean(amplitudes[band_ind] ** 2))
    fft_band_max = np.max(amplitudes[band_ind])
    fft_band_std = np.std(amplitudes[band_ind])

    return fft_band_rms, fft_band_max, fft_band_std


def get_frequncy_indices(freqs, bottom_frequncy, top_frequncy):
    """
    returns the indices of a frequency band in the list of frequencies extracted from Fourier Transform

    Parameters
    ----------
    freqs : list, float
       list of frequencies extracted from a real fast Fourier Transform (np.rfft)

    bottom_frequncy : float
        the bottom frequency of the frequency band of interest

    top_frequncy : float
        the top frequency of the frequency band of interest

    Returns
    ----------
    frequency_indices : list, int
        the indices of the in freqs that for the frequency band of interest
    """
    freqs_above_bottom = freqs[freqs > bottom_frequncy]
    freqs_below_top = freqs[freqs < top_frequncy]
    # the first index of freqs_above_bottom is the lowest frequncy in the desired range
    bottom_index = np.where(freqs == freqs_above_bottom[0])[0][0]
    # the last index of freqs_below_top is the highest frequnecy in the desired range
    top_index = np.where(freqs == freqs_below_top[-1])[0][0]
    # adding +1 to include the top index
    frequncy_indices = list(range(bottom_index, top_index + 1))

    return frequncy_indices


def create_eeg_feature_columns(df):
    """
    inserts columns of features for the feature extraction process to go over and enumerates them accoording to
    the eeg channel count (each feature is calculated for all electrodes)

    Parameters
    ----------
    df: pandas DataFrame
        the dataframe to add the columns to

    Returns
    ----------
    df_with_features: pandas DataFrame
        an extended DataFrame with columns corresponding to features for the extraction process
    """
    feature_list = [
        'mean_half_diff', 'mean_com_diff', 'mean_q1', 'mean_q2', 'mean_q3', 'mean_q4',
        'diff_q1_q2', 'diff_q1_q3', 'diff_q1_q4', 'diff_q2_q3', 'diff_q2_q4',
        'diff_q3_q4', 'std_com_diff', 'min_com_diff', 'max_com_diff', 'kurtosis',
        'skewness', 'fft_std', 'fft_alpha', 'fft_alpha_max', 'fft_alpha_std', 'fft_beta', 'fft_beta_max',
        'fft_beta_std', 'fft_gamma', 'fft_gamma_max', 'fft_gamma_std', 'fft_delta', 'fft_delta_max',
        'fft_delta_std', 'fft_theta', 'fft_theta_max', 'fft_theta_std', 'fft_hf', 'fft_hf_max', 'fft_hf_std'
    ]
    enumerated_feature_list = []
    for i in feature_list:
        for j in range(8):
            enumerated_feature_list.append(i + '_' + str(j))
    enumerated_feature_list.append('correlate_0_4')
    enumerated_feature_list.append('pcorr_0_4')
    enumerated_feature_list.append('correlate_1_3')
    enumerated_feature_list.append('pcorr_1_3')
    enumerated_feature_list.append('correlate_2_7')
    enumerated_feature_list.append('pcorr_2_7')

    for feature in enumerated_feature_list:
        df.insert(df.shape[1], column=feature, value=np.float32(0))

    return df


def extract_eeg_features(df, index):

    """
    Extracts features for a specific trial (each trial has 8 electrodes, thus 8 samples to analyze) and
    adds them to the DataFrame in-place

    Parameters
    ----------

     index : int
        index of the trial

     df : pd.DataFrame
        DataFrame to add features to

    """

    for i in range(np.shape(df.loc[index].eeg)[0]):
        sample = df.loc[index].eeg[i]
        df.at[index, 'sample_mean_' + str(i)] = np.mean(sample)
        df.at[index, 'sample_std_' + str(i)] = np.std(sample)
        df.at[index, 'mean_half_diff_' + str(i)] = mean_half_diff(sample)
        df.at[index, 'mean_com_diff_' + str(i)] = mean_com_diff(sample)
        df.at[index, 'mean_com_diff_' + str(i)] = mean_com_diff(sample)
        mean_quarts, diff_quarts = mean_quart(sample)
        df.at[index, 'mean_q1_' + str(i)] = mean_quarts[0]
        df.at[index, 'mean_q2_' + str(i)] = mean_quarts[1]
        df.at[index, 'mean_q3_' + str(i)] = mean_quarts[2]
        df.at[index, 'mean_q4_' + str(i)] = mean_quarts[3]
        df.at[index, 'diff_q1_q2_' + str(i)] = diff_quarts[0]
        df.at[index, 'diff_q1_q3_' + str(i)] = diff_quarts[1]
        df.at[index, 'diff_q1_q4_' + str(i)] = diff_quarts[2]
        df.at[index, 'diff_q2_q3_' + str(i)] = diff_quarts[3]
        df.at[index, 'diff_q2_q4_' + str(i)] = diff_quarts[4]
        df.at[index, 'diff_q3_q4_' + str(i)] = diff_quarts[5]
        df.at[index, 'std_com_diff_' + str(i)] = std_com_diff(sample)
        df.at[index, 'kurtosis_' + str(i)] = stats.kurtosis(sample)
        df.at[index, 'skewness_' + str(i)] = stats.skew(sample)
        df.at[index, 'sample_max_' + str(i)] = np.max(sample)
        df.at[index, 'sample_min_' + str(i)] = np.min(sample)
        df.at[index, 'min_com_diff_' + str(i)] = min_com_diff(sample)
        df.at[index, 'max_com_diff_' + str(i)] = max_com_diff(sample)

        ALPHA = [7, 13]
        BETA = [14, 30]
        GAMMA = [31, 100]
        DELTA = [0, 5]
        THETA = [4, 8]
        HF = [80, 500]

        fft_freqs, amplitudes, fft_std, peak_freq = fft_feature(sample)
        df.at[index, 'fft_std_' + str(i)] = fft_std
        df.at[index, 'peak_freq_' + str(i)] = peak_freq
        alpha_sum, alpha_max, alpha_std = frequency_band_analysis(fft_freqs, amplitudes, ALPHA)
        df.at[index, 'fft_alpha_' + str(i)] = alpha_sum
        df.at[index, 'fft_alpha_max_' + str(i)] = alpha_max
        df.at[index, 'fft_alpha_std_' + str(i)] = alpha_std
        beta_sum, beta_max, beta_std = frequency_band_analysis(fft_freqs, amplitudes, BETA)
        df.at[index, 'fft_beta_' + str(i)] = beta_sum
        df.at[index, 'fft_beta_max_' + str(i)] = beta_max
        df.at[index, 'fft_beta_std_' + str(i)] = beta_std
        gamma_sum, gamma_max, gamma_std = frequency_band_analysis(fft_freqs, amplitudes, GAMMA)
        df.at[index, 'fft_gamma_' + str(i)] = gamma_sum
        df.at[index, 'fft_gamma_max_' + str(i)] = gamma_max
        df.at[index, 'fft_gamma_std_' + str(i)] = gamma_std
        delta_sum, delta_max, delta_std = frequency_band_analysis(fft_freqs, amplitudes, DELTA)
        df.at[index, 'fft_delta_' + str(i)] = delta_sum
        df.at[index, 'fft_delta_max_' + str(i)] = delta_max
        df.at[index, 'fft_delta_std_' + str(i)] = delta_std
        theta_sum, theta_max, theta_std = frequency_band_analysis(fft_freqs, amplitudes, THETA)
        df.at[index, 'fft_theta_' + str(i)] = theta_sum
        df.at[index, 'fft_theta_max_' + str(i)] = theta_max
        df.at[index, 'fft_theta_std_' + str(i)] = theta_std
        hf_sum, hf_max, hf_std = frequency_band_analysis(fft_freqs, amplitudes, HF)
        df.at[index, 'fft_hf_' + str(i)] = hf_sum
        df.at[index, 'fft_hf_max_' + str(i)] = hf_max
        df.at[index, 'fft_hf_std_' + str(i)] = hf_std
    df.at[index, 'correlate_0_4'] = stats.stats.pearsonr(df.loc[index].eeg[0], df.loc[index].eeg[4])[0]
    df.at[index, 'pcorr_0_4'] = stats.stats.pearsonr(df.loc[index].eeg[0], df.loc[index].eeg[4])[1]
    df.at[index, 'correlate_1_3'] = stats.stats.pearsonr(df.loc[index].eeg[1], df.loc[index].eeg[3])[0]
    df.at[index, 'pcorr_1_3'] = stats.stats.pearsonr(df.loc[index].eeg[1], df.loc[index].eeg[3])[1]
    df.at[index, 'correlate_2_7'] = stats.stats.pearsonr(df.loc[index].eeg[2], df.loc[index].eeg[7])[0]
    df.at[index, 'pcorr_2_7'] = stats.stats.pearsonr(df.loc[index].eeg[2], df.loc[index].eeg[7])[1]


def enrich_eeg_data(df):
    """
    Extract features for the entire dataframe

    Parameters
    ----------

     df : pandas DataFrame
        DataFrame for feature extraction

    """
    df = create_eeg_feature_columns(df.copy())

    for i in tqdm.tqdm(df.index):
        extract_eeg_features(df, i)

    print('EEG features have been extracted for the DataFrame\n'
          'for a model which predicts liking scores based on EEG signals use "build_liking_based_model"\n'
          'for a model which predicts whether or not a commercial is effective use "build_diff_based_model"')
    return df


def build_diff_based_model(ad_df, bdm_df, corr_plots=False):
    """
    This function takes a parsed, enriched ad experiment DaraFrame and combines it with the BDM session data to create 
    a difference between product scores metric which is the target for classification - we want to try and predict the 
    whether or not a commercial had a positive effect on the score of an item within a subject.

    Parameters
    ----------
    ad_df : pandas DataFrame
        The advetisment database after cleaning and enhanching
    bdm_df : pandas DataFrame
        The BDM database after cleaning
    corr_plots : Binary
        should the function print a correlation matrix? defaults to False

    Returns
    ----------
    X_train: pandas DataFrame
        The training set Xs

    X_test: pandas DataFrame
        The test set Xs

    y_train: pandas series
        The training labels

    y_test: pandas series
        The test set labels

    rf_searcher:
        The resulting models of the hyperparameter tunnings

    best_rf: sklearn random forest trained model
        The random forest model with the best score out of the hyperparameterization paradigm
    features: list
        The features used for the model

    """
    # insert 'pre_label' & 'label_diff' columns to ad_df
    ad_df.insert(ad_df.shape[1], column='pre_label', value=np.float32(0))
    ad_df.insert(ad_df.shape[1], column='label_diff', value=np.float32(0))

    # sort bdm_df by item
    bdm_sorted = bdm_df.sort_values(by='item', inplace=False)

    # Extract before commercial preferences
    before_pref = bdm_sorted.drop(columns=['eeg', 'category', 'repitition'])
    pre_ad = before_pref.rename(columns={'sub_id': 'sub_id',
                                         'label': 'pre_label',
                                         'item': 'item'})

    # for every item -- sub_id pairing calculate the difference before and after the commercial:
    print('extracting diff data from BDM session')
    for i in tqdm.tqdm(range(len(pre_ad))):
        line = pre_ad.iloc[i]
        item_pre = line['item']
        score = line['pre_label']
        sub_id_pre = line['sub_id']
        # find it in data
        ad_df.at[ad_df.query('sub_id == @sub_id_pre & item == @item_pre').index, 'pre_label'] = score

    ad_df.label_diff = ad_df.label - ad_df.pre_label
    print('diff column setup done, setting up training...')
    # Rename ad_df => df
    df = ad_df.copy()

    # stamp binary diff column
    df.insert(df.shape[1], column='binary_diff', value=np.float64(0))
    df.at[df.query('label_diff > 0').index, 'binary_diff'] = 1
    df.at[df.query('label_diff <= 0').index, 'binary_diff'] = 0

    # Split Test & Train Sets
    y = df.loc[:, 'binary_diff'].copy()
    X = df.drop(columns=(['sub_id',
                          'eeg',
                          'ad_id',
                          'item',
                          'label',
                          'repitition',
                          'item_id',
                          'liking',
                          'binary_diff',
                          'label_diff',
                          'pre_label']))
    X_numerical = X.drop(columns=['category'])

    # one-hot encoding for the item category
    X_dummy = pd.get_dummies(X, columns=['category'])

    # plot the correlation matrix
    if corr_plots:
        corr_mat = X_numerical.corr()
        _, ax = plt.subplots(figsize=(20, 20))
        plt.suptitle("Correlation Matrix")
        _ = sns.heatmap(corr_mat, annot=False, cmap='mako')
    features = X_dummy.columns

    # Test / Train Split
    X_train, X_test, y_train, y_test = train_test_split(X_dummy, y, test_size=0.33)
    train_ind = X_train.index

    # Scale data before training
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # set up the training set for a groupKfold procedure
    groups = df.loc[train_ind, :].sub_id
    N_SPLITS = 5

    # define parameter grid for hyperparameter tunning
    total_num_features = len(X.columns)
    N_ESTIMATORS = range(200, 205)
    MAX_FEATURES = range(int(np.sqrt(total_num_features)) - 2, int(np.sqrt(total_num_features) + 2))
    PARAM_GRID = {'max_features': MAX_FEATURES, 'n_estimators': N_ESTIMATORS}
    print('Training random forest model... This might take a few minutes')
    # Train the random forest model
    group_cv = GroupKFold(n_splits=N_SPLITS).split(X_train, y_train, groups=groups)
    rf = RandomForestClassifier(random_state=0)
    rf_searcher = GridSearchCV(rf, param_grid=PARAM_GRID, cv=group_cv, scoring='roc_auc')
    rf_searcher.fit(X_train, y_train)
    best_rf = rf_searcher.best_estimator_
    print('The model is trained! \n'
          'use the "plot_ROC", "plot_confusion_matrix" and "plot_feature_importance" functions '
          'to asses the model\'s performance')
    return X_train, X_test, y_train, y_test, rf_searcher, best_rf, features


def build_liking_based_model(df, liking_thr_low=2, liking_thr_high=5, corr_plots=False):

    """
    This function takes a parsed, enriched DataFrame and build a random forest classification model
    that predicts whether or not a trial resulted in "loving it" (liking > 5) or not (liking < 2)
    Note: The thrsholds can be moved to accommodate different liking distributions

    Parameters
    ----------

    df : pandas DataFrame
       A cleaned up, feature enriched DataFrame containing all of the features from the .mat file and all features
       extracted from the EEG signals
    liking_thr_low : float
        where should the threshold for "not liking" a commercial be? default = < 2
    liking_thr_high : float
        where should the threshold for "liking" a commercial be? default =  > 5
    corr_plots : Binary
       should the process produce a correlations matrix? prints if True

    Returns
    ----------
    X_train: pandas DataFrame
        The training set Xs

    X_test: pandas DataFrame
        The test set Xs

    y_train: pandas series
        The training labels

    y_test: pandas series
        The test set labels

    rf_searcher:
        The resulting models of the hyperparameter tunnings

    best_rf: sklearn random forest trained model
        The random forest model with the best score out of the hyperparameterization paradigm
    features: list
        The features used for the model
    """

    # Insert binary "loving it" column
    if 'loving_it' not in df.columns:
        df.insert(df.shape[1], column='loving_it', value=np.float32(0))
        df.at[df.query('liking > @liking_thr_high').index, 'loving_it'] = 2
        df.at[df.query('liking < @liking_thr_low').index, 'loving_it'] = 1
        df = df.query('loving_it != 0')

    # Split Test & Train Sets
    y = df.loc[:, 'loving_it'].copy()
    X = df.drop(columns=(['sub_id',
                          'eeg',
                          'ad_id',
                          'item',
                          'label',
                          'repitition',
                          'item_id',
                          'liking',
                          'loving_it']))
    X_numerical = X.drop(columns=['category'])

    # one-hot encoding for the item category
    X_dummy = pd.get_dummies(X, columns=['category'])

    features = X_dummy.columns

    # plot the correlation matrix
    if corr_plots:
        corr_mat = X_numerical.corr()
        _, ax = plt.subplots(figsize=(20, 20))
        plt.suptitle("Correlation Matrix")
        _ = sns.heatmap(corr_mat, annot=False, cmap='mako')

    # Test / Train Split
    X_train, X_test, y_train, y_test = train_test_split(X_dummy, y, test_size=0.33)
    train_ind = X_train.index

    # Scale data before training
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # set up the training set for a groupKfold procedure
    groups = df.loc[train_ind, :].sub_id
    N_SPLITS = 5

    # define parameter grid for hyperparameter tunning
    total_num_features = len(X.columns)
    N_ESTIMATORS = range(200, 205)
    MAX_FEATURES = range(int(np.sqrt(total_num_features)) - 2, int(np.sqrt(total_num_features) + 2))
    PARAM_GRID = {'max_features': MAX_FEATURES, 'n_estimators': N_ESTIMATORS}

    # Train the random forest model
    group_cv = GroupKFold(n_splits=N_SPLITS).split(X_train, y_train, groups=groups)
    rf = RandomForestClassifier(random_state=0)
    rf_searcher = GridSearchCV(rf, param_grid=PARAM_GRID, cv=group_cv, scoring='roc_auc')
    rf_searcher.fit(X_train, y_train)
    best_rf = rf_searcher.best_estimator_

    print('The model is trained! \n'
          'use the "plot_ROC", "plot_confusion_matrix" and "plot_feature_importance" functions '
          'to asses the model\'s performance')
    return X_train, X_test, y_train, y_test, rf_searcher, best_rf, features


def plot_ROC(model, name, X_test, y_test):
    """
    Plots ROC for a model

    Parameters
    ----------
     model : sklearn model
        model with predict_proba() method

     name : str
        name of model

     X_test, y_test : float
        np.arrays of input samples (X_test) and response variable (y_test)
    """
    metrics.plot_roc_curve(model, X_test, y_test)
    auc_score = metrics.roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f'{name} ROC-AUC Score: {auc_score:.3f}')
    _ = plt.title(f'{name} ROC curve')


def plot_confusion_matrix(model, name, X_test, y_test):
    """
    Plots confusion matrix for a model

    Parameters
    ----------
     model : sklearn model
        model with predict_proba() method

     name : str
        name of model

     X_test, y_test : float
        np.arrays of input samples (X_test) and response variable (y_test)
    """
    # Random Forest confusion matrix
    y_predict = model.predict(X_test)
    confmat = metrics.confusion_matrix(y_test, y_predict)
    label_names = ["True Negative", "False Positive", "False Negative", "True Positive"]
    label_counts = confmat.flatten()
    labels = [f"{v1}\n\n{v2}" for v1, v2 in zip(label_names, label_counts)]
    labels = np.asarray(labels).reshape(2, 2)
    _ = sns.heatmap(confmat, annot=labels, fmt='', cmap='Blues', cbar=False,
                    xticklabels=['Predicted Negative', 'Predicted Positive'],
                    yticklabels=['True Negative', 'True Positive'])
    _ = plt.title(f'{name} Model Confusion Matrix')


def plot_feature_importance(model, features, name, num_features):
    """
    Plots horizontal bar plot for top feature importance for a model

    Parameters
    ----------
     model : RandomForestClassifier / GradientBosstingClassifier
        model with feature_importance attribute

     features : pd.DataFrame , float
        features of the model

     name : str
        name of model

     num_features : int
         number of features to present in barplot

    """
    feature_importance = model.feature_importances_
    importance_df = pd.DataFrame({"Feature": features, "Feature importance": feature_importance})
    importance_df = importance_df.sort_values(by="Feature importance", ascending=False)
    importance_df = importance_df.reset_index(drop=True)
    _ = plt.figure(figsize=(8, 7))
    _ = plt.barh(importance_df.loc[num_features:0:-1, 'Feature'],
                 importance_df.loc[num_features:0:-1, 'Feature importance'])
    _ = plt.xlabel('Feature Importance')
    _ = plt.title(f'{name} Top {num_features} Features by Importance')
    _ = plt.yticks(fontsize=10)
