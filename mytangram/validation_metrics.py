# Validation metric functions
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy import stats


def ssim(raw, impute, scale=None):

    ###This was used for calculating the SSIM value between two arrays.

    if scale == 'scale_max':
        raw = scale_max(raw)
        impute = scale_max(impute)
    else:
        print('Please note you do not scale data by max')
    if raw.shape[1] == impute.shape[1]:
        result = pd.DataFrame()
        for label in raw.columns:
            raw_col = raw.loc[:, label]
            impute_col = impute.loc[:, label]

            M = [raw_col.max(), impute_col.max()][raw_col.max() > impute_col.max()]
            raw_col_2 = np.array(raw_col)
            raw_col_2 = raw_col_2.reshape(raw_col_2.shape[0], 1)

            impute_col_2 = np.array(impute_col)
            impute_col_2 = impute_col_2.reshape(impute_col_2.shape[0], 1)

            ssim = cal_ssim(raw_col_2, impute_col_2, M)

            ssim_df = pd.DataFrame(ssim, index=["SSIM"], columns=[label])
            result = pd.concat([result, ssim_df], axis=1)
        return result
    else:
        print("columns error")

def pearsonr(raw, impute, scale=None):

    ###This was used for calculating the Pearson value between two arrays.

    if raw.shape[1] == impute.shape[1]:
        result = pd.DataFrame()
        for label in raw.columns:
            raw_col = raw.loc[:, label]
            impute_col = impute.loc[:, label]
            pearsonr, _ = st.pearsonr(raw_col, impute_col)
            pearson_df = pd.DataFrame(pearsonr, index=["Pearson"], columns=[label])
            result = pd.concat([result, pearson_df], axis=1)
        return result

def JS(raw, impute, scale=None):

    ###This was used for calculating the JS value between two arrays.

    if scale == 'scale_plus':
        raw = scale_plus(raw)
        impute = scale_plus(impute)
    else:
        print('Please note you do not scale data by plus')
    if raw.shape[1] == impute.shape[1]:
        result = pd.DataFrame()
        for label in raw.columns:
            raw_col = raw.loc[:, label]
            impute_col = impute.loc[:, label]

            #M = (raw_col + impute_col) / 2
            M = pd.Series(data=sum(raw_col.values, impute_col.values), index=impute_col.index)
            KL = 0.5 * st.entropy(raw_col, M) + 0.5 * st.entropy(impute_col, M)
            KL_df = pd.DataFrame(KL, index=["JS"], columns=[label])

            result = pd.concat([result, KL_df], axis=1)
        return result

def RMSE(raw, impute, scale=None):

    ###This was used for calculating the RMSE value between two arrays.

    if scale == 'zscore':
        raw = scale_z_score(raw)
        impute = scale_z_score(impute)
    else:
        print('Please note you do not scale data by zscore')
    if raw.shape[1] == impute.shape[1]:
        result = pd.DataFrame()
        for label in raw.columns:
            raw_col = raw.loc[:, label]
            impute_col = impute.loc[:, label]

            temp = pd.Series(data=sum(raw_col.values, -impute_col.values), index=impute_col.index)
            RMSE = np.sqrt((temp ** 2).mean())
            RMSE_df = pd.DataFrame(RMSE, index=["RMSE"], columns=[label])

            result = pd.concat([result, RMSE_df], axis=1)
        return result


def cal_ssim(im1, im2, M):
    """
        calculate the SSIM value between two arrays.

    Parameters
        -------
        im1: array1, shape dimension = 2
        im2: array2, shape dimension = 2
        M: the max value in [im1, im2]

    """

    assert len(im1.shape) == 2 and len(im2.shape) == 2
    assert im1.shape == im2.shape
    mu1 = im1.mean()
    mu2 = im2.mean()
    sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
    sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
    sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
    k1, k2, L = 0.01, 0.03, M
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    C3 = C2 / 2
    l12 = (2 * mu1 * mu2 + C1) / (mu1 ** 2 + mu2 ** 2 + C1)
    c12 = (2 * sigma1 * sigma2 + C2) / (sigma1 ** 2 + sigma2 ** 2 + C2)
    s12 = (sigma12 + C3) / (sigma1 * sigma2 + C3)
    ssim = l12 * c12 * s12

    return ssim


def scale_max(df):
    """
        Divided by maximum value to scale the data between [0,1].
        Please note that these datafrmae are scaled data by column.
        Detail usages can be found in PredictGenes.ipynb


        Parameters
        -------
        df: dataframe, each col is a feature.

    """

    result = pd.DataFrame()
    for label, content in df.items():
        content = content / content.max()
        result = pd.concat([result, content], axis=1)
    return result


def scale_z_score(df):
    """
        scale the data by Z-score to conform the data to the standard normal distribution, that is, the mean value is 0, the standard deviation is 1, and the conversion function is 0.
        Please note that these datafrmae are scaled data by column.
        Detail usages can be found in PredictGenes.ipynb


        Parameters
        -------
        df: dataframe, each col is a feature.

        """

    result = pd.DataFrame()
    for label, content in df.items():
        content = stats.zscore(content)
        content = pd.DataFrame(content, columns=[label])
        result = pd.concat([result, content], axis=1)
    return result


def scale_plus(df):
    """
        Divided by the sum of the data to scale the data between (0,1), and the sum of data is 1.
        Please note that these datafrmae are scaled data by column.
        Detail usages can be found in PredictGenes.ipynb


        Parameters
        -------
        df: dataframe, each col is a feature.

    """

    result = pd.DataFrame()
    for label, content in df.items():
        content = content / content.sum()
        result = pd.concat([result, content], axis=1)
    return result