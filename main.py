import datetime as dt
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import t
import statsmodels.api as sm
import collections
from sklearn.metrics import mean_squared_error
import argparse

from utils import *

import matplotlib.pyplot as plt
plt.style.use('ggplot') # fivethirtyeight, ggplot, dark_background, classic,  
import warnings
warnings.filterwarnings("ignore")


class ForecastingAnalysis:
    """
    This class examines various statistical time series forecasting methods using historical financial markets data. 
    """

    def __init__(self, sp500_full_df, bond_full_df, rf_ret_df):
        """
        Initialize the ForecastingAnalysis object with S&P 500, bond, and risk-free rate data.

        Parameters
        ----------
        sp500_full_df : pd.DataFrame
            Monthly S&P 500 stock index price data from Dec 1979 to Dec 2021.
        bond_full_df : pd.DataFrame
            Monthly Bloomberg Barclays U.S. Aggregate Bond Index price data from Dec 1979 to Dec 2021.
        rf_ret_df : pd.DataFrame
            Monthly risk-free rate data from Jan 1980 to Dec 2021.
        """
        self.sp500_full_df = sp500_full_df
        self.bond_full_df = bond_full_df
        self.rf_ret_df = rf_ret_df

    def question2(self, simp_ret_df, breakpoint, method, window_size=None):
        """
        Generate mean benchmark forecasts for the given data and parameters.

        Parameters
        ----------
        simp_ret_df : pd.DataFrame
            A DataFrame containing the simple excess returns data for two assets: 
            'SP500_excess' and 'LBUSTRUU_excess'
        breakpoint : int or str
            The breakpoint used for splitting the data into training and testing sets representing 
            a date in the format
        method : str
            The method used for generating the benchmark forecast. 
            Supported methods: 'recursive' or 'rolling'.
        window_size : int, optional
            The size of the rolling window for calculating rolling method. 
            This parameter is required if the method is 'rolling'. 
            Default is None.

        Returns
        -------
        mean_benchmark_forecasts_df : pd.DataFrame
            A DataFrame containing the mean benchmark forecasts for the given data and parameters.
        """
        mean_benchmark_forecasts_df = generate_mean_benchmark_forecast(simp_ret_df.loc[:, ['SP500_excess', 'LBUSTRUU_excess']], breakpoint, method=method, window_size=window_size if window_size is not None else None)
        return mean_benchmark_forecasts_df

    def question3(self, simp_ret_df, predictors_df, breakpoint, asset_tickers, mean_benchmark_forecasts_df, method, window_size=None):

        """
        Generate various forecasts, compute Mean Squared Forecast Errors (MSFE), and perform Diebold-Mariano (DM) tests for equal predictive ability.

        Paramters
        ---------
        simp_ret_df : pd.DataFrame
            A DataFrame containing the simple excess returns data for two assets: 
            'SP500_excess' and 'LBUSTRUU_excess'
        predictors_df: pd.DataFrame
            A DataFrame containing the predictor variables.
        breakpoint : int or str
            The breakpoint used for splitting the data into training and testing sets representing 
            a date in the format
        asset_tickers: list
            A list of asset tickers, e.g. ['SP500', 'LBUSTRUU'].
        mean_benchmark_forecasts_df: pd.DataFrame
            A DataFrame containing the mean benchmark forecasts for the assets.
        method : str
            The method used for generating the benchmark forecast. 
            Supported methods: 'recursive' or 'rolling'.
        window_size : int, optional
            The size of the rolling window for calculating rolling method. 
            This parameter is required if the method is 'rolling'. 
            Default is None.

        Returns
        -------
        A tuple containing the following DataFrames:
        1) ols_predictor_forecasts_df: Forecasts for each asset using OLS fit on predictors.
        2) combination_mean_forecasts_df: Forecasts for each asset using the combination mean method.
        3) plr_predictor_forecasts_df: Forecasts for each asset using PLR fit on predictors.
        4) msfe_df: Mean Squared Forecast Errors for all predictive models and corresponding benchmarks for both assets.
        5) msfe_ratios_to_benchmark: Ratios of MSFEs of predictive models to corresponding benchmark MSFE values.
        6) dm_test_stats: Diebold-Mariano test statistics and p-values for checking the equal predictive ability of all predictive models with the benchmark.
        7) sp500_all_models: DataFrame containing all forecasts for S&P 500.
        8) bond_all_models: DataFrame containing all forecasts for US Aggregate Bond Index.

        """




        # SP500 predictors
        sp500_predictors = predictors_df.loc[:, ['infl', 'b/m', 'svar', 'ntis']]
        sp500_predictors['div_payout'] = np.log(predictors_df['D12']/predictors_df['E12'])

        # Bond predictors
        bond_predictors = predictors_df.loc[:, ['infl', 'tbl', 'lty', 'ltr']]
        bond_predictors['DFY'] = predictors_df['BAA'] - predictors_df['AAA']

        # out-of-sample return forecasts for SP500 predicted using OLS fit on 5 different predictors
        sp500_ols_forecasts_df = generate_ols_predictor_forecast(simp_ret_df.loc[:, ['SP500_excess']], sp500_predictors,
                                                                 breakpoint=breakpoint, method=method, window_size=window_size if window_size is not None else None)
        # out-of-sample return forecasts for bonds predicted using OLS fit on 5 different predictors
        bond_ols_forecasts_df = generate_ols_predictor_forecast(simp_ret_df.loc[:, ['LBUSTRUU_excess']], bond_predictors,
                                                                breakpoint=breakpoint, method=method, window_size=window_size if window_size is not None else None)
        # Rename the columns in each data frame
        sp500_ols_forecasts_df.columns = pd.MultiIndex.from_product([[asset_tickers[0]], sp500_ols_forecasts_df.columns])
        bond_ols_forecasts_df.columns = pd.MultiIndex.from_product([[asset_tickers[1]], bond_ols_forecasts_df.columns])
        # Concatenate the data frames along the columns axis and create a multi-level column index
        ols_predictor_forecasts_df = pd.concat([sp500_ols_forecasts_df, bond_ols_forecasts_df], axis=1)


        # Compute Combination mean forecasts 
        combination_mean_forecasts_df = generate_combination_mean_forecasts(sp500_ols_forecasts_df, bond_ols_forecasts_df, assets_names=asset_tickers)

        # out-of-sample return forecasts for SP500 predicted using PLR fit on predictors 
        sp500_plr_forecasts_df = generate_plr_forecast(simp_ret_df.loc[:, ['SP500_excess']], sp500_predictors,
                                                                 breakpoint=breakpoint, method=method, window_size=window_size if window_size is not None else None)
        # out-of-sample return forecasts for bonds predicted using PLR fit on predictors
        bond_plr_forecasts_df = generate_plr_forecast(simp_ret_df.loc[:, ['LBUSTRUU_excess']], bond_predictors,
                                                                breakpoint=breakpoint, method=method, window_size=window_size if window_size is not None else None)
        # Rename the columns in each data frame
        sp500_plr_forecasts_df.columns = pd.MultiIndex.from_product([[asset_tickers[0]], sp500_plr_forecasts_df.columns])
        bond_plr_forecasts_df.columns = pd.MultiIndex.from_product([[asset_tickers[1]], bond_plr_forecasts_df.columns])
        # Concatenate the data frames along the columns axis and create a multi-level column index
        plr_predictor_forecasts_df = pd.concat([sp500_plr_forecasts_df, bond_plr_forecasts_df], axis=1)

        # combine benchamark & all 8 forecast model predictions for each asset class 
        sp500_all_models = pd.concat([mean_benchmark_forecasts_df.loc[:, ['SP500_MB']], sp500_ols_forecasts_df['SP500'], 
                                      combination_mean_forecasts_df.loc[:, ['SP500_Comb_Mean']], sp500_plr_forecasts_df['SP500']], axis=1)
        sp500_all_models.rename(columns={'SP500_MB': 'Benchmark', 'SP500_Comb_Mean': 'Combination_mean'}, inplace=True)
        bond_all_models = pd.concat([mean_benchmark_forecasts_df.loc[:, ['LBUSTRUU_MB']], bond_ols_forecasts_df['LBUSTRUU'], 
                                      combination_mean_forecasts_df.loc[:, ['LBUSTRUU_Comb_Mean']], bond_plr_forecasts_df['LBUSTRUU']], axis=1)
        bond_all_models.rename(columns={'LBUSTRUU_MB': 'Benchmark', 'LBUSTRUU_Comb_Mean': 'Combination_mean'}, inplace=True)   

        # Compute MSFE for all predictive models & corresponding benchmarks for both the assets
        msfe_df = pd.DataFrame(index=asset_tickers, columns=['Benchmark', 'OLS_1', 'OLS_2', 'OLS_3', 'OLS_4', 'OLS_5', 'CM', 'PLR_1', 'PLR_2'])
        msfe_df.loc[asset_tickers[0]] = [mean_squared_error(simp_ret_df.loc[simp_ret_df.index > breakpoint, ['SP500_excess']], sp500_all_models[x]) 
                                for x in sp500_all_models.columns]
        msfe_df.loc[asset_tickers[1]] = [mean_squared_error(simp_ret_df.loc[simp_ret_df.index > breakpoint, ['LBUSTRUU_excess']], bond_all_models[x]) 
                                for x in bond_all_models.columns]

        # Compute the ratios of MSFEs of predictive models to corresponding benchmark MSFE values
        msfe_ratios_to_benchmark = pd.DataFrame(index=asset_tickers, columns=['OLS_1', 'OLS_2', 'OLS_3', 'OLS_4', 'OLS_5', 'CM', 'PLR_1', 'PLR_2'])
        msfe_ratios_to_benchmark.loc[asset_tickers[0]] = [msfe_df.loc[asset_tickers[0], [x]].values[0] / msfe_df.loc[asset_tickers[0], ['Benchmark']].values[0] for x in msfe_df.columns[1:]]
        msfe_ratios_to_benchmark.loc[asset_tickers[1]] = [msfe_df.loc[asset_tickers[1], [x]].values[0] / msfe_df.loc[asset_tickers[1], ['Benchmark']].values[0] for x in msfe_df.columns[1:]]

        # Check for equal predictive ability of all 8 predictive models with the benchmark using DM test
        dm_test_stats = pd.DataFrame(index=pd.MultiIndex.from_product([asset_tickers, ['dm_tstat', 'p_val']]), 
                                              columns=['OLS_1', 'OLS_2', 'OLS_3', 'OLS_4', 'OLS_5', 'CM', 'PLR_1', 'PLR_2'])
        dm_test_stats.loc[(asset_tickers[0], 'dm_tstat')] = [dm_test(real_values=simp_ret_df.loc[simp_ret_df.index > breakpoint, 'SP500_excess'].values,
                                                                     pred1=sp500_all_models[col].values, pred2=sp500_all_models['Benchmark'].values)[0]
                                                                       for col in sp500_all_models.columns[1:]]
        dm_test_stats.loc[(asset_tickers[0], 'p_val')] = [dm_test(real_values=simp_ret_df.loc[simp_ret_df.index > breakpoint, 'SP500_excess'].values,
                                                                     pred1=sp500_all_models[col].values, pred2=sp500_all_models['Benchmark'].values)[1]
                                                                       for col in sp500_all_models.columns[1:]]
        dm_test_stats.loc[(asset_tickers[1], 'dm_tstat')] = [dm_test(real_values=simp_ret_df.loc[simp_ret_df.index > breakpoint, 'LBUSTRUU_excess'].values,
                                                                     pred1=bond_all_models[col].values, pred2=bond_all_models['Benchmark'].values)[0]
                                                                       for col in bond_all_models.columns[1:]]
        dm_test_stats.loc[(asset_tickers[1], 'p_val')] = [dm_test(real_values=simp_ret_df.loc[simp_ret_df.index > breakpoint, 'LBUSTRUU_excess'].values,
                                                                     pred1=bond_all_models[col].values, pred2=bond_all_models['Benchmark'].values)[1]
                                                                       for col in bond_all_models.columns[1:]]

        return ols_predictor_forecasts_df, combination_mean_forecasts_df, plr_predictor_forecasts_df, msfe_df, msfe_ratios_to_benchmark, dm_test_stats, sp500_all_models, bond_all_models

    def plot_forecast(self, sp500_all_models, bond_all_models, method):

        """
        Plot the out-of-sample forecasts for S&P 500 and US Aggregate Bond Index.

        Parameters
        ----------
        sp500_all_models : pd.DataFrame
            A DataFrame containing the out-of-sample forecasts for the S&P 500.
        bond_all_models : pd.DataFrame
            A DataFrame containing the out-of-sample forecasts for the US Aggregate Bond Index.
        method : str
            The method used for generating the benchmark forecast. 
            Supported methods: 'recursive' or 'rolling'.

        Returns
        -------
        This function returns a plot showing out-of-sample forecasts for S&P 500 and US Aggregate Bond Index.
        """

        plt.figure(figsize=(12, 12))
        desired_models = ['Benchmark', 'Combination_mean', 'Lasso', 'Ridge']

        plt.subplot(2, 1, 1)
        plt.plot(sp500_all_models.index, sp500_all_models.loc[:, desired_models].values)
        plt.title(f'Out of sample {method} forecasts for S&P 500', fontweight='bold', fontsize=14)
        plt.xlabel('Date', weight='bold', fontsize=12)
        plt.ylabel('Excess return', weight='bold', fontsize=12)
        plt.legend(desired_models)

        plt.subplot(2, 1, 2)
        plt.plot(bond_all_models.index, bond_all_models.loc[:, desired_models].values)
        plt.title(f'Out of sample {method} forecasts for US Aggregate Bond Index', fontweight='bold', fontsize=14)
        plt.xlabel('Date', weight='bold', fontsize=12)
        plt.ylabel('Excess return', weight='bold', fontsize=12)
        plt.legend(desired_models)

        plt.tight_layout(pad=2.0)
        plt.show()

    def question4(self, simp_ret_df, breakpoint, method, window_size=None):
        """
        Generates a timeseries of monthly out-of-sample variance-covariance 
        matrix forecasts for a portfolio of assets.

        Parameters
        ----------
        simp_ret_df : pd.DataFrame
            A DataFrame containing the simple excess returns data for two assets: 
            'SP500_excess' and 'LBUSTRUU_excess'
        breakpoint : int or str
            The breakpoint used for splitting the data into training and testing sets representing 
            a date in the format
        method : str
            The method used for generating the benchmark forecast. 
            Supported methods: 'recursive' or 'rolling'.
        window_size : int, optional
            The size of the rolling window for calculating rolling method. 
            This parameter is required if the method is 'rolling'. 
            Default is None.

        Returns
        -------
        portf_cov_mat_forecasts_df : pd.DataFrame
            Monthly out-of-sample forecast of variance-covariance matrices for a portfolio of assets.
        """

        portf_cov_mat_forecasts_df = generate_portfolio_var_cov_mat_forecast(simp_ret_df.loc[:, ['SP500_excess', 'LBUSTRUU_excess']], 
                                                                              breakpoint=breakpoint, method=method, window_size=window_size if window_size is not None else None)
        return portf_cov_mat_forecasts_df

    def question5(self, sp500_all_models, bond_all_models, portf_cov_mat_forecasts_df):

        """
        Generate OTP excess returns, summary statistics, and out-of-sample weights for various forecast models.

        Parameters
        ----------
        sp500_all_models : pd.DataFrame
            A DataFrame containing all forecasts for S&P 500.
        bond_all_models : pd.DataFrame
            A DataFrame containing all forecasts for US Aggregate Bond Index.
        portf_cov_mat_forecasts_df : pd.DataFrame
            A DataFrame containing the portfolio covariance matrix forecasts.

        Returns
        -------
        A tuple containing the following DataFrames:
        1) otp_excess_ret_all_models: A DataFrame with the OTP excess returns for each forecast model.
        2) portf_excess_ret_summary_stats: A DataFrame with the summary statistics of the portfolio excess returns for each forecast model.
        3) otp_oos_weights_all_models: A DataFrame with the out-of-sample weights for each forecast model.
        """


        # placeholder for storing list of individual DFs with asset return forecasts computed using all the models 
        temp_list = [pd.concat([sp500_all_models.iloc[:, i], bond_all_models.iloc[:, i]], axis=1) for i in range(len(sp500_all_models.columns))]
        # placeholder for OTP excess return (for all model forecasts) 
        otp_excess_ret_all_models = pd.DataFrame(index=sp500_all_models.index, 
                                                 columns=['Benchmark', 'OLS_1', 'OLS_2', 'OLS_3', 'OLS_4', 'OLS_5', 'CM', 'PLR_1', 'PLR_2'])
        # placeholder for OTP out-of-sample weights (for all model forecasts)
        otp_oos_weights_all_models = otp_excess_ret_all_models.copy()
        for i in range(len(temp_list)):
            # compute time-series of monthly OTP excess return for all the forecast models
            otp_excess_ret_all_models.iloc[:, i], otp_oos_weights_all_models.iloc[:, i] = generate_OTP_excess_ret(temp_list[i], portf_cov_mat_forecasts_df)

        # Compute Portfolio excess return summary statistics 
        portf_excess_ret_summary_stats = pd.DataFrame(index=['Mean', 'Volatility', 'Sharpe', 'Skew', 'Kurtosis'], 
                                                      columns=otp_excess_ret_all_models.columns)
        for i in range(len(portf_excess_ret_summary_stats.columns)):
            portf_excess_ret_summary_stats.iloc[:, i] = compute_stat_measures(otp_excess_ret_all_models.iloc[:, i].values)

        return otp_excess_ret_all_models, portf_excess_ret_summary_stats, otp_oos_weights_all_models

    def plot_portfolio_asset_allocation(self, otp_oos_weights_all_models, method):
        """
        Plot the portfolio asset allocation for various forecast models.

        Parameters
        ----------
        otp_oos_weights_all_models : pd.DataFrame
            A DataFrame containing the out-of-sample weights for each forecast model.
        method : str
            The method used for generating the benchmark forecast. 
            Supported methods: 'recursive' or 'rolling'.

        Returns
        -------
        This function returns a plot showing the asset allocation with the specified method's forecasts.

        """
        # Plot portfolio asset allocation for various forecast models 
        plt.figure(figsize=(10, 6))
        plt.title(f"Portfolio asset allocation with {method} forecasts (S&P 500 Index vs US aggregate Bond Index)\n", fontweight='bold', fontsize=12)
        plt.plot(otp_oos_weights_all_models.index, [x[0]/x[1] for x in otp_oos_weights_all_models['Benchmark'].values], label='Benchmark')
        plt.plot(otp_oos_weights_all_models.index, [x[0]/x[1] for x in otp_oos_weights_all_models['CM'].values], label='Combination mean')
        plt.plot(otp_oos_weights_all_models.index, [x[0]/x[1] for x in otp_oos_weights_all_models['PLR_1'].values], label='Lasso')
        plt.plot(otp_oos_weights_all_models.index, [x[0]/x[1] for x in otp_oos_weights_all_models['PLR_2'].values], label='Ridge')
        plt.ylabel("S&P 500 / US Bond weights ratio")
        plt.xlabel("Date")
        plt.tight_layout(pad=2.0)
        plt.legend(loc='best')
        plt.show()

    def plot_portfolio_returns(self, otp_excess_ret_all_models, method):
        """
        Plot the portfolio excess returns for various forecast models.

        Parameters
        ----------
        otp_excess_ret_all_models : pd.DataFrame
            A DataFrame containing the excess returns for each forecast model.
        method : str
            The method used for generating the benchmark forecast. 
            Supported methods: 'recursive' or 'rolling'.

        Returns
        -------
        This function returns a plot showing the portfolio excess returns for various forecast models.

        """
        # Plot portfolio daily and cummulative returns for various forecast models 
        plt.figure(figsize=(10, 6))
        plt.title(f"Portfolio cummulative excess return {method}\n", fontweight='bold', fontsize=12)
        plt.plot(otp_excess_ret_all_models.index, otp_excess_ret_all_models['Benchmark'].cumsum() * 100, label='Benchmark')
        plt.plot(otp_excess_ret_all_models.index, otp_excess_ret_all_models['CM'].cumsum() * 100, label='Combination mean')
        plt.plot(otp_excess_ret_all_models.index, otp_excess_ret_all_models['PLR_1'].cumsum() * 100, label='Lasso')
        plt.plot(otp_excess_ret_all_models.index, otp_excess_ret_all_models['PLR_2'].cumsum() * 100, label='Ridge')
        plt.ylabel("Percentage return (%)")
        plt.xlabel("Date")
        plt.tight_layout(pad=2.0)
        plt.legend(loc='best')
        plt.show()

def main(method, rolling_window_size):

    """
    Perform forecasting analysis for S&P 500 and US Aggregate Bond index using various predictive models.

    Parameters
    ----------
    method : str
        The method to use for forecasting analysis: either 'recursive' or 'rolling'.
    rolling_window_size : int
        The size of the rolling window for the rolling method. Ignored if method is 'recursive'.
    """

    # Load data
    sp500_full_df, bond_full_df, rf_ret_df, predictors_df = get_market_data()
    breakpoint = dt.datetime(2000, 1, 1)
    asset_tickers = ['SP500', 'LBUSTRUU']
    simp_ret_df = generate_asset_simple_rets(sp500_full_df, bond_full_df, rf_ret_df)

    # Create instance of ForecastingAnalysis class
    forecasting_analysis = ForecastingAnalysis(sp500_full_df, bond_full_df, rf_ret_df)


    ##---Question 1---##
    sp_mean, sp_vol, sp_sharpe, sp_skew, sp_kurt = compute_stat_measures(simp_ret_df['SP500_excess'].values)
    bond_mean, bond_vol, bond_sharpe, bond_skew, bond_kurt = compute_stat_measures(simp_ret_df['LBUSTRUU_excess'].values)

    asset_ret_summary_stats = pd.DataFrame({'SP500': [sp_mean, sp_vol, sp_sharpe, sp_skew, sp_kurt], 
                                            'LBUSTRUU': [bond_mean, bond_vol, bond_sharpe, bond_skew, bond_kurt]}, 
                                            index=['Mean', 'Volatility', 'Sharpe', 'Skew', 'Kurtosis'])
    print('\nAsset class excess return summary statistics (total time period)')
    print('-' * 100)
    print(asset_ret_summary_stats)
    ##---End of Question 1---##

    if method == 'recursive':
        ##---Question 2---##
        mean_benchmark_forecasts_df = forecasting_analysis.question2(simp_ret_df, breakpoint, method=method)
        print(f'\nOut-of-sample Mean Benchmark {method} forecasts')
        print('-' * 100)
        print(mean_benchmark_forecasts_df)
        ##---End of Question 2---##

        ##---Question 3---##
        ols_predictor_forecasts_df, combination_mean_forecasts_df, plr_predictor_forecasts_df, msfe_df, msfe_ratios_to_benchmark, dm_test_stats, sp500_all_models, bond_all_models = forecasting_analysis.question3(simp_ret_df, predictors_df, breakpoint, asset_tickers, mean_benchmark_forecasts_df, method)
        print(f'\nOut-of-sample OLS Predictor {method} forecasts')
        print('-' * 100)
        print(ols_predictor_forecasts_df)
        print(f'\nOut-of-sample Combination Mean {method} forecasts')
        print('-' * 100)
        print(combination_mean_forecasts_df)
        print(f'\nOut-of-sample Penalised Linear Regression {method} forecasts (using all Predictors)')
        print('-' * 100)
        print(plr_predictor_forecasts_df)
        print(f'\nMSFE values for all 9 {method} predictive models including benchmark (both asset classes)')
        print('-' * 100)
        print(msfe_df)
        print(f'\nRatios of MSFE values of all 8 {method} predictive models to corresponding benchmark forecasts MSFEs (both asset classes)')
        print('-' * 100)
        print(msfe_ratios_to_benchmark)
        print(f'\nDM test to check for equal predictive ability relative to mean benchmark forecasts (all 8 {method} predictive models for both asset classes)')
        print('-' * 100)
        print(dm_test_stats)
        forecasting_analysis.plot_forecast(sp500_all_models, bond_all_models, method)
        ##---End of Question 3---##

        ##---Question 4---##
        portf_cov_mat_forecasts_df = forecasting_analysis.question4(simp_ret_df, breakpoint, method)
        print(f'\nOut-of-sample variance-covariance matrix {method} forecasts')
        print('-' * 100)
        print(portf_cov_mat_forecasts_df)
        ##---End of Question 4---##


        ##---Question 5---##
        otp_excess_ret_all_models, portf_excess_ret_summary_stats, otp_oos_weights_all_models = forecasting_analysis.question5(sp500_all_models, bond_all_models, portf_cov_mat_forecasts_df)
        print(f'\nOTP out-of-sample excess returns (all 9 {method} predictive models)')
        print('-' * 100)
        print(otp_excess_ret_all_models)
        print(f'\nOTP out-of-sample excess returns summary statistics (all 9 {method} predictive models)')
        print('-' * 100)
        print(portf_excess_ret_summary_stats)
        forecasting_analysis.plot_portfolio_asset_allocation(otp_oos_weights_all_models, method)
        forecasting_analysis.plot_portfolio_returns(otp_excess_ret_all_models, method)
        ##---End of Question 5---##

    else:

        ##---Question 6---##
        mean_benchmark_forecasts_df = forecasting_analysis.question2(simp_ret_df, breakpoint, method=method, window_size=rolling_window_size)
        print(f'\nOut-of-sample Mean Benchmark {method} forecasts window size {rolling_window_size}')
        print('-' * 100)
        print(mean_benchmark_forecasts_df)

        ols_predictor_forecasts_df, combination_mean_forecasts_df, plr_predictor_forecasts_df, msfe_df, msfe_ratios_to_benchmark, dm_test_stats, sp500_all_models, bond_all_models = forecasting_analysis.question3(simp_ret_df, predictors_df, breakpoint, asset_tickers, mean_benchmark_forecasts_df, method, rolling_window_size)
        print(f'\nOut-of-sample OLS Predictor {method} forecasts window size {rolling_window_size}')
        print('-' * 100)
        print(ols_predictor_forecasts_df)
        print(f'\nOut-of-sample Combination Mean {method} forecasts window size {rolling_window_size}')
        print('-' * 100)
        print(combination_mean_forecasts_df)
        print(f'\nOut-of-sample Penalised Linear Regression {method} forecasts (using all Predictors; window size {rolling_window_size})')
        print('-' * 100)
        print(plr_predictor_forecasts_df)
        print(f'\nMSFE values for all 9 {method} predictive models including benchmark (both asset classes)')
        print('-' * 100)
        print(msfe_df)
        print(f'\nRatios of MSFE values of all 8 {method} predictive models to corresponding benchmark forecasts MSFEs (both asset classes)')
        print('-' * 100)
        print(msfe_ratios_to_benchmark)
        print(f'\nDM test to check for equal predictive ability relative to mean benchmark forecasts (all 8 {method} predictive models for both asset classes with window size {rolling_window_size})')
        print('-' * 100)
        print(dm_test_stats)
        forecasting_analysis.plot_forecast(sp500_all_models, bond_all_models, method)

        portf_cov_mat_forecasts_df = forecasting_analysis.question4(simp_ret_df, breakpoint, method, rolling_window_size)
        print(f'\nOut-of-sample variance-covariance matrix {method} forecasts window size {rolling_window_size}')
        print('-' * 100)
        print(portf_cov_mat_forecasts_df)

        otp_excess_ret_all_models, portf_excess_ret_summary_stats, otp_oos_weights_all_models = forecasting_analysis.question5(sp500_all_models, bond_all_models, portf_cov_mat_forecasts_df)
        print(f'\nOTP out-of-sample excess returns (all 9 {method} predictive models with window size {rolling_window_size})')
        print('-' * 100)
        print(otp_excess_ret_all_models)
        print(f'\nOTP out-of-sample excess returns summary statistics (all 9 {method} predictive models with window size {rolling_window_size})')
        print('-' * 100)
        print(portf_excess_ret_summary_stats)
        forecasting_analysis.plot_portfolio_asset_allocation(otp_oos_weights_all_models, method)
        forecasting_analysis.plot_portfolio_returns(otp_excess_ret_all_models, method)
        ##---End of Question 6---##


def parse_arguments():
    """
    Parse command-line arguments for the forecasting analysis script.
    
    Returns
    -------
    args : argparse.Namespace
        An object containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Forecasting Analysis")
    parser.add_argument("-m", "--method", choices=["recursive", "rolling"], default="recursive", help="Forecasting method: 'recursive' or 'rolling'. Default is 'recursive'.")
    parser.add_argument("-w", "--window", type=int, default=240, help="Rolling window size (only used if method is 'rolling'). Default is 240.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Parse command-line arguments and call the main function with the provided arguments.
    args = parse_arguments()
    main(args.method, args.window)

