import datetime as dt
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import t
import statsmodels.api as sm
import collections
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error

def get_market_data():
    """
    Fetch historical market data on asset classes (stock & bond index), 
    risk-free rate & predictors.

    Returns: 
    --------
    sp500_full_df : pd.Dataframe
        Monthly S&P 500 stock index price data from Dec 1979 to Dec 2021.
    bond_full_df : pd.Dataframe
        Monthly Bloomberg Barclays U.S. Aggregate Bond Index price data from Dec 1979 to Dec 2021.
    rf_ret_df : pd.Dataframe
        Monthly risk-free rate data from Jan 1980 to Dec 2021.
    predictors_df : pd.Dataframe
        Monthly predictors data from Jan 1980 to Dec 2021.
    """
    # fetch data for S&P 500 stock index (SP500) & Bloomberg Barclays U.S. Aggregate Bond Index (LBUSTRUU) 
    sp500_full_df = pd.read_excel('./data/S&P_500_stock_index.xlsx')
    bond_full_df = pd.read_excel('./data/US_Aggregate_Bond_index.xlsx')
    # fetch risk-free rate from Professor Kenneth French’s data library
    rf_ret_df = pd.read_excel('./data/Risk-free_rate_of_return.xlsx')

    # fetch predictors data 
    predictors_df = pd.read_excel('./data/PredictorData2022.xlsx', sheet_name='Monthly')
    predictors_df['Date'] = pd.to_datetime(predictors_df['yyyymm'], format='%Y%m')
    predictors_df.drop(columns=['yyyymm'], inplace=True)
    predictors_df.set_index(['Date'], drop=True, inplace=True)
    predictors_df = predictors_df[(predictors_df.index > dt.datetime(1979, 12, 1)) & (predictors_df.index < dt.datetime(2022, 1, 1))]

    return sp500_full_df, bond_full_df, rf_ret_df, predictors_df

def generate_asset_simple_rets(sp500_data: pd.DataFrame, bond_data: pd.DataFrame, rf_data: pd.DataFrame):
    """
    Generate simple returns and excess returns for the defined asset classes.

    Parameters:
    -----------
    sp500_data : pd.DataFrame
        S&P 500 stock index price data.
    bond_data : pd.DataFrame
        Bloomberg Barclays U.S. Aggregate Bond Index price data.
    rf_data : pd.DataFrame
        risk-free rate data.

    Returns: 
    --------
    simp_ret_df : pd.Dataframe
        Monthly simple return & excess return data for both the assets from Jan 1980 to Dec 2021.
        All data combined in a single dataframe for ease of use.
    """
    assets_df = pd.merge(left=sp500_data, right=bond_data, on='Dates', how='inner')
    assets_df.set_index(['Dates'], drop=True, inplace=True)
    # compute simple percentage return
    simp_ret_df = assets_df.pct_change()
    simp_ret_df.dropna(inplace=True)
    simp_ret_df = pd.merge(left=simp_ret_df, right=rf_data, left_on='Dates', right_on='Date', how='inner')
    simp_ret_df.set_index(['Date'], drop=True, inplace=True)
    # compute excess return over risk-free rate
    simp_ret_df['SP500_excess'] = simp_ret_df['SP500 index'] - simp_ret_df['Risk free rate of return ']
    simp_ret_df['LBUSTRUU_excess'] = simp_ret_df['LBUSTRUU Index '] - simp_ret_df['Risk free rate of return ']
    
    return simp_ret_df

def compute_stat_measures(values):
    """
    Computes summary statistics for input return data.

    Parameters:
    -----------
    values : ndarray
        Asset (excess) returns data.

    Returns: 
    --------
    mean : float
        Annualised mean of excess return.
    vol : float
        Annualised volatility of excess return.
    sharpe : float
        Annualised sharpe ratio.
    skew : float
        Skewness of excess return.
    kurt : float
        Kurtosis (Pearson) of excess return.
    """
    # Compute annualised mean: mean
    mean = np.nanmean(values) * 12
    # Compute annualised vol: std
    vol = np.nanstd(values, ddof=1) * np.sqrt(12)
    # Compute annualised sharpe ratio: mean/std
    sharpe = mean / vol
    # Compute skew: skew
    skew = stats.skew(values)
    # Compute kurtosis: kurtosis
    kurt = stats.kurtosis(values, fisher=False)

    return mean, vol, sharpe, skew, kurt

def generate_mean_benchmark_forecast(excess_ret: pd.DataFrame, breakpoint: dt.datetime):
    """
    Generates a timeseries of monthly out-of-sample mean excess return forecasts
    for the defined assets using recursive estimation approach.

    Parameters:
    -----------
    excess_ret : pd.DataFrame
        Assets (excess) returns data (for both asset classes).
    breakpoint : datetime
        datetime to split the data into in-sample & out-of-sample.

    Returns: 
    --------
    mean_forecast : pd.DataFrame
        Monthly out-of-sample mean excess forecasts for all assets.
    """
    # define out-of-sample period by splitting data at the breakpoint
    out_of_sample_period = excess_ret[excess_ret.index > breakpoint].index
    # placeholder for out-of-sample mean forecasts 
    mean_forecast = pd.DataFrame(index=out_of_sample_period)

    # Loop over the out-of-sample period and compute
    # mean forecasts using recursive estimation approach
    for t in out_of_sample_period:
        # Update the in-sample period to be considered 
        # recursively to include all data up to period t
        curr_in_sample = excess_ret[excess_ret.index < t]
        # Compute the expected (mean) excess return for each asset using the current in-sample period
        for asset in curr_in_sample.columns:
            mean_forecast.loc[t, f"{asset.split('_')[0]}_MB"] = np.nanmean(curr_in_sample[asset].values)

    return mean_forecast

def generate_ols_predictor_forecast(asset_excess_ret: pd.DataFrame, predictors: pd.DataFrame, breakpoint: dt.datetime):
    """
    Generates a timeseries of monthly out-of-sample OLS return forecasts
    using the defined predictors for a specific asset class.

    Parameters:
    -----------
    asset_excess_ret : pd.DataFrame
        Asset (excess) returns data (for one asset class).
    predictors : pd.DataFrame
        Predictors data.
    breakpoint : datetime
        datetime to split the data into in-sample & out-of-sample.

    Returns: 
    --------
    ols_forecasts : pd.DataFrame
        Monthly out-of-sample OLS forecasts generated by 
        fitting over all individual predictors.
    """
    # define out-of-sample period by splitting data at the breakpoint
    out_of_sample_period = asset_excess_ret[asset_excess_ret.index > breakpoint].index 
    # placeholder for out-of-sample OLS forecasts
    ols_forecasts = pd.DataFrame(index=out_of_sample_period, columns=predictors.columns)

    for pred in predictors.columns:
        pred_data = predictors.loc[:, [pred]]
        pred_forecasts = []

        for t in out_of_sample_period:
            # Update the in-sample period to be considered 
            # recursively to include all data up to period t
            curr_in_sample_asset_ret = asset_excess_ret[asset_excess_ret.index < t]
            curr_in_sample_pred = pred_data.iloc[:asset_excess_ret.index.get_loc(t)]
            curr_in_sample_len = len(curr_in_sample_pred)

            regressor = curr_in_sample_pred.iloc[:curr_in_sample_len-1]
            regressand = curr_in_sample_asset_ret.iloc[1:curr_in_sample_len]
            X = sm.add_constant(regressor.values) # add constant to regressor
            y = regressand.values
            ols_model = sm.OLS(y, X) # define OLS model
            ols_res = ols_model.fit() # fit data with defined model
            # predict the very next out-of-sample return forecast (using the last in-sample predictor value to forecast)
            X_hat = sm.add_constant(curr_in_sample_pred.iloc[curr_in_sample_len-1].values, has_constant='add')
            forecast = ols_res.predict(X_hat) # ols_model.predict(params=[ols_res.params[0], ols_res.params[1]], exog=X_hat)
            pred_forecasts.append(forecast[0])
        
        ols_forecasts[pred] = pred_forecasts
    return ols_forecasts

def generate_combination_mean_forecasts(asset1_pred_forecasts: pd.DataFrame, asset2_pred_forecasts: pd.DataFrame, assets_names: list):
    """
    Generates a timeseries of monthly out-of-sample return forecasts by taking
    mean of the combination of corresponding predictor forecasts of each asset class.
    Finally returns a dataframe with combination mean forecasts for both asset classes.

    Parameters:
    -----------
    asset1_pred_forecasts : pd.DataFrame
        Monthly out-of-sample OLS predictor forecats (5 models) for asset class 1 (SP500).
    asset2_pred_forecasts : pd.DataFrame
        Monthly out-of-sample OLS predictor forecats (5 models) for asset class 2 (LBUSTRUU).
    assets_names : list
        List of asset names (tickers).

    Returns: 
    --------
    combination_mean_forecasts : pd.DataFrame
        Monthly out-of-sample Combination mean forecasts generated by 
        taking mean of individual OLS predictor forecasts. (both asset classes)
    """
    assert len(asset1_pred_forecasts) == len(asset2_pred_forecasts)
    # rename the asset names to reflect that these correspond to combination mean forecasts
    assets = [f'{asset}_Comb_Mean' for asset in assets_names]
    combination_mean_forecasts = pd.DataFrame(index=asset1_pred_forecasts.index, columns=assets)
    combination_mean_forecasts[assets[0]] = np.nanmean(asset1_pred_forecasts, axis=1)
    combination_mean_forecasts[assets[1]] = np.nanmean(asset2_pred_forecasts, axis=1)

    return combination_mean_forecasts

def generate_plr_forecast(asset_excess_ret: pd.DataFrame, predictors: pd.DataFrame, breakpoint: dt.datetime):
    """
    Generates a timeseries of monthly out-of-sample return forecasts
    predicted using penalised linear regression models fit on 
    defined predictors for a specific asset class

    Parameters:
    -----------
    asset_excess_ret : pd.DataFrame
        Asset (excess) returns data (for one asset class).
    predictors : pd.DataFrame
        Predictors data.
    breakpoint : datetime
        datetime to split the data into in-sample & out-of-sample.

    Returns: 
    --------
    plr_forecasts : pd.DataFrame
        Monthly out-of-sample Penalised Linear Regression forecasts generated by 
        fitting over all predictors (both models Lasso & Ridge - for one asset class only).
    """
    # define out-of-sample period by splitting data at the breakpoint
    out_of_sample_period = asset_excess_ret[asset_excess_ret.index > breakpoint].index
    # placeholder for out-of-sample lasso & ridge (plr) forecasts
    plr_forecasts = pd.DataFrame(index=out_of_sample_period, columns=['Lasso', 'Ridge'])
    forecasts = []

    for t in out_of_sample_period:
        # Update the in-sample period to be considered
        # recursively to include all data up to period t
        curr_in_sample_asset_ret = asset_excess_ret.loc[asset_excess_ret.index < t]
        curr_in_sample_pred = predictors.iloc[:asset_excess_ret.index.get_loc(t)]
        curr_in_sample_len = len(curr_in_sample_pred)
        X = curr_in_sample_pred.iloc[:curr_in_sample_len-1].values
        y = curr_in_sample_asset_ret.iloc[1:curr_in_sample_len].values
        
        # predict the very next out-of-sample return forecast (using the last in-sample predictor values to forecast)
        X_hat = curr_in_sample_pred.iloc[curr_in_sample_len-1].values.reshape(1, -1)
        # fit Lasso model on data
        lasso_model = LassoCV()
        lasso_model.fit(X, y)
        lasso_forecast = lasso_model.predict(X_hat)
        # fit Ridge model on data
        ridge_model = RidgeCV()
        ridge_model.fit(X, y)
        ridge_forecast = ridge_model.predict(X_hat)

        forecasts.append([lasso_forecast[0], ridge_forecast[0][0]])

    plr_forecasts.loc[:, :] = forecasts    
    return plr_forecasts

def dm_test(real_values, pred1, pred2, h=1):
    """
    Diebold-Mariano test for equal predictive ability.
    
    Parameters
    ----------
    real_values : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    pred1 : array-like of shape (n_samples,)
        Predicted values from model 1.
    pred2 : array-like of shape (n_samples,)
        Predicted values from model 2.
    h : int, optional (default=1)
        Forecast horizon.
        
    Returns
    -------
    tuple : named tuple
        DM test statistics and p-value.
    """
    # placeholders for square differences & loss differential 
    e1_lst = []
    e2_lst = []
    d_lst = []
    real_values = pd.Series(real_values).apply(lambda x: float(x)).tolist()
    pred1 = pd.Series(pred1).apply(lambda x: float(x)).tolist()
    pred2 = pd.Series(pred2).apply(lambda x: float(x)).tolist()

    # Length of forecasts
    T = float(len(real_values))
    # Construct loss differential according to error criterion (MSE)
    for real, p1, p2 in zip(real_values, pred1, pred2):
        e1_lst.append((real - p1)**2)
        e2_lst.append((real - p2)**2)
    for e1, e2 in zip(e1_lst, e2_lst):
        d_lst.append(e1 - e2)
    # Mean of loss differential
    mean_d = pd.Series(d_lst).mean()

    # Calculate autocovariance
    def autocovariance(Xi, N, k, Xs):
        autoCov = 0
        T = float(N)
        for i in np.arange(0, N-k):
            autoCov += ((Xi[i+k])-Xs) * (Xi[i]-Xs)
        return (1/(T))*autoCov

    # Calculate the denominator of DM stat
    gamma = []
    for lag in range(0, h):
        gamma.append(autocovariance(d_lst, len(d_lst), lag, mean_d))
    V_d = (gamma[0] + 2*sum(gamma[1:]))/T

    # Calculate DM stat
    DM_stat = V_d**(-0.5) * mean_d

    # Calculate and apply Harvey adjustement
    # It applies a correction for small sample
    harvey_adj = ((T+1 - 2*h + h*(h-1)/T)/T)**(0.5)
    DM_stat = harvey_adj*DM_stat
    # Calculate p-value
    p_value = 2 * t.cdf(-abs(DM_stat), df=T - 1)

    dm_return = collections.namedtuple('dm_return', 'DM p_value')
    result = dm_return(DM=DM_stat, p_value=p_value)

    return result

def generate_portfolio_var_cov_mat_forecast(excess_ret: pd.DataFrame, breakpoint: dt.datetime):
    """
    Generates a timeseries of monthly out-of-sample variance-covariance 
    matrix forecasts for a portfolio of assets.

    Parameters:
    -----------
    excess_ret : pd.DataFrame
        Assets (excess) returns data (for both asset classes).
    breakpoint : datetime
        datetime to split the data into in-sample & out-of-sample.

    Returns: 
    --------
    portf_var_cov_forecast : pd.DataFrame
        Monthly out-of-sample forecast of variance-covariance matrices for a portfolio of assets.
    """
    out_of_sample_period = excess_ret[excess_ret.index > breakpoint].index # define out-of-sample period by splitting data at the breakpoint
    portf_var_cov_forecast = pd.DataFrame(index=out_of_sample_period) # placeholder for out-of-sample portfolio var-cov forecasts
    cov_mats = []

    # Loop over the out-of-sample period and compute
    # portfolio var-cov matrice forecasts using recursive estimation approach
    for t in out_of_sample_period:
        # Update the in-sample period to be considered 
        # recursively to include all data up to period t
        curr_in_sample = excess_ret[excess_ret.index < t]
        # Compute the portfolio var-cov matrices using the current in-sample period
        cov_mat = np.cov(curr_in_sample.values, rowvar=False, ddof=1)
        cov_mats.append(cov_mat)
        
    portf_var_cov_forecast['portf_asset_cov_mat'] = cov_mats
    return portf_var_cov_forecast

def generate_OTP_weights(excessMean, cov):
    """
    Calculate the weights of the Optimal Tangency Portfolio.

    Parameters:
    -----------
    excessMean : ndarray
        Excess mean returns array.
    
    cov : ndarray
        Covariance of asset returns.

    Returns: 
    --------
    weights_TP : array_like
        The weights computed for OTP using the input excesss mean & cov matrix.
    """
    inv_cov = np.linalg.inv(cov)
    num = inv_cov @ excessMean.T
    den = np.ones(cov.shape[0]) @ inv_cov @ excessMean.T
    weights_TP = num/den
    return weights_TP

def generate_OTP_excess_ret(assets_ret_forecasts: pd.DataFrame, cov_mat_forecasts: pd.DataFrame):
    """
    Generates a timeseries of monthly out-of-sample Optimal
    Tangency Portfolio excess return forecasts for a portfolio of assets.

    Parameters:
    -----------
    assets_ret_forecasts : pd.DataFrame
        Monthly out-of-sample return forecasts predicted by a specific model (for both assets).
    breakpoint : datetime
        datetime to split the data into in-sample & out-of-sample.

    Returns: 
    --------
    otp_excess_ret : NDArray
        Monthly out-of-sample excess portfolio returns.
    all_weights : List of portfolio weights
        Monthly out-of-sample portfolio weights.
    """
    # define a placeholder for time-series of OTP excess return
    otp_excess_ret = pd.Series(index=assets_ret_forecasts.index)
    excess_ret_mean = np.array(assets_ret_forecasts.mean())
    all_weights = []

    # iterate over the entire period
    for i in range(len(otp_excess_ret)):
        # generate out-of-sample OTP weights
        otp_weights = generate_OTP_weights(excessMean=excess_ret_mean, cov=cov_mat_forecasts.iloc[i].values[0])
        all_weights.append(otp_weights)
        # Compute the portfolio excess return
        otp_excess_ret.iloc[i] = assets_ret_forecasts.iloc[i].values @ otp_weights.T

    return otp_excess_ret.values, all_weights   