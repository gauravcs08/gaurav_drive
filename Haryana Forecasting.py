# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 09:19:49 2020

@author: gauravrai
"""
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf,pacf
import statsmodels.api as sm
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt


def datapreperation(ts):
    d1=pd.DataFrame()
    for key in ts:
        k=ts[key].T
        d1=pd.concat([d1,k])
    d1=d1[d1[0]!='Time']
    d1=d1[d1[0]!='Average']
    d1=d1[d1[0]!='AVG']
    d1=d1[d1[0]!='Maximum']
    d1=d1.rename(columns={0:'Date'})
    d1['Date']=pd.to_datetime(d1['Date'])
    d1.set_index('Date',inplace=True)
    d1['DC(MW)']=d1.sum(axis=1)
    return(d1)

def modelverification(ts,forecast1,actual1):
    
    forecast1=pd.DataFrame(forecast1)
    actual1=actual1.reset_index()
    forecast1.reset_index(inplace=True)
   
    forecast=forecast1[forecast1.columns[1]]
    actual=actual1[actual1.columns[1]]

    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
   
    print("mape:",mape)
    print("me:",me)
    print("mae:",mae)
    print("mpe:",mpe)
    print("RMSE:",rmse)
    print("Correlation:",corr)
    
    #Validation through residuals
    resd_ar=pd.DataFrame(ts.resid)
    plt.figure(figsize=(10,5))
    plt.plot(resd_ar,color='red')
    plt.title("Residual plot of Model")
    plt.show()
    print("Mean of Residual is:\n")
    print(resd_ar.mean())
    
    plt.figure(figsize=(10,5))
    resd_ar.hist()
    plt.title("Histogram of Residual plot of Model")
    plt.show()
    
    #ACF Graph
    lag_acf=acf(ts.resid,nlags=20)
    lag_pacf=pacf(ts.resid,nlags=20,method='ols')

    #plot ACF
    plt.figure(figsize=(15,5))
    plt.subplot(121)
    plt.plot(lag_acf)
    plt.axhline(y=0,linestyle='--',color='grey')
    plt.axhline(y=-1.96/np.sqrt(len(ts.resid)),linestyle='--',color='red')
    plt.axhline(y=1.96/np.sqrt(len(ts.resid)),linestyle='--',color='red')
    plt.title('Auto Correlation Function')
    
    # plot PACF
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0,linestyle='--',color='grey')
    plt.axhline(y=-1.96/np.sqrt(len(ts.resid)),linestyle='--',color='red')
    plt.axhline(y=1.96/np.sqrt(len(ts.resid)),linestyle='--',color='red')
    plt.title('Partial Auto Correlation Function')
    plt.tight_layout()
    plt.show()
    '''-------------Lung-box test-----------'''
    ltest=sm.stats.acorr_ljungbox(ts.resid, lags=[10])
    print(ltest)

def HLM_winter_model(train,test):
    #alpha=smoothing_level and beta=smoothing slope

    fit1 = ExponentialSmoothing(train,seasonal_periods=365,trend='mul',seasonal='mul',damped=False).fit(optimized=True, use_boxcox=False, remove_bias=True)
    fcast=fit1.forecast(len(test))
    plt.figure(figsize=(18,8))
    plt.plot(train,label='train data',color='black')
    plt.plot(test,label='test data',color='green')
    plt.plot(fcast,label='forecast',color='red')
    plt.legend(loc='best')
    plt.title('Load Forecast using HLM winter Method',fontsize=15)
    plt.xlabel('day----->')
    plt.ylabel('Consumption in Mwh')
    plt.show()
    results=pd.DataFrame(index=[r"$\alpha$",r"$\beta$",r"$\phi$",r"$\gamma$",r"$l_0$","$b_0$","SSE"])
    params = ['smoothing_level', 'smoothing_slope', 'damping_slope', 'smoothing_seasonal', 'initial_level', 'initial_slope']
    results["Additive"]= [fit1.params[p] for p in params] + [fit1.sse]
    print(results)
    print("Verification of HLM winter Forecasting Model")
    modelverification(fit1,fcast,test)
    return(fit1)


def forecast_model(model,train,test):
    fcast=model.forecast(650)
    plt.figure(figsize=(40,8))
    plt.plot(train,label='train data',color='black')
    plt.plot(test,label='test data',color='green')
    plt.plot(fcast,label='forecast',color='red')
    plt.legend(loc='best',fontsize=20)
    plt.title('Load Forecast using HLM winter Method: Haryana',fontsize=35)
    plt.xlabel('day----->',fontsize=25)
    plt.xticks(fontsize=25)
    plt.ylabel('Load in MW',fontsize=25)
    plt.yticks(fontsize=25)
    plt.show()
'''----------------------Main Program-------------------------'''

os.chdir(r'C:\Users\gauravrai\Desktop\IPBA\Time Series Model')
data1=pd.read_excel('Haryana 2017-18.xlsx',sheet_name=None)
data2=pd.read_excel('Haryana 2018-19.xlsx',sheet_name=None)

d1=datapreperation(data1)
d2=datapreperation(data2)
fd=pd.concat([d1,d2])
train=fd[:700]
test=fd[600:]

model=HLM_winter_model(train['DC(MW)'],test['DC(MW)'])
forecast_model(model,train['DC(MW)'],test['DC(MW)'])