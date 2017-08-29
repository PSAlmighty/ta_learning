import urllib
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os
import talib
"""
get technical data for training

"""
ticker = input("predict what stock?: ")
url="https://www.google.com/finance/historical?output=csv&q="+ticker
stock=ticker+".csv"
urllib.request.urlretrieve(url,stock)
df = pd.read_csv(stock)

df.iloc[:] = df.iloc[::-1].values

rsi = pd.DataFrame(talib.RSI(df['Close'].values,timeperiod=14),columns=['RSI_14'])
atr = pd.DataFrame(talib.ATR(df['High'].values,df['Low'].values,df['Close'].values,timeperiod=14),columns=["ATR_14"])
roc = pd.DataFrame(talib.ROC(df['Close'].values,timeperiod=14),columns=["ROC_14"])
wilr = pd.DataFrame(talib.WILLR(df['High'].values,df['Low'].values,df['Close'].values,timeperiod=14),columns=["WILLR"])
    
frames = [df,rsi,atr,roc,wilr]
data = pd.concat(frames,axis=1)
data = data.loc[14:,~data.columns.duplicated()].reset_index()
close = data['Close'].values

Target = []
for i in range(len(close)):
    try:
        if close[i]<close[i+1]:
            Target.append("UP")
        else:
            Target.append("DOWN")
    except:
        Target.append("-")         


forest = RandomForestClassifier(n_estimators=5, random_state=2)  
X=data[['RSI_14','ATR_14','ROC_14','WILLR']].copy()
train_X = X[:len(X)-1]
train_Y = Target[:len(Target)-1] #.reshape(len(Target),)
forest.fit(train_X,train_Y)
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gint', max_depth=None, max_features='auto', max_leaf_nodes=None,min_samples_leaf=1,min_samples_split=2,min_weight_fraction_leaf=0.0, n_estimators=5,n_jobs=1,oob_score=False, random_state=2, verbose= 2, warm_start=False)
y_pred_frt = forest.predict(X)
#print(np.mean(y_pred_frt == train_Y))
print (data['Date'].values[-1])
print(y_pred_frt[-1])

os.system("del /f %s" %stock)
