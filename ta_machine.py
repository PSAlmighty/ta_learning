import urllib
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
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
mom = pd.DataFrame(talib.MOM(df['Close'].values,timeperiod=14),columns=["MOM_14"])
aaron = pd.DataFrame(talib.AROONOSC(df['High'].values,df['Low'].values,timeperiod=14),columns=["ARN_14"])
    
frames = [df,rsi,atr,roc,wilr,mom,aaron]
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
X=data[['RSI_14','ATR_14','ROC_14','WILLR',"MOM_14","ARN_14"]].copy()
train_X = X[:len(X)-1]
train_Y = Target[:len(Target)-1] #.reshape(len(Target),)
forest.fit(train_X,train_Y)
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gint', max_depth=None, max_features='auto', max_leaf_nodes=None,min_samples_leaf=1,min_samples_split=2,min_weight_fraction_leaf=0.0, n_estimators=5,n_jobs=1,oob_score=False, random_state=2, verbose= 2, warm_start=False)
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(100,100,100,100,100,100,100, ), random_state=1)
clf.fit(train_X,train_Y)
y_pred_nn = clf.predict(X)
y_eval_nn =clf.predict(train_X)
y_eval_frt = forest.predict(train_X)
print("Training accuracy: \r")
print(np.mean(y_eval_frt == train_Y))
print(np.mean(y_eval_nn == train_Y))
y_pred_frt = forest.predict(X)
print ("using " + data['Date'].values[-1] +" data")
print("Random Forest:")
print(y_pred_frt[-1])
print("Neural Network:")
print(y_pred_nn[-1])
if y_pred_nn[-1]!= y_pred_frt[-1]:
    print("it's split decision..")

os.system("del /f %s" %stock)
