import urllib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
import os
import talib
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from matplotlib import pyplot as plt
from candleplot import *
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
import sys


class technical_prediction(object):
    def __init__(self,ticker):
        self.ticker = ticker
        
    def prediksi(self):
        ticker = self.ticker
        try:
            url="https://www.google.com/finance/historical?output=csv&q="+ticker
            stock=ticker+".csv"
            urllib.request.urlretrieve(url,stock)
        except:
            try:
                print("Retrying..")
                url="https://www.google.com/finance/historical?output=csv&q="+ticker
                stock=ticker+".csv"
                urllib.request.urlretrieve(url,stock)
            except:
                print("Connection Error..")
                exit(0)
    
            
        df = pd.read_csv(stock).dropna(how='any') 
        for i in df:
            df = df[~df[i].isin(["-"])]
            
        df.iloc[:] = df.iloc[::-1].values
            
        try:
            rsi = pd.DataFrame(talib.RSI(df['Close'].values,timeperiod=14),columns=['RSI_14'])
            atr = pd.DataFrame(talib.ATR(df['High'].values.astype(float),df['Low'].values.astype(float),df['Close'].values.astype(float),timeperiod=14),columns=["ATR_14"])
            roc = pd.DataFrame(talib.ROC(df['Close'].values,timeperiod=14),columns=["ROC_14"])
            wilr = pd.DataFrame(talib.WILLR(df['High'].values.astype(float),df['Low'].values.astype(float),df['Close'].values,timeperiod=14),columns=["WILLR"])
            mom = pd.DataFrame(talib.MOM(df['Close'].values,timeperiod=14),columns=["MOM_14"])
            aaron = pd.DataFrame(talib.AROONOSC(df['High'].values.astype(float),df['Low'].values.astype(float),timeperiod=14),columns=["ARN_14"])
           
        except Exception as e:
            print(e)   
            
        frames = [df,rsi,atr,roc,wilr,mom,aaron]
        data = pd.concat(frames,axis=1)
        data = data.loc[14:,~data.columns.duplicated()].reset_index()
        data = data.dropna(how='any')
        close = data['Close'].values
        
        threshold = float(input("threshold (%): "))
        
        up_t = 1+threshold/100
        down_t = 1-threshold/100
        Target = []
        for i in range(len(close)):
            try:
                if (close[i]*up_t)<close[i+1]:
                    Target.append("UP")
                elif (close[i]*down_t<=close[i+1]) and (close[i]*up_t)>=close[i+1]:
                    Target.append("STABLE")
                elif (close[i]*down_t)>close[i+1]:
                    Target.append("DOWN")
                    
            except:
                Target.append("-")         
        
        forest = RandomForestClassifier(n_estimators=5, random_state=2)  
        X=data[['RSI_14','ATR_14','ROC_14','WILLR',"MOM_14","ARN_14"]].copy()
        train_X = X[:len(X)-1]
        train_Y = Target[:len(Target)-1] 
        forest.fit(train_X,train_Y)
        
        RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gint', max_depth=None, max_features='auto', max_leaf_nodes=None,min_samples_leaf=1,min_samples_split=2,min_weight_fraction_leaf=0.0, n_estimators=100,n_jobs=1,oob_score=False, random_state=2, verbose= 2, warm_start=False)
        
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(100,100,100,100,100,100,100, ), random_state=1)
        clf.fit(train_X,train_Y)
        
        gnb = GaussianNB()
        gnb.fit(train_X,train_Y)
        
        vm = svm.SVC()
        vm.fit(train_X,train_Y)
        y_eval_vm = vm.predict(train_X)
        y_pred_vm = vm.predict(X)
        
        
        y_eval_gnb = gnb.predict(train_X)
        y_pred_gnb = gnb.predict(X)
        
        y_pred_nn = clf.predict(X)
        y_eval_nn = clf.predict(train_X)
        y_eval_frt = forest.predict(train_X)
        print("Training accuracy: \r")
        print("Random Forest: %.3f "%(np.mean(y_eval_frt == train_Y)))
        print("Neural Network: %.3f "%(np.mean(y_eval_nn == train_Y)))
        print("Naive Bayes: %.3f " %(np.mean(y_eval_gnb == train_Y)))
        print("Suppor Vector Machine: %.3f " %(np.mean(y_eval_vm == train_Y)))
        y_pred_frt = forest.predict(X)
        print ("using " + data['Date'].values[-1] +" data")
        
        print("Random Forest: %s" %(y_pred_frt[-1]))
        print("Neural Network: %s" %(y_pred_nn[-1]))
        print("Naive Bayes: %s" %(y_pred_gnb[-1]))
        print("Support Vector Machine: %s \n" %(y_pred_gnb[-1]))
    
        hasil = y_pred_nn[-1][0]+y_pred_gnb[-1][0]+y_pred_frt[-1][0]+y_pred_vm[-1][0]
                
        os.system("del /f %s" %stock)
        self.data = data
        self.train_X = train_X
        self.target = Target
        return close,Target,train_X,hasil,data
    
    def pattern_checker(self):
        data = self.data
        thrusting = talib.CDLTHRUSTING(data['Open'].values, data['High'].values,\
                                     data['Low'].values, data['Close'].values)
        #bullish
        morning = talib.CDLMORNINGSTAR(data['Open'].values, data['High'].values,\
                                     data['Low'].values, data['Close'].values,penetration=0)
        #bullish
        threeline = talib.CDL3LINESTRIKE(data['Open'].values, data['High'].values,\
                                     data['Low'].values, data['Close'].values)
        #bearish
        black = talib.CDL3BLACKCROWS(data['Open'].values, data['High'].values,\
                                     data['Low'].values, data['Close'].values)
        #bearish
        eveningstar = talib.CDLEVENINGDOJISTAR(data['Open'].values, data['High'].values,\
                                     data['Low'].values, data['Close'].values,penetration=0)
        #bullish
        abandonedbaby = talib.CDLABANDONEDBABY(data['Open'].values, data['High'].values,\
                                     data['Low'].values, data['Close'].values,penetration=0)
        
        
        talist = ['thrusting','morning','threeline','black','eveningstar','abandonedbaby']
        ta_name = thrusting,morning,threeline,black,eveningstar,abandonedbaby
        self.ta_dict = dict([(talist[i],ta_name[i]) for i in range(len(talist))])
        return thrusting,morning,threeline,black,eveningstar,abandonedbaby
    
    def plot(self,ta_indicator=None):
        df =  self.train_X
        data_plot = self.data
        target = self.target
        
        if ta_indicator:
            morning  = self.ta_dict[ta_indicator]
        
        fig, axes = plt.subplots(nrows=len(df.columns)+1,sharex=True)
        index = 0
        fig.tight_layout()
        for i in df:
            df[i].plot(ax=axes[index],title=i,figsize=(9,9))
            index+=1
            
        ticks = []
        for i in target:
            if i=="STABLE":
                ticks.append(0.0)
            elif i == "UP":    
                ticks.append(10)
            else:    
                ticks.append(-10)
                
        axes[index].plot(ticks)
        axes[index].set_title("Price Movement")
        plt.show()
        
        data = []
        for i in range(len(data_plot)):
            temp=(i, data_plot['Open'][i], data_plot['Close'][i], data_plot['Low'][i], data_plot['High'][i])
            temp=tuple(temp)
            data.append(temp)
            
        item = CandlestickItem(data)
        plt_qt = pg.plot(title='price candleplot')
        plt_qt.addItem(item)
        
        max_price = max(data_plot['Close'])
        min_price = min(data_plot['Close'])
        
        #print indicator marker
        if ta_indicator:
            for i in range(len(morning)):
                if morning[i]!=0:
                    plt_qt.plot([i,i],[min_price,max_price])
        
                

if __name__ == '__main__':
    ticker = input("predict what stock?: ")
    stock =  technical_prediction(ticker)
    stock.prediksi()
    #patern available thrusting,morning,threeline,black,eveningstar,abandonedbaby
    pattern = stock.pattern_checker()
    stock.plot(ta_indicator='morning')
        
    #required for pyqt plot    
    try:
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()
    except:
        sys.exit(0)  
