import os, json, random, math, numpy as np, pandas as pd

import plotly.graph_objects as go, numpy as np
from plotly.subplots import make_subplots
from plotly.offline import plot
from glob import glob

from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score
from tensorflow.keras import utils as tfutils 
from attrdict import AttrDict
from datetime import datetime, timedelta
from shutil import copy2
# =======================================================================

def timelinemetrics(yOutput, yTarget, paths, configuration, architecture):
    dataframe = pd.DataFrame.from_records({'targets':yTarget, 'predictions':yOutput})
    dataframe['APE'] = np.abs((dataframe['targets']-dataframe['predictions']) / dataframe['targets'] )*100.0 if ~np.any(yTarget==0) else "-NA-"
    testfiles = glob(paths['data']+f'/test/*')
    timeframes = [pd.read_csv(testfile, parse_dates=['timestamp'], usecols=['timestamp']) for testfile in testfiles]
    timeframes = pd.concat(timeframes).reset_index(drop=True)
    timeframes = timeframes.loc[168:, :].reset_index(drop=True)
    dataframe['timestamp'] = timeframes['timestamp']
    dataframe['monthName'] = dataframe['timestamp'].dt.month_name()
    dataframe['dayName'] = dataframe['timestamp'].dt.day.apply(lambda x: f"D{x:02}")
    dataframe['HourName'] = dataframe['timestamp'].dt.hour.apply(lambda x: f"H{x:02}")    

    timestamps = dataframe['timestamp'].astype(str).to_numpy()
    targets = np.round(dataframe['targets'].values, 1)
    predictions = np.round(dataframe['predictions'].values, 1)
    fig1 = go.Figure()
    fig1.add_trace(
        go.Scatter(
            x=timestamps, 
            y=predictions, 
            name="Predictions", 
            line_color='red'
        )
    )
    fig1.add_trace(
        go.Scatter(
            x=timestamps, 
            y=targets, 
            name="Targets", 
            line_color='blue'
        )
    )
    fig1.update_layout(title_text=f'Time Series Forecasting - {configuration}', xaxis_rangeslider_visible=True)

    avgdataframe = dataframe.groupby(by=[dataframe['monthName']]).mean().reset_index()
    xvalues = avgdataframe['monthName'].tolist()
    yvalues = np.round(avgdataframe['APE'].values, 1)
    fig2 = go.Figure()
    fig2.add_trace(
        go.Bar(
            x=xvalues, 
            y=yvalues,
            text=yvalues.astype(str),
            textposition='outside',
            name='MAPE'
        )
    )
    fig2.update_layout(title=f"MEAN ABSOLUTE PERCENTAGE ERROR - MONTHWISE", xaxis_title="MONTH", yaxis_title="MAPE")

    avgdataframe = dataframe.groupby(by=[dataframe['dayName']]).mean().reset_index()
    xvalues = avgdataframe['dayName'].tolist()
    yvalues = np.round(avgdataframe['APE'].values, 1)
    fig3 = go.Figure()
    fig3.add_trace(
        go.Bar(
            x=xvalues, 
            y=yvalues,
            text=yvalues.astype(str),
            textposition='outside',
            name='MAPE'
        )
    )
    fig3.update_layout(title=f"MEAN ABSOLUTE PERCENTAGE ERROR - DAYWISE", xaxis_title="DAY", yaxis_title="MAPE")

    avgdataframe = dataframe.groupby(by=[dataframe['HourName']]).mean().reset_index()
    xvalues = avgdataframe['HourName'].tolist()
    yvalues = np.round(avgdataframe['APE'].values, 1)
    fig4 = go.Figure()
    fig4.add_trace(
        go.Bar(
            x=xvalues, 
            y=yvalues,
            text=yvalues.astype(str),
            textposition='outside',
            name='MAPE'
        )
    )
    fig4.update_layout(title=f"MEAN ABSOLUTE PERCENTAGE ERROR - HOURWISE", xaxis_title="HOUR", yaxis_title="MAPE")

    with open(paths['result'] + f'/{configuration}_{architecture}_MAPE.html', 'w') as f:
        f.write(fig1.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write(fig2.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write(fig3.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write(fig4.to_html(full_html=False, include_plotlyjs='cdn'))    

def randomSpilter(data, ratio=0.8):
    nsamples = len(data['y'])
    idx = np.random.permutation(nsamples)
    for keyname in list(data.keys()):
        data[keyname] = data[keyname][idx]

    splitidx = int(ratio*nsamples)
    trndata, valdata = {}, {}
    for keyname in list(data.keys()):
        trndata[keyname] = data[keyname][:splitidx]
        valdata[keyname] = data[keyname][splitidx:] 

    return trndata, valdata

def randomSeed(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    random.seed(seed)

def SaveBestModel(checkpointlink, buildlink, architecture):
    trnlog = pd.read_csv(buildlink + f'/train{architecture}.log')
    bestepoch = int(trnlog.loc[trnlog['valloss']==trnlog['valloss'].min(), 'epoch'].values)

    src = checkpointlink + f'/{architecture}_{bestepoch:02d}.h5'
    dst = buildlink + f'/{architecture}_savedmodel.h5'
    copy2(src, dst)

class tanhScaler(object):
    def __init__(self, mean=None, std=None):
        super(tanhScaler, self).__init__()
        self.mean_ = mean
        self.std_  = std

    def fit(self, data):
        data = np.array(data, dtype=np.float32)
        self.mean_ = data.mean()
        self.std_  = data.std()

    def fitTransform(self, data):
        data = np.array(data, dtype=np.float32)
        self.mean_ = data.mean()
        self.std_  = data.std()

        xscaled = (data - self.mean_)/self.std_
        scaledData = 0.5*( np.tanh(0.01*xscaled, dtype=np.float32)+1 )
        return scaledData

    def Transform(self, data):
        data = np.array(data, dtype=np.float32)
        xscaled = (data - self.mean_)/self.std_
        scaledData = 0.5*( np.tanh(0.01*xscaled, dtype=np.float32)+1 )
        return scaledData

    def invTransform(self, data):
        data = np.array(data, dtype=np.float32)
        xscaled = np.arctanh((data/0.5) - 1, dtype=np.float32)
        scaledData = (xscaled*self.std_*100) + self.mean_
        return scaledData

class normalScaler:
    def __init__(self, mean=None, std=None):
        super(normalScaler, self).__init__()
        self.mean_ = mean
        self.std_  = std

    def fit(self, data):
        data = np.array(data, dtype=np.float32)
        self.mean_ = data.mean()
        self.std_  = data.std()

    def fitTransform(self, data):
        data = np.array(data, dtype=np.float32)
        self.mean_ = data.mean()
        self.std_  = data.std()

        scaledData = (data - self.mean_)/self.std_
        return scaledData

    def Transform(self, data):
        data = np.array(data, dtype=np.float32)
        scaledData = (data - self.mean_)/self.std_
        return scaledData

    def invTransform(self, data):
        data = np.array(data, dtype=np.float32)
        scaledData = (data*self.std_) + self.mean_
        return scaledData

    def __repr__(self):
        return repr(f'NormalScaler: ({self.mean_},{self.std_})')

class rangeScaler(object):
    def __init__(self, srange=(-1,1)):
        self.min_ = None
        self.max_ = None
        self.smin_ = srange[0]
        self.smax_ = srange[1]

    def fit(self, data):
        data = np.array(data, dtype=np.float32)
        self.min_ = data.min()
        self.max_ = data.max()

    def fitTransform(self, data):
        data = np.array(data, dtype=np.float32)
        self.min_ = data.min()
        self.max_ = data.max()

        scaledData = ( ((data - self.min_)*(self.smax_ - self.smin_)) / (self.max_ - self.min_) ) + self.smin_
        return scaledData

    def Transform(self, data):
        data = np.array(data, dtype=np.float32)
        scaledData = ( (data - self.min_)*(self.smax_ - self.smin_) / (self.max_ - self.min_) ) + self.smin_
        return scaledData

    def invTransform(self, data):
        data = np.array(data, dtype=np.float32)
        scaledData = ( (data - self.smin_)*(self.max_ - self.min_) / (self.smax_ - self.smin_) ) + self.min_
        return scaledData

class avgmeter(object):
    def __init__(self):
        self.valsum = 0
        self.avg = 0
        self.count = 0

    def update(self, value, size):
        self.valsum = self.valsum + value*size
        self.count = self.count + size
        self.avg = self.valsum/self.count

    def reset(self):
        self.valsum = 0
        self.avg = 0
        self.count = 0

class infologger(object):
    def __init__(self, logkeys, logdir):
        self.logkeys = logkeys
        self.logdir  = logdir
        if not os.path.exists(logdir):
            with open(logdir, 'w') as f:
                f.write(self.list2str(logkeys))
                f.write('\n')
    def list2str(self, nlist):
        nstring = ""
        for value in nlist:
            nstring += f"{value},"
        return nstring[0:-1]
    def update(self, logvalues):
        logrow = self.list2str(logvalues)
        with open(self.logdir, 'a') as f:
            f.write(logrow)
            f.write('\n')