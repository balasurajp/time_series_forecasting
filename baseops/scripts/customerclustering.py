import json, os, joblib, sys
import pandas as pd, numpy as np
from pymongo import MongoClient
from glob import glob

from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler

import plotly.graph_objects as go
import plotly as ply
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

filecustomers = [
 'ppusage_BrooksideHomes',
 'ppusage_CentervilleHomes',
 'ppusage_DowntownOffices',
 'ppusage_EastsideOffices',
 'ppusage_FrostyStorage',
 'ppusage_HextraChemical',
 'ppusage_MedicalCenter-1',
 'ppusage_OfficeComplex 1 NS Base',
 'ppusage_OfficeComplex 1 NS Controllable',
 'ppusage_OfficeComplex 1 SS Base',
 'ppusage_OfficeComplex 1 SS Controllable',
 'ppusage_OfficeComplex 2 NS Base',
 'ppusage_OfficeComplex 2 NS Controllable',
 'ppusage_OfficeComplex 2 SS Base',
 'ppusage_OfficeComplex 2 SS Controllable',
 'ppusage_Village 1 NS Base',
 'ppusage_Village 1 NS Controllable',
 'ppusage_Village 1 RaS Base',
 'ppusage_Village 1 RaS Controllable',
 'ppusage_Village 1 ReS Base',
 'ppusage_Village 1 ReS Controllable',
 'ppusage_Village 1 SS Base',
 'ppusage_Village 1 SS Controllable',
 'ppusage_Village 2 NS Base',
 'ppusage_Village 2 NS Controllable',
 'ppusage_Village 2 RaS Base',
 'ppusage_Village 2 RaS Controllable',
 'ppusage_Village 2 ReS Base',
 'ppusage_Village 2 ReS Controllable',
 'ppusage_Village 2 SS Base',
 'ppusage_Village 2 SS Controllable',
 'ppusage_fc2',
 'ppusage_fc3',
 'ppusage_freezeco-1',
 'ppusage_freezeco-2',
 'ppusage_freezeco-3',
 'ppusage_seafood-1',
 'ppusage_seafood-2',
 'ppusage_sf2',
 'ppusage_sf3'
]

files = glob("../../database/ptaccu/*_train.csv")
files = [file for file in files if 'RAND' not in file]
files = [os.path.abspath(file) for file in files]

book = {name:[] for name in filecustomers}
for file in files:
    df = pd.read_csv(file, parse_dates=['timestamp'])
    dates = df['timestamp'].dt.date.unique()
    np.random.shuffle(dates)
    
    checkdates = dates[0:30]
    df = df[df['timestamp'].dt.date.isin(checkdates)].reset_index(drop=True)
    
    for name in filecustomers:
        for date in checkdates:
            timeseries = df.loc[df['timestamp'].dt.date==date, name].to_numpy()
            book[name].append(timeseries)
    
for idno, name in enumerate(filecustomers):
    book[name] = np.expand_dims(np.stack(book[name], axis=0), axis=2)
    book[name] = ( TimeSeriesScalerMeanVariance().fit_transform(book[name]), idno*np.ones(shape=(len(book[name]),1)) )

X, Y = [], []
for name in filecustomers:
    X.append(book[name][0])
    Y.append(book[name][1])
X = np.concatenate(X, axis=0)
Y = np.concatenate(Y, axis=0)
print(X.shape, Y.shape)
    
model = TimeSeriesKMeans(n_clusters=4, metric="softdtw", max_iter=40, metric_params={"gamma": .01}, verbose=True)
C = model.fit_predict(X)


classnames = [filecustomers[int(value.item())] for value in Y]
clusternames = [f"cluster_{value}" for value in C]
dataframe = pd.DataFrame.from_records({'customers': classnames, 'clustername': clusternames})

fig = go.Figure()
for clustername in dataframe['clustername'].unique():
    df = dataframe.loc[dataframe['clustername']==clustername, ['customers']].reset_index(drop=True)
    df['count'] = 1
    df = df.groupby(by=['customers']).sum().reset_index()
    df = pd.merge(df, pd.DataFrame.from_records({'customers':filecustomers}), on=['customers'], how='right', validate='1:1')
    df.fillna(0, inplace=True)
    df.sort_values(by=['customers'], key=lambda x: x.map({name:idno for idno,name in enumerate(filecustomers)}), inplace=True)

    fig.add_trace( go.Bar(x=df['customers'], y=df['count'], text=df['count'], textposition='outside', visible=True, name=clustername) )

fig.update_layout(title=f"PowerTAC Customers Clustering", xaxis_title="Customers", yaxis_title="Cluster Selection")
with open(f'cc.html', 'w') as f:
    f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))