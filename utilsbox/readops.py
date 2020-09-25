import __init__
import multiprocessing
import json, joblib, os, argparse
import pandas as pd, numpy as np
from utilsbox.misc import normalScaler, tanhScaler
from joblib import Parallel, delayed
from glob import glob
    
def NormalizeColumns(dataframe, columns, scalebook):
    columns = [column for column in columns if column!='timestamp']
    checker = len(scalebook)

    for column in columns:
        if checker==0:
            scalebook[column] = normalScaler()
            dataframe = dataframe.assign(**{column : scalebook[column].fitTransform( dataframe[str(column)].values )})
        else:
            dataframe = dataframe.assign(**{column : scalebook[column].Transform( dataframe[str(column)].values )})
    return dataframe

def NormalizeTensor(X, X_mu=None, X_std=None):
    if X_mu is None and X_std is None:
        X_mu = np.mean(X, axis=1, keepdims=True)
        X_std = np.std(X, axis=1, keepdims=True)
        X_normalised = (X-X_mu)/X_std
        X_std = np.clip(X_std, 0.5, np.inf)
        return X_normalised, X_mu, X_std
    else:
        X_normalised = (X-X_mu)/X_std
        return X_normalised

def ProcessTimestamp(dataframe, columns, method='default'):
    columns = [column for column in columns if column!='timestamp']
    dataframe['month']  = dataframe['timestamp'].dt.month -1
    dataframe['mday']   = dataframe['timestamp'].dt.day -1
    dataframe['wday']   = dataframe['timestamp'].dt.weekday
    dataframe['hour']   = dataframe['timestamp'].dt.hour
    if method=='default':
        dataframe['month'] = (dataframe['month'] - 6.0)/6.0
        dataframe['mday'] = (dataframe['mday'] - 15.0)/15.0
        dataframe['wday'] = (dataframe['wday'] - 3.0)/3.0
        dataframe['hour'] = (dataframe['hour'] - 12.0)/12.0
        tcolumns = ['month', 'mday', 'wday', 'hour']
    elif method=='base2encode':
        for featureno in [1,2,3,4]:
            dataframe[f'month_{featureno}'] = dataframe['month'].apply(lambda x: int(format(x,'04b')[featureno-1]))
        for featureno in [1,2,3,4,5]:
            dataframe[f'mday_{featureno}'] = dataframe['mday'].apply(lambda x: int(format(x,'05b')[featureno-1]))
        for featureno in [1,2,3]:
            dataframe[f'wday_{featureno}'] = dataframe['wday'].apply(lambda x: int(format(x,'03b')[featureno-1]))
        for featureno in [1,2,3,4,5]:
            dataframe[f'hour_{featureno}'] = dataframe['hour'].apply(lambda x: int(format(x,'05b')[featureno-1]))
        tcolumns = [f'month_{i}' for i in range(1,5)] + [f'mday_{i}' for i in range(1,6)] + [f'wday_{i}' for i in range(1,4)] + [f'hour_{i}' for i in range(1,6)]       
    else:
        raise NotImplementedError(f'must implement {method} for timestamp processing')

    dataframe.drop(columns=['timestamp'], inplace=True)
    dataframe = dataframe[tcolumns+columns]
    return dataframe, tcolumns

def SupervisedSampler(dataframe, params, timecolumns, mode):
    inputs     = params['inputs']
    exogenous  = params['exogenous']
    calendar   = timecolumns

    X1, E1, C1, X2, E2, C2, Y = list(),list(),list(),list(),list(),list(),list()
    samplestep = params['samplestep'] if mode=='train' else params['outputlength']
    for i in range(0, len(dataframe)-params['outputlength']-params['inputlength']+1, samplestep):
        x1data = dataframe.loc[i:i+params['inputlength']-1, inputs].to_numpy()
        c1data = dataframe.loc[i:i+params['inputlength']-1, calendar].to_numpy()
        e1data = dataframe.loc[i:i+params['inputlength']-1, exogenous].to_numpy()
        X1.append( x1data )
        E1.append( e1data )
        C1.append( c1data )

        if params['extrainfo']=='simpleavg':
            x2data = x1data.reshape(params['inputlength']//params['outputlength'], params['outputlength'], len(inputs)).mean(axis=0)
            e2data = e1data.reshape(params['inputlength']//params['outputlength'], params['outputlength'], len(exogenous)).mean(axis=0)
        elif params['extrainfo']=='teachforce_exopred':
            x2data = dataframe.loc[i+params['inputlength']:i+params['inputlength']+params['outputlength']-1, inputs].to_numpy()
            e2data = e1data.reshape(params['inputlength']//params['outputlength'], params['outputlength'], len(exogenous)).mean(axis=0)
        elif params['extrainfo']=='teachforce_exoinfer':
            x2data = dataframe.loc[i+params['inputlength']:i+params['inputlength']+params['outputlength']-1, inputs].to_numpy()
            e2data = dataframe.loc[i+params['inputlength']:i+params['inputlength']+params['outputlength']-1, exogenous].to_numpy()
        else:
            raise NotImplementedError(f"must implement {params['extrainfo']} for data sample processing")

        c2data = dataframe.loc[i+params['inputlength']:i+params['inputlength']+params['outputlength']-1, calendar].to_numpy()
        X2.append( x2data )
        E2.append( e2data )
        C2.append( c2data )

        ydata = dataframe.loc[i+params['inputlength']:i+params['inputlength']+params['outputlength']-1, inputs].to_numpy()
        Y.append( ydata )

    X1, E1, C1, X2, E2, C2, Y = np.stack(X1, axis=0), np.stack(E1, axis=0), np.stack(C1, axis=0), np.stack(X2, axis=0), np.stack(E2, axis=0), np.stack(C2, axis=0), np.stack(Y, axis=0)
    X1, X_mu, X_std = NormalizeTensor(X1)
    E1, E_mu, E_std = NormalizeTensor(E1)
    X2 = NormalizeTensor(X2, X_mu, X_std)
    E2 = NormalizeTensor(E2, E_mu, E_std)
    Y = NormalizeTensor(Y, X_mu, X_std)

    checks = [~np.isnan(X1[i]).any() and ~np.isnan(E1[i]).any() and ~np.isnan(C1[i]).any() and ~np.isnan(X2[i]).any() and ~np.isnan(E2[i]).any() and ~np.isnan(C2[i]).any() and ~np.isnan(Y[i]).any() for i in range(len(Y))] 
    X1, E1, C1, X2, E2, C2, Y = X1[checks], E1[checks], C1[checks], X2[checks], E2[checks], C2[checks], Y[checks]
    X_mu, X_std = X_mu[checks], X_std[checks]

    EC1 = np.concatenate([E1,C1], axis=2)
    EC2 = np.concatenate([E2,C2], axis=2)
    inputs = [X1, EC1, X2, EC2]
    targets = Y
    scalers = [X_mu, X_std]
    return inputs, targets, scalers

def reading_default(paths, params, encodingtime='default', mode='train'):
    datafiles = glob(f"{paths['data']}/*_train.csv")
    datacolumns = ['timestamp'] + params['inputs'] + params['exogenous']

    # scalebook = joblib.load(f"{paths['data']}/scalebook.pkl") if mode=='test' else {}
    def processfile(fileno, datafile):
        filename = os.path.basename(datafile)
        print(f"processing file {fileno} : {filename}......")
        
        dataframe = pd.read_csv(datafile, parse_dates=['timestamp'], usecols=datacolumns)
        dataframe = dataframe[datacolumns]

        # ndays = len(dataframe)//24
        # if mode=='train':
        #     dataframe = dataframe.head(int(ndays*0.8)*24).reset_index(drop=True)
        # else: 
        #     dataframe = dataframe.tail(168*24+int(ndays*0.2)*24).reset_index(drop=True)

        # dataframe = NormalizeColumns(dataframe, datacolumns, scalebook)
        dataframe, timecolumns = ProcessTimestamp(dataframe, datacolumns, method=encodingtime)
        inputs, targets, scalers = SupervisedSampler(dataframe, params, timecolumns, mode)
        processbook = {'filename':filename, 'inputs': inputs, 'targets': targets, 'scalers': scalers}
        return processbook

    processedfiles = Parallel(n_jobs=multiprocessing.cpu_count()-2)(delayed(processfile)(fileno,file) for fileno,file in enumerate(datafiles))
    databook = {book['filename']:{'inputs': book['inputs'], 'targets': book['targets'], 'scalers': book['scalers']} for book in processedfiles}
    
    # if mode=='train':
    #     joblib.dump(scalebook, f"{paths['data']}/scalebook.pkl")
    return databook

def combine_databook(databook):
    inputparts = 4
    scalerparts = 2

    filenames = list(databook.keys())
    inputbook, targetbook, scalerbook = [list() for i in range(inputparts)], list(), [list() for i in range(scalerparts)]
    for filename in filenames:
        filedata = databook[filename]

        for index, inputdata in enumerate(filedata['inputs']): 
            inputbook[index].append(inputdata)
        targetbook.append(filedata['targets'])
        for index, scalerdata in enumerate(filedata['scalers']): 
            scalerbook[index].append(scalerdata)

    for index in range(len(inputbook)):
        inputbook[index] = np.concatenate(inputbook[index], axis=0)
    targetbook = np.concatenate(targetbook, axis=0)
    for index in range(len(scalerbook)):
        scalerbook[index] = np.concatenate(scalerbook[index], axis=0)

    mybook = {'inputs': inputbook, 'targets':targetbook, 'scalers':scalerbook}
    return mybook

