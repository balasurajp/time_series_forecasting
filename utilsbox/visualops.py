import __init__
import numpy as np, pandas as pd
import json, joblib, os
import plotly, plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, r2_score
# ********************************************************************************************************

def EvaluationMetrics(outputs, targets, filename):
    MAE  = np.abs(targets-outputs).mean()
    RMSE  = np.sqrt((np.abs(targets-outputs)**2).mean())
    EVS = explained_variance_score(targets, outputs)
    R2S = r2_score(targets, outputs)

    idx = [(target>0).all() and (output>0).all() for target,output in zip(targets, outputs)]
    targets[idx] = 1.0
    outputs[idx] = 1.0
    MAPE = (np.abs((targets-outputs)/targets)).mean() * 100.0
    return {'filename':filename, 'mae':MAE, 'rmse':RMSE, 'mape':MAPE, 'evs':EVS, 'r2s':R2S}

def VisualBook(databook, config, use_model):
    targetseries = {name:[] for name in config['params']['inputs']}
    outputseries = {name:[] for name in config['params']['inputs']}
    metricbook = {name:[] for name in config['params']['inputs']}
    for name in databook.keys():
        outputs = databook[name]['outputs']
        targets = databook[name]['targets']

        outputs_norm = (outputs)# - databook[name]['scalers'][0])/databook[name]['scalers'][1]
        targets_norm  = (targets)# - databook[name]['scalers'][0])/databook[name]['scalers'][1]

        for index, inputname in enumerate(config['params']['inputs']):
            targetseries[inputname] += targets_norm[:,:,index].reshape(-1).tolist()
            outputseries[inputname] += outputs_norm[:,:,index].reshape(-1).tolist()
            metricbook[inputname].append(EvaluationMetrics(outputs[:,:,index], targets[:,:,index], name.replace("_test.csv","")))

    seriesfigures = {}
    for inputname in config['params']['inputs']:    
        seriesfigures[inputname] = go.Figure()
        seriesfigures[inputname].add_trace(go.Scatter(x=np.arange(len(targetseries[inputname])), y=targetseries[inputname], name="Targets", line_color='red'))
        seriesfigures[inputname].add_trace(go.Scatter(x=np.arange(len(outputseries[inputname])), y=outputseries[inputname], name="Outputs", line_color='blue'))
        seriesfigures[inputname].update_layout(title_text=f"TimeSeries - {inputname.upper()}", xaxis_rangeslider_visible=True)

    metricsfigures = {}
    for inputname in config['params']['inputs']:    
        df = pd.DataFrame.from_records(metricbook[inputname])
        df = df.append({'filename':'TOTAL', 'mae':df['mae'].mean(), 'rmse':df['rmse'].mean(), 'mape':df['mape'].mean(), 'evs':df['evs'].mean(), 'r2s':df['r2s'].mean()}, ignore_index=True)
        df.columns = [name.upper() for name in list(df.columns)]

        metricsfigures[inputname] = go.Figure()
        metricsfigures[inputname].add_trace(
            go.Table(header=dict(values=list(df.columns)), cells=dict(values=[df['FILENAME'], df['MAE'].round(2), df['RMSE'].round(2), df['MAPE'].round(2), df['EVS'].round(2), df['R2S'].round(2)]) )
        )
        metricsfigures[inputname].update_layout(title=f"Quantitative Metrics - {inputname.upper()}")
    
    with open(f"{config['paths']['logs']}/{use_model}/visualbook.html", 'w') as f:
        for figurename in seriesfigures.keys():
            f.write(seriesfigures[figurename].to_html(full_html=False, include_plotlyjs='cdn'))
    with open(f"{config['paths']['logs']}/{use_model}/metricbook.html", 'w') as f:
        for figurename in metricsfigures.keys():
            f.write(metricsfigures[figurename].to_html(full_html=False, include_plotlyjs='cdn'))