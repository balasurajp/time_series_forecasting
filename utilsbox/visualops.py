import __init__
import numpy as np, pandas as pd
import json, joblib, os
import plotly, plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score
# ********************************************************************************************************

def EvaluationMetrics(outputs, targets, filename):
    MAE  = np.abs(targets-outputs).mean()
    RMSE  = np.sqrt((np.abs(targets-outputs)**2).mean())
    if (targets==0).any():
        targets = targets + 1.0
        outputs = outputs + 1.0
    MAPE = (np.abs((targets-outputs)/targets)).mean() * 100.0

    EVS = 0.0
    for index in range(outputs.shape[2]):
        EVS += explained_variance_score(targets[:,:,index], outputs[:,:,index])
    EVS /= outputs.shape[2]
    return {'filename':filename, 'mae':MAE, 'rmse':RMSE, 'mape':MAPE, 'evs':EVS}

def VisualBook(databook, config, use_model):
    targetseries = {name:[] for name in config['params']['inputs']}
    outputseries = {name:[] for name in config['params']['inputs']}
    metricbook = []
    for name in databook.keys():
        outputs = databook[name]['outputs']
        targets = databook[name]['targets']

        metrics = EvaluationMetrics(outputs, targets, name)
        metricbook.append(metrics)

        outputs = (outputs - databook[name]['scalers'][0])/databook[name]['scalers'][1]
        targets = (targets - databook[name]['scalers'][0])/databook[name]['scalers'][1]

        for index, inputname in enumerate(config['params']['inputs']):
            targetseries[inputname] += targets[:,:,index].reshape(-1).tolist()
            outputseries[inputname] += outputs[:,:,index].reshape(-1).tolist()

    figures = {}
    for inputname in config['params']['inputs']:    
        figures[inputname] = go.Figure()
        figures[inputname].add_trace(go.Scatter(x=np.arange(len(targetseries[inputname])), y=targetseries[inputname], name="Targets", line_color='red'))
        figures[inputname].add_trace(go.Scatter(x=np.arange(len(outputseries[inputname])), y=outputseries[inputname], name="Outputs", line_color='blue'))
        figures[inputname].update_layout(title_text=f"TimeSeries-{inputname}", xaxis_rangeslider_visible=True)

    df = pd.DataFrame.from_records(metricbook)
    df = df.append({'filename':'TOTAL', 'mae':df['mae'].mean(), 'rmse':df['rmse'].mean(), 'mape':df['mape'].mean(), 'evs':df['evs'].mean()}, ignore_index=True)
    df.columns = [name.upper() for name in list(df.columns)]

    figures['table'] = go.Figure()
    figures['table'].add_trace(
        go.Table(header=dict(values=list(df.columns)), cells=dict(values=[df['FILENAME'], df['MAE'].round(2), df['RMSE'].round(2), df['MAPE'].round(2), df['EVS'].round(2)]) )
    )
    figures['table'].update_layout(title=f"Quantitative Metrics")
    
    with open(f"{config['paths']['logs']}/{use_model}/visualbook.html", 'w') as f:
        for figurename in figures.keys():
            f.write(figures[figurename].to_html(full_html=False, include_plotlyjs='cdn'))
    