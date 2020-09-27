import __init__
import tensorflow as tf
tf.keras.backend.set_floatx('float32')

import numpy as np, pandas as pd
import argparse, json, joblib, os, logging
from algobox.model import TimeSeriesModel
from utilsbox.loadops import PowerTAC

# *********************************************************************************************************
def getConfig(configName, arguments):
    filedir = os.path.dirname(os.path.realpath(__file__))
    rootdir = os.path.split(filedir)[0]

    config = json.load(open(f"{rootdir}/configbox/{configName}.txt", 'r'))
    for name in config['paths'].keys():
        if name!='type':
            config['paths'][name] = f"{rootdir}{config['paths'][name]}"
            os.makedirs(config['paths'][name], exist_ok=True)

    if arguments['inputlength'] is not None:
        config['params']['inputlength'] = arguments['inputlength']
    if arguments['outputlength'] is not None:
        config['params']['outputlength'] = arguments['outputlength']
    if arguments['samplestep'] is not None:
        config['params']['samplestep'] = arguments['samplestep']
    return config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TIME SERIES FORECASTING TRAIN MODULE')
    parser.add_argument('-alg',     '--algorithm',        type=str,       required=True,      default=None,     help='TSF algorithm')
    parser.add_argument('-tfm',     '--tensorflowmode',   type=str,       required=False,     default='eager',  help='Tensorflow mode')
    parser.add_argument('-cfg',     '--configuration',    type=str,       required=True,      default=None,     help='configuration file')
    parser.add_argument('-gpu',     '--gpuids',           type=str,       required=False,     default='0',      help='gpu devices id')
    parser.add_argument('-isl',     '--inputlength',      type=int,       required=False,     default=None,     help='Timeseries Input Sequence length')
    parser.add_argument('-osl',     '--outputlength',     type=int,       required=False,     default=None,     help='Timeseries Output Sequence length')
    parser.add_argument('-sst',     '--samplestep',       type=int,       required=False,     default=None,     help='Timeseries Sampling Step')
    args = vars(parser.parse_args())
    os.environ["CUDA_VISIBLE_DEVICES"]=args['gpuids']

    config = getConfig(configName=args['configuration'], arguments=args)
    identitytag = config['params']['identity']

    if 'ptac' in identitytag:
        dataloader = PowerTAC(config, 'train')
    else: 
        raise NotImplementedError(f'must implement dataloader for Identity {identitytag}')
    
    network = TimeSeriesModel(config, use_model=args['algorithm'], custom_model_params=config['algoparams'])
    if args['tensorflowmode']=='fit':
        network.train(dataloader, 'fit')
    else:
        network.train(dataloader, 'eager')