import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Concatenate 
from tensorflow.keras.layers import Input, Dense, Lambda, Subtract, Add, Reshape, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from algobox.layers.nbeats import seasonality_model, trend_model

defaultparams = {
        "dropout_rate": 0.4,
        "hidden_units": 256,
        "stack_types": ["seasonality", "seasonality", "generic", "generic"],
        "thetas_dim": [8, 8, 8, 8],
        "blocks_per_stack": 4,
        "share_blockweights": True,
        "harmonics": False
}

class NBeatsNet:
    GENERIC_BLOCK = 'generic'
    TREND_BLOCK = 'trend'
    SEASONALITY_BLOCK = 'seasonality'
    
    def __init__(self, inputdimension, exogdimension, inputlength, outputlength, custom_params={}):
        self.backcast_length = inputlength
        self.forecast_length = outputlength
        self.input_dim = inputdimension
        self.exog_dim = exogdimension
        self.input_shape = (self.backcast_length, self.input_dim)
        self.exog_shape = (self.backcast_length, self.exog_dim)
        self.output_shape = (self.forecast_length, self.input_dim)
        
        self.weights = {}
        defaultparams.update(custom_params)
        self.dropoutrate = defaultparams['dropout_rate']        
        self.units = defaultparams['hidden_units']
        self.stack_types = defaultparams['stack_types']
        self.thetas_dim = defaultparams['thetas_dim']
        self.nb_blocks_per_stack = defaultparams['blocks_per_stack']
        self.share_weights_in_stack = defaultparams['share_blockweights']
        self.nb_harmonics = defaultparams['harmonics']
        assert len(self.stack_types) == len(self.thetas_dim), "Stack dimension doesn't match Theta dimension"

    def has_exog(self):
        return self.exog_dim > 0

    def __call__(self, training):
        # Input Variables Placeholder
        x = Input(shape=self.input_shape, name='input_variable')
        x_ = {}
        for k in range(self.input_dim):
            x_[k] = Lambda(lambda z: z[..., k])(x)
        
        # Exogenous Variables Placeholder
        e_ = {}
        if self.has_exog():
            e = Input(shape=self.exog_shape, name='exog_variables')
            for k in range(self.exog_dim):
                e_[k] = Lambda(lambda z: z[..., k])(e)
        else:
            e = None
        
        # Output Variables Placeholder
        y_ = {}

        for stack_id in range(len(self.stack_types)):
            stack_type = self.stack_types[stack_id]
            nb_poly = self.thetas_dim[stack_id]
            for block_id in range(self.nb_blocks_per_stack):
                backcast, forecast = self.create_block(x_, e_, stack_id, block_id, stack_type, nb_poly)
                for k in range(self.input_dim):
                    x_[k] = Subtract()([x_[k], backcast[k]])
                    if stack_id == 0 and block_id == 0:
                        y_[k] = forecast[k]
                    else:
                        y_[k] = Add()([y_[k], forecast[k]])

        for k in range(self.input_dim):
            y_[k] = Reshape(target_shape=(self.forecast_length, 1))(y_[k])
        if self.input_dim>1:
            y_ = Concatenate(axis=-1)([y_[index] for index in range(self.input_dim)])  
        else:
            y_ = y_[0]

        nbeats = Model([x, e], y_) if self.has_exog() else Model(x, y_)
        return nbeats

    # Register layer weights (useful when share_weights_in_stack=True)
    def LayerRegister(self, layer_with_weights, stack_id):
        # mechanism to restore weights when block share the same weights.
        # only useful when share_weights_in_stack=True.
        if self.share_weights_in_stack:
            layer_name = layer_with_weights.name.split('/')[-1]
            try:
                reused_weights = self.weights[stack_id][layer_name]
                return reused_weights
            except KeyError:
                pass
            if stack_id not in self.weights:
                self.weights[stack_id] = {}
            self.weights[stack_id][layer_name] = layer_with_weights
        return layer_with_weights

    def create_block(self, x, e, stack_id, block_id, stack_type, nb_poly):
        # update layer name (useful when share_weights_in_stack=True)
        def CreateIdentity(layer_name):
            return '/'.join([str(stack_id), str(block_id), stack_type, layer_name])

        backcast_ = {}
        forecast_ = {}
        d1 = self.LayerRegister(Dense(self.units, activation='relu', name=CreateIdentity('d1')), stack_id)
        d2 = self.LayerRegister(Dense(self.units, activation='relu', name=CreateIdentity('d2')), stack_id)
        d3 = self.LayerRegister(Dense(self.units, activation='relu', name=CreateIdentity('d3')), stack_id)
        d4 = self.LayerRegister(Dense(self.units, activation='relu', name=CreateIdentity('d4')), stack_id)
        if stack_type == 'generic':
            theta_b = self.LayerRegister(Dense(nb_poly, activation='linear', use_bias=False, name=CreateIdentity('theta_b')), stack_id)
            theta_f = self.LayerRegister(Dense(nb_poly, activation='linear', use_bias=False, name=CreateIdentity('theta_f')), stack_id)
            backcast = self.LayerRegister(Dense(self.backcast_length, activation='linear', name=CreateIdentity('backcast')), stack_id)
            forecast = self.LayerRegister(Dense(self.forecast_length, activation='linear', name=CreateIdentity('forecast')), stack_id)
        elif stack_type == 'trend':
            theta_f = theta_b = self.LayerRegister(Dense(nb_poly, activation='linear', use_bias=False, name=CreateIdentity('theta_f_b')), stack_id)
            backcast = Lambda(trend_model, arguments={"is_forecast": False, "backcast_length": self.backcast_length,
                                                      "forecast_length": self.forecast_length})
            forecast = Lambda(trend_model, arguments={"is_forecast": True, "backcast_length": self.backcast_length,
                                                      "forecast_length": self.forecast_length})
        elif stack_type == 'seasonality':
            if self.nb_harmonics:
                theta_b = self.LayerRegister(Dense(self.nb_harmonics, activation='linear', use_bias=False, name=CreateIdentity('theta_b')), stack_id)
            else:
                theta_b = self.LayerRegister(Dense(self.forecast_length, activation='linear', use_bias=False, name=CreateIdentity('theta_b')), stack_id)
            theta_f = self.LayerRegister(Dense(self.forecast_length, activation='linear', use_bias=False, name=CreateIdentity('theta_f')), stack_id)
            backcast = Lambda(seasonality_model,
                              arguments={"is_forecast": False, "backcast_length": self.backcast_length,
                                         "forecast_length": self.forecast_length})
            forecast = Lambda(seasonality_model,
                              arguments={"is_forecast": True, "backcast_length": self.backcast_length,
                                         "forecast_length": self.forecast_length})
        else:
            raise ValueError("Unsupported Stacktype for building Nbeats")

        for k in range(self.input_dim):
            if self.has_exog():
                d0 = Concatenate()([x[k]] + [e[index] for index in range(self.exog_dim)])
            else:
                d0 = x[k]
            d1_ = d1(d0)
            d2_ = d2(d1_)
            d2_ = Dropout(self.dropoutrate)(d2_)
            d3_ = d3(d2_)
            d3_ = Dropout(self.dropoutrate)(d3_)
            d4_ = d4(d3_)
            d4_ = Dropout(self.dropoutrate)(d4_)

            theta_f_ = theta_f(d4_)
            theta_b_ = theta_b(d4_)

            backcast_[k] = backcast(theta_b_)
            forecast_[k] = forecast(theta_f_)

        return backcast_, forecast_