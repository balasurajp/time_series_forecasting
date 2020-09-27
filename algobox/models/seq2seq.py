import tensorflow as tf
from tensorflow.keras.layers import Dense, GRUCell, LSTMCell, RNN, Input, Lambda, Concatenate
from tensorflow.keras.models import Model
from algobox.layers.attention import Attention

defaultparams = {
    'hidden_size': 256,
    'use_attention': True,
    'num_heads': 8,
    'hidden_dropout': 0.4,
    'attention_dropout': 0.4
}


class Seq2seqNet(object):
    def __init__(self, inputdimension, exogdimension, inputlength, outputlength, custom_params):
        defaultparams.update(custom_params)
        self.params = defaultparams
        self.params['inputdim'] = inputdimension
        self.params['exogdimension'] = exogdimension
        self.params['inputlength'] = inputlength
        self.params['outputlength'] = outputlength
        self.hiddenunits = self.params['hidden_size']

        self.encoder = Encoder(self.params)
        self.decoder = Decoder(self.params)

    def __call__(self, training):
        mx1 = Input([self.params['inputlength'], self.params['inputdim']])
        e1 = Input([self.params['inputlength'], self.params['exogdimension']])
        mx2 = Input([self.params['outputlength'], self.params['inputdim']])
        e2 = Input([self.params['outputlength'], self.params['exogdimension']])

        # x1_container, x2_container, y_container = {}, {}, {}
        # for inputno in range(self.params['inputdim']):
        #     x1_container[inputno] = Lambda(lambda z: z[..., inputno])(mx1)
        #     x2_container[inputno] = Lambda(lambda z: z[..., inputno])(mx2)

        teacherforcing=False
        # for inputno in range(self.params['inputdim']):
            # x1 = tf.expand_dims(x1_container[inputno], axis=-1)
            # x2 = tf.expand_dims(x2_container[inputno], axis=-1)

        encoder_feature = tf.concat([mx1, e1], axis=-1)
        decoder_feature = e2
        if training and teacherforcing:
            teacherforce = mx2
        else:
            teacherforce = None

        encoder_output, encoder_state = self.encoder(encoder_feature, training)
        decoder_output = self.decoder(
            training=training,
            decoder_feature=decoder_feature, 
            init_state=encoder_state, 
            init_value=mx1[:,-1,:],
            encoder_output=encoder_output,
            outputlength=self.params['outputlength'],
            teacher=teacherforce,
            use_attention=self.params['use_attention']
        )
        my = decoder_output

        # y_container[inputno] = decoder_output
        # if self.params['inputdim']>1:
        #     my = Concatenate(axis=-1)([y_container[inputno] for inputno in range(self.params['inputdim'])])  
        # else:
        #     my = y_container[0]

        seq2seqmodel = Model([mx1, e1, mx2, e2], my)
        return seq2seqmodel


class Encoder(object):
    def __init__(self, params):
        self.params = params
        self.rnn_cell = GRUCell(units=self.params['hidden_size'])
        self.rnn = RNN(self.rnn_cell, return_state=True, return_sequences=True)
        self.dense = Dense(units=self.params['inputdim'])

    def __call__(self, inputs, training):
        outputs, state = self.rnn(inputs)
        if training:
            outputs = tf.nn.dropout(outputs, rate=self.params['hidden_dropout']) 
        outputs = self.dense(outputs)
        return outputs, state


class Decoder(object):
    def __init__(self, params):
        self.params = params
        self.rnn_cell = GRUCell(self.params['hidden_size'])
        self.rnn = RNN(self.rnn_cell, return_state=True, return_sequences=True)
        self.dense = Dense(units=self.params['inputdim'])
        self.attention = Attention(hidden_size=self.params['hidden_size'], num_heads=self.params['num_heads'], attention_dropout=self.params['attention_dropout'])

    def forward(self, training, decoder_feature, init_state, init_value, encoder_output, outputlength, teacher, use_attention):
        def cond_fn(time, prev_output, prev_state, decoder_output_ta):
            return time < outputlength
        def body(time, prev_output, prev_state, decoder_output_ta):
            if time == 0 or teacher is None:
                this_input = prev_output
            else:
                this_input = teacher[:, time-1, :]

            if decoder_feature is not None:
                this_feature = decoder_feature[:, time, :]
                this_input = tf.concat([this_input, this_feature], axis=1)

            if use_attention:
                attention_feature = self.attention(tf.expand_dims(prev_state[-1], axis=1), encoder_output)
                attention_feature = tf.squeeze(attention_feature, axis=1)
                this_input = tf.concat([this_input, attention_feature], axis=-1)

            this_output, this_state = self.rnn_cell(this_input, prev_state)
            if training:
                this_output = tf.nn.dropout(this_output, rate=self.params['hidden_dropout']) 
            project_output = self.dense(this_output)
            decoder_output_ta = decoder_output_ta.write(time, project_output)
            return time+1, project_output, this_state, decoder_output_ta

        loop_init = [tf.constant(0, dtype=tf.int32), 
                     init_value, 
                     init_state, 
                     tf.TensorArray(dtype=tf.float32, size=outputlength)]
        _, _, _, decoder_outputs_ta = tf.while_loop(cond_fn, body, loop_init)

        decoder_outputs = decoder_outputs_ta.stack()
        decoder_outputs = tf.transpose(decoder_outputs, [1, 0, 2])
        return decoder_outputs

    def __call__(self, training, decoder_feature, init_state, init_value, encoder_output, outputlength=1, teacher=None, use_attention=False):
        return self.forward(training=training, 
                            decoder_feature=decoder_feature,
                            init_state=[init_state],
                            init_value=init_value,
                            encoder_output=encoder_output,
                            outputlength=outputlength,
                            teacher=teacher,
                            use_attention=use_attention)

