import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Concatenate, Dense
from tensorflow.keras.models import Model
from algobox.layers.attention import *

defaultparams = {
    'n_layers': 2,
    'attention_hidden_size': 16*8,
    'num_heads': 8,
    'ffn_hidden_size': 16*8,
    'ffn_filter_size': 16*8,
    'attention_dropout': 0.1,
    'relu_dropout': 0.1,
    'layer_postprocess_dropout': 0.1,
}


class TransformerNet(object):
    def __init__(self, inputdimension, exogdimension, inputlength, outputlength, custom_params={}):
        defaultparams.update(custom_params)
        self.params = defaultparams
        self.params['inputdim'] = inputdimension
        self.params['exogdimension'] = exogdimension
        self.params['inputlength'] = inputlength
        self.params['outputlength'] = outputlength
        self.hiddenunits = self.params['attention_hidden_size']

        self.embedding_layer = EmbeddingLayer(embedding_size=self.params['attention_hidden_size'])
        self.encoder_stack = EncoderStack(self.params)
        self.decoder_stack = DecoderStack(self.params)
        self.projection = Dense(units=1)

    def get_config(self):
        return {'params': self.params}

    def __call__(self, training):
        mx1 = Input([self.params['inputlength'], self.params['inputdim']])
        e1 = Input([self.params['inputlength'], self.params['exogdimension']])
        mx2 = Input([self.params['outputlength'], self.params['inputdim']])
        e2 = Input([self.params['outputlength'], self.params['exogdimension']])

        teacherforcing = False
        x1_container, x2_container, y_container = {}, {}, {}
        for inputno in range(self.params['inputdim']):
            x1_container[inputno] = Lambda(lambda z: z[..., inputno])(mx1)
            x2_container[inputno] = Lambda(lambda z: z[..., inputno])(mx2)

        for inputno in range(self.params['inputdim']):
            x1 = tf.expand_dims(x1_container[inputno], axis=-1)
            x2 = tf.expand_dims(x2_container[inputno], axis=-1)

            encoder_feature = tf.concat([x1, e1], axis=-1)
            decoder_feature = e2
            teacherforce = x2

            self.position_encoding_layer_enc = PositionEncoding(max_len=self.params['inputlength'])
            self.position_encoding_layer_dec = PositionEncoding(max_len=self.params['outputlength'])

            src_mask = self.get_src_mask(encoder_feature)  # => batch_size * input_sequence_length
            src_mask = self.get_src_mask_bias(src_mask)  # => batch_size * 1 * 1 * input_sequence_length
            memory = self.encoder(encoder_inputs=encoder_feature, mask=src_mask, training=training)

            if training and teacherforcing:
                decoder_inputs = tf.concat([x1[:, -1:, :], teacherforce[:, :-1, :]], axis=1)
                decoder_inputs = tf.concat([decoder_inputs, decoder_feature], axis=-1)

                decoder_output = self.decoder(decoder_inputs, memory, src_mask, training=training, outputlength=self.params['outputlength'])
                y = self.projection(decoder_output)
            else:
                decoder_inputs = decoder_inputs_update = tf.cast(x1[:, -1:, :], tf.float32)
                for i in range(self.params['outputlength']):
                    decoder_inputs_update = tf.concat([decoder_inputs_update, decoder_feature[:, :i+1, :]], axis=-1)
                    decoder_inputs_update = self.decoder(decoder_inputs_update, memory, src_mask, training, outputlength=1)
                    decoder_inputs_update = self.projection(decoder_inputs_update)
                    decoder_inputs_update = tf.concat([decoder_inputs, decoder_inputs_update], axis=1)
                y = decoder_inputs_update[:, 1:, :]
            y_container[inputno] = y

        if self.params['inputdim']>1:
            my = Concatenate(axis=-1)([y_container[inputno] for inputno in range(self.params['inputdim'])])  
        else:
            my = y_container[0]
        
        transformermodel = Model([mx1, e1, mx2, e2], my)
        return transformermodel

    def encoder(self, encoder_inputs, mask, training):
        with tf.name_scope("encoder"):
            src = self.embedding_layer(encoder_inputs)  # batch_size * sequence_length * embedding_size
            src += self.position_encoding_layer_enc(src)

            if training:
                src = tf.nn.dropout(src, rate=self.params['attention_dropout'])  # batch_size * sequence_length * attention_hidden_size

            return self.encoder_stack(src, mask, training)

    def decoder(self, decoder_inputs, memory, src_mask, training, outputlength):
        with tf.name_scope("shift_targets"):
            tgt_mask = self.get_tgt_mask_bias(outputlength)
            tgt = self.embedding_layer(decoder_inputs)

        with tf.name_scope("add_pos_encoding"):
            pos_encoding = self.position_encoding_layer_dec(tgt)
            tgt += pos_encoding

        if training:
            tgt = tf.nn.dropout(tgt, rate=self.params["layer_postprocess_dropout"])

        with tf.name_scope('decoder'):
            dsoutputs = self.decoder_stack(tgt, memory, src_mask, tgt_mask, training)  
        return dsoutputs

    def get_src_mask(self, x, pad=0):
        src_mask = tf.reduce_all(tf.math.equal(x, pad), axis=-1)
        return src_mask

    def get_src_mask_bias(self, mask):
        attention_bias = tf.cast(mask, tf.float32)
        attention_bias = attention_bias * tf.constant(-1e9, dtype=tf.float32)
        attention_bias = tf.expand_dims(tf.expand_dims(attention_bias, 1), 1)  # => batch_size * 1 * 1 * input_length
        return attention_bias

    def get_tgt_mask_bias(self,length):
        valid_locs = tf.linalg.band_part(tf.ones([length, length], dtype=tf.float32),-1, 0)
        valid_locs = tf.reshape(valid_locs, [1, 1, length, length])
        decoder_bias = -1e9 * (1.0 - valid_locs)
        return decoder_bias


class EncoderStack(tf.keras.layers.Layer):
    def __init__(self, params):
        super(EncoderStack, self).__init__()
        self.params = params
        self.layers = []

    def build(self, input_shape):
        for _ in range(self.params['n_layers']):
            attention_layer = Attention(self.params['attention_hidden_size'],
                                        self.params['num_heads'],
                                        self.params['attention_dropout'])
            feed_forward_layer = FeedForwardNetwork(self.params['ffn_hidden_size'],
                                                    self.params['ffn_filter_size'],
                                                    self.params['relu_dropout'])
            post_attention_layer = SublayerConnection(attention_layer, self.params)
            post_feed_forward_layer = SublayerConnection(feed_forward_layer, self.params)
            self.layers.append([post_attention_layer, post_feed_forward_layer])
        self.output_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype="float32")
        super(EncoderStack, self).build(input_shape)

    def get_config(self):
        return {'params': self.params}

    def call(self, encoder_inputs, src_mask, training):
        for n, layer in enumerate(self.layers):
            attention_layer = layer[0]
            ffn_layer = layer[1]

            with tf.name_scope('layer_{}'.format(n)):
                with tf.name_scope('self_attention'):
                    encoder_inputs = attention_layer(encoder_inputs, encoder_inputs, src_mask, training=training)
                with tf.name_scope('ffn'):
                    encoder_inputs = ffn_layer(encoder_inputs, training=training)
        return self.output_norm(encoder_inputs)


class DecoderStack(tf.keras.layers.Layer):
    def __init__(self, params):
        super(DecoderStack, self).__init__()
        self.params = params
        self.layers = []

    def build(self,input_shape):
        for _ in range(self.params['n_layers']):
            self_attention_layer = Attention(self.params['attention_hidden_size'],
                                             self.params['num_heads'],
                                             self.params['attention_dropout'])
            enc_dec_attention_layer = Attention(self.params['attention_hidden_size'],
                                                self.params['num_heads'],
                                                self.params['attention_dropout'])
            feed_forward_layer = FeedForwardNetwork(self.params['ffn_hidden_size'],
                                                    self.params['ffn_filter_size'],
                                                    self.params['relu_dropout'])
            post_self_attention_layer = SublayerConnection(self_attention_layer, self.params)
            post_enc_dec_attention_layer = SublayerConnection(enc_dec_attention_layer, self.params)
            post_feed_forward_layer = SublayerConnection(feed_forward_layer, self.params)
            self.layers.append([post_self_attention_layer, post_enc_dec_attention_layer, post_feed_forward_layer])
        self.output_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype="float32")
        super(DecoderStack, self).build(input_shape)

    def get_config(self):
        return {'params': self.params}

    def call(self, decoder_inputs, encoder_outputs, src_mask, tgt_mask, training):
        for n, layer in enumerate(self.layers):
            self_attention_layer = layer[0]
            enc_dec_attention_layer = layer[1]
            ffn_layer = layer[2]

            with tf.name_scope("dec_layer_{}".format(n)):
                with tf.name_scope('self_attention'):
                    decoder_inputs = self_attention_layer(decoder_inputs, decoder_inputs, tgt_mask, training=training)
                with tf.name_scope('enc_dec_attention'):
                    decoder_inputs = enc_dec_attention_layer(decoder_inputs, encoder_outputs, src_mask, training=training)
                with tf.name_scope('ffn'):
                    decoder_inputs = ffn_layer(decoder_inputs, training=training)
        return self.output_norm(decoder_inputs)