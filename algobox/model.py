import __init__
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping,  ModelCheckpoint, TensorBoard
from algobox.models.nbeats import NBeatsNet
from algobox.models.transformer import TransformerNet
from utilsbox.misc import avgmeter
from tqdm import tqdm

class Loss(object):
    def __init__(self, use_loss):
        self.use_loss = use_loss

    def __call__(self,):
        if self.use_loss == 'mse':
            return tf.keras.losses.MeanSquaredError()
        elif self.use_loss == 'rmse':
            return tf.math.sqrt(tf.keras.losses.MeanSquaredError())
        elif self.use_loss == 'mae':
            return tf.keras.losses.MeanAbsoluteError()
        elif self.use_loss == 'huber':
            return tf.keras.losses.Huber(delta=1.0)
        else:
            raise ValueError(f"Not supported function: {self.use_loss}")


class Optimizer(object):
    def __init__(self, use_optimizer):
        self.use_optimizer = use_optimizer

    def __call__(self, learning_rate):
        if self.use_optimizer == 'adam':
            return tf.keras.optimizers.Adam(lr=learning_rate)
        elif self.use_optimizer == 'sgd':
            return tf.keras.optimizers.SGD(lr=learning_rate)
        else:
            raise ValueError(f"Not supported optimizer: {self.use_optimizer}")


class TimeSeriesModel(object):
    def __init__(self, config, use_model, custom_model_params):
        self.paths = config['paths']
        self.params = config['params']

        self.use_model = use_model
        self.custom_model_params = custom_model_params

        self.use_loss = self.params['loss']
        self.use_optimizer = self.params['optimizer']

        self.input_dimension = len(self.params['inputs'])
        self.exog_dimension = len(self.params['exogenous']) + 17 #calender variables
        self.input_length = self.params['inputlength']
        self.output_length = self.params['outputlength']

    def build_model(self):
        if self.use_model == 'nbeats':
            tsnetwork = NBeatsNet(self.input_dimension, self.exog_dimension, self.input_length, self.output_length, self.custom_model_params)
        elif self.use_model == 'transformer':
            tsnetwork = TransformerNet(self.input_dimension, self.exog_dimension, self.input_length, self.output_length, self.custom_model_params)
        else:
            raise ValueError(f"unsupported use_model named {self.use_model} yet")

        self.tsmodel = tsnetwork(training=True)
        self.loss_fn = Loss(self.use_loss)()
        self.optimizer_fn = Optimizer(self.use_optimizer)(learning_rate=self.params['learning_rate'])

    @tf.function
    def trnstep(self, data):
        inputs, targets = data['inputs'], data['targets']
        with tf.GradientTape() as tape:
            outputs = self.tsmodel(inputs, training=True)
            batchloss = self.loss_fn(targets, outputs)

        gradients = tape.gradient(batchloss, self.tsmodel.trainable_variables)
        gradients = [(tf.clip_by_value(gradient, -10.0, 10.0)) for gradient in gradients]
        self.optimizer_fn.apply_gradients(zip(gradients, self.tsmodel.trainable_variables))
        
        self.trnsteps.assign_add(1)
        return batchloss
    
    @tf.function
    def valstep(self, data):
        inputs, targets = data['inputs'], data['targets']
        outputs = self.tsmodel(inputs, training=False)
        batchloss = self.loss_fn(targets, outputs)
        
        self.valsteps.assign_add(1)
        return batchloss

    def train(self, dataloader, mode='eager'):
        print("-" * 35)
        print(f"Starting training {self.use_model} in {mode} mode")
        print("-" * 35)
        self.build_model()
        dataset = dataloader.get_samples()
        dataset.pop('scalers')
        if self.custom_model_params['netdesign']=='oneshot':
            inputs, targets = tuple(dataset['inputs'][0:2]), dataset['targets']
        else:
            inputs, targets = tuple(dataset['inputs']), dataset['targets']

        if mode == 'fit':
            self.tsmodel.compile(loss=self.loss_fn, optimizer=self.optimizer_fn)
            callbacks = [TensorBoard(log_dir=f"{self.paths['logs']}/{self.use_model}"),
                         ModelCheckpoint(f"{self.paths['ckpts']}/{self.use_model}_weights.h5", verbose=1, save_weights_only=True)]
    
            self.tsmodel.fit(x=inputs, y=targets, batch_size=self.params['batchsize'], epochs=self.params['epochs'], 
                             validation_split=0.2, shuffle=True, callbacks=callbacks)

        elif mode == 'eager':
            self.summarywriter = tf.summary.create_file_writer(f"{self.paths['logs']}/{self.use_model}")
            self.trnsteps = tf.Variable(1, trainable=False, dtype=tf.int64)
            self.valsteps = tf.Variable(1, trainable=False, dtype=tf.int64)
            trnloss = avgmeter()
            valloss = avgmeter()

            nbatches = len(dataset['targets'])//self.params['batchsize']
            dataset = tf.data.Dataset.from_tensor_slices({'inputs':inputs, 'targets':targets}).shuffle(nbatches, reshuffle_each_iteration=False).batch(self.params['batchsize']) 

            comparisonmetric = 1e9
            for epoch in range(1, self.params['epochs'] + 1):
                trnloss.reset()
                valloss.reset()
                with tqdm(total=nbatches) as bar:
                    bar.set_description(f'Epoch:{epoch}')
                    for index, batchdata in dataset.enumerate():
                        if (index+1)/nbatches < 0.8:
                            batchloss = self.trnstep(batchdata)
                            trnloss.update(batchloss.numpy(), self.params['batchsize'])
                            bar.update(1)
                            bar.set_postfix(trainloss=trnloss.avg, validloss=valloss.avg)
                            with self.summarywriter.as_default():
                                tf.summary.scalar('trainloss-iterations', batchloss, step=self.trnsteps)
                        else:
                            batchloss = self.valstep(batchdata)
                            valloss.update(batchloss.numpy(), self.params['batchsize'])
                            bar.update(1)
                            bar.set_postfix(trainloss=trnloss.avg, validloss=valloss.avg)
                            with self.summarywriter.as_default():
                                tf.summary.scalar('validationloss-iterations', batchloss, step=self.valsteps)
                            
                    with self.summarywriter.as_default():
                        tf.summary.scalar('Lossfunction/trn', trnloss.avg, step=epoch)
                        tf.summary.scalar('Lossfunction/val', valloss.avg, step=epoch)

                if valloss.avg < comparisonmetric:
                    self.tsmodel.save_weights(f"{self.paths['ckpts']}/{self.use_model}_weights.h5")
                    comparisonmetric = valloss.avg
                else:
                    print('Validation loss increased!')

        else:
            raise ValueError(f"Unsupported training mode: {mode}, choose 'eager' or 'fit'")
        self.exportmodel()

    def test(self, dataloader):
        print("-" * 35)
        print(f"Starting testing {self.use_model}")
        print("-" * 35)
        self.build_model()
        self.importmodel()

        dataset = dataloader.get_databook()
        for name in dataset.keys():
            print(f'Predicting {name}')
            inputs, targets, scalers = dataset[name]['inputs'], dataset[name]['targets'], dataset[name]['scalers']
            outputs = self.tsmodel(inputs, training=False).numpy()
            dataset[name]['outputs'] = outputs 
            
            dataset[name].pop('inputs')
            dataset[name]['outputs'] = (dataset[name]['outputs']*dataset[name]['scalers'][1]) + dataset[name]['scalers'][0] 
            dataset[name]['targets'] = (dataset[name]['targets']*dataset[name]['scalers'][1]) + dataset[name]['scalers'][0] 
            # for index, inputname in enumerate(self.params['inputs']):
            #     dataset[name]['targets'][:,:,index] = scalebook[inputname].invTransform(dataset[name]['targets'][:,:,index])
            #     dataset[name]['outputs'][:,:,index] = scalebook[inputname].invTransform(dataset[name]['outputs'][:,:,index])
        return dataset

    def exportmodel(self):
        self.tsmodel.load_weights(f"{self.paths['ckpts']}/{self.use_model}_weights.h5")
        self.tsmodel.save_weights(f"{self.paths['models']}/{self.use_model}_model.h5")

    def importmodel(self):
        self.tsmodel.load_weights(f"{self.paths['models']}/{self.use_model}_model.h5")
