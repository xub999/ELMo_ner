# -*- coding: utf-8 -*-
import glob

"""# config.py"""
USE_google_drive = True

import os

config = dict()

elmo = dict()

dir = "/content/ELMo_ner/"
if USE_google_drive:
    dir = "/content/drive/My Drive/ai-gg_drive/ner/ELMo_ner/"

elmo['data_path'] = dir + "data/ner_dataset.csv"
elmo['modelCheckpoint_file'] = dir + "record/modelCheckpoint_file.cpt"
elmo['have_trained_nb_epoch_file'] = dir + "record/have_trained_nb_epoch.dat.npy"
elmo['tensorboard_dir'] = dir + "record/tensorboard"
elmo['hub_model_file'] = dir + "record/hub_elmo_module"


elmo['batch_size'] = 128
elmo['maxlen'] = 50

elmo['test_rate'] = 0.1
elmo['val_rate'] = 0.1
elmo["n_epochs"] = 20

elmo['n_tags'] = 0

# for google drive


config['elmo'] = elmo

"""# Data.py"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K
from keras.models import Model, Input, load_model
from keras.layers.merge import add
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Lambda
from keras.preprocessing.sequence import pad_sequences
import random


class SentenceGetter(object):
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


class Data(object):
    def __init__(self):
        self.batch_size = elmo["batch_size"]
        test_split_rate = elmo["test_rate"]
        val_split_rate = elmo["val_rate"]
        max_len = elmo['maxlen']

        data = pd.read_csv(elmo['data_path'], encoding="latin1")
        data = data.fillna(method="ffill")

        words = list(set(data["Word"].values))
        words.append("ENDPAD")
        tags = list(set(data["Tag"].values))
        elmo['n_tags'] = len(tags)
        getter = SentenceGetter(data)

        sentences = getter.sentences
        tag2idx = {t: i for i, t in enumerate(tags)}
        X = [[w[0] for w in s] for s in sentences]
        new_X = []
        for seq in X:
            new_seq = []
            for i in range(max_len):
                try:
                    new_seq.append(seq[i])
                except:
                    new_seq.append("__PAD__")
            new_X.append(new_seq)
        X = np.array(new_X)

        y = [[tag2idx[w[2]] for w in s] for s in sentences]
        y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_split_rate, random_state=2018)
        X_tr = np.array(X_tr)
        self.X_te = np.array(X_te)

        self.X_tr, self.X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=val_split_rate, random_state=2018)
        # self.X_tr, self.X_val = X_tr[:1213 * self.batch_size], X_tr[-135 * self.batch_size:]
        # y_tr, y_val = y_tr[:1213 * self.batch_size], y_tr[-135 * self.batch_size:]

        self.y_tr = y_tr.reshape(y_tr.shape[0], y_tr.shape[1], 1)
        self.y_val = y_val.reshape(y_val.shape[0], y_val.shape[1], 1)
        self.y_te = y_te.reshape(y_te.shape[0], y_te.shape[1], 1)

        # train paths shuffle
        self.shuffle_train_data()

        self.X_tr = np.array(self.X_tr)
        self.X_val = np.array(self.X_val)

        self.total_nb_batch_train = int(self.X_tr.shape[0] / self.batch_size)
        self.now_nb_batch_index_train = 0
        self.total_nb_batch_validate = int(self.X_val.shape[0] / self.batch_size)
        self.now_nb_batch_index_validate = 0

    def shuffle_train_data(self):
        nb_train_data = self.X_tr.shape[0]
        random_index = np.linspace(0, nb_train_data - 1, nb_train_data, dtype=np.int32)
        random.shuffle(random_index)
        self.X_tr = self.X_tr[random_index]
        self.y_tr = self.y_tr[random_index]

    def next_batch_train(self):
        data_X_batch_train = self.X_tr[self.now_nb_batch_index_train*self.batch_size:(self.now_nb_batch_index_train+1)*self.batch_size]
        data_y_batch_train = self.y_tr[self.now_nb_batch_index_train*self.batch_size:(self.now_nb_batch_index_train+1)*self.batch_size]

        self.now_nb_batch_index_train += 1
        if self.now_nb_batch_index_train == self.total_nb_batch_train:
            self.now_nb_batch_index_train = 0
            self.shuffle_train_data()

        return np.array(data_X_batch_train), np.array(data_y_batch_train)

    def next_batch_validate(self):
        data_X_batch_validate = self.X_val[self.now_nb_batch_index_validate * self.batch_size:(self.now_nb_batch_index_validate + 1) * self.batch_size]
        data_y_batch_validate = self.y_val[self.now_nb_batch_index_validate * self.batch_size:(self.now_nb_batch_index_validate + 1) * self.batch_size]

        self.now_nb_batch_index_validate += 1
        if self.now_nb_batch_index_validate == self.total_nb_batch_validate:
            self.now_nb_batch_index_validate = 0

        return np.array(data_X_batch_validate), np.array(data_y_batch_validate)


"""# model.py"""
from keras.callbacks import Callback, ModelCheckpoint, TensorBoard
from keras.utils import plot_model


class Save_crt_epoch_nb(Callback):
    def __init__(self, filepath):
        super(Save_crt_epoch_nb, self).__init__()
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs=None):
        np.save(self.filepath, np.array(epoch))


class Save_records(Callback):
    def __init__(self, filepath, val_filepath):
        super(Save_records, self).__init__()
        self.filepath = filepath
        self.val_filepath = val_filepath
        # self.t_logs = t_logs
        # self.v_logs = v_logs

    '''
    def on_batch_end(self, batch, logs=None):
        print('\t\t\tbatch %d:\tloss: %.5f\tdice: %.4f' %
              (logs['batch'], logs['loss'],
               logs['dice_coef_except_background']))
    '''

    def on_epoch_end(self, epoch, logs=None):
        print('epoch %d:\tloss: %.5f\t6_dice: %.4f' %
              (epoch, logs['loss'],
               logs['dice_coef_except_background']))
        print('\t\t\tval_loss: %.5f\tval_6_dice: %.4f' %
              (logs['val_loss'],
               logs['val_dice_coef_except_background']))
        print()


elmo_model = None

if os.path.exists(elmo['hub_model_file']):
    elmo_model = hub.Module(elmo['hub_model_file'])
else:
    elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        elmo_model.export(elmo['hub_model_file'], sess)


def ElmoEmbedding(x):
    return elmo_model(inputs={
                            "tokens": tf.squeeze(tf.cast(x, tf.string)),
                            "sequence_len": tf.constant(elmo['batch_size']*[elmo['maxlen']])
                      },
                      signature="tokens",
                      as_dict=True)["elmo"]


class ELMo(object):
    def __init__(self):
        self.myData = Data()
        self.epoches = elmo["n_epochs"]
        self.batch_size = elmo['batch_size']

        # load elmo model
        self.elmo_net = None
        if os.path.exists(config['elmo']['modelCheckpoint_file']):
            self.elmo_net = load_model(config['elmo']["modelCheckpoint_file"])
            print('loading elmo model and weights from file')
            print("got elmo")
        else:
            self.elmo_net = self.get_elmo()
            print('no elmo model file exists, creating model')

        # load have_trained_nb_epoch
        if os.path.exists(config['elmo']['have_trained_nb_epoch_file']):
            print("file not exist: " + config['elmo']['have_trained_nb_epoch_file'])
            self.have_trained_nb_epoch = np.load(config['elmo']['have_trained_nb_epoch_file']) + 1
        else:
            self.have_trained_nb_epoch = 0

    def next_batch_data_train(self):
        data_X_batch_train, data_y_batch_train = self.myData.next_batch_train()
        return data_X_batch_train, data_y_batch_train

    def next_batch_data_validate(self):
        data_X_batch_validate, data_y_batch_validate = self.myData.next_batch_validate()
        return data_X_batch_validate, data_y_batch_validate

    def generator_data_train_fine(self):
        while True:
            X_batch_train, y_batch_train = self.next_batch_data_train()

            yield ({'input': X_batch_train},
                   {'output': y_batch_train})

    def generator_data_validate_fine(self):
        while True:
            X_batch_validate, y_batch_validate = self.next_batch_data_validate()

            yield ({'input': X_batch_validate},
                   {'output': y_batch_validate})

    def get_elmo(self):
        input_text = Input(shape=(elmo['maxlen'],), dtype='string', name='input')
        embedding = Lambda(ElmoEmbedding, output_shape=(elmo['maxlen'], 1024))(input_text)
        x = Bidirectional(LSTM(units=512, return_sequences=True,
                               recurrent_dropout=0.2, dropout=0.2))(embedding)
        x_rnn = Bidirectional(LSTM(units=512, return_sequences=True,
                                   recurrent_dropout=0.2, dropout=0.2))(x)
        x = add([x, x_rnn])  # residual connection to the first biLSTM
        out = TimeDistributed(Dense(elmo['n_tags'], activation="softmax"), name='output')(x)

        model = Model(input_text, out)
        # model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

        return model

    def train_elmo_generator(self):
        elmo_net = self.elmo_net

        save_crt_epoch_nb = Save_crt_epoch_nb(config['elmo']['have_trained_nb_epoch_file'])
        # save_records = Save_records(config['unet']['logges_file'], config['unet']['validate_loss_file'])
        checkpointer = ModelCheckpoint(filepath=config['elmo']['modelCheckpoint_file'],
                                       verbose=1, save_best_only=False, save_weights_only=False)
        tensorboard = TensorBoard(log_dir=config['elmo']['tensorboard_dir'])

        print('Fitting model...')

        elmo_net.fit_generator(
            generator=self.generator_data_train_fine(),
            steps_per_epoch=self.myData.total_nb_batch_train,
            epochs=self.epoches,
            verbose=1,
            # callbacks=[save_crt_epoch_nb, save_records, checkpointer, tensorboard],
            callbacks=[save_crt_epoch_nb, checkpointer, tensorboard],
            validation_data=self.generator_data_validate_fine(),
            validation_steps=self.myData.total_nb_batch_validate,
            class_weight=None,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
            shuffle=False,
            initial_epoch=self.have_trained_nb_epoch
            )

    def plot_model(self):
        model = self.get_elmo()
        plot_model(model, to_file='model.png')


if __name__ == '__main__':
    # '''
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    my_elmo_model = ELMo()
    my_elmo_model.train_elmo_generator()
