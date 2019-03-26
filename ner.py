# -*- coding: utf-8 -*-
import glob

"""# config.py"""
USE_google_drive = True
USE_checkpoint_model = True

MODE = "TEST"

import os
import numpy as np


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
elmo['model_h5'] = dir + "record/keras_model/"
elmo['val_metrics_file'] = dir + "record/val_metrics.dat"

elmo['batch_size'] = 96
elmo['maxlen'] = 50

elmo['test_rate'] = 0.1
elmo['val_rate'] = 0.1
elmo["n_epochs"] = 20

elmo['tags'] = ['O', 'I-eve', 'I-tim', 'I-org', 'I-gpe', 'B-tim', 'B-eve', 'I-per', 'B-gpe', 'B-per', 'B-geo', 'I-geo', 'I-nat', 'B-org', 'I-art', 'B-nat', 'B-art']
elmo['n_tags'] = len(elmo['tags'])

elmo['pad_wording'] = '__PAD__'
elmo['prepared_X'] = np.array([elmo['pad_wording']] * elmo['maxlen'])

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
import tensorflow.keras as keras
from keras import backend as K
from keras.models import Model, Input, load_model
from keras.layers.merge import add
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Lambda
from keras.preprocessing.sequence import pad_sequences
from keras import metrics
import random


def my_train_test_split(X, y, split_rate):
    split_index = int(X.shape[0] * (1.0-split_rate))
    print("split_index = " + str(split_index))
    return X[:split_index], X[split_index:], y[:split_index], y[split_index:]

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
        tags = elmo['tags']
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

        # X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_split_rate, random_state=2018)
        X_tr, X_te, y_tr, y_te = my_train_test_split(X, y, test_split_rate)
        X_tr = np.array(X_tr)
        self.X_te = np.array(X_te)

        # self.X_tr, self.X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=val_split_rate, random_state=2018)
        self.X_tr, self.X_val, y_tr, y_val = my_train_test_split(X_tr, y_tr, val_split_rate)
        # self.X_tr = self.X_tr[:4 * self.batch_size]
        self.X_val = self.X_val[:self.X_val.shape[0]//self.batch_size * self.batch_size]
        # y_tr = y_tr[:4 * self.batch_size]
        y_val = y_val[:y_val.shape[0]//self.batch_size * self.batch_size]

        self.y_tr = y_tr.reshape(y_tr.shape[0], y_tr.shape[1], 1)
        self.y_val = y_val.reshape(y_val.shape[0], y_val.shape[1], 1)
        self.y_te = y_te.reshape(y_te.shape[0], y_te.shape[1], 1)

        self.X_te = self.X_te[:self.X_te.shape[0] // self.batch_size * self.batch_size]
        self.y_te = self.y_te[:self.y_te.shape[0] // self.batch_size * self.batch_size]

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


def my_sparse_categorical_accuracy(y_true, y_pred):
    return K.mean(
        K.cast(K.equal(K.flatten(y_true), K.flatten(K.cast(K.argmax(y_pred, axis=-1), K.floatx()))), K.floatx()))


from sklearn.metrics import f1_score, precision_score, recall_score
import pickle


val_metrics = dict()


class My_Metrics(Callback):
    def __init__(self, val_data):
        super(My_Metrics, self).__init__()

        self.val_data = val_data
        self.val_metrics = dict()
        if os.path.exists(elmo['val_metrics_file']):
            with open(elmo['val_metrics_file'], 'rb') as f:
                self.val_metrics = pickle.load(f)
        else:
            self.val_metrics = {"val_f1s": [], "val_recalls": [], "val_precisions": [],
                                "micro_val_f1s": [], "micro_val_recalls": [], "micro_val_precisions": []}

        self.val_f1s = self.val_metrics["val_f1s"]
        self.val_recalls = self.val_metrics["val_recalls"]
        self.val_precisions = self.val_metrics["val_precisions"]
        self.micro_val_f1s = self.val_metrics["micro_val_f1s"]
        self.micro_val_recalls = self.val_metrics["micro_val_recalls"]
        self.micro_val_precisions = self.val_metrics["micro_val_precisions"]

    def on_epoch_end(self, epoch, logs=None):
        val_predict = np.expand_dims(np.argmax(np.asarray(self.model.predict(self.val_data[0], batch_size=elmo['batch_size'], verbose=1)), -1).flatten(), -1)
        val_targ = np.expand_dims(self.val_data[1].flatten(), -1)

        _val_f1 = f1_score(val_targ, val_predict, average=None)
        _val_recall = recall_score(val_targ, val_predict, average=None)
        _val_precision = precision_score(val_targ, val_predict, average=None)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print("val_f1s =")
        for fls in self.val_f1s:
            print("\t".join(["%.4f" % i for i in fls]))
        print("val_recalls =")
        for recalls in self.val_recalls:
            print("\t".join(["%.4f" % i for i in recalls]))
        print("val_precisions =")
        for precisions in self.val_precisions:
            print("\t".join(["%.4f" % i for i in precisions]))

        _val_f1 = f1_score(val_targ, val_predict, average='micro')
        _val_recall = recall_score(val_targ, val_predict, average='micro')
        _val_precision = precision_score(val_targ, val_predict, average='micro')
        self.micro_val_f1s.append(_val_f1)
        self.micro_val_recalls.append(_val_recall)
        self.micro_val_precisions.append(_val_precision)
        print("\nmicro_val_f1s = \t" + "\t".join(["%.4f" % i for i in self.micro_val_f1s]))
        print("micro_val_recalls = \t" + "\t".join(["%.4f" % i for i in self.micro_val_recalls]))
        print("micro_val_precisions = \t" + "\t".join(["%.4f" % i for i in self.micro_val_precisions]))

        self.val_metrics["val_f1s"] = self.val_f1s
        self.val_metrics["val_recalls"] = self.val_recalls
        self.val_metrics["val_precisions"] = self.val_precisions
        self.val_metrics["micro_val_f1s"] = self.micro_val_f1s
        self.val_metrics["micro_val_recalls"] = self.micro_val_recalls
        self.val_metrics["micro_val_precisions"] = self.micro_val_precisions

        with open(elmo['val_metrics_file'], 'wb') as f:
            pickle.dump(self.val_metrics, f)


class Save_crt_epoch_nb(Callback):
    def __init__(self, filepath):
        super(Save_crt_epoch_nb, self).__init__()
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs=None):
        np.save(self.filepath, np.array(epoch))


class Save_keras_model(Callback):
    def __init__(self, filepath):
        super(Save_keras_model, self).__init__()
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs=None):
        tf.contrib.saved_model.save_keras_model(self.model, self.filepath)


class Save_records(Callback):
    def __init__(self):
        super(Save_records, self).__init__()
        # self.t_logs = t_logs
        # self.v_logs = v_logs

    def on_epoch_end(self, epoch, logs=None):
        print('logs = ' + str(logs))
        '''
        print('epoch %d:\tloss: %.5f\t6_dice: %.4f' %
              (epoch, logs['loss'],
               logs['dice_coef_except_background']))
        print('\t\t\tval_loss: %.5f\tval_6_dice: %.4f' %
              (logs['val_loss'],
               logs['val_dice_coef_except_background']))
        print()
        '''


elmo_model = None
if os.path.exists(elmo['hub_model_file']):
    print("load hub model from file")
    elmo_model = hub.Module(elmo['hub_model_file'])
else:
    print("load hub model from URL")
    elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        elmo_model.export(elmo['hub_model_file'], sess)


def ElmoEmbedding(x):
    return elmo_model(inputs={
        "tokens": tf.squeeze(tf.cast(x, tf.string)),
        "sequence_len": tf.constant(elmo['batch_size'] * [elmo['maxlen']])
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
        # self.elmo_net = self.get_elmo()
        model_path = config['elmo']['modelCheckpoint_file'] if USE_checkpoint_model else config['elmo']['model_h5']
        if os.path.exists(model_path):
            print(model_path + " exists")
            if USE_checkpoint_model:
                global elmo_model
                self.elmo_net = load_model(model_path, custom_objects={'elmo_model': elmo_model, 'tf': tf, 'elmo': elmo,
                                                                       'my_sparse_categorical_accuracy': my_sparse_categorical_accuracy})
                # self.elmo_net = load_model(model_path)
                self.elmo_net.summary()
                # self.elmo_net.load_weights(model_path)
            else:
                self.elmo_net = tf.contrib.saved_model.load_keras_model(elmo['model_h5'])
                self.elmo_net.summary()

            print('loading elmo model and weights from file')
            print("got elmo")
        else:
            self.elmo_net = self.get_elmo()
            self.elmo_net.summary()
            print('no elmo model file exists, creating model')

        # load have_trained_nb_epoch
        if os.path.exists(config['elmo']['have_trained_nb_epoch_file']):
            self.have_trained_nb_epoch = np.load(config['elmo']['have_trained_nb_epoch_file']) + 1
            print("loaded have_trained_nb_epoch")
        else:
            print("file not exist: " + config['elmo']['have_trained_nb_epoch_file'])
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
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=[my_sparse_categorical_accuracy])

        return model

    def train_elmo_generator(self):
        elmo_net = self.elmo_net

        save_crt_epoch_nb = Save_crt_epoch_nb(config['elmo']['have_trained_nb_epoch_file'])
        save_keras_model = Save_keras_model(config['elmo']['model_h5'])
        checkpointer = ModelCheckpoint(filepath=config['elmo']['modelCheckpoint_file'],
                                       verbose=1, save_best_only=False, save_weights_only=False)
        tensorboard = TensorBoard(log_dir=config['elmo']['tensorboard_dir'])

        val_data = [np.array(self.myData.X_val), np.array(self.myData.y_val)]
        my_metrics = My_Metrics(val_data)

        saveRecords = Save_records()

        print('Fitting model...')

        elmo_net.fit_generator(
            generator=self.generator_data_train_fine(),
            steps_per_epoch=self.myData.total_nb_batch_train,
            epochs=self.epoches,
            verbose=1,
            # callbacks=[save_crt_epoch_nb, checkpointer, tensorboard],
            callbacks=[my_metrics, save_crt_epoch_nb, checkpointer, tensorboard],
            # validation_data=self.generator_data_validate_fine(),
            # validation_steps=self.myData.total_nb_batch_validate,
            class_weight=None,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
            shuffle=False,
            initial_epoch=self.have_trained_nb_epoch
            )

    def predict_test_data(self):
        model_path = config['elmo']['modelCheckpoint_file'] if USE_checkpoint_model else config['elmo']['model_h5']
        if not os.path.exists(model_path):
            print("elmo model not trained.")

        test_data = [np.array(self.myData.X_te), np.array(self.myData.y_te)]
        y_pred = np.argmax(np.asarray(self.elmo_net.predict(test_data[0], batch_size=elmo['batch_size'], verbose=1)), -1)  # (None, 50)
        y_true = np.squeeze(test_data[1])  # (None, 50)
        X_true = test_data[0]  # (None, 50)

        index = 100
        self.print_one_sample_and_result(X_true[index], y_true[index], y_pred[index])

        return y_pred

    def print_one_sample_and_result(self, one_X, one_y_pred, one_y_true=None):
        tags = elmo['tags']
        if one_y_true is not None:
            print("{:15} {:5}: ({})".format("Word", "Pred", "True"))
            print("=" * 30)
            for w, pred, true in zip(one_X, one_y_pred, one_y_true):
                if w != "PADword":
                    print("{:15}:{:5} ({})".format(w, tags[pred], tags[true]))
        else:
            print("{:15} {:5}: ({})".format("Word", "Pred", "True"))
            print("=" * 30)
            for w, pred in zip(one_X, one_y_pred):
                if w != "PADword":
                    print("{:15}:{:5}".format(w, tags[pred]))

    def predict_one_sample(self, X):
        X = np.array(X)
        y_pred = np.argmax(np.asarray(self.elmo_net.predict(X, batch_size=elmo['batch_size'], verbose=1)), -1)  # (1, 50)

        self.print_one_sample_and_result(X[0], y_pred[0])

        return y_pred

    def predict_samples(self, X):
        X = np.array(X)
        y_pred = np.argmax(np.asarray(self.elmo_net.predict(X, batch_size=elmo['batch_size'], verbose=1)), -1)  # (None, 50)

        index = random.randint(0, X.shape[0])
        self.print_one_sample_and_result(X[index], y_pred[index])

        return y_pred

    def plot_model(self):
        model = self.get_elmo()
        plot_model(model, to_file='model.png')

    def predict(self, X):
        y_pred = np.argmax(np.asarray(self.elmo_net.predict(X, batch_size=elmo['batch_size'], verbose=1)),
                           -1)  # (None, 50)
        return y_pred


class ELMo_test():
    def __init__(self):
        self.batch_size = elmo['batch_size']
        model_path = config['elmo']['modelCheckpoint_file']
        self.elmo_net = None
        if os.path.exists(model_path):
            global elmo_model
            self.elmo_net = load_model(model_path, custom_objects={'elmo_model': elmo_model, 'tf': tf, 'elmo': elmo,
                                                                   'my_sparse_categorical_accuracy': my_sparse_categorical_accuracy})
            print("load keras model ELMo success")
            self.elmo_net.summary()
        else:
            print("not exist model file")

    def predict(self, X):
        y_pred = np.argmax(np.asarray(self.elmo_net.predict(X, batch_size=elmo['batch_size'], verbose=1)),
                           -1)  # (None, 50)
        return y_pred

import math
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')


def predict_one_sample(elmo_model_test, raw_x_str):
    x_tokens = [word_tokenize(raw_x_str)]
    words_num = len(x_tokens[0])

    x_processed = process_input(x_tokens)
    y_pred = elmo_model_test.predict(x_processed)

    # print_one_sample_and_result(x_processed[0][:words_num], y_pred[0][:words_num])
    new_y_pred = [y_pred[0][:words_num]]
    return x_tokens, new_y_pred


def predict_samples(elmo_model_test, raw_samples_str):
    # raw_samples_str: ["", "", ""]
    samples_token = [word_tokenize(raw_x_str) for raw_x_str in raw_samples_str]  # [['', '', ''],[],[]]
    words_num = [len(sample_token) for sample_token in samples_token]
    samples_num = len(raw_samples_str)

    x_processed = process_input(samples_token)
    y_pred = elmo_model_test.predict(x_processed)

    new_y_pred = [y_pred[i][:words_num[i]] for i in range(samples_num)]

    return samples_token, new_y_pred


def print_one_sample_and_result(one_X, one_y_pred, one_y_true=None):
    tags = elmo['tags']
    if one_y_true is not None:
        print("{:15} {:5}: ({})".format("Word", "Pred", "True"))
        print("=" * 30)
        for w, pred, true in zip(one_X, one_y_pred, one_y_true):
            if w != "PADword":
                print("{:15}:{:5} ({})".format(w, tags[pred], tags[true]))
    else:
        print("{:15} {:5}: ({})".format("Word", "Pred", "True"))
        print("=" * 30)
        for w, pred in zip(one_X, one_y_pred):
            if w != "PADword":
                print("{:15}:{:5}".format(w, tags[pred]))


def process_input(X):
    # X: [['', '', ''], [], []]
    num_samples = len(X)
    expected_num_samples = math.ceil(num_samples / elmo['batch_size']) * elmo['batch_size']

    new_X = []
    for i in range(expected_num_samples):
        if i < num_samples:
            seq = X[i]
            new_seq = []
            for j in range(elmo['maxlen']):
                try:
                    new_seq.append(seq[j])
                except:
                    new_seq.append(elmo['pad_wording'])
            new_X.append(np.array(new_seq))
        else:
            new_X.append(elmo['prepared_X'])
    new_X = np.array(new_X)
    return new_X


if __name__ == '__main__':
    # '''
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    if MODE == "TRAIN":
        my_elmo_model = ELMo()
        my_elmo_model.train_elmo_generator()
    else:
        my_elmo_mode_test = ELMo_test()
        raw_str = "I am Barton Xu, he is Aspire Tan, and we live in Nanjing Jiangsu."
        x_tokens, y_pred = predict_one_sample(my_elmo_mode_test, raw_str)
        print_one_sample_and_result(x_tokens[0], y_pred[0])

        x_tokens, y_pred = predict_samples(my_elmo_mode_test, [raw_str])
        print_one_sample_and_result(x_tokens[0], y_pred[0])
