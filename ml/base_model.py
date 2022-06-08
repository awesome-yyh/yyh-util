# -*- coding: utf-8 -*-
import datetime
import numpy as np
from sklearn import metrics


class BaseModel:
    def __init__(self):
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        self.model = None
        self.pred = np.array([])
        # self.k_train = []
        # self.k_test = []
        self.ratio = 1
        self.num_boost_round = 1000

    def feed(self, data_train=None, data_test=None):
        if data_test is None:
            data_test = []
        assert len(data_train) == 2
        self.x_train = data_train[0]
        self.y_train = data_train[1]
        assert len(data_test) == 2
        self.x_test = data_test[0]
        self.y_test = data_test[1]

    def eval(self, data_type='test', dump=False):
        x_eval = self.x_test
        y_eval = self.y_test
        if data_type == 'train':
            x_eval = self.x_train
            y_eval = self.y_train
        self.predict(x_eval)
        pred_prob = self.pred
        # print(pred_prob)
        pred_label = np.argmax(pred_prob, axis=1)
        acc = metrics.accuracy_score(y_eval, pred_label)
        fscore = metrics.f1_score(y_eval, pred_label, average='macro')
        recall = metrics.recall_score(y_eval, pred_label, average='macro')
        precision = metrics.precision_score(y_eval, pred_label, average='macro')
        # with open(path, 'a+') as fo:
        print('======== %s ========' % datetime.datetime.now())
        print('data_type: %s' % data_type)
        print('acc: %.3f' % acc)
        print('fscore: %.3f' % fscore)
        print('recall: %.3f' % recall)
        print('precision: %.3f' % precision)


    def train(self):
        pass

    def predict(self, x_pred):
        pass

    def load(self, path):
        pass

    def save(self, path, features):
        pass

    def show(self, path):
        pass
