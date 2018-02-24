import os
import sys
from time import time
import argparse
from abc import ABC, abstractmethod, ABCMeta
import numpy as np
import scipy
import pandas as pd
from scipy.stats import chisquare
from scipy.stats import mannwhitneyu
from PyQt5.QtCore import (QCoreApplication)
from PyQt5.QtSql import QSqlDatabase, QSqlQuery, QSqlTableModel
from PyQt5.QtCore import QElapsedTimer
import sqlalchemy
from keras.models import Sequential, load_model, model_from_json, model_from_yaml, Model
from keras.layers import Concatenate, Input, Dense, Add
from keras.layers.core import Dense, Dropout, Activation
from keras.utils.vis_utils import plot_model
from keras.callbacks import TensorBoard
from keras.layers.normalization import BatchNormalization
from sklearn import preprocessing
import keras


class Feature:
    def __init__(self, name: str, significance: float = None, resulting: bool = False) -> object:
        self._name = name
        self._significance = significance
        self._resulting = resulting
        self._type = None
        self._sensitivity = 0.0

    def get_name(self):
        return self._name

    def set_name(self, name):
        self._name = name

    def set_significance(self, significance):
        self._significance = significance

    def get_significance(self):
        return self._significance

    def get_pvalue(self):
        return self._significance.pvalue

    def get_statistic(self):
        return self._significance.statistic

    def is_valuable(self):
        return self._significance.pvalue < 0.05

    def is_resulting(self):
        return self._resulting

    def resulting(self):
        self._resulting = True

    def not_resulting(self):
        self._resulting = False

    def get_type(self):
        return self._type

    def get_sensitivity(self):
        return self._sensitivity


class ContinuousFeature(Feature):
    def __init__(self, name: str, significance: float = None, resulting: bool = False) -> object:
        super(ContinuousFeature, self).__init__(name, significance, resulting)
        self._type = "Continuous"


class CategoricalFeature(Feature):
    def __init__(self, name, significance=None, categories=None, resulting=False):
        super(CategoricalFeature, self).__init__(name, significance, resulting)
        self.__categories = categories
        self._type = "Categorical"

    def get_categories(self):
        return self.__categories

    def set_categories(self, categories):
        self.__categories = categories

    def add_categories(self, categories):
        self.__categories.append(categories)

    def drop_categories(self, categories):
        self.__categories = list(filter(lambda cat: cat not in categories, self.__categories))


class InputHandler(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def feature_classes(self):
        """Read classes for represented features"""
        pass

    @abstractmethod
    def data(self):
        """Retrieve data from the input source and return an object."""
        pass


class DataFileHandler(InputHandler):
    def __init__(self, file_address: str, delimiter: str, header_line: int, feature_classes_line: int, na_values: list) -> object:
        self.__file = open(file_address)
        self.__delimiter = delimiter
        self.__header_line = header_line
        self.__feature_classes_line = feature_classes_line
        self.__na_values = na_values

    @property
    def feature_classes(self) -> list:
        self.__file.seek(0)
        for i in range(self.__feature_classes_line):
            self.__file.readline()
        return self.__file.readline().strip().split(self.__delimiter)

    @property
    def data(self) -> pd.DataFrame:
        self.__file.seek(0)
        return pd.read_csv(self.__file, delimiter=self.__delimiter, header=self.__header_line, na_values = self.__na_values)

    @property
    def file(self):
        return self.__file

    @file.setter
    def open_file(self, file_address):
        self.__file = open(file_address)


class QtSqlDBHandler(InputHandler):
    def __init__(self, dbname: str, username: str, password: str, port, table_name, hostname: str = "127.0.0.1"):
        self.__db = QSqlDatabase.addDatabase("QPSQL")
        self.__table_name = table_name
        self.__hostname = hostname
        self.__dbname = dbname
        self.__port = port
        self.__username = username
        self.__password = password
        self.__status = False

    def configure(self):
        self.__db.setHostName(self.__hostname)
        self.__db.setDatabaseName(self.__dbname)
        self.__db.setUserName(self.__username)
        self.__db.setPassword(self.__password)
        self.__db.setPort(self.__port)

    def open(self):
        self.__status = self.__db.open()
        if not self.__status:
            print(self.__db.lastError().text())

    def close(self):
        self.__db.close()
        self.__status = False

    def data(self, *features) -> pd.DataFrame:
        request = "SELECT * FROM " + self.__table_name + " ;"
        query = QSqlQuery()
        if self.__status:
            query.setForwardOnly(True)
            query.exec(request)
            d = {el: [] for el in features}
            query.seek(0)
            while query.next():
                for i in features:
                    d[i].append(query.value(i))
            return pd.DataFrame(data=d)

    @property
    def future_classes(self):
        pass


class SqlAlchemyDBHandler(InputHandler):
    def __init__(self, db_dialect, dbname: str, username: str, password: str, port, table_name, hostname: str = "127.0.0.1"):
        self.__db = None
        self.__db_dialect = db_dialect
        self.__table_name = table_name
        self.__hostname = hostname
        self.__dbname = dbname
        self.__port = port
        self.__username = username
        self.__password = password
        self.__connection = None

    def configure(self):
        self.__db = sqlalchemy.create_engine("{}://{}:{}@{}:{}/{}".format(self.__db_dialect, self.__username, self.__password, self.__hostname, self.__port, self.__dbname))

    def open(self):
        self.__connection = self.__db.connect()

    def close(self):
        if self.__connection is not None:
            self.__connection.close()

    def data(self, *features) -> pd.DataFrame:
        if not features:
            return pd.read_sql_table(self.__table_name, self.__db)
        else:
            features = ",".join(features)
            return pd.read_sql_query("SELECT {} FROM {};".format(features, self.__table_name), self.__connection)

    @property
    def future_classes(self):
        pass


class NeuralNetwork:
    def __init__(self, model):
        self.__model = model

    def get_model(self):
        return self.__model

    @classmethod
    def from_scratch(cls, input_dim: int, hidden_units: int, kernel_initializer: str="uniform"):
        model = NeuralNetwork.build(input_dim, hidden_units, kernel_initializer)
        return cls(model)

    @classmethod
    def from_file(cls, from_file: str):
        model = load_model(from_file)
        return cls(model)

    @classmethod
    def merged(cls, *models):
        model = Concatenate(models,axis=1)
        return cls(model)

    @staticmethod
    def build(input_dim: int, hidden_units: int, kernel_initializer: str="uniform" ):
        output_units = 1
        activation = 'relu'
        output_activation = 'sigmoid'
        model = Sequential()
        model.add(Dense(hidden_units, input_dim=input_dim, kernel_initializer=kernel_initializer))
        model.add(Activation(activation))
        model.add(Dropout(0.2))
        model.add(Dense(hidden_units, input_dim=input_dim, kernel_initializer=kernel_initializer))
        model.add(Activation(activation))
        # model.add(Dropout(0.2))
        model.add(Dense(output_units))
        model.add(Activation(output_activation))
        return model

    def save_plot(self, to_file='model_plot.png', shapes=True, layer_names=True):
        plot_model(self.__model, to_file=to_file, show_shapes=shapes, show_layer_names=layer_names)

    def compile(self, loss='binary_crossentropy', optimizer='adam'):
        self.__model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    def to_h5(self, to_file='my_model.h5'):
        self.__model.save(to_file)

    def to_json(self, to_file='my_model.json'):
        model_json = self.__model.to_json()
        with(to_file, 'w') as json_file:
            json_file.write(model_json)

    def to_yaml(self, to_file='my_model.yaml'):
        model_yaml = self.__model.to_yaml()
        with(to_file, 'w') as yaml_file:
            yaml_file.write(model_yaml)


class PreprocessedData:
    def __init__(self, dataset, start_position=None, stop_position=None, without_resulting_feature=False, resulting_feature=None):
        self.__dataset = dataset.get_data(start_position, stop_position, without_resulting_feature)
        self.__features = dataset.get_features()
        self.__resulting_feature = resulting_feature
        self.__drop_invalid_data()

    def bucketize(self, column_name, categories_num, labels):
        col = self.__dataset[column_name]
        self.__dataset.drop(column_name, inplace=True, axis=1)
        bucketized_col = pd.cut(col, categories_num, labels=labels)
        self.__dataset = self.__dataset.join(bucketized_col)

    def __drop_invalid_data(self):
        self.__dataset = self.__dataset.dropna(axis=0, how='any')

    def combine_classes(self, feature_name, from_classes, to_class):
        if feature_name in self.__features:
            self.__dataset[feature_name].cat.remove_categories(from_classes, inplace=True)
            self.__dataset[feature_name].fillna(value=to_class, inplace=True)

    def normalize(self):
        cont_features = self.__dataset.select_dtypes(exclude='category').columns.tolist()
        normalized_data = preprocessing.normalize(self.__dataset[cont_features])
        self.__dataset[cont_features] = normalized_data
        # continuous_features = self.__dataset.select_dtypes(exclude='category')
        # normalized_cont_f = (continuous_features-continuous_features.mean())/continuous_features.std()
        # self.__dataset.update(normalized_cont_f)

    def one_hot_encode(self):
        categorical_features = self.__dataset.select_dtypes(include='category')
        if self.__resulting_feature:
            categorical_features = categorical_features.drop(columns=self.__resulting_feature)
        features_to_remove = categorical_features.columns.tolist()
        self.__dataset.drop(features_to_remove, inplace=True, axis=1)
        normalized_cat_f = pd.get_dummies(categorical_features, columns=categorical_features.columns.tolist())
        self.__dataset = self.__dataset.join(normalized_cat_f)

    def get_dataset(self):
        return self.__dataset

    def get_columns(self):
        return self.__dataset.columns.tolist()

    def add_noise(self, column: str, rate: float):
        copy = self.__dataset.copy()
        copy[column] *= (1 + rate)
        return copy


class NNTrainer:
    def __init__(self, nnet: NeuralNetwork, training_data, target_data, batch_size=16, epochs=1000):
        self.__nnet = nnet
        self.__training_data = training_data
        self.__target_data = target_data
        self.__batch_size = batch_size
        self.__epochs = epochs
        self.__score = None

    def train(self, verbose=1):
        tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
        self.__nnet.get_model().fit(self.__training_data, self.__target_data, batch_size=self.__batch_size,
                                    epochs=self.__epochs, verbose=verbose, callbacks=[tensorboard])

    def evaluate(self, verbose=1):
        self.__score = self.__nnet.get_model().evaluate(self.__training_data, self.__target_data,
                                                        batch_size=self.__batch_size, verbose=verbose)

    def get_score(self):
        return self.__score


class NNPredictor:
    def __init__(self, nnet: NeuralNetwork, dataset):
        self.__nnet = nnet
        self.__dataset = dataset
        self.__score = {}
        self.__prediction = []

    def predict(self):
        self.__prediction = self.__nnet.get_model().predict(self.__dataset).flatten()

    def evaluate(self, real_values):
        if len(self.__prediction) > 0:
            compared = np.equal(np.round(self.__prediction), real_values)
            self.__score['correct'] = np.count_nonzero(compared)
            self.__score['total'] = len(self.__dataset)
            self.__score['wrong'] = self.__score['total'] - self.__score['correct']
            self.__score['score'] = self.__score['correct'] / self.__score['total']

    def get_score(self):
        return self.__score

    def get_prediction(self):
        return self.__prediction


class DataSet:
    def __init__(self, res_feature: str, high_risk: list, low_risk: list, input_handler: InputHandler):
        self.__input_handler = input_handler
        self.__resulting_feature = res_feature
        self.__low_risk = low_risk
        self.__high_risk = high_risk
        self.__data = None
        self.__feature_classes = None
        self.__features = None

    def read_data(self):
        self.__data = self.__input_handler.data
        self.__feature_classes = self.__input_handler.feature_classes
        self.__features = self.__create_feature_objects()
        self.__set_valuable_features()
        self.__set_resulting_feature()

    def get_feature(self, name):
        if name in self.__features:
            return self.__features[name]
        else:
            return None

    def shuffle(self):
        self.__data = self.__data.sample(frac=1).reset_index(drop=True)

    def drop_columns(self, columns):
        self.__data.drop(columns, inplace=True, axis=1)

    def __create_feature_objects(self):
        feature_names = self.__data.columns.values.tolist()
        feature_classes = self.__feature_classes
        features_list = {}
        for idx, f in enumerate(feature_names):
            if feature_classes[idx] == "cont":
                features_list[f] = ContinuousFeature(name=f)
            elif feature_classes[idx] == "cat":
                self.__data[f] = self.__data[f].astype('category')
                features_list[f] = CategoricalFeature(name=f)
        return features_list

    def get_data(self, start=None, stop=None, without_resulting_feature=False):
        if without_resulting_feature:
            return self.__data.ix[start:stop].drop(columns=self.__resulting_feature)
        return self.__data.ix[start:stop]

    def get_features(self):
        return self.__features

    def get_feature_classes(self):
        return self.__feature_classes

    def get_categorical_features(self):
        categorical_features = []
        for name, feature in self.__features.items():
            if feature.get_type=="Categorical":
                categorical_features.append(feature)
        return categorical_features

    @property
    def low_risk_groups(self):
        return self.__data.loc[self.__data['num'].isin(self.__low_risk)]

    @property
    def high_risk_groups(self):
        return self.__data.loc[self.__data['num'].isin(self.__high_risk)]

    def __set_valuable_features(self):
        harm = self.high_risk_groups
        no_harm = self.low_risk_groups
        for name, feature in self.__features.items():
            feature_name = feature.get_name()
            feature_type = feature.get_type()
            result = None
            if feature_type == "Continuous":
                result = mannwhitneyu(no_harm[feature_name], harm[feature_name], alternative='two-sided')
            elif feature_type == "Categorical":
                cp_harm_categories = harm[feature_name].value_counts(sort=False).tolist()
                cp_no_harm_categories = no_harm[feature_name].value_counts(sort=False).tolist()
                result = chisquare(cp_harm_categories, cp_no_harm_categories)
            feature.set_significance(result)

    def __set_resulting_feature(self):
        self.get_feature(self.__resulting_feature).resulting()

    def get_resulting_feature(self):
        for name, feature in self.__features.items():
            if feature.is_resulting():
                return feature
        return None

    def drop_invalid_data(self):
        self.__data = self.__data.dropna(axis=0, how='any')

    def combine_classes(self, feature_name, from_classes, to_class):
        if feature_name in self.__features:
            column = self.__data[feature_name]
            column[column not in from_classes] = to_class


if __name__ == '__main__':
    app = QCoreApplication(sys.argv)

    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    # parser.add_argument('model')
    args = parser.parse_args()

    def alchemy():
        alchemy = SqlAlchemyDBHandler("postgresql", "postgres", "nurlan", "your_password", 5432, "dataset")
        alchemy.configure()
        alchemy.open()
        t = QElapsedTimer()
        t.start()
        ddd = alchemy.data()
        print(t.elapsed())
        alchemy.close()
        print(ddd)

    def qtsql():
        qtdb = QtSqlDBHandler("postgres", "nurlan", "your_password", 5432, "dataset")
        qtdb.configure()
        qtdb.open()
        t1 = QElapsedTimer()
        t1.start()
        ddd = qtdb.data("age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num")
        print(t1.elapsed())
        qtdb.close()
        print(ddd)

    def files():
        ihandler = DataFileHandler(args.input, ',', 1, 0, ['?'])
        sample = DataSet("num", [1, 2, 3], [0], ihandler)
        sample.read_data()
        print(sample.get_feature("num").is_resulting())

    if args.input:
        ihandler = DataFileHandler(args.input, ',', 1, 0, ['?'])
        dataset = DataSet("num", [1, 2, 3], [0], ihandler)
        dataset.read_data()

        preprocessed_data = PreprocessedData(dataset, stop_position=241, resulting_feature='num')
        preprocessed_data.combine_classes(feature_name='num', from_classes=[2,3,4], to_class=1)
        preprocessed_data.bucketize('age', 10, list(range(1,11)))
        preprocessed_data.one_hot_encode()

        training_data = preprocessed_data.get_dataset()
        training_data = training_data.loc[:, training_data.columns != 'num'].values
        training_target = preprocessed_data.get_dataset()['num'].values

        test_data = PreprocessedData(dataset, start_position=242, without_resulting_feature=True)
        test_data.bucketize('age', 10, list(range(1,11)))
        test_data.one_hot_encode()
        test_data = test_data.get_dataset().values

        test_target = dataset.get_data(start=242).dropna(axis=0, how='any')['num']
        test_target.cat.remove_categories([2,3,4], inplace=True)
        test_target.fillna(value=1, inplace=True)
        test_target = test_target.values

        network = NeuralNetwork.from_scratch(training_data.shape[1], training_data.shape[1])
        network.save_plot('model_plot.png')
        network.compile(loss='binary_crossentropy', optimizer='adam')

        trainer = NNTrainer(network, training_data, training_target, epochs=100)
        trainer.train(verbose=0)
        trainer.evaluate()

        network.to_h5('my_model.h5')

        predictor = NNPredictor(network, test_data)
        predictor.predict()
        predictor.evaluate(test_target)
        print(predictor.get_score())

        predictionX = predictor.get_prediction()
        # print(predictionX)

        columns = preprocessed_data.get_columns()
        for column in columns:
            if column == 'num':
                continue
            print(column)
            training_data = preprocessed_data.add_noise(column, 0.01)
            training_data = training_data.loc[:, training_data.columns != 'num'].values
            training_target = preprocessed_data.get_dataset()['num'].values

            network = NeuralNetwork.from_scratch(training_data.shape[1], training_data.shape[1])
            network.compile(loss='binary_crossentropy', optimizer='adam')

            trainer = NNTrainer(network, training_data, training_target, epochs=100)
            trainer.train(verbose=0)
            trainer.evaluate()

            predictor = NNPredictor(network, test_data)
            predictor.predict()
            predictor.evaluate(test_target)
            print(predictor.get_score())

            noisy_prediction = predictor.get_prediction()
            # print(noisy_prediction)
            print(abs(np.sum(noisy_prediction)-np.sum(predictionX))/predictionX.size)

        ###
        # for k, v in dataset.get_features().items():
        #     print(v.is_valuable())
        # print(dataset.get_data())
        # dataset.drop_columns('trestbps')
        ###

        ###
        # input1 = Input(shape=(5,))
        # dense1 = Dense(5, activation='relu')(input1)
        # out1 = Dense(1, activation='sigmoid')(dense1)
        # model1 = Model(inputs=input1, outputs=out1)
        #
        # input2 = Input(shape=(23,))
        # dense2 = Dense(23, activation='relu')(input2)
        # out2 = Dense(1, activation='sigmoid')(dense2)
        # model2 = Model(inputs=input2, outputs=out2)
        #
        # merged = Add()([model1.output, model2.output])
        # merged = Dense(2, activation='relu')(merged)
        # merged = Dense(1, activation='sigmoid')(merged)
        # model = Model([model1.input, model2.input], merged)
        ###
