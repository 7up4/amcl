import tensorflow as tf
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np

from sklearn.metrics import roc_curve, auc
from keras.layers import Concatenate, Input, Embedding, Lambda
from keras.layers.core import Dense, Dropout, Reshape
from keras.models import load_model, model_from_json, model_from_yaml, Model
from keras.utils.vis_utils import plot_model

from .datasets import DataSet
from .importing_modules import *


class NeuralNetworkConfig:
    def __init__(self, categorical_input: str="cat_input", continuous_input: str="cont_input", output: str="output",
                 reshaped_output: str="reshaped_output", noisy_layer: str="noisy", kernel_initializer: str="uniform",
                 hidden: str = "hidden", reshaped: str="reshaped", dropout: str="dropout", merge: str="merge",
                 activation: str="relu", output_activation: str="sigmoid"):
        self.kernel_initializer = kernel_initializer
        self.activation = activation
        self.output_activation = output_activation
        self.cont_input = continuous_input
        self.cat_input = categorical_input
        self.hidden = hidden
        self.noisy_layer = noisy_layer
        self.reshaped = reshaped
        self.merge = merge
        self.dropout = dropout
        self.output = output
        self.reshaped_output = reshaped_output


class NeuralNetwork:
    def __init__(self, model):
        self.__model = model

    def get_model(self):
        return self.__model

    @classmethod
    def from_file(cls, from_file: str):
        model = load_model(from_file)
        return cls(model)

    def get_layer(self, name):
        return self.__model.get_layer(name)

    def get_weights(self):
        return self.__model.get_weights()

    def get_weights_for_feature(self, feature):
        res = self.__model.get_layer(feature).get_weights()
        return res

    def get_weights_with_name(self):
        model = self.__model
        names = [layer.name for layer in model.layers]
        weights = []
        for name in names:
            weights.append(model.get_layer(name).get_weights())
        return dict(zip(names, weights))

    def set_weights_by_name(self, weigths):
        for name, weigth in weigths.items():
            self.__model.get_layer(name).set_weights(weigth)

    def save_plot(self, to_file='model_plot.svg', shapes=False, layer_names=False):
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


class FullyConnectedNeuralNetwork(NeuralNetwork):
    @classmethod
    def from_scratch(cls, config: NeuralNetworkConfig, dataset, hidden_units: int,
                     embedding_size: int = 10, dropout_rate: float = 0.2,
                     output_units=1, embedding_layers_trainable=True):
        categorical_data = dataset.get_data(without_resulting_feature=True).select_dtypes(include='category')
        continuous_features = dataset.get_data(without_resulting_feature=True).select_dtypes(
            exclude='category').columns.size

        if isinstance(categorical_data, pd.DataFrame):
            categorical_data_categories = {}
            for column in categorical_data:
                categorical_data_categories[column] = categorical_data[column].cat.categories.size
            categorical_data = categorical_data_categories

        model = FullyConnectedNeuralNetwork._build(config, categorical_data, continuous_features, hidden_units, embedding_size,
                                     dropout_rate, output_units, embedding_layers_trainable)
        return cls(model)


    @staticmethod
    def _build(config, categorical_data_categories, continuous_features: int, hidden_units: int, embedding_size: int,
               dropout_rate, output_units: int, embedding_layers_trainable):

        # create input layer for continuous data
        continuous_input = Input(shape=(continuous_features,), name=config.cont_input)
        reshaped_continuous_input = Reshape((1, continuous_features),
                                            name=config.reshaped)(continuous_input)

        # create input layers complemented by embedding layers to handle categorical features
        embedding_layers = []
        categorical_inputs = []
        for feature, size in categorical_data_categories.items():
            categorical_input = Input((1,), name=config.cat_input + feature)
            categorical_inputs.append(categorical_input)
            embedding_layer = Embedding(size + 1, embedding_size, name=feature, trainable=embedding_layers_trainable)(
                categorical_input)
            embedding_layers.append(embedding_layer)

        # merge all inputs
        merge_layer = Concatenate(name=config.merge)(embedding_layers + [reshaped_continuous_input])

        # hidden layers
        hidden_layer1 = Dense(hidden_units, activation=config.activation, kernel_initializer=config.kernel_initializer,
                              name=config.hidden + "1")(merge_layer)
        dropout_layer1 = Dropout(dropout_rate, name=config.dropout + "1")(hidden_layer1)

        # output_layer
        output_layer = Dense(output_units, activation=config.output_activation, name=config.output)(dropout_layer1)

        # add reshape layer since output should be vector
        output_layer = Reshape((1,), name=config.reshaped_output)(output_layer)

        # create final model
        model = Model(inputs=categorical_inputs + [continuous_input], outputs=output_layer)
        return model


class OptimizedNeuralNetwork(NeuralNetwork):
    @classmethod
    def from_scratch(cls, config: NeuralNetworkConfig, dataset: DataSet, correlation_info: list, embedding_size: int=10,
                     dropout_rate: float=0.2, output_units=1):
        flatten_correlation = [item for sublist in correlation_info for item in sublist]
        features = dataset.get_data(without_resulting_feature=True).columns
        diff = list(set(features) - set(flatten_correlation))
        diff = [[item] for item in diff]
        correlation_info.extend(diff)
        categorical_data = dataset.get_data(without_resulting_feature=True).select_dtypes(include='category')
        continuous_features = dataset.get_data(without_resulting_feature=True).select_dtypes(exclude='category').columns

        if isinstance(categorical_data, pd.DataFrame):
            categorical_data_categories = {}
            for column in categorical_data:
                categorical_data_categories[column] = categorical_data[column].cat.categories.size
            categorical_data = categorical_data_categories

        model = OptimizedNeuralNetwork._build(config, categorical_data, continuous_features, correlation_info,
                                              embedding_size, dropout_rate, output_units)
        return cls(model)

    @staticmethod
    def _build(config: NeuralNetworkConfig, categorical_data_categories: dict, continuous_features: list,
               correlation_info: list,embedding_size: int, dropout_rate, output_units: int):
        feature_layers = {}
        hidden_layers = []
        inputs = []
        for feature, size in categorical_data_categories.items():
            categorical_input = Input((1,), name=config.cat_input + feature)
            inputs.append(categorical_input)
            embedding_layer = Embedding(size + 1, embedding_size, name=feature)(categorical_input)
            feature_layers[feature] = embedding_layer
        for feature in continuous_features:
            continuous_input = Input((1,), name=config.cont_input + feature)
            inputs.append(continuous_input)
            reshaped_continuous_input = Reshape((1, 1), name=feature)(continuous_input)
            feature_layers[feature] = reshaped_continuous_input
        for couple in correlation_info:
            coupled_layers = [feature_layers[feature] for feature in couple]
            if len(couple) > 1:
                merge_layer = Concatenate()(coupled_layers)
                hidden_layer = Dense(1, activation=config.activation, kernel_initializer=config.kernel_initializer)(merge_layer)
            else:
                hidden_layer = Dense(1, activation=config.activation, kernel_initializer=config.kernel_initializer)(coupled_layers[0])
            hidden_layers.append(hidden_layer)
        merge_layer = Concatenate()(hidden_layers)
        dropout_layer = Dropout(dropout_rate, name=config.dropout)(merge_layer)
        # output_layer
        output_layer = Dense(1, activation=config.output_activation, name=config.output)(dropout_layer)
        # add reshape layer since output should be vector
        output_layer = Reshape((output_units,), name=config.reshaped_output)(output_layer)
        # create final model
        model = Model(inputs=inputs, outputs=output_layer)
        return model


class Trainer:
    def __init__(self, nnet: NeuralNetwork, training_dataset, training_target, batch_size=16, epochs=1000):
        self.__nnet = nnet
        self.__training_dataset = training_dataset
        self.__training_target = training_target
        self.__batch_size = batch_size
        self.__epochs = epochs
        self.__score = None
        self._preprocess_dataset()

    def _preprocess_dataset(self):
        categorical_data = DataSet.dataframe_to_series(self.__training_dataset.get_data(without_resulting_feature=True).select_dtypes(include='category'))
        if isinstance(self.__nnet, OptimizedNeuralNetwork):
            continuous_data = DataSet.dataframe_to_series(self.__training_dataset.get_data(without_resulting_feature=True).select_dtypes(exclude='category'))
            self.__training_dataset = [*categorical_data, *continuous_data]
        else:
            continuous_data = self.__training_dataset.get_data().select_dtypes(exclude='category').values
            self.__training_dataset = [*categorical_data, continuous_data]

    def train(self, verbose=1):
        # tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
        self.__nnet.get_model().fit(self.__training_dataset, self.__training_target, batch_size=self.__batch_size,
                                    epochs=self.__epochs, verbose=verbose)

    def evaluate(self, verbose=1):
        self.__score = self.__nnet.get_model().evaluate(self.__training_dataset, self.__training_target,
                                                        batch_size=self.__batch_size, verbose=verbose)

    def get_score(self):
        return self.__score


class Predictor:
    def __init__(self, nnet: NeuralNetwork, dataset: DataSet):
        self._nnet = nnet
        self._dataset = dataset
        self._score = {}
        self._prediction = []
        self._preprocess()

    def _preprocess(self):
        categorical_data = DataSet.dataframe_to_series(self._dataset.get_data().select_dtypes(include='category'))
        if isinstance(self._nnet, OptimizedNeuralNetwork):
            continuous_data = DataSet.dataframe_to_series(self._dataset.get_data().select_dtypes(exclude='category'))
            self._dataset = [*categorical_data, *continuous_data]
        else:
            continuous_data = self._dataset.get_data().select_dtypes(exclude='category').values
            self._dataset = [*categorical_data, continuous_data]

    def predict(self):
        self._prediction = self._nnet.get_model().predict(self._dataset).flatten()
        return self._prediction

    def evaluate(self, real_values):
        if len(self._prediction) > 0:
            rounded_pred = np.round(self._prediction)
            tp = np.sum(np.logical_and(rounded_pred == 1, real_values == 1))
            tn = np.sum(np.logical_and(rounded_pred == 0, real_values == 0))
            fp = np.sum(np.logical_and(rounded_pred == 1, real_values == 0))
            fn = np.sum(np.logical_and(rounded_pred == 0, real_values == 1))
            accuracy = (tp + tn) / (tp + fp + fn + tn)
            self._score['ppv'] = tp / (tp + fp)
            self._score['npv'] = tn / (tn + fn)
            self._score['recall'] = tp / (tp + fn)
            self._score['specificity'] = tn / (tn + fp)
            self._score['accuracy'] = accuracy
            self._score['tp'] = tp
            self._score['tn'] = tn
            self._score['fp'] = fp
            self._score['fn'] = fn
            self._roc(real_values, np.unique(real_values).size)

    def _roc(self, real_values, n_classes):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(real_values, self._prediction)
            roc_auc[i] = auc(fpr[i], tpr[i])
        plt.figure()
        lw = 1
        plt.plot(fpr[1], tpr[1], color='darkorange',
                 lw=lw, label='AUC = %0.2f' % roc_auc[1])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('Ложно-положительные решения')
        plt.ylabel('Истино-положительные решения')
        plt.title('Кривая ошибок')
        plt.legend(loc="lower right")
        # plt.show()

    def get_score(self):
        return self._score

    def get_prediction(self):
        return self._prediction


class FeatureSelector:
    def __init__(self, config: NeuralNetworkConfig, nnet: FullyConnectedNeuralNetwork, training_dataset):
        self._source_model = nnet
        self._config = config
        self._training_dataset = training_dataset
        categorical_columns = training_dataset.get_data(without_resulting_feature=True).select_dtypes(include='category').columns
        self._weights = self._source_model.get_weights_with_name()
        self._cat_input_shape = self._source_model.get_layer(config.cat_input+categorical_columns[0]).get_input_shape_at(0)
        self._cont_input_shape = self._source_model.get_layer(config.cont_input).get_input_shape_at(0)[-1]
        self._hid_size = self._source_model.get_layer(config.hidden+"1").get_output_shape_at(0)[-1]
        self._emb_size = self._source_model.get_layer(categorical_columns[0]).get_output_shape_at(0)[-1]
        self._dropout_rate = self._source_model.get_layer(config.dropout+"1").get_config()['rate']
        self._cat_data = {}
        for x in categorical_columns:
            self._cat_data[x] = self._source_model.get_layer(x).get_config()["input_dim"] - 1

    def _build_network(self, config, dataset):
        noisy_model = FullyConnectedNeuralNetwork.from_scratch(config=config, dataset=dataset,
                                                 hidden_units=self._hid_size, embedding_size=self._emb_size,
                                                 dropout_rate=self._dropout_rate)
        # noisy_model.save_plot("noisy_model_plot.png", shapes=True, layer_names=True)
        noisy_model.set_weights_by_name(self._weights)
        return noisy_model

    def run(self, training_dataset, training_target, test_dataset, test_target, noise_rate=0.01, training_epochs=100):
        training_dataset = DataSet.copy(training_dataset)
        test_dataset = DataSet.copy(test_dataset)
        predictor = Predictor(self._source_model, test_dataset)
        prediction = predictor.predict()
        predictor.evaluate(test_target)
        prev_accuracy = predictor.get_score()['accuracy']
        curr_accuracy = prev_accuracy
        features_to_remove = []
        noise_rate = random.uniform(0, noise_rate)
        while curr_accuracy >= prev_accuracy:
            for column in training_dataset.get_data().columns:
                if test_dataset.get_data()[column].dtype.name == 'category':
                    noisy_dataset = DataSet.copy(test_dataset)
                    noisy_dataset.add_noise_to_categorical_columns(column, noise_rate)
                    noisy_model = self._source_model
                    predictor = Predictor(noisy_model, noisy_dataset)
                else:
                    noisy_dataset = DataSet.copy(test_dataset)
                    noisy_dataset.add_noise_to_column(column, noise_rate)
                    noisy_model = self._source_model
                    predictor = Predictor(noisy_model, noisy_dataset)
                noisy_prediction = predictor.predict()
                sensitivity = abs(np.sum(noisy_prediction) - np.sum(prediction))
                test_dataset.get_features().set_sensitivity(column, sensitivity)
                training_dataset.get_features().set_sensitivity(column, sensitivity)
                print("Sensitivity of %s: %f" % (column, training_dataset.get_features().get_sensitivity(column)))
            less_sensitive_feature = test_dataset.get_features().get_less_sensitive_feature()
            features_to_remove.append(less_sensitive_feature)
            test_dataset.rm_less_sensitive()
            training_dataset.rm_less_sensitive()
            self._source_model = FullyConnectedNeuralNetwork.from_scratch(self._config, training_dataset, embedding_size=self._emb_size,
                                                 hidden_units=self._hid_size, dropout_rate=self._dropout_rate)
            self._source_model.compile()
            trainer = Trainer(self._source_model, training_dataset, training_target, epochs=training_epochs)
            trainer.train()
            trainer.evaluate()
            self._weights = self._source_model.get_weights_with_name()
            predictor = Predictor(self._source_model, test_dataset)
            prediction = predictor.predict()
            predictor.evaluate(test_target)
            prev_accuracy, curr_accuracy = curr_accuracy, predictor.get_score()['accuracy']
        return features_to_remove[:-1]


class CorrelationAnalyzer:
    def __init__(self, config: NeuralNetworkConfig, nnet: FullyConnectedNeuralNetwork, training_dataset):
        self._source_model = nnet
        self._config = config
        self._training_dataset = training_dataset
        self._columns = self._training_dataset.get_data().columns
        categorical_columns = training_dataset.get_data(without_resulting_feature=True).select_dtypes(
            include='category').columns
        self._weights = None
        self._emb_weights = None
        self._cat_input_shape = self._source_model.get_layer(config.cat_input+categorical_columns[0]).get_input_shape_at(0)
        self._cont_input_shape = self._source_model.get_layer(config.cont_input).get_input_shape_at(0)[-1]
        self._hid_size = self._source_model.get_layer(config.hidden+"1").get_output_shape_at(0)[-1]
        self._emb_size = self._source_model.get_layer(categorical_columns[0]).get_output_shape_at(0)[-1]
        self._dropout_rate = self._source_model.get_layer(config.dropout+"1").get_config()['rate']
        self._table = np.empty([len(categorical_columns)+self._cont_input_shape+1, len(categorical_columns)+self._cont_input_shape+1])
        self._cat_data = {}
        for x in categorical_columns:
            self._cat_data[x] = self._source_model.get_layer(x).get_config()["input_dim"] - 1

    def _build_network(self, config, dataset, full_copy: bool = False):
        noisy_model = FullyConnectedNeuralNetwork.from_scratch(config=config, dataset=dataset,
                                                 hidden_units=self._hid_size, embedding_size=self._emb_size,
                                                 dropout_rate=self._dropout_rate,embedding_layers_trainable=False)
        if not full_copy:
            noisy_model.set_weights_by_name(self._emb_weights)
        else:
            noisy_model.set_weights_by_name(self._weights)
        return noisy_model

    def run(self, test_dataset, training_dataset, training_target, noise_rate=0.01, training_epochs=100):
        training_dataset = DataSet.copy(training_dataset)
        trainer = Trainer(self._source_model, training_dataset, training_target, epochs=training_epochs)
        trainer.train()
        trainer.evaluate()
        self._weights = self._source_model.get_weights_with_name()
        self._emb_weights = {feature: self._weights[feature] for feature in list(self._cat_data.keys())}
        predictor = Predictor(self._source_model, test_dataset)
        self._table[0][0] = np.sum(predictor.predict())
        noise_rate = random.uniform(0, noise_rate)
        for idx, column in enumerate(self._columns):
            if training_dataset.get_data()[column].dtype.name == 'category':
                noisy_dataset = DataSet.copy(training_dataset)
                noisy_dataset.add_noise_to_categorical_columns(column, noise_rate)
                noisy_model = self._build_network(self._config, training_dataset)
                noisy_model.compile()
                trainer = Trainer(noisy_model, noisy_dataset, training_target, epochs=training_epochs)
                trainer.train()
                trainer.evaluate()
                noisy_test_dataset = DataSet.copy(test_dataset)
                noisy_test_dataset.add_noise_to_categorical_columns(column, noise_rate)
                predictor = Predictor(noisy_model, noisy_test_dataset)
            else:
                noisy_dataset = DataSet.copy(training_dataset)
                noisy_dataset.add_noise_to_column(column, noise_rate)
                noisy_model = self._build_network(self._config, training_dataset)
                noisy_model.compile()
                trainer = Trainer(noisy_model,noisy_dataset, training_target, epochs=training_epochs)
                trainer.train()
                trainer.evaluate()
                noisy_test_dataset = DataSet.copy(test_dataset)
                noisy_test_dataset.add_noise_to_column(column, noise_rate)
                predictor = Predictor(noisy_model, noisy_test_dataset)
            noisy_prediction = predictor.predict()
            self._table[0][idx+1] = abs(np.sum(noisy_prediction) - self._table[0][0])

        for idx, column in enumerate(self._columns):
            if test_dataset.get_data()[column].dtype.name == 'category':
                noisy_dataset = DataSet.copy(test_dataset)
                noisy_dataset.add_noise_to_categorical_columns(column, noise_rate)
                noisy_model = self._source_model
                predictor = Predictor(noisy_model, test_dataset)
            else:
                noisy_dataset = DataSet.copy(test_dataset)
                noisy_dataset.add_noise_to_column(column, noise_rate)
                noisy_model = self._source_model
                predictor = Predictor(noisy_model, noisy_dataset)
            noisy_prediction = predictor.predict()
            self._table[idx + 1][0] = abs(np.sum(noisy_prediction) - self._table[0][0])

        for c in range(len(self._cat_data)+self._cont_input_shape):
            for idx in range(len(self._cat_data)+self._cont_input_shape):
                self._table[idx+1][c+1] = abs(self._table[idx+1][0] - self._table[0][c+1])
        self._table = np.delete(self._table, 0, 0)
        self._table = np.delete(self._table, 0, 1)
        self._table = pd.DataFrame(data=self._table, index=self._columns, columns=self._columns)
        self._table.loc['mean'] = self._table.mean()
        return self._table

    def select_candidates(self):
        candidates = pd.DataFrame(columns=self._columns)
        fcandidates = dict()
        for column in self._table:
            candidates[column] = (self._table.loc[self._table[column] > self._table[column]['mean']]).index.tolist()
        for column in candidates:
            fcandidates[column] = []
            for row in range(candidates.shape[0]):
                if column in candidates[candidates[column][row]].tolist():
                    fcandidates[column].append(candidates[column][row])
        fcandidates = list(filter(None, fcandidates.values()))
        fcandidates = [list(x) for x in set(tuple(x) for x in fcandidates)]
        return fcandidates
