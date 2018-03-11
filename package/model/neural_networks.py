from .importing_modules import *
import re
from keras.models import load_model, model_from_json, model_from_yaml, Model, clone_model
from keras.layers import Concatenate, Input, Embedding, Lambda
from keras.layers.core import Dense, Dropout, Reshape
from keras.utils.vis_utils import plot_model
import tensorflow as tf
from keras.callbacks import TensorBoard


class NeuralNetwork:
    def __init__(self, model):
        self.__model = model

    def get_model(self):
        return self.__model

    @classmethod
    def from_scratch(cls, categorical_data, continuous_features: int, hidden_units: int,
                     noise_rate: float=None, noisy_column=None, embedding_size: int=10, dropout_rate: float=0.2,
                     kernel_initializer="uniform", output_units=1):

        if isinstance(categorical_data, pd.DataFrame):
            categorical_data_categories = {}
            for column in categorical_data:
                categorical_data_categories[column] = categorical_data[column].cat.categories.size
            categorical_data = categorical_data_categories

        model = NeuralNetwork._build(categorical_data, continuous_features, hidden_units, embedding_size, noisy_column,
                                     noise_rate, dropout_rate, kernel_initializer, output_units)
        return cls(model)

    @classmethod
    def from_file(cls, from_file: str):
        model = load_model(from_file)
        return cls(model)

    @staticmethod
    def _add_noise_to_column(x, column, rate):
        if column == -1:
            return x
        noised_column = tf.slice(x, begin=[0, column], size=[-1, 1])
        left_part = tf.slice(x, [0, 0], [-1, column])
        right_part = tf.slice(x, [0, column + 1], [-1, -1])
        noised_column = NeuralNetwork._add_noise(noised_column, rate)
        return tf.concat(values=[left_part, noised_column, right_part], axis=1)

    @staticmethod
    def _add_noise(x, rate):
        return x * (1 + rate)

    @staticmethod
    def _build(categorical_data_categories, continuous_features: int, hidden_units: int, embedding_size: int,
               noisy_column, noise_rate: float, dropout_rate: float, kernel_initializer: str, output_units: int):
        activation = 'relu'
        output_activation = 'sigmoid'

        # create input layer for continuous data
        continuous_input = Input(shape=(continuous_features,), name="continuous_input")
        if noisy_column and isinstance(noisy_column, int):
            noise_layer = Lambda(NeuralNetwork._add_noise_to_column, arguments={'column': noisy_column,
                                                                                'rate': noise_rate},
                                 name=str(noisy_column)+"noisy_layer")

            continuous_input = noise_layer(continuous_input)
            noisy_column = None
        reshaped_continuous_input = Reshape((1, continuous_features),
                                            name="reshaped_continuous_input")(continuous_input)

        # create input layers complemented by embedding layers to handle categorical features
        embedding_layers = []
        categorical_inputs = []
        for feature, size in categorical_data_categories.items():
            categorical_input = Input((1,), name="cat_input_"+feature)
            categorical_inputs.append(categorical_input)
            embedding_layer = Embedding(size + 1, embedding_size, name=feature)(categorical_input)
            if noisy_column == feature:
                categorical_noisy_layer = Lambda(NeuralNetwork._add_noise, arguments={'rate': noise_rate},
                                                 name=str(noisy_column)+"noisy_layer")
                embedding_layer = categorical_noisy_layer(embedding_layer)
            embedding_layers.append(embedding_layer)

        # merge all inputs
        merge_layer = Concatenate(name="merge_layer")(embedding_layers + [reshaped_continuous_input])

        # hidden layers
        hidden_layer1 = Dense(hidden_units, activation=activation, kernel_initializer=kernel_initializer,
                              name="hidden_layer1")(merge_layer)
        dropout_layer1 = Dropout(dropout_rate, name="dropout_layer1")(hidden_layer1)

        # output_layer
        output_layer = Dense(output_units, activation=output_activation, name="output_layer")(dropout_layer1)

        # add reshape layer since output should be vector
        output_layer = Reshape((1,), name="reshaped_output_layer")(output_layer)

        # create final model
        model = Model(inputs=categorical_inputs + [continuous_input], outputs=output_layer)
        return model

    def get_weights(self):
        weights = []
        for layer in self.__model.layers:
            layer_weights = layer.get_weights()
            weights.append(layer_weights)
        return weights

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


class FeatureSelector:
    def __init__(self, nnet: NeuralNetwork, categorical_data):
        self._nnet = nnet
        self._weights = FeatureSelector._get_w(nnet)
        self._cat_data = categorical_data
        self._cat_input_shape = self._nnet.get_model().get_layer("cat_input_age").get_input_shape_at(0)
        self._cont_input_shape = self._nnet.get_model().get_layer("continuous_input").get_input_shape_at(0)[-1]
        self._hid_size = self._nnet.get_model().get_layer("hidden_layer1").get_output_shape_at(0)[-1]
        self._emb_size = self._nnet.get_model().get_layer("age").get_output_shape_at(0)[-1]
        self._dropout_rate = self._nnet.get_model().get_layer("dropout_layer1").get_config()['rate']
        self._build_network()

    def _build_network(self):
        m = NeuralNetwork.from_scratch(categorical_data=self._cat_data, continuous_features=self._cont_input_shape,
                                       hidden_units=self._hid_size, embedding_size=self._emb_size,
                                       dropout_rate=self._dropout_rate, noise_rate=100, noisy_column="age")
        m.save_plot("noisy_model_plot.png", shapes=True, layer_names=True)
        FeatureSelector._set_w(m, self._weights)

    @staticmethod
    def _get_w(nnet):
        model = nnet.get_model()
        names = [re.match("[^/]*", weight.name).group() for layer in model.layers for weight in layer.weights]
        weights = []
        for i in names:
            weights.append(model.get_layer(i).get_weights())
        return dict(zip(names, weights))

    @staticmethod
    def _set_w(nnet, weigths):
        model = nnet.get_model()
        for name, weigth in weigths.items():
            model.get_layer(name).set_weights(weigth)


class NNTrainer:
    def __init__(self, nnet: NeuralNetwork, training_data, target_data, batch_size=16, epochs=1000):
        self.__nnet = nnet
        self.__training_data = training_data
        self.__target_data = target_data
        self.__batch_size = batch_size
        self.__epochs = epochs
        self.__score = None

    def train(self, verbose=1):
        # tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
        self.__nnet.get_model().fit(self.__training_data, self.__target_data, batch_size=self.__batch_size,
                                    epochs=self.__epochs, verbose=verbose)

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
            self.__score['total'] = len(self.__prediction)
            self.__score['wrong'] = self.__score['total'] - self.__score['correct']
            self.__score['score'] = self.__score['correct'] / self.__score['total']

    def get_score(self):
        return self.__score

    def get_prediction(self):
        return self.__prediction
