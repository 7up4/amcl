import tensorflow as tf
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
    def from_scratch(cls, config: NeuralNetworkConfig, categorical_data, continuous_features: int, hidden_units: int,
                     noise_rate: float=None, noisy_column=None, embedding_size: int=10, dropout_rate: float=0.2,
                     output_units=1):

        if isinstance(categorical_data, pd.DataFrame):
            categorical_data_categories = {}
            for column in categorical_data:
                categorical_data_categories[column] = categorical_data[column].cat.categories.size
            categorical_data = categorical_data_categories

        model = NeuralNetwork._build(config, categorical_data, continuous_features, hidden_units, embedding_size,
                                     noisy_column, noise_rate, dropout_rate, output_units)
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
    def _build(config, categorical_data_categories, continuous_features: int, hidden_units: int, embedding_size: int,
               noisy_column, noise_rate, dropout_rate, output_units: int):

        # create input layer for continuous data
        continuous_input = Input(shape=(continuous_features,), name=config.cont_input)
        if noisy_column and isinstance(noisy_column, int):
            noise_layer = Lambda(NeuralNetwork._add_noise_to_column, arguments={'column': noisy_column,
                                                                                'rate': noise_rate},
                                 name=str(noisy_column)+config.noisy_layer)

            continuous_input = noise_layer(continuous_input)
            noisy_column = None
        reshaped_continuous_input = Reshape((1, continuous_features),
                                            name=config.reshaped)(continuous_input)

        # create input layers complemented by embedding layers to handle categorical features
        embedding_layers = []
        categorical_inputs = []
        for feature, size in categorical_data_categories.items():
            categorical_input = Input((1,), name=config.cat_input+feature)
            categorical_inputs.append(categorical_input)
            embedding_layer = Embedding(size + 1, embedding_size, name=feature)(categorical_input)
            if noisy_column == feature:
                categorical_noisy_layer = Lambda(NeuralNetwork._add_noise, arguments={'rate': noise_rate},
                                                 name=str(noisy_column)+config.noisy_layer)
                embedding_layer = categorical_noisy_layer(embedding_layer)
            embedding_layers.append(embedding_layer)

        # merge all inputs
        merge_layer = Concatenate(name=config.merge)(embedding_layers + [reshaped_continuous_input])

        # hidden layers
        hidden_layer1 = Dense(hidden_units, activation=config.activation, kernel_initializer=config.kernel_initializer,
                              name=config.hidden+"1")(merge_layer)
        dropout_layer1 = Dropout(dropout_rate, name=config.dropout+"1")(hidden_layer1)

        # output_layer
        output_layer = Dense(output_units, activation=config.output_activation, name=config.output)(dropout_layer1)

        # add reshape layer since output should be vector
        output_layer = Reshape((1,), name=config.reshaped_output)(output_layer)

        # create final model
        model = Model(inputs=categorical_inputs + [continuous_input], outputs=output_layer)
        return model

    def get_layer(self, name):
        return self.__model.get_layer(name)

    def get_weights(self):
        return self.__model.get_weights()

    def get_weights_by_name(self):
        model = self.__model
        names = [layer.name for layer in model.layers]
        weights = []
        for name in names:
            weights.append(model.get_layer(name).get_weights())
        return dict(zip(names, weights))

    def set_weights_by_name(self, weigths):
        for name, weigth in weigths.items():
            self.__model.get_layer(name).set_weights(weigth)

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
    def __init__(self, config: NeuralNetworkConfig, nnet: NeuralNetwork, noisy_columns, cat_columns: list):
        self._source_model = nnet
        self._config = config
        self._weights = self._source_model.get_weights_by_name()
        self._cat_input_shape = self._source_model.get_layer(config.cat_input+"age").get_input_shape_at(0)
        self._cont_input_shape = self._source_model.get_layer(config.cont_input).get_input_shape_at(0)[-1]
        self._hid_size = self._source_model.get_layer(config.hidden+"1").get_output_shape_at(0)[-1]
        self._emb_size = self._source_model.get_layer("age").get_output_shape_at(0)[-1]
        self._dropout_rate = self._source_model.get_layer(config.dropout+"1").get_config()['rate']
        self._noisy_columns = noisy_columns
        self._cat_data = {}
        for x in cat_columns:
            self._cat_data[x] = self._source_model.get_layer(x).get_config()["input_dim"] - 1

    def _build_network(self, config, noisy_column, noise_rate=0.1):
        noisy_model = NeuralNetwork.from_scratch(config=config, categorical_data=self._cat_data,
                                                 continuous_features=self._cont_input_shape,
                                                 hidden_units=self._hid_size, embedding_size=self._emb_size,
                                                 dropout_rate=self._dropout_rate, noise_rate=noise_rate,
                                                 noisy_column=noisy_column)
        noisy_model.save_plot("noisy_model_plot.png", shapes=True, layer_names=True)
        noisy_model.set_weights_by_name(self._weights)
        return noisy_model

    def run(self, dataset, test_target):
        for column in self._noisy_columns:
            noisy_model = self._build_network(self._config, noisy_column=column)
            predictor = NNPredictor(noisy_model, dataset)
            predictor.predict()
            predictor.evaluate(test_target)
            print(predictor.get_score())


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
    def __init__(self, nnet: NeuralNetwork, dataset: pd.DataFrame):
        self._nnet = nnet
        self._dataset = dataset
        self._score = {}
        self._prediction = []
        self._preprocess()

    def _preprocess(self):
        test_data_cont = self._dataset.get_data().select_dtypes(exclude='category').values
        test_data_cat = self._dataset.get_data().select_dtypes('category')
        test_data_cat = DataSet.dataframe_to_series(test_data_cat)
        self._dataset = [*test_data_cat, test_data_cont]

    def predict(self):
        self._prediction = self._nnet.get_model().predict(self._dataset).flatten()

    def evaluate(self, real_values):
        if len(self._prediction) > 0:
            compared = np.equal(np.round(self._prediction), real_values)
            self._score['correct'] = np.count_nonzero(compared)
            self._score['total'] = len(self._prediction)
            self._score['wrong'] = self._score['total'] - self._score['correct']
            self._score['score'] = self._score['correct'] / self._score['total']

    def get_score(self):
        return self._score

    def get_prediction(self):
        return self._prediction
