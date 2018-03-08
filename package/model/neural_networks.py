from .importing_modules import *
from keras.models import load_model, model_from_json, model_from_yaml, Model
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
    def from_scratch(cls, categorical_data: pd.DataFrame, continuous_features: int, hidden_units: int,
                     noise_rate: float, noisy_column=None, dropout_rate: float=0.2):
        model = NeuralNetwork.build(categorical_data, continuous_features, hidden_units, noisy_column, noise_rate,
                                    dropout_rate)
        return cls(model)

    @classmethod
    def from_file(cls, from_file: str):
        model = load_model(from_file)
        return cls(model)

    @staticmethod
    def __add_noise_to_column(x, column, rate):
        if column == -1:
            return x
        noised_column = tf.slice(x, begin=[0, column], size=[-1, 1])
        left_part = tf.slice(x, [0, 0], [-1, column])
        right_part = tf.slice(x, [0, column + 1], [-1, -1])
        noised_column *= (1 + rate)
        return tf.concat(values=[left_part, noised_column, right_part], axis=1)

    @staticmethod
    def __add_noise(x, rate):
        return x * (1 + rate)

    @staticmethod
    def build(categorical_data: pd.DataFrame, continuous_features: int, hidden_units: int, noisy_column: int,
              noise_rate: float, dropout_rate: float):
        kernel_initializer = "uniform"
        output_units = 1
        activation = 'relu'
        output_activation = 'sigmoid'
        if noisy_column:
            noise_layer = Lambda(NeuralNetwork.__add_noise_to_column, arguments={'column': noisy_column,
                                                                                 'rate': noise_rate})

        # create input layers complemented by embedding layers to handle categorical features
        embedding_layers = []
        categorical_inputs = []
        for i in categorical_data:
            categorical_input = Input((1,))
            categorical_inputs.append(categorical_input)
            categorical_noisy_layer = Lambda(NeuralNetwork.__add_noise, arguments={'rate': noise_rate})
            embedding_layer = Embedding(categorical_data[i].cat.categories.size + 1, 10)(categorical_input)
            noisy_embedding_input = categorical_noisy_layer(embedding_layer)
            embedding_layers.append(noisy_embedding_input)

        # create input layer for continuous data
        continuous_input = Input(shape=(continuous_features,))
        if noisy_column:
            continuous_input = noise_layer(continuous_input)
        reshaped_continuous_input = Reshape((1, continuous_features))(continuous_input)

        # merge all inputs
        merge_layer = Concatenate()(embedding_layers + [reshaped_continuous_input])

        # hidden layers
        hidden_layer1 = Dense(hidden_units, activation=activation, kernel_initializer=kernel_initializer)(merge_layer)
        dropout_layer1 = Dropout(dropout_rate)(hidden_layer1)

        # output_layer
        output_layer = Dense(output_units, activation=output_activation)(dropout_layer1)

        # add reshape layer since output should be vector
        output_layer = Reshape((1,))(output_layer)

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
