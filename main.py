import argparse
import sys
import pandas as pd
import numpy as np
from PyQt5.QtCore import QCoreApplication

from package.model.datasets import DataSet
from package.model.input_handlers import SqlAlchemyDBHandler, QtSqlDBHandler, DataFileHandler
from package.model.neural_networks import FullyConnectedNeuralNetwork, OptimizedNeuralNetwork, Trainer, FeatureSelector, NeuralNetworkConfig, Predictor, CorrelationAnalyzer

from keras.layers import Concatenate, Input, Embedding, Lambda
from keras.layers.core import Dense, Dropout, Reshape
from keras.models import load_model, model_from_json, model_from_yaml, Model
from keras.utils.vis_utils import plot_model

if __name__ == '__main__':
    app = QCoreApplication(sys.argv)

    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    args = parser.parse_args()
    pd.set_option('display.max_columns', None)
    def alchemy():
        alchemy = SqlAlchemyDBHandler("postgresql", "postgres", "nurlan", "your_password", 5432, "dataset")
        alchemy.configure()
        alchemy.open()
        ddd = alchemy.data()
        alchemy.close()
        print(ddd)

    def qtsql():
        qtdb = QtSqlDBHandler("postgres", "nurlan", "your_password", 5432, "dataset")
        qtdb.configure()
        qtdb.open()
        ddd = qtdb.data("age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak",
                        "slope", "ca", "thal", "num")
        qtdb.close()
        print(ddd)

    if args.input:
        ihandler = DataFileHandler(args.input, ',', 1, 0, ['?'])
        dataset = DataSet.load("num", ihandler)
        dataset.drop_invalid_data()
        # dataset.shuffle()
        dataset.combine_classes(feature_name='num', from_classes=[2, 3, 4], to_class=1)
        dataset.bucketize('age', 10, list(range(0, 10)))
        dataset.calculate_statistics([1, 2, 3], [0])
        # print(dataset.get_features().get_table())
        # print(dataset.get_invaluable_features())
        dataset.remove_invaluable_features()

        # Preprocess input data
        preprocessed_data = DataSet.copy(dataset, stop=210)
        preprocessed_data.normalize()
        preprocessed_data.label_categorical_data()
        # preprocessed_data.drop_columns(columns=['chol','cp'])

        debug = 1
        if debug == 1:
            # Create neural network model
            config = NeuralNetworkConfig()
            # network = FullyConnectedNeuralNetwork.from_file('my_model1.h5')
            # network.get_model().summary()

            test_data = DataSet.copy(dataset, start=211, without_resulting_feature=True)
            test_data.normalize()
            test_data.label_categorical_data()
            test_target = dataset.get_data(start=211).dropna(axis=0, how='any')['num']
            test_target = test_target.values

            training_data = DataSet.copy(preprocessed_data)
            training_target = training_data.get_resulting_series().values
            training_data.drop_resulting_feature()

            # Create predictor and make some predictions
            # predictor = Predictor(network, test_data)
            # prediction = predictor.predict()
            # predictor.evaluate(test_target)
            # print("Prediction accuracy: %0.2f %%" % (predictor.get_score()['accuracy'] * 100))

            # feature_selector = FeatureSelector(config, network, training_data)
            # less_sensitive_features = feature_selector.run(training_data, training_target, test_data, test_target, noise_rate=0.001, training_epochs=100)
            # print(less_sensitive_features)
            # training_data.drop_columns(less_sensitive_features)
            # test_data.drop_columns(less_sensitive_features)

            # network = FullyConnectedNeuralNetwork.from_scratch(config, training_data, embedding_size=3,
            #                                      hidden_units=13, dropout_rate=0.2)
            # network.compile()
            # network.to_h5('after_feature_selector.h5')

            # network = FullyConnectedNeuralNetwork.from_file('after_feature_selector.h5')
            #
            # correlation_analyzer = CorrelationAnalyzer(config, network, training_data)
            # table = correlation_analyzer.run(test_data, training_data, training_target, noise_rate=0.001, training_epochs=100)
            # correlation_info = correlation_analyzer.select_candidates()
            # print(correlation_info)
            correlation_info = [['sex', 'trestbps', 'chol', 'thalach', 'exang', 'oldpeak', 'slope', 'thal']]
            # correlation_info = [['age', 'trestbps', 'chol', 'thalach', 'oldpeak']]
            network = OptimizedNeuralNetwork.from_scratch(config, training_data, correlation_info, embedding_size=3, dropout_rate=0.2, output_units=1)
            network.compile()
            # network.save_plot('optimized_model.png')

            trainer = Trainer(network, training_data, training_target, epochs=100)
            trainer.train()
            trainer.evaluate()

            predictor = Predictor(network, test_data)
            prediction = predictor.predict()
            predictor.evaluate(test_target)
            print("Prediction accuracy for %d rows: %0.2f %%" % (len(test_data.index()),
                                                                 (predictor.get_score()['accuracy'] * 100)))
        else:
            # Prepare data for training
            training_data = DataSet.copy(preprocessed_data)
            training_target = training_data.get_resulting_series().values

            # Create neural network model
            config = NeuralNetworkConfig()
            network = FullyConnectedNeuralNetwork.from_scratch(config, training_data, embedding_size=3, hidden_units=13, dropout_rate=0.2)
            network.save_plot()
            network.compile()

            # Create trainer and train a little
            trainer = Trainer(network, training_data, training_target, epochs=100)
            trainer.train()
            trainer.evaluate()
            network.to_h5()
            # network.get_model().summary()

            test_data = DataSet.copy(dataset, start=211, without_resulting_feature=True)
            test_data.normalize()
            test_data.label_categorical_data()

            test_target = dataset.get_data(start=211).dropna(axis=0, how='any')['num']
            test_target = test_target.values

            # Create predictor and make some predictions
            predictor = Predictor(network, test_data)
            prediction = predictor.predict()
            predictor.evaluate(test_target)
            print("Prediction accuracy for %d rows: %0.2f %%" % (len(test_data.index()),
                                                                 (predictor.get_score()['accuracy'] * 100)))
