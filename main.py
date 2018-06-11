"""amcl
Usage:
    main.py --dataset=dir [options]

Options:
    --debug                     Debugging mode (temporary)
    --use-batch-normalization   Use Batch Normalization to speed up learning
    --dataset=dir               Dataset directory
    --embedding-size=<value>    Embedding size for categorical features [default: 3]
    --dropout-rate=<float>      Dropot rate for regularization [default: 0.0]
    --training-epochs=<value>   Number of training epochs [default: 100]
    --hidden-units=<value>      Number of units in hidden layer [default: 4]
"""
from docopt import docopt
import sys
import pandas as pd
import numpy as np
from PyQt5.QtCore import QCoreApplication
from keras import optimizers
from package.model.datasets import DataSet
from package.model.input_handlers import SqlAlchemyDBHandler, QtSqlDBHandler, FSHandler
from package.model.neural_networks import DenseNeuralNetwork, OptimizedNeuralNetwork, Trainer, FeatureSelector, NeuralNetworkConfig, Predictor, CorrelationAnalyzer

if __name__ == '__main__':
    app = QCoreApplication(sys.argv)

    argv = docopt(__doc__)
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

    if argv['--dataset']:
        emb_size = int(argv['--embedding-size'])
        dropout_rate = float(argv['--dropout-rate'])
        training_epochs = int(argv['--training-epochs'])
        hidden_units = int(argv['--hidden-units'])
        batch_normalization = argv['--use-batch-normalization']

        ihandler = FSHandler(argv['--dataset'], ',', 1, 0, ['?'])
        dataset = DataSet.load("num", ihandler)
        dataset.drop_invalid_data()
        # dataset.shuffle()
        dataset.combine_classes(feature_name='num', from_classes=[2, 3, 4], to_class=1)
        # dataset.bucketize('age', 5, list(range(0, 5)))
        dataset.calculate_statistics([1, 2, 3], [0])
        # print(dataset.get_features().get_table())
        # print(dataset.get_invaluable_features())
        dataset.remove_invaluable_features()

        # Preprocess input data
        preprocessed_data = DataSet.copy(dataset, stop=210)
        preprocessed_data.normalize()
        preprocessed_data.label_categorical_data()
        # preprocessed_data.drop_columns(columns=['chol','cp'])

        if argv['--debug']:
            # Create neural network model
            config = NeuralNetworkConfig(batch_normalization=batch_normalization)
            network = DenseNeuralNetwork.from_file('my_model.h5')
            network.get_model().summary()

            test_data = DataSet.copy(dataset, start=211, without_resulting_feature=True)
            test_data.normalize()
            test_data.label_categorical_data()
            test_target = dataset.get_data(start=211).dropna(axis=0, how='any')['num']
            test_target = test_target.values

            training_data = DataSet.copy(preprocessed_data)
            training_target = training_data.get_resulting_series().values
            training_data.drop_resulting_feature()

            # Create predictor and make some predictions
            predictor = Predictor(network, test_data)
            prediction = predictor.predict()
            predictor.evaluate(test_target)
            print("Prediction accuracy: %0.2f %%" % (predictor.get_score()['accuracy'] * 100))

            feature_selector = FeatureSelector(config, network, training_data)
            less_sensitive_features = feature_selector.run(training_data, training_target, test_data, test_target, noise_rate=0.1, training_epochs=training_epochs)
            print(less_sensitive_features)
            training_data.drop_columns(less_sensitive_features)
            test_data.drop_columns(less_sensitive_features)

            network = DenseNeuralNetwork.from_scratch(config, training_data, embedding_size=emb_size,
                                                 hidden_units=13, dropout_rate=dropout_rate)
            network.compile()
            network.to_h5('after_feature_selector.h5')

            correlation_analyzer = CorrelationAnalyzer(config, network, training_data)
            table = correlation_analyzer.run(test_data, training_data, training_target, noise_rate=0.1, training_epochs=training_epochs)
            correlation_info = correlation_analyzer.select_candidates()
            # correlation_info = [['trestbps', 'chol', 'thalach', 'oldpeak', 'ca']]
            print(correlation_info)

            network = OptimizedNeuralNetwork.from_scratch(config, training_data, correlation_info, embedding_size=emb_size, dropout_rate=dropout_rate, output_units=1)
            network.compile()
            network.save_plot('optimized_network.png', layer_names=True)
            trainer = Trainer(network, training_data, training_target, epochs=training_epochs)
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
            config = NeuralNetworkConfig(batch_normalization=batch_normalization)
            network = DenseNeuralNetwork.from_scratch(config, training_data, embedding_size=emb_size, hidden_units=hidden_units, dropout_rate=dropout_rate)
            network.save_plot(layer_names=True)
            network.compile()

            # Create trainer and train a little
            trainer = Trainer(network, training_data, training_target, epochs=training_epochs)
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
