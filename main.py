import argparse
import sys
import pandas as pd
from PyQt5.QtCore import QCoreApplication

from package.model.datasets import DataSet
from package.model.input_handlers import SqlAlchemyDBHandler, QtSqlDBHandler, DataFileHandler
from package.model.neural_networks import NeuralNetwork, Trainer, FeatureSelector, NeuralNetworkConfig, Predictor

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

        debug = 0
        if debug:
            # Create neural network model
            config = NeuralNetworkConfig()
            network = NeuralNetwork.from_file('my_model.h5')
            network.get_model().summary()

            # preprocessed_data.drop_columns(columns='num')
            test_data = DataSet.copy(dataset, start=211, without_resulting_feature=True)
            # test_data.drop_columns(columns=['chol','cp'])
            test_data.normalize()
            test_data.label_categorical_data()

            test_target = dataset.get_data(start=211).dropna(axis=0, how='any')['num']
            test_target = test_target.values

            # Create predictor and make some predictions
            predictor = Predictor(network, test_data)
            prediction = predictor.predict()
            predictor.evaluate(test_target)
            print("Prediction accuracy: %0.2f %%" % (predictor.get_score()['accuracy'] * 100))

            feature_selector = FeatureSelector(config, network, test_data.get_data().columns.tolist(),
                                               test_data.get_data().select_dtypes(include='category').columns.tolist())
            feature_selector.run(test_data, prediction, n=10, noise_rate=0.001)
        else:
            # Prepare data for training
            training_data = preprocessed_data.get_data()
            cont_data = training_data.select_dtypes(exclude='category').values
            cat_data = training_data.select_dtypes(include='category').drop(columns='num')
            training_target = preprocessed_data.get_data()['num'].values

            # Create neural network model
            config = NeuralNetworkConfig()
            network = NeuralNetwork.from_scratch(config, cat_data, cont_data.shape[-1], embedding_size=3,
                                                 hidden_units=13, dropout_rate=0.2)
            network.save_plot()
            network.compile(loss='binary_crossentropy', optimizer='adam')

            cat_data = DataSet.dataframe_to_series(cat_data)

            # Create trainer and train a little
            trainer = Trainer(network, [*cat_data, cont_data], training_target, epochs=100)
            trainer.train(verbose=1)
            trainer.evaluate()
            network.to_h5()
            network.get_model().summary()

            test_data = DataSet.copy(dataset, start=211, without_resulting_feature=True)
            # test_data.drop_columns(columns=['chol','cp'])
            print(test_data.get_features().get_table())
            test_data.normalize()
            test_data.label_categorical_data()

            test_target = dataset.get_data(start=211).dropna(axis=0, how='any')['num']
            test_target = test_target.values

            # Create predictor and make some predictions
            predictor = Predictor(network, test_data)
            prediction = predictor.predict()
            predictor.evaluate(test_target)
            print("Prediction accuracy for %d rows: %0.2f %%" % (
            len(test_data.index()), (predictor.get_score()['accuracy'] * 100)))
