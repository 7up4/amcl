import argparse
import sys

from PyQt5.QtCore import QCoreApplication

from package.model.datasets import DataSet
from package.model.input_handlers import SqlAlchemyDBHandler, QtSqlDBHandler, DataFileHandler
from package.model.neural_networks import NeuralNetwork, NNTrainer, FeatureSelector, NeuralNetworkConfig, NNPredictor

if __name__ == '__main__':
    app = QCoreApplication(sys.argv)

    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    args = parser.parse_args()

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
        dataset.combine_classes(feature_name='num', from_classes=[2, 3, 4], to_class=1)
        dataset.bucketize('age', 10, list(range(0, 10)))
        dataset.calculate_statistics([1, 2, 3], [0])
        dataset.remove_invaluable_features()

        # Preprocess input data
        preprocessed_data = DataSet.copy(dataset, stop=241)
        preprocessed_data.normalize()
        preprocessed_data.label_categorical_data()

        # Prepare data for training
        training_data = preprocessed_data.get_data()
        cont_data = training_data.select_dtypes(exclude='category').values
        cat_data = training_data.select_dtypes(include='category').drop(columns='num')
        training_target = preprocessed_data.get_data()['num'].values

        # Create neural network model
        config = NeuralNetworkConfig()
        network = NeuralNetwork.from_scratch(config, cat_data, cont_data.shape[-1], hidden_units=95, dropout_rate=0.2)
        network.save_plot('model_plot.png')
        network.compile(loss='binary_crossentropy', optimizer='adam')

        cat_data = DataSet.dataframe_to_series(cat_data)

        # Create trainer and train a little
        trainer = NNTrainer(network, [*cat_data, cont_data], training_target, epochs=100)
        trainer.train(verbose=1)
        trainer.evaluate()

        test_data = DataSet.copy(dataset, start=242, without_resulting_feature=True)
        test_data.update_features()
        test_data.normalize()
        test_data.label_categorical_data()

        test_target = dataset.get_data(start=242).dropna(axis=0, how='any')['num']
        test_target = test_target.values

        feature_selector = FeatureSelector(config, network, training_data.select_dtypes(include='category').
                                           drop(columns='num').columns.tolist(),
                                           training_data.select_dtypes(include='category').drop(
                                               columns='num').columns.tolist())
        feature_selector.run(test_data, test_target)

        # # Create predictor and make some predictions
        # predictor = NNPredictor(network, test_data)
        # predictor.predict()
        # predictor.evaluate(test_target)
        # print(predictor.get_score())

        # columns = preprocessed_data.get_columns()
        # for column in columns:
        #     if column == 'num':
        #         continue
        #     print(column)
        #     training_data = preprocessed_data.add_noise(column, 0.01)
        #     training_data = training_data.loc[:, training_data.columns != 'num'].values
        #     training_target = preprocessed_data.get_dataset()['num'].values
        #
        #     network = NeuralNetwork.from_scratch(training_data.shape[1], training_data.shape[1])
        #     network.compile(loss='binary_crossentropy', optimizer='adam')
        #
        #     trainer = NNTrainer(network, training_data, training_target, epochs=100)
        #     trainer.train(verbose=0)
        #     trainer.evaluate()
        #
        #     predictor = NNPredictor(network, test_data)
        #     predictor.predict()
        #     predictor.evaluate(test_target)
        #     print(predictor.get_score())
        #
        #     noisy_prediction = predictor.get_prediction()
        #     # print(noisy_prediction)
        #     print(abs(np.sum(noisy_prediction)-np.sum(predictionX))/predictionX.size)
