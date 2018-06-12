"""amcl
Usage:
    main.py (predict|optimize|create-naive-model) --model-path=<str> --dataset=<str> --resulting-feature=<str> [--training-data=<float>|--test-data=<float>] [options]

Options:
    --debug                     Debugging mode (temporary)
    --use-batch-normalization   Use Batch Normalization to speed up learning
    --dataset=dir               Dataset directory
    --embedding-size=<int>      Embedding size for categorical features [default: 3]
    --dropout-rate=<float>      Dropout rate for regularization [default: 0.0]
    --training-epochs=<int>     Number of training epochs [default: 100]
    --hidden-units=<int>        Number of units in hidden layer [default: 4]
    --noise-rate=<float>        Rate of noise added to dataset for some optimizations [default: 0.01]
    --model-path=path           Model path to export or import
    --naive-plot-path=path      Naive model structure image path
    --optimized-plot-path=path  Optimized model structure image path
    --delimiter=<char>          Delimiter [default: ,]
    --header-line=<int>         Header line [default: 1]
    --classes-line=<int>        Classes line [default: 0]
    --resulting-feature=<str>   Resulting feature
    --shuffle                   Shuffle
    --na-values=<chr>           Empty values [default: ['?']]
    --training-data=<float>     Percentage of training data [default: 0.8]
    --test-data=<float>         Percentage of test data [default: 0.2]
"""
from docopt import docopt
from numpy import random
from keras import optimizers, backend
from random import seed
from tensorflow import set_random_seed, ConfigProto, Session, get_default_graph
from os import environ
from pandas import set_option
from ast import literal_eval
from package.model.datasets import DataSet
from package.model.input_handlers import SqlAlchemyDBHandler, QtSqlDBHandler, FSHandler
from package.model.neural_networks import NeuralNetwork, DenseNeuralNetwork, OptimizedNeuralNetwork, Trainer, FeatureSelector, NeuralNetworkConfig, Predictor, CorrelationAnalyzer

# def alchemy():
#     alchemy = SqlAlchemyDBHandler("postgresql", "postgres", "nurlan", "your_password", 5432, "dataset")
#     alchemy.configure()
#     alchemy.open()
#     ddd = alchemy.data()
#     alchemy.close()
#     print(ddd)
#
# def qtsql():
#     qtdb = QtSqlDBHandler("postgres", "nurlan", "your_password", 5432, "dataset")
#     qtdb.configure()
#     qtdb.open()
#     ddd = qtdb.data("age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak",
#                     "slope", "ca", "thal", "num")
#     qtdb.close()
#     print(ddd)

def enable_reproducible_mode():
    environ['PYTHONHASHSEED'] = '0'
    random.seed(32)
    seed(1254)
    set_random_seed(0)
    session_conf = ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = Session(graph=get_default_graph(), config=session_conf)
    backend.set_session(sess)

if __name__ == '__main__':
    set_option('display.max_columns', None)
    set_option('display.max_rows', None)
    argv = docopt(__doc__)

    dataset_path = argv['--dataset']
    emb_size = int(argv['--embedding-size'])
    dropout_rate = float(argv['--dropout-rate'])
    training_epochs = int(argv['--training-epochs'])
    hidden_units = int(argv['--hidden-units'])
    batch_normalization = argv['--use-batch-normalization']
    noise_rate = float(argv['--noise-rate'])
    model_path = argv['--model-path']
    predicting = argv['predict']
    optimizing = argv['optimize']
    creating_naive = argv['create-naive-model']
    naive_plot_path = argv['--naive-plot-path']
    optimized_plot_path = argv['--optimized-plot-path']
    delimiter = argv['--delimiter']
    header_line = int(argv['--header-line'])
    classes_line = int(argv['--classes-line'])
    resulting_feature = argv['--resulting-feature']
    shuffle = argv['--shuffle']
    na_values = literal_eval(argv['--na-values'])
    training_sample = float(argv['--training-data'])
    test_sample = float(argv['--test-data'])

    ihandler = FSHandler(dataset_path, delimiter, header_line, classes_line, na_values)
    dataset = DataSet.load(resulting_feature, ihandler)
    dataset.drop_invalid_data()
    if shuffle:
        dataset.shuffle()
    dataset.combine_classes(feature_name=resulting_feature, from_classes=[2, 3, 4], to_class=1)
    dataset.calculate_statistics([1], [0])
    print(dataset.get_features().get_table())
    print(dataset.get_invaluable_features())
    dataset.remove_invaluable_features()

    training_data = DataSet.copy(dataset, stop=210)
    training_data.normalize()
    training_data.label_categorical_data()
    training_target = training_data.get_resulting_series().values

    test_data = DataSet.copy(dataset, start=211, without_resulting_feature=True)
    test_data.normalize()
    test_data.label_categorical_data()
    test_target = dataset.get_data(start=211).dropna(axis=0, how='any')[resulting_feature].values

    config = NeuralNetworkConfig(batch_normalization=batch_normalization)

    if optimizing:
        enable_reproducible_mode()
        training_data.drop_resulting_feature()
        # Create neural network model
        network = NeuralNetwork.from_file(model_path)

        feature_selector = FeatureSelector(config, network, training_data)
        less_sensitive_features = feature_selector.run(training_data, training_target, test_data, test_target, noise_rate=noise_rate, training_epochs=training_epochs)
        print(less_sensitive_features)

        training_data.drop_columns(less_sensitive_features)
        test_data.drop_columns(less_sensitive_features)

        network = DenseNeuralNetwork.from_scratch(config, training_data, embedding_size=emb_size,
                                             hidden_units=hidden_units, dropout_rate=dropout_rate)
        network.compile()

        correlation_analyzer = CorrelationAnalyzer(config, network, training_data)
        table = correlation_analyzer.run(test_data, training_data, training_target, noise_rate=noise_rate, training_epochs=training_epochs)
        correlation_info = correlation_analyzer.select_candidates()
        print(correlation_info)

        network = OptimizedNeuralNetwork.from_scratch(config, training_data, correlation_info, embedding_size=emb_size, dropout_rate=dropout_rate, output_units=1)
        network.compile()
        network.save_plot(optimized_plot_path, layer_names=True)

        trainer = Trainer(network, training_data, training_target, epochs=training_epochs)
        trainer.train()

        predictor = Predictor(network, test_data)
        prediction = predictor.predict()
        predictor.evaluate(test_target)

        print("Prediction accuracy for %d rows: %0.2f %%" % (len(test_data.index()), (predictor.get_score()['accuracy'] * 100)))
    if creating_naive:
        network = DenseNeuralNetwork.from_scratch(config, training_data, embedding_size=emb_size, hidden_units=hidden_units, dropout_rate=dropout_rate)
        if naive_plot_path:
            network.save_plot(naive_plot_path, layer_names=True)
        network.compile()

        trainer = Trainer(network, training_data, training_target, epochs=training_epochs)
        trainer.train()
        network.export(model_path)

        predictor = Predictor(network, test_data)
        prediction = predictor.predict()
        predictor.evaluate(test_target)
        print("Prediction accuracy for %d rows: %0.2f %%" % (len(test_data.index()), (predictor.get_score()['accuracy'] * 100)))

    if predicting:
        network = NeuralNetwork.from_file(model_path)
        predictor = Predictor(network, test_data)
        prediction = predictor.predict()
        print(prediction)
        predictor.evaluate(test_target)
