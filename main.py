"""amcl
Usage:
    main.py optimize --import-model=<str> --export-model=<str> --dataset=<str> --resulting-feature=<str> [--train-size=<float>|--test-size=<float>] [options]
    main.py predict --import-model=<str> --dataset=<str> --resulting-feature=<str> [--train-size=<float>|--test-size=<float>] [options]
    main.py create-naive-model --dataset=<str> --resulting-feature=<str> [--export-model=<str>] [--train-size=<float>|--test-size=<float>] [options]
    main.py (--svm|--naive-bayes) --dataset=<str> --resulting-feature=<str> [options]
    main.py create-optimized-model --dataset=<str> --resulting-feature=<str> --correlation-info=<list> [--export-model=<str>] [--train-size=<float>|--test-size=<float>] [options]

Options:
    --debug                     Debugging mode (temporary)
    --use-batch-normalization   Use Batch Normalization to speed up learning
    --dataset=dir               Dataset directory
    --embedding-size=<int>      Embedding size for categorical features [default: 3]
    --dropout-rate=<float>      Dropout rate for regularization [default: 0.0]
    --training-epochs=<int>     Number of training epochs [default: 100]
    --hidden-units=<int>        Number of units in hidden layer [default: 4]
    --noise-rate=<float>        Rate of noise added to dataset for some optimizations [default: 0.01]
    --import-model=<str>        Model path to import
    --export-model=<str>        Model path to export
    --naive-plot=path           Naive model structure image path
    --optimized-plot=path       Optimized model structure image path
    --delimiter=<char>          Delimiter [default: ,]
    --header-line=<int>         Header line [default: 1]
    --classes-line=<int>        Classes line [default: 0]
    --resulting-feature=<str>   Resulting feature
    --shuffle                   Shuffle
    --na-values=<chr>           Empty values [default: ['?']]
    --train-size=<float>        Percentage of training data [default: 0.8]
    --test-size=<float>         Percentage of test data [default: 0.2]
    --correlation-info=<list>   Correlation info [default: []]
    --drop-features=<list>      Drop features from dataset [default: []]
    --svm
    --naive-bayes
    --show-plot
    --evaluate
"""
from docopt import docopt
from numpy import random, sum, logical_and
from keras import optimizers, backend
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from tensorflow import set_random_seed, ConfigProto, Session, get_default_graph
from os import environ
from pandas import set_option
from ast import literal_eval
from package.model.datasets import DataSet
from package.model.input_handlers import SqlAlchemyDBHandler, QtSqlDBHandler, FSHandler
from package.model.neural_networks import NeuralNetwork, DenseNeuralNetwork, OptimizedNeuralNetwork, Trainer, FeatureSelector, NeuralNetworkConfig, Predictor, CorrelationAnalyzer
import random as rn
from random import randint

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

def enable_reproducible_mode(seed=795, skip_tf: bool = False):
    environ['PYTHONHASHSEED'] = '0'
    random.seed(seed)
    rn.seed(1254)
    if not skip_tf:
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
    imported_model = argv['--import-model']
    model_to_export = argv['--export-model']
    predicting = argv['predict']
    optimizing = argv['optimize']
    creating_naive = argv['create-naive-model']
    creating_optimized = argv['create-optimized-model']
    naive_plot_path = argv['--naive-plot']
    optimized_plot_path = argv['--optimized-plot']
    delimiter = argv['--delimiter']
    header_line = int(argv['--header-line'])
    classes_line = int(argv['--classes-line'])
    resulting_feature = argv['--resulting-feature']
    shuffle = argv['--shuffle']
    na_values = literal_eval(argv['--na-values'])
    training_sample = float(argv['--train-size'])
    test_sample = float(argv['--test-size'])
    correlation_info = literal_eval(argv['--correlation-info'])
    features_to_drop = literal_eval(argv['--drop-features'])
    use_svm = argv['--svm']
    naive_bayes = argv['--naive-bayes']
    show_plot = argv['--show-plot']
    evaluate = argv['--evaluate']

    ihandler = FSHandler(dataset_path, delimiter, header_line, classes_line, na_values)
    dataset = DataSet.load(resulting_feature, ihandler)
    dataset.drop_invalid_data()
    dataset.calculate_statistics([1], [0])
    dataset.remove_invaluable_features()
    dataset.drop_columns(features_to_drop)
    dataset.normalize()
    dataset.label_categorical_data()

    training_data, test_data, training_target, test_target = train_test_split(dataset.get_data().drop(columns=resulting_feature), dataset.get_resulting_series(), train_size=0.7, shuffle=shuffle)
    training_data = DataSet.dataframe_to_dataset(training_data)
    test_data = DataSet.dataframe_to_dataset(test_data)

    if use_svm:
        x = training_data.get_data().values
        y = training_target
        clf = svm.SVC()
        clf.fit(x, y)
        prediction = clf.predict(test_data.get_data().values)
        print(prediction)
        if (evaluate) & (len(prediction) > 0):
            tp = sum(logical_and(prediction == 1, test_target == 1))
            tn = sum(logical_and(prediction == 0, test_target == 0))
            fp = sum(logical_and(prediction == 1, test_target == 0))
            fn = sum(logical_and(prediction == 0, test_target == 1))
            accuracy = (tp + tn) / (tp + fp + fn + tn)
            ppv = tp / (tp + fp)
            npv = tn / (tn + fn)
            recall = tp / (tp + fn)
            specificity = tn / (tn + fp)
            print("Prediction accuracy for %d rows: %0.2f %%" % (len(test_data.index()), accuracy * 100))
            print("PPV:", ppv)
            print("NPV:", npv)
        exit()

    if naive_bayes:
        x = training_data.get_data().values
        y = training_target
        gnb = GaussianNB()
        prediction = gnb.fit(x, y).predict(test_data.get_data().values)
        print(prediction)
        if (evaluate) & (len(prediction) > 0):
            tp = sum(logical_and(prediction == 1, test_target == 1))
            tn = sum(logical_and(prediction == 0, test_target == 0))
            fp = sum(logical_and(prediction == 1, test_target == 0))
            fn = sum(logical_and(prediction == 0, test_target == 1))
            accuracy = (tp + tn) / (tp + fp + fn + tn)
            ppv = tp / (tp + fp)
            npv = tn / (tn + fn)
            recall = tp / (tp + fn)
            specificity = tn / (tn + fp)
            print("Prediction accuracy for %d rows: %0.2f %%" % (len(test_data.index()), accuracy * 100))
            print("PPV:", ppv)
            print("NPV:", npv)
        exit()

    config = NeuralNetworkConfig(batch_normalization=batch_normalization)

    if optimizing:
        if not len(correlation_info):
            enable_reproducible_mode()
            network = NeuralNetwork.from_file(imported_model)
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
            print(table)
            correlation_info = correlation_analyzer.select_candidates()
            print(correlation_info)

        network = OptimizedNeuralNetwork.from_scratch(config, training_data, correlation_info, embedding_size=emb_size, dropout_rate=dropout_rate, output_units=1)
        network.compile(lr=0.03)
        network.save_plot(optimized_plot_path)

        trainer = Trainer(network, training_data, training_target, epochs=training_epochs, batch_size=32)
        trainer.train()

        predictor = Predictor(network, test_data)
        prediction = predictor.predict()
        print(prediction)

        if evaluate:
            predictor.evaluate(test_target, show_plot)
            print("Prediction accuracy for %d rows: %0.2f %%" % (len(test_data.index()), (predictor.get_score()['accuracy'] * 100)))
            print("PPV: %0.2f" % predictor.get_score()['ppv'])
            print("NPV: %0.2f" % predictor.get_score()['npv'])
        if model_to_export:
            network.export(model_to_export)

    if creating_optimized:
        # enable_reproducible_mode()
        network = OptimizedNeuralNetwork.from_scratch(config, training_data, correlation_info, embedding_size=emb_size, dropout_rate=dropout_rate, output_units=1)
        network.compile(lr=0.03)
        network.save_plot(optimized_plot_path)

        trainer = Trainer(network, training_data, training_target, epochs=training_epochs, batch_size=32)
        trainer.train()

        predictor = Predictor(network, test_data)
        prediction = predictor.predict()
        print(prediction)
        if evaluate:
            predictor.evaluate(test_target, show_plot)
            print("Prediction accuracy for %d rows: %0.2f %%" % (len(test_data.index()), (predictor.get_score()['accuracy'] * 100)))
            print("PPV: %0.2f" % predictor.get_score()['ppv'])
            print("NPV: %0.2f" % predictor.get_score()['npv'])
        if model_to_export:
            network.export(model_to_export)

    if creating_naive:
        # seed = randint(0,2000)
        # print(seed)
        # enable_reproducible_mode()
        network = DenseNeuralNetwork.from_scratch(config, training_data, embedding_size=emb_size, hidden_units=hidden_units, dropout_rate=dropout_rate)
        if naive_plot_path:
            network.save_plot(naive_plot_path, layer_names=True)
        network.compile()

        trainer = Trainer(network, training_data, training_target, epochs=training_epochs, batch_size=16)
        trainer.train()
        network.export(model_to_export)

        predictor = Predictor(network, test_data)
        prediction = predictor.predict()
        print(prediction)

        if evaluate:
            predictor.evaluate(test_target, show_plot)
            print("Prediction accuracy for %d rows: %0.2f %%" % (len(test_data.index()), (predictor.get_score()['accuracy'] * 100)))
            print("PPV: %0.2f" % predictor.get_score()['ppv'])
            print("NPV: %0.2f" % predictor.get_score()['npv'])
        if model_to_export:
            network.export(model_to_export)

    if predicting:
        network = NeuralNetwork.from_file(imported_model)
        predictor = Predictor(network, test_data)
        prediction = predictor.predict()
        print(prediction)
        if evaluate:
            predictor.evaluate(test_target, show_plot)
            print("Prediction accuracy for %d rows: %0.2f %%" % (len(test_data.index()), (predictor.get_score()['accuracy'] * 100)))
            print("PPV: %0.2f" % predictor.get_score()['ppv'])
            print("NPV: %0.2f" % predictor.get_score()['npv'])
