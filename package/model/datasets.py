from .importing_modules import *
from .features import ContinuousFeature, CategoricalFeature
from .input_handlers import InputHandler
from scipy.stats import chisquare
from scipy.stats import mannwhitneyu
from sklearn import preprocessing


class DataSet:
    def __init__(self, res_feature: str, high_risk: list, low_risk: list, input_handler: InputHandler):
        self.__input_handler = input_handler
        self.__resulting_feature = res_feature
        self.__low_risk = low_risk
        self.__high_risk = high_risk
        self.__data = None
        self.__feature_classes = None
        self.__features = None

    def read_data(self):
        self.__data = self.__input_handler.data
        self.__feature_classes = self.__input_handler.feature_classes
        self.__features = self.__create_feature_objects()
        self.__set_valuable_features()
        self.__set_resulting_feature()

    def get_feature(self, name):
        if name in self.__features:
            return self.__features[name]
        else:
            return None

    def shuffle(self):
        self.__data = self.__data.sample(frac=1).reset_index(drop=True)

    def drop_columns(self, columns):
        self.__data.drop(columns, inplace=True, axis=1)

    def __create_feature_objects(self):
        feature_names = self.__data.columns.values.tolist()
        feature_classes = self.__feature_classes
        features_list = {}
        for idx, f in enumerate(feature_names):
            if feature_classes[idx] == "cont":
                features_list[f] = ContinuousFeature(name=f)
            elif feature_classes[idx] == "cat":
                self.__data[f] = self.__data[f].astype('category')
                features_list[f] = CategoricalFeature(name=f)
        return features_list

    def get_data(self, start=None, stop=None, without_resulting_feature=False):
        if without_resulting_feature:
            return self.__data.ix[start:stop].drop(columns=self.__resulting_feature)
        return self.__data.ix[start:stop]

    def get_features(self):
        return self.__features

    def get_feature_classes(self):
        return self.__feature_classes

    def get_categorical_features(self):
        categorical_features = []
        for name, feature in self.__features.items():
            if feature.get_type == "Categorical":
                categorical_features.append(feature)
        return categorical_features

    @property
    def low_risk_groups(self):
        return self.__data.loc[self.__data['num'].isin(self.__low_risk)]

    @property
    def high_risk_groups(self):
        return self.__data.loc[self.__data['num'].isin(self.__high_risk)]

    def __set_valuable_features(self):
        harm = self.high_risk_groups
        no_harm = self.low_risk_groups
        for name, feature in self.__features.items():
            feature_name = feature.get_name()
            feature_type = feature.get_type()
            result = None
            if feature_type == "Continuous":
                result = mannwhitneyu(no_harm[feature_name], harm[feature_name], alternative='two-sided')
            elif feature_type == "Categorical":
                cp_harm_categories = harm[feature_name].value_counts(sort=False).tolist()
                cp_no_harm_categories = no_harm[feature_name].value_counts(sort=False).tolist()
                result = chisquare(cp_harm_categories, cp_no_harm_categories)
            feature.set_significance(result)

    def __set_resulting_feature(self):
        self.get_feature(self.__resulting_feature).resulting()

    def get_resulting_feature(self):
        for name, feature in self.__features.items():
            if feature.is_resulting():
                return feature
        return None

    def drop_invalid_data(self):
        self.__data = self.__data.dropna(axis=0, how='any')

    def combine_classes(self, feature_name, from_classes, to_class):
        if feature_name in self.__features:
            column = self.__data[feature_name]
            column[column not in from_classes] = to_class

class PreprocessedData:
    def __init__(self, dataset, start_position=None, stop_position=None, without_resulting_feature=False, resulting_feature=None):
        self.__dataset = dataset.get_data(start_position, stop_position, without_resulting_feature)
        self.__features = dataset.get_features()
        self.__resulting_feature = resulting_feature
        self.__drop_invalid_data()

    def bucketize(self, column_name, categories_num, labels):
        col = self.__dataset[column_name]
        self.__dataset.drop(column_name, inplace=True, axis=1)
        bucketized_col = pd.cut(col, categories_num, labels=labels)
        self.__dataset = self.__dataset.join(bucketized_col)

    def label_categorical_data(self):
        categorical_features = self.__dataset.select_dtypes('category').columns
        try:
            categorical_features.drop(self.__resulting_feature)
        except ValueError:
            pass
        for column in categorical_features:
            self.__dataset[column] = self.__dataset[column].cat.codes.astype('category')

    def __drop_invalid_data(self):
        self.__dataset = self.__dataset.dropna(axis=0, how='any')

    def combine_classes(self, feature_name, from_classes, to_class):
        if feature_name in self.__features:
            self.__dataset[feature_name].cat.remove_categories(from_classes, inplace=True)
            self.__dataset[feature_name].fillna(value=to_class, inplace=True)

    def normalize(self):
        # cont_features = self.__dataset.select_dtypes(exclude='category').columns.tolist()
        # normalized_data = preprocessing.normalize(self.__dataset[cont_features])
        # self.__dataset[cont_features] = normalized_data
        continuous_features = self.__dataset.select_dtypes(exclude='category')
        normalized_cont_f = (continuous_features-continuous_features.mean())/continuous_features.std()
        self.__dataset.update(normalized_cont_f)

    def one_hot_encode(self):
        categorical_features = self.__dataset.select_dtypes(include='category')
        if self.__resulting_feature:
            categorical_features = categorical_features.drop(columns=self.__resulting_feature)
        features_to_remove = categorical_features.columns.tolist()
        self.__dataset.drop(features_to_remove, inplace=True, axis=1)
        normalized_cat_f = pd.get_dummies(categorical_features, columns=categorical_features.columns.tolist())
        self.__dataset = self.__dataset.join(normalized_cat_f)

    def get_dataset(self):
        return self.__dataset

    def get_columns(self):
        return self.__dataset.columns.tolist()

    def add_noise(self, column: str, rate: float):
        copy = self.__dataset.copy()
        copy[column] *= (1 + rate)
        return copy