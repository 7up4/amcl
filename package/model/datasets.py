from .importing_modules import *
from .metadata import Metadata
import copy
import math
import random
import numpy as np
from .input_handlers import InputHandler
from scipy.stats import chisquare, chi2_contingency
from scipy.stats import mannwhitneyu
from sklearn import preprocessing


class DataSet:
    def __init__(self, data: pd.DataFrame, features: Metadata):
        self.__data = data
        self.__features = features

    @classmethod
    def copy(cls, dataset, start=None, stop=None, without_resulting_feature=False):
        data = dataset.get_data(start, stop, without_resulting_feature)
        features = copy.deepcopy(dataset.get_features())
        return cls(copy.copy(data), features)

    @classmethod
    def load(cls, resulting_feature: str, input_handler: InputHandler):
        dataframe = input_handler.data
        feature_classes = input_handler.feature_classes
        feature_names = dataframe.columns.tolist()
        for idx, f in enumerate(feature_names):
            if feature_classes[idx] == "cat":
                dataframe[f] = dataframe[f].astype('category')
        features = Metadata(feature_names)
        features.resulting(resulting_feature)
        return cls(dataframe, features)

    @classmethod
    def dataframe_to_dataset(cls, dataframe, resulting_feature:str=None):
        features = Metadata(dataframe.columns.tolist())
        if resulting_feature:
            features.resulting(resulting_feature)
        return cls(dataframe, features)

    def calculate_statistics(self, high_risk: list, low_risk: list):
        data = self.__data
        resulting_feature = self.resulting_feature
        harm = data.loc[data[resulting_feature].isin(high_risk)]
        no_harm = data.loc[data[resulting_feature].isin(low_risk)]
        for feature in self.__features.get_columns():
            feature_type = data[feature].dtype.name
            if feature_type == "category":
                harm_categories = harm[feature].value_counts(sort=False)
                no_harm_categories = no_harm[feature].value_counts(sort=False)
                print(feature)
                print("no_harm")
                print(no_harm_categories)
                print("harm")
                print(harm_categories)
                observed = pd.concat([no_harm_categories, harm_categories], axis=1).values.transpose()
                _, _, _, expected = chi2_contingency(observed)
                result = chisquare(observed.flatten(), expected.flatten())
                print(result)
            else:
                result = mannwhitneyu(no_harm[feature], harm[feature], alternative='two-sided')
                print(no_harm[feature].mean())
                print(harm[feature].mean())
                print(result)
            if result:
                self.__features.set(feature, "statistic", result.statistic)
                self.__features.set(feature, "pvalue", result.pvalue)

    def index(self):
        return self.__data.index

    def update_features(self):
        self.__features.update(self.__data.columns.tolist())

    def get_feature(self, feature):
        return self.__features.get_feature(feature)

    def shuffle(self):
        self.__data = self.__data.sample(frac=1).reset_index(drop=True)

    def drop_columns(self, columns):
        if columns:
            self.__features.drop_features(columns)
            self.__data.drop(columns, inplace=True, axis=1)

    def get_data(self, start=None, stop=None, without_resulting_feature=False):
        if without_resulting_feature and self.resulting_feature:
            return self.__data.ix[start:stop].drop(columns=self.resulting_feature)
        return self.__data.ix[start:stop]

    def get_features(self):
        return self.__features

    def rm_less_sensitive(self):
        rm_feature = self.__features.get_less_sensitive_feature()
        self.drop_columns(rm_feature)
        return rm_feature

    def get_resulting_series(self):
        resulting_feature = self.resulting_feature
        if resulting_feature:
            return self.__data[resulting_feature]

    def drop_resulting_feature(self):
        resulting_feature = self.resulting_feature
        if resulting_feature:
            self.drop_columns(resulting_feature)

    def remove_invaluable_features(self):
        for feature in self.__features.get_columns():
            if not self.__features.is_valuable(feature):
                self.__data.drop(columns=feature, inplace=True)
        self.update_features()

    def get_invaluable_features(self):
        invaluable_features = []
        for feature in self.__features.get_columns():
            if not self.__features.is_valuable(feature):
                invaluable_features.append(feature)
        return invaluable_features

    @property
    def resulting_feature(self):
        for feature in self.__features.get_columns():
            if self.__features.get(feature, 'resulting') == True:
                return feature
        return None

    def get_categorical_features_size(self):
        return [self.__data[x].cat.categories.size for x in self.__data.select_dtypes(include='category')]

    def drop_invalid_data(self):
        self.__data = self.__data.dropna(axis=0, how='any')

    def combine_classes(self, feature_name, from_classes, to_class):
        if feature_name in self.__features.get_columns():
            self.__data[feature_name].cat.remove_categories(from_classes, inplace=True)
            self.__data[feature_name].fillna(value=to_class, inplace=True)

    def bucketize(self, column_name, categories_num, labels):
        col = self.__data[column_name]
        self.__data.drop(column_name, inplace=True, axis=1)
        bucketized_col = pd.cut(col, categories_num, labels=labels)
        self.__data = self.__data.join(bucketized_col)

    def label_categorical_data(self):
        categorical_features = self.__data.select_dtypes('category').columns
        for column in categorical_features:
            self.__data[column] = self.__data[column].cat.codes.astype('category')

    def get_features_of_dtype(self, dtype: str):
        return self.__data.select_dtypes(include=dtype)

    def normalize(self):
        continuous_features = self.__data.select_dtypes(exclude='category')
        normalized_cont_f = (continuous_features-continuous_features.mean())/continuous_features.std()
        self.__data.update(normalized_cont_f)

    def add_noise_to_column(self, column, noise_rate=0.01):
        self.__data[column] *= (1+noise_rate)

    def add_noise_to_categorical_columns(self, column, noise_rate=0.01):
        orig_column = self.__data[column]
        categories = orig_column.cat.categories.tolist()
        random_rows = random.sample(list(orig_column.index), math.ceil(orig_column.size*noise_rate))
        for row in random_rows:
            self.__data[column][row] = random.choice(list(set(categories) - {orig_column.loc[row]}))

    @staticmethod
    def dataframe_to_series(dataframe):
        return list(np.transpose(dataframe.values))
