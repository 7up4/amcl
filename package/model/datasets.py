from .importing_modules import *
from .metadata import Metadata
import copy
from .input_handlers import InputHandler
from scipy.stats import chisquare
from scipy.stats import mannwhitneyu
from sklearn import preprocessing


class DataSet:
    def __init__(self, data: pd.DataFrame, features: Metadata):
        self.__data = data
        self.__features = features

    @classmethod
    def copy(cls, dataset, start=None, stop=None, without_resulting_feature=False):
        data = dataset.get_data(start, stop, without_resulting_feature)
        features = copy.copy(dataset.get_features())
        return cls(copy.copy(data), features)

    @classmethod
    def load(cls, resulting_feature: str, input_handler: InputHandler):
        dataset = input_handler.data
        feature_classes = input_handler.feature_classes
        feature_names = dataset.columns.tolist()
        for idx, f in enumerate(feature_names):
            if feature_classes[idx] == "cat":
                dataset[f] = dataset[f].astype('category')
        features = Metadata(feature_names)
        features.resulting(resulting_feature)
        return cls(dataset, features)

    def calculate_statistics(self, high_risk: list, low_risk: list):
        data = self.__data
        harm = data.loc[data['num'].isin(high_risk)]
        no_harm = data.loc[data['num'].isin(low_risk)]
        for feature in self.__features.get_columns():
            feature_type = data[feature].dtype.name
            if feature_type == "category":
                cp_harm_categories = harm[feature].value_counts(sort=False).tolist()
                cp_no_harm_categories = no_harm[feature].value_counts(sort=False).tolist()
                result = chisquare(cp_harm_categories, cp_no_harm_categories)
            else:
                result = mannwhitneyu(no_harm[feature], harm[feature], alternative='two-sided')
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
        self.__features.drop_features(columns)
        self.__data.drop(columns, inplace=True, axis=1)

    def get_data(self, start=None, stop=None, without_resulting_feature=False):
        if without_resulting_feature:
            return self.__data.ix[start:stop].drop(columns=self.resulting_feature)
        return self.__data.ix[start:stop]

    def get_features(self):
        return self.__features

    def rm_less_sensitive(self):
        rm_feature = self.__features.get_less_sensitive_feature()
        self.drop_columns(rm_feature)
        print("Just removed", rm_feature)

    def remove_invaluable_features(self):
        for feature in self.__features.get_columns():
            if not self.__features.is_valuable(feature):
                self.__data.drop(columns=feature, inplace=True)
        self.update_features()

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
        # cont_features = self.__dataset.select_dtypes(exclude='category').columns.tolist()
        # normalized_data = preprocessing.normalize(self.__dataset[cont_features])
        # self.__dataset[cont_features] = normalized_data
        continuous_features = self.__data.select_dtypes(exclude='category')
        normalized_cont_f = (continuous_features-continuous_features.mean())/continuous_features.std()
        self.__data.update(normalized_cont_f)

    @staticmethod
    def dataframe_to_series(dataframe):
        return list(np.transpose(dataframe.values))
