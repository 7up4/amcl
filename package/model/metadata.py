import pandas as pd


class Metadata:
    def __init__(self, columns: list) -> object:
        self._significance = 'significance'
        self._statistic = 'statistic'
        self._resulting = 'resulting'
        self._sensitivity = 'sensitivity'
        self._pvalue = 'pvalue'
        self._table = pd.DataFrame(columns=columns, index=[self._significance, self._statistic, self._resulting,
                                                           self._sensitivity, self._pvalue])

    def get_table(self):
        return self._table

    def update(self, updated_columns: list):
        current_columns = self._table.columns.tolist()
        dropped_columns = list(set(current_columns) - set(updated_columns))
        added_columns = list(set(updated_columns) - set(current_columns))
        self._table.drop(columns=dropped_columns, inplace=True)
        for col in added_columns:
            self._table[col] = None

    def get_columns(self):
        return self._table.columns.tolist()

    def get_feature(self, feature):
        return self.__table[feature]

    def has_feature(self, feature):
        return feature in self._table.columns.tolist()

    def get_features(self, features: list):
        return self._table[features]

    def get_attribute_names(self):
        return self._table.index

    def rename_features(self, feature_names, new_names):
        delta = dict(zip(feature_names, new_names))
        self._table.rename(columns=delta, inplace=True)

    def rename_attrs(self, attrs_names, new_names):
        delta = dict(zip(attrs_names, new_names))
        self._table.rename(index=delta, inplace=True)

    def set(self, feature, attribute, value):
        self._table[feature][attribute] = value

    def get(self, feature, attribute):
        return self._table[feature][attribute]

    def get_significance(self):
        return self._significance

    def get_pvalue(self, feature):
        return self._table[feature][self._pvalue]

    def get_statistic(self, feature):
        return self._table[feature][self._statistic]

    def is_valuable(self, feature):
        return self._table[feature][self._pvalue] < 0.05

    def resulting(self, feature):
        self._table[feature]['resulting'] = True

    def not_resulting(self, feature):
        self._table[feature]['resulting'] = False

