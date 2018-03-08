# class FeaturesList(list):
#     def __getitem__(self, key):
#         if isinstance(key, str):
#             for item in self:
#                 if item.get_name() == key:
#                     return item
#             raise IndexError('no object named {!r}'.format(key))
#         return list.__getitem__(self, key)
#
#     def __missing__(self, key):
#         if isinstance(key, str):
#             self[key] = None
#
#     def __setitem__(self, key, value):
#         if isinstance(key, str) and isinstance(value, Feature):
#             item = self[key]
#             item[key]=value
#
#     def __delitem__(self, key):
#         if isinstance(key, str):
#             del self[key]
#

class Feature:
    def __init__(self, name: str, significance: float = None, resulting: bool = False) -> object:
        self._name = name
        self._significance = significance
        self._resulting = resulting
        self._type = None
        self._sensitivity = 0.0

    def get_name(self):
        return self._name

    def set_name(self, name):
        self._name = name

    def set_significance(self, significance):
        self._significance = significance

    def get_significance(self):
        return self._significance

    def get_pvalue(self):
        return self._significance.pvalue

    def get_statistic(self):
        return self._significance.statistic

    def is_valuable(self):
        return self._significance.pvalue < 0.05

    def is_resulting(self):
        return self._resulting

    def resulting(self):
        self._resulting = True

    def not_resulting(self):
        self._resulting = False

    def get_type(self):
        return self._type

    def get_sensitivity(self):
        return self._sensitivity


class ContinuousFeature(Feature):
    def __init__(self, name: str, significance: float = None, resulting: bool = False) -> object:
        super(ContinuousFeature, self).__init__(name, significance, resulting)
        self._type = "Continuous"


class CategoricalFeature(Feature):
    def __init__(self, name, significance=None, categories=None, resulting=False):
        super(CategoricalFeature, self).__init__(name, significance, resulting)
        self.__categories = categories
        self._type = "Categorical"

    def get_categories(self):
        return self.__categories

    def set_categories(self, categories):
        self.__categories = categories

    def add_categories(self, categories):
        self.__categories.append(categories)

    def drop_categories(self, categories):
        self.__categories = list(filter(lambda cat: cat not in categories, self.__categories))