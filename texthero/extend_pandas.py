"""

"""

import pandas as pd


@pd.api.extensions.register_dataframe_accessor("hero")
class TextheroPandas:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        # verify there is a column latitude and a column longitude
        # if 'latitude' not in obj.columns or 'longitude' not in obj.columns:
        #    raise AttributeError("Must have 'latitude' and 'longitude'.")
        pass

    @property
    def feature_(self):
        # return the geographic center point of this DataFrame
        return (float(1), float(2))

    def plot(self):
        # plot this array's data on a map, e.g., using Cartopy
        pass
