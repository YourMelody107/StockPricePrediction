import xgboost as xgb
from xgboost import DMatrix
from log import log
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd
import numpy as np
import operator
import matplotlib.pyplot as plt

class MeowModel(object):
    def __init__(self, cacheDir):
        self.params = {
            'booster': 'gbtree',
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'lambda': 3,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'min_child_weight': 3,
            'eta': 0.05,
            'seed': 1000,
            'nthread': 4,
            'tree_method': 'hist',
            'device': 'cuda'
        }
        self.num_boost_round = 90
        self.bst = None
        self.cacheDir = cacheDir

    def fit(self, xdf, ydf):
        # Convert pandas DataFrame to DMatrix object
        dtrain = DMatrix(data=xdf.to_numpy(), label=ydf.to_numpy())
        self.bst = xgb.train(self.params, dtrain, num_boost_round=self.num_boost_round)
        log.inf("Done fitting")
        self.plot_feature_importance()

    def predict(self, xdf):
        dtest = DMatrix(data=xdf.to_numpy())
        return self.bst.predict(dtest)

    def feature_selection(self, xdf, ydf):
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            booster='gbtree',
            max_depth=self.params['max_depth'],
            subsample=self.params['subsample'],
            colsample_bytree=self.params['colsample_bytree'],
            min_child_weight=self.params['min_child_weight'],
            eta=self.params['eta'],
            seed=self.params['seed'],
            n_jobs=self.params['nthread'],
            tree_method=self.params['tree_method']
        )

        # Setup the RFECV
        selector = RFECV(
            estimator=model,
            step=1,
            cv=StratifiedKFold(3),
            scoring='neg_mean_squared_error',
            verbose=2
        )

        # Fit RFECV
        selector.fit(xdf, ydf)

        # Log the best features
        log.inf("Optimal number of features : %d" % selector.n_features_)
        log.inf('Selected features: %s' % list(xdf.columns[selector.support_]))

        # Transform the data to keep only the selected features
        xdf_selected = selector.transform(xdf)

        return xdf_selected, ydf

    def plot_feature_importance(self):
        importance = self.bst.get_fscore()
        importance_sorted = sorted(importance.items(), key=operator.itemgetter(1))
        
        df = pd.DataFrame(importance_sorted, columns=['feature', 'fscore'])
        df['fscore'] = df['fscore'] / df['fscore'].sum()

        plt.figure()
        df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(12, 10))
        plt.title('XGBoost Feature Importance')
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300)
        plt.show()
