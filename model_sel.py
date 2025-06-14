import os
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor
from joblib import dump

import warnings
warnings.simplefilter('ignore')

class ModelSelector:
    def __init__(self, data_input_dir= r"outputs", input_dir= r"outputs/saved_models"):
        self.input_dir = input_dir
        self.data_input_dir = data_input_dir
        

    def unified_param_search(self, models_dict, param_grids_dict, cv=5, scoring={'r2': 'r2', 'rmse': 'neg_root_mean_squared_error'}, refit='r2'):
        
        self.X = pd.read_csv(os.path.join(self.data_input_dir, "X.csv"), index_col= 0)
        self.y = pd.read_csv(os.path.join(self.data_input_dir, "y.csv"), index_col= 0)

        results = {}

        for name in models_dict:
            print(f"\noptimizing {name}...")
            model_template = models_dict[name]
            param_grid = param_grids_dict[name]

            grid_search = GridSearchCV(
                estimator=model_template,
                param_grid=param_grid,
                cv=cv,
                scoring=scoring, 
                refit=refit,
                error_score='raise'
            )
            grid_search.fit(self.X, self.y.iloc[:, 0])

            cv_results = grid_search.cv_results_

            results[name] = {
                'best_r2': grid_search.best_score_,
                'best_rmse': abs(cv_results[f'mean_test_rmse'][grid_search.best_index_]),
                'best_params': grid_search.best_params_,
                'best_estimator': grid_search.best_estimator_,
            }

            model_path = f"{self.input_dir}/{name}_best_model.joblib"
            dump(grid_search.best_estimator_, model_path)

            print(f"Best R²: {results[name]['best_r2']:.4f}")
            print(f"Best RMSE: {results[name]['best_rmse']:.4f}")

        return results

    def save_results_to_csv(self, results):

        records = []
        for model_name, result in results.items():
            records.append({
                "model": model_name,
                "best_r2": result["best_r2"],
                "best_rmse": result["best_rmse"],
                "best_params": str(result["best_params"]),
            })

        df = pd.DataFrame(records, columns=["model", "best_r2", "best_rmse", "best_params"])
        df.to_csv(os.path.join(self.input_dir, "model_res.csv"), index=False)

    def exe(self):

        models = {
            'rf': RandomForestRegressor(random_state=1234),
            'gbdt': GradientBoostingRegressor(random_state=1234),
            'adaboost': AdaBoostRegressor(random_state=1234),
            'svr': SVR(),
            'lr': BayesianRidge(),
            'xgb': XGBRegressor(random_state=1234),   
            'lgbm': LGBMRegressor(random_state=1234, force_col_wise=True, verbose=-1),
            'catboost': CatBoostRegressor(verbose=0),     
            'elastic': ElasticNet(random_state=1234)
        }

        param_grids = {
            'rf': {
                'n_estimators': [50, 100, 150, 200],
                'max_depth': [None, 5, 8, 10],
                'min_samples_split': [2, 5, 10]
            },
            'gbdt': {
                'n_estimators': [50, 100, 150, 200],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'adaboost': {
                'n_estimators': [50, 100, 150, 200],
                'learning_rate': [0.01, 0.1, 0.5, 1.0]
            },
            'svr': {
                'kernel': ['rbf', 'poly'],
                'C': [0.1, 1, 5],
                'epsilon': [0.01, 0.1]
            },
            'lr': {
                'alpha_1': [1e-6, 1e-5],
                'alpha_2': [1e-6, 1e-5]
            },
            'xgb': {
                'n_estimators': [50, 100, 150, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1]
            },
            'lgbm': {
                'learning_rate': [0.01, 0.05],
                'n_estimators': [50, 100, 150, 200]
            },
            'catboost': {
                'iterations': [500],
                'depth': [4, 6],
                'learning_rate': [0.01, 0.05]
            },
            'elastic': {
                'alpha': [0.001, 0.01, 0.1],
                'l1_ratio': [0.2, 0.5, 0.8]
            },
            'mlp': {
                'hidden_layer_sizes': [(50,), (100,)],
                'activation': ['relu', 'tanh'],
                'solver': ['adam']
            }
        }
        if not os.path.exists(self.input_dir):
            os.makedirs(self.input_dir)
        search_results = self.unified_param_search(models, param_grids, cv=5, scoring={'r2': 'r2', 'rmse': 'neg_root_mean_squared_error'}, refit='r2')
        
        self.save_results_to_csv(search_results)
