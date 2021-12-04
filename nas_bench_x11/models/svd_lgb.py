import logging
import os
import joblib
import numpy as np
import lightgbm as lgb

from sklearn.multioutput import RegressorChain
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde

from nas_bench_x11.utils import utils
from nas_bench_x11.surrogate_model import SurrogateModel


class SVDLGBModel(SurrogateModel):
    def __init__(self, data_root, log_dir, seed, model_config, data_config, search_space, nb101_api):

        super().__init__(data_root, log_dir, seed, model_config, data_config, search_space, nb101_api)

        self.model = None
        self.model_config["param:objective"] = "reg:squarederror"
        self.model_config["param:eval_metric"] = "rmse"

    def parse_param_config(self):
        identifier = "param:"
        param_config = dict()
        for key, val in self.model_config.items():
            if key.startswith(identifier):
                param_config[key.replace(identifier, "")] = val
        return param_config

    def train(self):
        # matrices (e.g. X) are capitalized, vectors (e.g. y) are uncapitalized
        X_train, y_train, _ = self.load_dataset(dataset_type='train', use_full_lc=True)
        X_val, y_val, _ = self.load_dataset(dataset_type='val', use_full_lc=True)

        param_config = self.parse_param_config()
        param_config["seed"] = self.seed

        self.num_components = param_config["num_components"]
        self.ss = StandardScaler()
        u, s, vh = np.linalg.svd(y_train, full_matrices=False)

        self.svd_s = s
        self.svd_vh = vh
        self.model = RegressorChain(lgb.LGBMRegressor(
            num_rounds=param_config["num_rounds"],
            boosting_type=param_config["boosting_type"],
            num_leaves=param_config["num_leaves"],
            max_depth=param_config["max_depth"],
            learning_rate=param_config["learning_rate"],
            min_child_weight=param_config["min_child_weight"],
            reg_alpha=param_config["lambda_l1"],
            reg_lambda=param_config["lambda_l2"],
        ))

        # the labels are the first n components of the SVD on the training data
        labels = u[:, :self.num_components].copy()
        fitted_labels = self.ss.fit_transform(labels)

        self.model.fit(X_train, fitted_labels, verbose=True)

        train_pred = self.ss.inverse_transform(self.model.predict(X_train))\
            @ np.diag(self.svd_s[:self.num_components])@self.svd_vh[:self.num_components, :]

        val_pred = self.ss.inverse_transform(self.model.predict(X_val))\
            @ np.diag(self.svd_s[:self.num_components])@self.svd_vh[:self.num_components, :]

        residuals = u[:, :self.num_components]@np.diag(self.svd_s[:self.num_components])@self.svd_vh[:self.num_components, :] - np.stack(y_train)
        self.kernel = gaussian_kde(residuals.T+np.random.randn(*residuals.T.shape)*1e-8)

        # metrics for final prediction
        train_pred_final = np.array(train_pred)
        val_pred_final = np.array(val_pred)
        y_train_final = y_train
        y_val_final = y_val

        train_metrics = utils.evaluate_learning_curve_metrics(y_train_final, train_pred_final, prediction_is_first_arg=False)
        valid_metrics = utils.evaluate_learning_curve_metrics(y_val_final, val_pred_final, prediction_is_first_arg=False)

        logging.info('train metrics: %s', train_metrics)
        logging.info('valid metrics: %s', valid_metrics)
        return valid_metrics

    def test(self):
        X_test, y_test, _ = self.load_dataset(dataset_type='test', use_full_lc=True)

        test_pred = self.ss.inverse_transform(self.model.predict(X_test))\
            @ np.diag(self.svd_s[:self.num_components])@self.svd_vh[:self.num_components, :]

        y_test_final = y_test
        test_pred_final = np.array(test_pred)
        test_metrics = utils.evaluate_learning_curve_metrics(y_test_final, test_pred_final, prediction_is_first_arg=False)

        logging.info('test metrics %s', test_metrics)

        return test_metrics

    def validate(self):
        X_val, y_val, _ = self.load_dataset(dataset_type='val', use_full_lc=True)

        val_pred = self.ss.inverse_transform(self.model.predict(X_val))\
            @ np.diag(self.svd_s[:self.num_components])@self.svd_vh[:self.num_components, :]

        y_val_final = y_val
        val_pred_final = np.array(val_pred)
        valid_metrics = utils.evaluate_learning_curve_metrics(y_val_final, val_pred_final, prediction_is_first_arg=False)

        logging.info('test metrics %s', valid_metrics)

        return valid_metrics

    def save(self):
        save_list = [self.model, self.ss, self.svd_s, self.svd_vh, self.num_components, self.kernel]
        joblib.dump(save_list, os.path.join(self.log_dir, 'surrogate_model.model'))

    def load(self, model_path):
        if len(joblib.load(model_path)) == 5:
            # load without noise model
            model, ss, svd_s, svd_vh, num_components = joblib.load(model_path)
            self.model = model
            self.ss = ss
            self.svd_s = svd_s
            self.svd_vh = svd_vh
            self.num_components = num_components

        else:
            # load with noise model
            logging.info('loading model with noise kernel')
            model, ss, svd_s, svd_vh, num_components, kernel = joblib.load(model_path)
            self.model = model
            self.ss = ss
            self.svd_s = svd_s
            self.svd_vh = svd_vh
            self.num_components = num_components
            self.kernel = kernel

    def evaluate(self):
        X_test, y_test, _ = self.load_dataset(dataset_type='test', use_full_lc=True)

        test_pred = self.ss.inverse_transform(self.model.predict(X_test))\
            @ np.diag(self.svd_s[:self.num_components])@self.svd_vh[:self.num_components, :]

        y_test_final = y_test
        test_pred_final = np.array(test_pred)
        test_metrics = utils.evaluate_learning_curve_metrics(y_test_final, test_pred_final, prediction_is_first_arg=False)

        return test_metrics, test_pred, y_test

    def query(self, config_dict, search_space='darts', use_noise=False, components=False):
        if search_space == 'darts':
            config_space_instance = self.config_loader.query_config_dict(config_dict)
            X = config_space_instance.get_array().reshape(1, -1)
            idx = np.isnan(X)
            X[idx] = -1
            X = X.reshape(1, -1)

        else:
            X = np.array([config_dict])

        comp = self.model.predict(X)        
        if components:
            return self.ss.inverse_transform(comp)

        ypred = self.ss.inverse_transform(comp) @ np.diag(self.svd_s[:self.num_components])\
            @ self.svd_vh[:self.num_components, :]

        if use_noise:
            noise = np.squeeze(self.kernel.resample(1))
            noise = np.clip(noise, -3, 3)  # clip outliers
            return ypred[0] + noise / 2

        return ypred[0]