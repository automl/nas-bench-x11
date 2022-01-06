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
        self.noise_model = {"type":"sliding_window", "windowsize":9}
        
        if search_space == "darts":
            # Create config loader
            root = utils.get_project_root()
            self.config_loader = utils.ConfigLoader(os.path.join(root, "configs/data_configs/nb301_configspace.json"))

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

        # run singular value decomposition
        self.num_components = param_config["num_components"]
        self.ss = StandardScaler()
        u, s, vh = np.linalg.svd(y_train, full_matrices=False)
        self.svd_s = s
        self.svd_vh = vh
        
        # initialize LGBoost
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

        # train LGBoost and make predictions on the train and validation sets
        self.model.fit(X_train, fitted_labels, verbose=True)

        train_pred = self.ss.inverse_transform(self.model.predict(X_train))\
            @ np.diag(self.svd_s[:self.num_components])@self.svd_vh[:self.num_components, :]

        val_pred = self.ss.inverse_transform(self.model.predict(X_val))\
            @ np.diag(self.svd_s[:self.num_components])@self.svd_vh[:self.num_components, :]

        # train the noise model
        residuals = u[:, :self.num_components]@np.diag(self.svd_s[:self.num_components])@self.svd_vh[:self.num_components, :] - np.stack(y_train)

        if self.noise_model["type"] == 'gkde':
            self.noise_model["kernel"] = gaussian_kde(residuals.T+np.random.randn(*residuals.T.shape)*1e-8)
        if self.noise_model["type"] == 'sliding_window':
            windowsize = self.noise_model["windowsize"]
            windows = np.lib.stride_tricks.sliding_window_view(residuals, windowsize, axis=1)
            windowstd = np.zeros_like(residuals)
            regular_case_start = windowsize // 2
            regular_case_end = residuals.shape[1] - windowsize // 2
            windowstd[:, regular_case_start:regular_case_end] = np.std(windows, axis=-1)

            # add edge cases: first few epochs and last few epochs
            for e in range(1, regular_case_start):
                # these epochs have smaller windows since they're on the edge 
                special_windowsize = e * 2 + 1
                windowstd[:, e] = np.std(residuals[:, :special_windowsize], axis=-1)
                end_index = residuals.shape[1] - 1 - e
                windowstd[:, end_index] = np.std(residuals[:, -special_windowsize:], axis=-1)
            # for the first and last epoch, take the std of just the first two and last two epochs
            windowstd[:, 0] = np.std(residuals[:, :2], axis=-1)
            windowstd[:, -1] = np.std(residuals[:, -2:], axis=-1)

            # train an lgb model using features = U[:, :K] (i.e., principal components), labels = window stds
            self.noise_model["windowstd_model"] = RegressorChain(lgb.LGBMRegressor(random_state=101)).fit(u[:, :self.num_components], windowstd.copy())

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
        save_list = [self.model, self.ss, self.svd_s, self.svd_vh, self.num_components, self.noise_model]
        joblib.dump(save_list, os.path.join(self.log_dir, 'surrogate_model.model'))

    def load(self, model_path):
        # load a model

        if len(joblib.load(model_path)) == 5:
            # to be backwards compatible with nas-bench-301 models,
            # we can load a model without a noise model
            print('loading model without noise model')
            model, ss, svd_s, svd_vh, num_components = joblib.load(model_path)
            self.model = model
            self.ss = ss
            self.svd_s = svd_s
            self.svd_vh = svd_vh
            self.num_components = num_components

        else:
            # load a model with a noise model
            print('loading model with noise model')
            model, ss, svd_s, svd_vh, num_components, noise_model = joblib.load(model_path)
            self.model = model
            self.ss = ss
            self.svd_s = svd_s
            self.svd_vh = svd_vh
            self.num_components = num_components
            if type(noise_model) is not dict:
                # this is left in to be compatible with v0.5 surrogates
                self.noise_model["type"] = "gkde"
                self.noise_model["kernel"] = noise_model
            else:
                self.noise_model = noise_model

    def evaluate(self):
        X_test, y_test, _ = self.load_dataset(dataset_type='test', use_full_lc=True)

        test_pred = self.ss.inverse_transform(self.model.predict(X_test))\
            @ np.diag(self.svd_s[:self.num_components])@self.svd_vh[:self.num_components, :]

        y_test_final = y_test
        test_pred_final = np.array(test_pred)
        test_metrics = utils.evaluate_learning_curve_metrics(y_test_final, test_pred_final, prediction_is_first_arg=False)

        return test_metrics, test_pred, y_test

    def query(self, config_dict, search_space='darts', use_noise=False, components=False):
        # TODO: get rid of special case for darts
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
            noise_model_type = self.noise_model["type"]
            if noise_model_type == "gkde":
                noise = np.squeeze(self.noise_model["kernel"].resample(1))
                noise = np.clip(noise, -3, 3)  # clip outliers
                return ypred[0] + noise / 2
            elif noise_model_type == "sliding_window":
                windowsize = self.noise_model["windowsize"]
                pred_stds = np.maximum(1e-4, self.noise_model["windowstd_model"].predict(comp))
                noise = np.squeeze([np.random.normal() * std for std in pred_stds])
                return ypred[0] + noise

        else:
            return ypred[0]