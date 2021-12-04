import logging
import os
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

from nas_bench_x11.utils import utils
from nas_bench_x11.surrogate_model import SurrogateModel


class NNSurrogateModel(nn.Module):

    def __init__(self, input_shape, hidden_dim, num_layers, out_dim, dropout_p):
        super().__init__()
        self.inlayer = nn.Linear(input_shape[1], hidden_dim)
        self.fclayers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        self.bnlayers = [nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)]
        self.droplayers = [nn.Dropout(dropout_p) for _ in range(num_layers)]
        self.outlayer = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.inlayer(x))
        for fc, bn, drop in zip(self.fclayers, self.bnlayers, self.droplayers):
            x = F.relu(fc(x))
            x = drop(x)
        return self.outlayer(x)


class SVDNNModel(SurrogateModel):
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

        X_train, y_train, _ = self.load_dataset(dataset_type='train', use_full_lc=True)
        X_val, y_val, _ = self.load_dataset(dataset_type='val', use_full_lc=True)

        pX = torch.tensor(X_train, dtype=torch.float32)

        param_config = self.parse_param_config()
        param_config["seed"] = self.seed

        self.num_components = param_config["num_components"]
        self.ss = StandardScaler()
        u, s, vh = np.linalg.svd(y_train, full_matrices=False)

        self.svd_s = s
        self.svd_vh = vh
        self.model = NNSurrogateModel(pX.shape, param_config["hidden_dim"], param_config["num_layers"], self.num_components, param_config["dropout_p"])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=param_config["learning_rate"])

        # the labels are the first n components of the SVD on the training data
        labels = u[:, :self.num_components].copy()
        fitted_labels = self.ss.fit_transform(labels)

        py = torch.tensor(fitted_labels, dtype=torch.float32)
        train_dataset = TensorDataset(pX, py)
        train_dataloader = DataLoader(train_dataset, batch_size=32)

        self.model.train()
        for epoch in range(param_config["num_epochs"]):
            running_loss = 0.0
            for i, (x, y) in enumerate(train_dataloader, 0):
                optimizer.zero_grad()
                outputs = self.model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if not i % 500:
                    print(f'[{epoch+1}, {i+1}] loss: {running_loss / 1000:.3f}')
                    running_loss = 0.0

        self.model.eval()
        with torch.no_grad():
            train_pred = self.ss.inverse_transform(self.model(torch.tensor(X_train, dtype=torch.float32)))\
                @ np.diag(self.svd_s[:self.num_components])@self.svd_vh[:self.num_components, :]

            val_pred = self.ss.inverse_transform(self.model(torch.tensor(X_val, dtype=torch.float32)))\
                @ np.diag(self.svd_s[:self.num_components])@self.svd_vh[:self.num_components, :]

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

        self.model.eval()
        with torch.no_grad():
            test_pred = self.ss.inverse_transform(self.model(torch.tensor(X_test, dtype=torch.float32)))\
                @ np.diag(self.svd_s[:self.num_components])@self.svd_vh[:self.num_components, :]

        y_test_final = y_test
        test_pred_final = np.array(test_pred)
        test_metrics = utils.evaluate_learning_curve_metrics(y_test_final, test_pred_final, prediction_is_first_arg=False)

        logging.info('test metrics %s', test_metrics)

        return test_metrics

    def validate(self):
        X_val, y_val, _ = self.load_dataset(dataset_type='val', use_full_lc=True)

        self.model.eval()
        with torch.no_grad():
            val_pred = self.ss.inverse_transform(self.model(torch.tensor(X_val, dtype=torch.float32)))\
                @ np.diag(self.svd_s[:self.num_components])@self.svd_vh[:self.num_components, :]

        y_val_final = y_val
        val_pred_final = np.array(val_pred)
        valid_metrics = utils.evaluate_learning_curve_metrics(y_val_final, val_pred_final, prediction_is_first_arg=False)

        logging.info('test metrics %s', valid_metrics)

        return valid_metrics

    def save(self):
        save_list = [self.model, self.ss, self.svd_s, self.svd_vh, self.num_components]
        joblib.dump(save_list, os.path.join(self.log_dir, 'surrogate_model.model'))

    def load(self, model_path):
        model, ss, svd_s, svd_vh, num_components = joblib.load(model_path)
        self.model = model
        self.ss = ss
        self.svd_s = svd_s
        self.svd_vh = svd_vh
        self.num_components = num_components

    def evaluate(self, result_paths):
        X_test, y_test, _ = self.load_dataset(dataset_type='test', use_full_lc=True)

        self.model.eval()
        with torch.no_grad():
            test_pred = self.ss.inverse_transform(self.model(torch.tensor(X_test, dtype=torch.float32)))\
                @ np.diag(self.svd_s[:self.num_components])@self.svd_vh[:self.num_components, :]

        y_test_final = y_test
        test_pred_final = np.array(test_pred)
        test_metrics = utils.evaluate_learning_curve_metrics(y_test_final, test_pred_final, prediction_is_first_arg=False)

        return test_metrics, test_pred, y_test

    def query(self, config_dict, search_space='darts', epoch=98, use_noise=False):
        if search_space == 'darts':
            config_space_instance = self.config_loader.query_config_dict(config_dict)
            X = config_space_instance.get_array().reshape(1, -1)
            idx = np.isnan(X)
            X[idx] = -1
            X = X.reshape(1, -1)
        else:
            X = np.array([config_dict])

        self.model.eval()
        with torch.no_grad():
            ypred = self.ss.inverse_transform(self.model(torch.tensor(X, dtype=torch.float32)))\
                @ np.diag(self.svd_s[:self.num_components])@self.svd_vh[:self.num_components, :]
        return ypred[0]
