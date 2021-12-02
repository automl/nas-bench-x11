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

import utils
from surrogate_model import SurrogateModel


class VAE(nn.Module):

    def __init__(self, input_shape, hidden_dim, num_enc_layers, num_dec_layers, z_dim, dropout_p):
        super().__init__()
        self.enc_in_layer = nn.Linear(input_shape[1], hidden_dim)
        self.enc_fc_layers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_enc_layers)]
        self.enc_bn_layers = [nn.BatchNorm1d(hidden_dim) for _ in range(num_enc_layers)]
        self.enc_drop_layers = [nn.Dropout(dropout_p) for _ in range(num_enc_layers)]
        self.enc_out_layer = nn.Linear(hidden_dim, z_dim*2)

        self.dec_in_layer = nn.Linear(z_dim, hidden_dim)
        self.dec_fc_layers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_dec_layers)]
        self.dec_bn_layers = [nn.BatchNorm1d(hidden_dim) for _ in range(num_dec_layers)]
        self.dec_drop_layers = [nn.Dropout(dropout_p) for _ in range(num_dec_layers)]
        self.dec_out_layer = nn.Linear(hidden_dim, input_shape[1])

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn(*mu.size())
        z = mu + std * eps
        return z

    def encode(self, tensor):
        enc_tensor = self.enc_in_layer(tensor)
        for fc, bn, drop in zip(self.enc_fc_layers, self.enc_bn_layers, self.enc_drop_layers):
            enc_tensor = self.LeakyReLU(fc(enc_tensor))
            enc_tensor = drop(bn(enc_tensor))
        mu, logvar = torch.chunk(self.enc_out_layer(enc_tensor), 2, dim=1)
        return mu, logvar

    def decode(self, tensor):
        dec_tensor = self.dec_in_layer(tensor)
        for fc, bn, drop in zip(self.dec_fc_layers, self.dec_bn_layers, self.dec_drop_layers):
            dec_tensor = F.relu(fc(dec_tensor))
            dec_tensor = drop(dec_tensor)
        return self.dec_out_layer(dec_tensor)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        output = self.decode(z)
        return output, mu, logvar


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


class VAENNModel(SurrogateModel):
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
        X_train, y_train = torch.Tensor(X_train), torch.Tensor(y_train)

        param_config = self.parse_param_config()
        param_config["seed"] = self.seed

        self.num_components = param_config["num_components"]
        self.vae = VAE(y_train.shape, param_config["vae_hidden_dim"],
                       param_config["enc_num_layers"], param_config["dec_num_layers"],
                       self.num_components, param_config["vae_dropout_p"])
        vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=param_config["vae_learning_rate"])

        def vae_loss_fn(x, recon_x, mu, logvar):
            MSE = nn.MSELoss(reduction='sum')(recon_x, x)
            KLD = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
            return MSE + KLD

        self.model = NNSurrogateModel(X_train.shape, param_config["nn_hidden_dim"], param_config["nn_num_layers"], self.num_components, param_config["nn_dropout_p"])
        nn_criterion = nn.MSELoss()
        nn_optimizer = optim.Adam(self.model.parameters(), lr=param_config["nn_learning_rate"])

        self.ss = StandardScaler()
        pX, py = torch.tensor(X_train, dtype=torch.float32), torch.tensor(self.ss.fit_transform(y_train), dtype=torch.float32)
        train_dataset = TensorDataset(pX, py)
        train_dataloader = DataLoader(train_dataset, batch_size=32)

        self.vae.train()
        num_epochs = param_config["vae_num_epochs"]
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, (_, y) in enumerate(train_dataloader, 0):
                vae_optimizer.zero_grad()
                outputs = self.vae(y)
                loss = vae_loss_fn(y, *outputs)
                loss.backward()
                vae_optimizer.step()

                running_loss += loss.item()
                if not i % 500:
                    print(f'VAE [{epoch+1}, {i+1}] loss: {running_loss / 1000:.3f}')
                    running_loss = 0.0

        self.vae.eval()
        self.model.train()
        num_epochs = param_config["nn_num_epochs"]
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, (X, y) in enumerate(train_dataloader, 0):
                nn_optimizer.zero_grad()
                with torch.no_grad():
                    _, z, _ = self.vae(y)
                loss = nn_criterion(self.model(X), z)
                loss.backward()
                nn_optimizer.step()

                running_loss += loss.item()
                if not i % 500:
                    print(f'NN [{epoch+1}, {i+1}] loss: {running_loss / 1000:.3f}')
                    running_loss = 0.0

        train_pred, val_pred = self.eval(X_train), self.eval(X_val)

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

    def eval(self, X):
        self.vae.eval()
        self.model.eval()
        with torch.no_grad():
            return self.ss.inverse_transform(self.vae.decode(self.model(torch.tensor(X, dtype=torch.float32))))

    def test(self):
        X_test, y_test, _ = self.load_dataset(dataset_type='test', use_full_lc=True)

        test_pred = self.eval(X_test)

        y_test_final = y_test
        test_pred_final = np.array(test_pred)
        test_metrics = utils.evaluate_learning_curve_metrics(y_test_final, test_pred_final, prediction_is_first_arg=False)

        # todo: compute metrics for the full learning curve
        logging.info('test metrics %s', test_metrics)

        return test_metrics

    def validate(self):
        X_val, y_val, _ = self.load_dataset(dataset_type='val', use_full_lc=True)

        val_pred = self.eval(X_val)

        y_val_final = y_val
        val_pred_final = np.array(val_pred)
        valid_metrics = utils.evaluate_learning_curve_metrics(y_val_final, val_pred_final, prediction_is_first_arg=False)

        # todo: compute metrics for the full learning curve
        logging.info('test metrics %s', valid_metrics)

        return valid_metrics

    def save(self):
        save_list = [self.model, self.ss, self.vae]
        joblib.dump(save_list, os.path.join(self.log_dir, 'surrogate_model.model'))

    def load(self, model_path):
        model, ss, vae = joblib.load(model_path)
        self.model = model
        self.ss = ss
        self.vae = vae

    def evaluate(self, result_paths):
        X_test, y_test, _ = self.load_dataset(dataset_type='test', use_full_lc=True)

        test_pred = self.ss.inverse_transform(self.vae.decode(self.model.predict(X_test)))

        y_test_final = y_test
        test_pred_final = np.array(test_pred)
        test_metrics = utils.evaluate_learning_curve_metrics(y_test_final, test_pred_final, prediction_is_first_arg=False)

        # todo: compute metrics for the full learning curve
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
            print(X.shape)

        ypred = self.eval(X)
        return ypred[0]
