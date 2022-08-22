import torch
from skorch import NeuralNetRegressor
from skorch.callbacks import EarlyStopping


class LSTM(torch.nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=2, output_dim=1, dropout=0.2):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
                                  batch_first=True, dropout=dropout, bidirectional=False)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out


def prepare_data(df, target_col, window_len=10, zero_base=True):
    X_train = tools.extract_window_data(df, window_len, zero_base)
    y_train = df[target_col][window_len:].values
    if zero_base:
        y_train = y_train / df[target_col][:-window_len].values - 1
    return X_train, y_train


if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from vector_analytica_forecast import tools
    from sklearn.metrics import mean_absolute_error
    from tqdm import tqdm

    df = pd.read_csv('../data/BTCUSDT_last_2000days_20220818.csv', parse_dates=['Open_time'], index_col='Open_time')
    # df = pd.read_csv('data/BTCUSDT_last_2000days_20220818.csv', parse_dates=['Open_time'], index_col='Open_time')
    print(df.tail())

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    target_col = 'Close'
    window = 7
    train, test, X_train, X_test, y_train, y_test = tools.prepare_data(df=df, target_col=target_col, window_len=window,
                                                                       zero_base=True, test_size=0.1)
    print(X_train.shape, y_train.shape, y_train.min(), y_train.max())
    X_train_torch = torch.from_numpy(X_train).float()
    y_train_torch = torch.from_numpy(y_train.reshape(-1, 1)).float()
    model = NeuralNetRegressor(module=LSTM(input_dim=X_train.shape[-1], output_dim=1),
                               criterion=torch.nn.HuberLoss,
                               max_epochs=500, batch_size=128, lr=0.001, optimizer=torch.optim.Adam,
                               callbacks=[('early_stop', EarlyStopping(patience=25))], verbose=0)
    model.fit(X=X_train_torch, y=y_train_torch)

    # Get predictions
    # final = train.iloc[-window:]
    # x_new = np.array(tools.normalise_zero_base(final))
    # y_new = model.predict(torch.tensor(x_new).unsqueeze(0).float())
    # y_new = np.squeeze(final[target_col].values[0] * (y_new + 1))
    # print(f'o(t+1): {test[target_col].values[0]}')
    # print(f'y(t+1): {y_new}')
    # o(t+1): 39974.44
    # y(t+1): 39944.92578125

    # pd.concat([self.S_train, S], ignore_index=False)
    n_points = 20
    last_x = train[-window:]
    last_history = train
    preds = np.zeros(n_points)
    for k in tqdm(range(n_points), total=n_points, desc='Get prediction and update the model'):

        # get next day prediction
        x_new = np.array(tools.normalise_zero_base(last_x))
        y_new = model.predict(torch.tensor(x_new).unsqueeze(0).float())
        y_new = np.squeeze(last_x[target_col].values[0] * (y_new + 1))
        preds[k] = y_new

        # refit
        X_train, y_train = prepare_data(last_history, target_col, window_len=window, zero_base=True)
        X_train_torch = torch.from_numpy(X_train).float()
        y_train_torch = torch.from_numpy(y_train.reshape(-1, 1)).float()
        model.fit(X=X_train_torch, y=y_train_torch)

        # update data for next iteration
        last_x = pd.concat([last_x[1:], test[k:k+1]], ignore_index=False)
        last_history = pd.concat([last_history[1:], test[k:k + 1]], ignore_index=False)

    preds_ts = pd.Series(data=preds, index=test[:n_points].index)
    target_ts = test[target_col][:n_points]

    fig, ax = plt.subplots(1, figsize=(16, 9))
    ax.plot(target_ts, label='real', linewidth=3)
    ax.plot(preds_ts, label='pred', linewidth=3)
    ax.set_ylabel('Price [USDT]', fontsize=14)
    ax.set_title('BTC prediction using LSTM with pytorch', fontsize=18)
    ax.legend(loc='best', fontsize=18)
    # plt.savefig('../figs/lstm_results.png')
    plt.show()

    print(f'mean_absolute_error (test): {mean_absolute_error(target_ts, preds_ts)}')

    # y_test_pred = np.squeeze(model.predict(X=torch.as_tensor(X_test, dtype=torch.float32)))
    # print(f'mean_absolute_error (test): {mean_absolute_error(y_test, y_test_pred)}')
    #
    # targets = test[target_col][window:]
    # preds_ts = pd.Series(data=test[target_col].values[:-window] * (y_test_pred + 1),
    #                      index=targets.index)
    #
    # n_points = 20
    # fig, ax = plt.subplots(1, figsize=(16, 9))
    # ax.plot(targets[-n_points:][:-1], label='real', linewidth=3)
    # ax.plot(preds_ts[-n_points:].shift(-1), label='pred', linewidth=3)
    # # ax.plot(targets[-n_points:], label='real', linewidth=3)
    # # ax.plot(preds_ts[-n_points:], label='pred', linewidth=3)
    # ax.set_ylabel('price [USD]', fontsize=14)
    # ax.set_title('BTC prediction using LSTM with pytorch', fontsize=18)
    # ax.legend(loc='best', fontsize=18)
    # # plt.savefig('../figs/lstm_results.png')
    # plt.show()

    # mean_absolute_error(test): 0.03463914442557751
    # mean_absolute_error (test): 0.027315723472412533
