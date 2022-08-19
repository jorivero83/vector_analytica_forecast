import numpy as np
import pandas as pd
from tqdm.auto import trange
import torch
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
import torch.nn.functional as fn


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

    def fit(self, X, y, epochs=500):
        X_torch = Variable(torch.Tensor(X).float())
        y_torch = Variable(torch.Tensor(y).float())
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005)
        loss_fn = torch.nn.MSELoss()
        history = []
        with trange(1, epochs + 1, desc='Training', leave=True) as steps:
            for k in steps:
                y_pred = self.forward(x=X_torch)
                loss = loss_fn(y_pred, y_torch)
                status = {'loss': loss.item()}
                history.append(status['loss'])
                steps.set_postfix(status)
                optimizer.zero_grad()  # Zero gradients
                loss.backward()  # Gradients
                optimizer.step()  # Update

        return history


if __name__ == '__main__':
    # from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.metrics import mean_absolute_error
    import pandas as pd
    import matplotlib.pyplot as plt
    from vector_analytica_forecast import tools

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    df = pd.read_csv('../data/BTCUSDT_last_2000days_20220818.csv', parse_dates=['Open_time'], index_col='Open_time')
    print(df.head())

    target_col = 'Close'
    window = 7

    # train test split
    train_data, test_data = tools.train_test_split(df, test_size=0.1)

    x_scaler = StandardScaler()
    x_scaler.fit(train_data)

    # extract targets
    y_train = train_data[target_col][window:].values.reshape(-1, 1)
    y_test = test_data[target_col][window:].values.reshape(-1, 1)

    # extract window data
    X_train = tools.extract_window_data(x_scaler.transform(train_data), window, False)
    X_test = tools.extract_window_data(x_scaler.transform(test_data), window, False)

    # train, test, X_train, X_test, y_train, y_test = tools.prepare_data(df=df, target_col=target_col, window_len=window,
    #                                                                    zero_base=False, test_size=0.1)

    print(X_train.shape, y_train.shape, y_train.min(), y_train.max())

    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_scaler.fit(y_train)
    y_train_scaled = y_scaler.transform(y_train)

    model = LSTM(input_dim=X_train.shape[-1], output_dim=1)
    model.fit(X=X_train, y=y_train_scaled, epochs=2000)

    y_test_pred = np.squeeze(model(torch.as_tensor(X_test, dtype=torch.float32)).detach().numpy())

    # assert 1==3, "Fin"
    y_test_scaled = y_scaler.transform(y_test)
    print(f'mean_absolute_error (test): {mean_absolute_error(y_test_scaled, y_test_pred)}')

    # y_scaler = StandardScaler().fit(y_train)
    # y_train_torch = torch.from_numpy(y_scaler.transform(y_train)).float()
    # y_test_torch = torch.from_numpy(y_scaler.transform(y_test)).float()

    idx_fin = 100
    fig, ax = plt.subplots(figsize=(16, 8))
    x_vec = np.arange(len(X_test))
    ax.fill_between(x_vec[:idx_fin], y_test_pred[:idx_fin] - np.std(y_train_scaled),
                    y_test_pred[:idx_fin] + np.std(y_train_scaled),
                    alpha=0.5, label='+/- 1 std dev')
    ax.plot(x_vec[:idx_fin], y_test_pred[:idx_fin], label='prediction', color='blue')
    ax.plot(x_vec[:idx_fin], np.squeeze(y_test_scaled)[:idx_fin], 'ko', label='true')
    ax.legend(loc='best', fontsize=18)
    plt.savefig('../figs/lstm_scaled_results.png')
    plt.show()

    targets = test_data[target_col][window:]
    # preds_ts = pd.Series(data=test_data[target_col].values[:-window] * (y_test_pred + 1),
    #                      index=targets.index)
    preds_ts = pd.Series(data=np.squeeze(y_scaler.inverse_transform(y_test_pred.reshape(-1, 1))),
                         index=targets.index)

    n_points = 200
    fig, ax = plt.subplots(1, figsize=(16, 9))
    ax.plot(targets[-n_points:][:-1], label='real', linewidth=3)
    ax.plot(preds_ts[-n_points:].shift(-1), label='pred', linewidth=3)
    ax.set_ylabel('price [USD]', fontsize=14)
    ax.set_title('BTC prediction using LSTM with pytorch', fontsize=18)
    ax.legend(loc='best', fontsize=18)
    plt.savefig('../figs/lstm_results.png')
    plt.show()

