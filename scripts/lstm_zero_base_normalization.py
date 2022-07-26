import torch
from torch.autograd import Variable
from tqdm.auto import trange


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
                y_pred = torch.squeeze(self.forward(x=X_torch))
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
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from vector_analytica_forecast import tools

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    df = pd.read_csv('../data/BTCUSDT_last_2000days_20220818.csv', parse_dates=['Open_time'], index_col='Open_time')
    print(df.tail())
    target_col = 'Close'
    window = 7
    train, test, X_train, X_test, y_train, y_test = tools.prepare_data(df=df, target_col=target_col, window_len=window,
                                                                       zero_base=True, test_size=0.1)
    print(X_train.shape, y_train.shape, y_train.min(), y_train.max())
    model = LSTM(input_dim=X_train.shape[-1], output_dim=1)
    model.fit(X=X_train, y=y_train, epochs=500)
    y_test_pred = np.squeeze(model(torch.as_tensor(X_test, dtype=torch.float32)).detach().numpy())
    print(f'mean_absolute_error (test): {mean_absolute_error(y_test, y_test_pred)}')

    idx_fin = 100
    fig, ax = plt.subplots(figsize=(16, 8))
    x_vec = np.arange(len(X_test))
    ax.fill_between(x_vec[:idx_fin], y_test_pred[:idx_fin] - np.std(y_train), y_test_pred[:idx_fin] + np.std(y_train),
                    alpha=0.5, label='+/- 1 std dev')
    ax.plot(x_vec[:idx_fin], y_test_pred[:idx_fin], label='prediction', color='blue')
    ax.plot(x_vec[:idx_fin], np.squeeze(y_test)[:idx_fin], 'ko', label='true')
    ax.legend()
    ax.legend(loc='best', fontsize=18)
    plt.savefig('../figs/lstm_scaled_results.png')
    plt.show()

    targets = test[target_col][window:]
    preds_ts = pd.Series(data=test[target_col].values[:-window] * (y_test_pred + 1),
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

    x_new = np.array(tools.normalise_zero_base(df.iloc[-window:]))
    y_new = model(torch.tensor(x_new).unsqueeze(0).float()).detach().numpy()
    y_new = np.squeeze(df[target_col].values[-window] * (y_new+1))
    print(f'LSTM Prediction for next day: {y_new} USDT')
