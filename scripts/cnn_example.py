import torch


class Cnn(torch.nn.Module):
    def __init__(self, dropout=0.5):
        super(Cnn, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_drop = torch.nn.Dropout2d(p=dropout)
        self.fc1 = torch.nn.Linear(1600, 100)  # 1600 = number channels * width * height
        self.fc2 = torch.nn.Linear(100, 10)
        self.fc1_drop = torch.nn.Dropout(p=dropout)

    def forward(self, x):
        x = torch.relu(torch.nn.functional.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.nn.functional.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        # flatten over channel, height and width = 1600
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))

        x = torch.relu(self.fc1_drop(self.fc1(x)))
        x = torch.softmax(self.fc2(x), dim=-1)
        return x