import torch
import torch.nn as nn



class Model(nn.Module):
    def __init__(self, num_classes=2, num_features=10, p=0.5):
        super(Model, self).__init__()

        # --- First layer: ---------------------------------------------------
        out_features_1 = 512
        self.fc1 = nn.Linear(in_features=num_features, out_features=out_features_1)
        self.bn1 = nn.BatchNorm1d(out_features_1)

        # --- Second layer: --------------------------------------------------
        out_features_2 = 256
        self.fc2 = nn.Linear(in_features=out_features_1, out_features=out_features_2)
        self.bn2 = nn.BatchNorm1d(out_features_2)

        # --- Third layer: ---------------------------------------------------
        out_features_3 = 128
        self.fc3 = nn.Linear(in_features=out_features_2, out_features=out_features_3)
        self.bn3 = nn.BatchNorm1d(out_features_3)

        # --- Fourth layer: ---------------------------------------------------
        self.fc4 = nn.Linear(in_features=out_features_3, out_features=num_classes)

        # --- The activation function: ---------------------------------------
        self.activation = nn.LeakyReLU(negative_slope=0.1)

        # --- The dropout: ---------------------------------------------------
        self.dropout = nn.Dropout(p=p)

        # --- Initializing the weights: --------------------------------------
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)

    def forward(self, x):

        # --- First layer: ---------------------------------------------------
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.bn1(x)

        # --- Second layer: ---------------------------------------------------
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.bn2(x)

        # --- Third layer: ---------------------------------------------------
        x = self.fc3(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.bn3(x)

        # --- Fourth layer: --------------------------------------------------
        x = self.fc4(x)

        return x







