import torch
import torch.nn as nn

class MyCNN(nn.Module):

    def __init__(self, num_conv_layers: int, num_filters: int, dropout: float = 0.5):
        super().__init__()
        self.num_conv_layers = num_conv_layers
        self.num_filters = num_filters
        self.dropout = dropout
        self.build_network()

    def build_network(self):

        layers = []
        input_dim = 3

        pool_layer = 0
        max_pool = 2

        for i in range(self.num_conv_layers):
            layers.append(
                nn.Conv2d(
                    input_dim,
                    self.num_filters,
                    kernel_size=3,
                    padding=1
                )
            )
            layers.append(nn.BatchNorm2d(self.num_filters))
            layers.append(nn.ReLU())

            if pool_layer < max_pool:
                layers.append(nn.MaxPool2d(kernel_size=2))
                pool_layer += 1

            input_dim = self.num_filters

        layers.append(nn.Conv2d(self.num_filters, 32, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(32))
        layers.append(nn.ReLU())
        layers.append(nn.Flatten())

        out_size = int((32 / (2 ** pool_layer)) ** 2 * 32)

        self.features = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.Linear(out_size, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_network(config: dict) -> nn.Module:
    return MyCNN(
        num_conv_layers=config["num_conv_layers"],
        num_filters=config["num_filters"],
        dropout=config.get("dropout", 0.5)
    )








