
import torch
import torch.nn as nn


class SmartAirCNN(nn.Module):

    def __init__(self):
        super(SmartAirCNN, self).__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()
        )

    def forward(self, x):

        x = self.features(x)
        x = self.regressor(x)

        return x.squeeze(1)


if __name__ == "__main__":

    model = SmartAirCNN()

    test_input = torch.randn(2, 3, 320, 320)

    output = model(test_input)

    print("SMARTAIR CNN MODEL")
    print("Model created successfully!")
    print("Input shape:", test_input.shape)
    print("Output shape:", output.shape)
    print("Sample predictions:", output.detach().numpy())
    print("Total parameters:", sum(p.numel() for p in model.parameters()))
