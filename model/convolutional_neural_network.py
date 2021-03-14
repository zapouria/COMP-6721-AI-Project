import torch.nn as nn

"""
in_channels -> rgb ~ 3
out_channels -> not fixed depends on system resource, better learning with high number 
padding = (kernel_size - 1) // 2
kernel_size equals to the size of the filter filter size 
"""


class convolutional_neural_network(nn.Module):
    def __init__(self):
        super(convolutional_neural_network, self).__init__()
        """
        kernel size of the filter 
        """
        # First block
        conv1_b1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        batchNorm1_b1 = nn.BatchNorm2d(32)
        leakyRLU1_b1 = nn.LeakyReLU(inplace=True)
        conv2_b1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        batchNorm2_b1 = nn.BatchNorm2d(32)
        leakyRLU2_b1 = nn.LeakyReLU(inplace=True)
        maxpool1_b1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second block
        conv1_b2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        batchNorm1_b2 = nn.BatchNorm2d(64)
        leakyRLU1_b2 = nn.LeakyReLU(inplace=True)
        conv2_b2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        batchNorm2_b2 = nn.BatchNorm2d(64)
        leakyRLU2_b2 = nn.LeakyReLU(inplace=True)
        maxpool1_b2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Third block
        conv1_b3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        batchNorm1_b3 = nn.BatchNorm2d(128)
        leakyRLU1_b3 = nn.LeakyReLU(inplace=True)
        conv2_b3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        batchNorm2_b3 = nn.BatchNorm2d(128)
        leakyRLU2_b3 = nn.LeakyReLU(inplace=True)
        maxpool1_b3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.clayer = nn.Sequential(
            conv1_b1,
            batchNorm1_b1,
            leakyRLU1_b1,
            conv2_b1,
            batchNorm2_b1,
            leakyRLU2_b1,
            maxpool1_b1,

            conv1_b2,
            batchNorm1_b2,
            leakyRLU1_b2,
            conv2_b2,
            batchNorm2_b2,
            leakyRLU2_b2,
            maxpool1_b2,

            conv1_b3,
            batchNorm1_b3,
            leakyRLU1_b3,
            conv2_b3,
            batchNorm2_b3,
            leakyRLU2_b3,
            maxpool1_b3
        )

        fdropout1 = nn.Dropout(p=0.1)
        fLinear1 = nn.Linear(64 * 32, 1000)
        fRelu1 = nn.ReLU(inplace=True)

        fLinear2 = nn.Linear(1000, 512)
        fRelu2 = nn.ReLU(inplace=True)
        fdropout2 = nn.Dropout(p=0.1)

        fLinear3 = nn.Linear(512, 128)
        fRelu3 = nn.ReLU(inplace=True)
        fdropout3 = nn.Dropout(p=0.1)

        fLinear4 = nn.Linear(128, 3)

        self.complete_connected_layer = nn.Sequential(
            fdropout1,
            fLinear1,
            fRelu1,
            fLinear2,
            fRelu2,
            fdropout2,
            fLinear3,
            fRelu3,
            fdropout3,
            fLinear4
        )

    def forward(self, data):
        data = self.clayer(data)
        size = data.size(0)
        data = data.view(size, -1)
        data = self.complete_connected_layer(data)
        return data
