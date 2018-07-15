import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    '''
    batchnormalization adds extra parameters per each weight update
    disable for now
    '''

    def __init__(self, action_space, in_channels=3, num_features=4):

        super(Model, self).__init__()
        self.action_space = action_space

        self.main = nn.Sequential(
            # chnage in_channels to 3 or 1 if rgb or grayscale
            # in_channels, out_channels, kernel_size, stride=1, padding=0
            nn.Conv2d(3, num_features, 4, stride=2, padding=1, bias=False),
            nn.ELU(inplace=True),

            nn.Conv2d(num_features, num_features * 2, 4,
                      stride=2, padding=1, bias=False),
            nn.ELU(inplace=True),

            nn.Conv2d(num_features * 2, num_features * 4,
                      4, stride=2, padding=1, bias=False),
            nn.ELU(inplace=True),

            nn.Conv2d(num_features * 4, num_features * 8,
                      4, stride=2, padding=1, bias=False),
            nn.ELU(inplace=True),

            nn.Conv2d(num_features * 8, num_features *
                      16, 4, stride=2, padding=1, bias=False),
            nn.ELU(inplace=True),

            nn.Conv2d(num_features * 16, self.action_space, 4,
                      stride=1, padding=0, bias=False),
            nn.Softmax(1)
        )

    def forward(self, input):
        main = self.main(input)
        return main

    def count_parameters(self):
        return sum(p.numel() for p in self.main.parameters() if p.requires_grad)
