import paddle as torch
import numpy as np
from paddle import nn

class FrozenBatchNorm2d(nn.Layer):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    """

    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.register_buffer("weight", torch.to_tensor(np.ones(num_features)))
        self.register_buffer("bias", torch.to_tensor(np.zeros(num_features)))
        self.register_buffer("running_mean", torch.to_tensor(np.zeros(num_features)))
        self.register_buffer("running_var", torch.to_tensor(np.ones(num_features)) - eps)

    def forward(self, x):
        scale = self.weight * (self.running_var + self.eps).rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape((1, -1, 1, 1))
        bias = bias.reshape((1, -1, 1, 1))
        return x * scale + bias

    def __repr__(self):
        return "FrozenBatchNorm2d(num_features={}, eps={})".format(self.num_features, self.eps)

