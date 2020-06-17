import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
class R4Conv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        # _ConvNd __init__ 中初始化 weight
        super(R4Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

    def r4conv2d_forward(self, input, weight):
        """

        :param input:
        :param weight: (out_channels, in_channels, kernel_size, kernel_size)
        :return:
        """
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)

            return torch.cat([F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            w, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
                       for w in [weight, weight.flip([-2]).T, weight.flip([-2,-1]), weight.flip([-1]).T]]
                      , dim=-3)

            # return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
            #                 weight, self.bias, self.stride,
            #                 _pair(0), self.dilation, self.groups)

        return torch.cat([F.conv2d(input, w, self.bias, self.stride,
                                   self.padding, self.dilation, self.groups)
                          for w in [weight, weight.flip([-2]).transpose(-2,-1), weight.flip([-2,-1]), weight.flip([-1]).transpose(-2,-1)]]
                         , dim=-3)

        # return F.conv2d(input, weight, self.bias, self.stride,
        #                 self.padding, self.dilation, self.groups)

    def forward(self, input):
        return self.r4conv2d_forward(input, self.weight)








if __name__ == '__main__':
    c = R4Conv2d(3, 10, 5)
    print(c.weight.shape)
    ca = torch.rand(2,3,4)
    print(ca.transpose(0, 1).shape)
    print(c.weight.grad)

    input = torch.ones(1,3, 28, 28)
    o = c(input)
    print(o.shape)