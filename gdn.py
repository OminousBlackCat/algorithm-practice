import torch
import torch.utils.data
from torch.autograd import Function
import torchvision as tv
from torch import nn
import matplotlib.pyplot as plt


class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones(inputs.size()) * bound
        b = b.to(inputs.device)
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors

        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


lower_bound = LowerBound.apply


class GDN(nn.Module):
    """Generalized divisive normalization layer.
    y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]))
    """

    def __init__(self,
                 ch,
                 inverse=False,
                 beta_min=1e-6,
                 gamma_init=.1,
                 reparam_offset=2 ** -18):
        super(GDN, self).__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.reparam_offset_tensor = torch.FloatTensor([reparam_offset])
        self.register_buffer('reparam_offset', self.reparam_offset_tensor)

        self.build(ch)

    def build(self, ch):
        self.pedestal_tensor = self.reparam_offset ** 2
        self.register_buffer('pedestal', self.pedestal_tensor)
        self.beta_bound = (self.beta_min + self.reparam_offset ** 2) ** .5
        self.gamma_bound = self.reparam_offset

        # Create beta param
        beta = torch.sqrt(torch.ones(ch) + self.pedestal)
        self.beta = nn.Parameter(beta)
        self.register_parameter('beta', self.beta)

        # Create gamma param
        eye = torch.eye(ch)
        g = self.gamma_init * eye
        g = g + self.pedestal
        gamma = torch.sqrt(g)

        self.gamma = nn.Parameter(gamma)
        self.register_parameter('gamma', self.gamma)

    def forward(self, inputs):
        _, ch, _, _ = inputs.size()

        # Beta bound and reparam

        beta = lower_bound(self.beta, self.beta_bound)
        beta = beta ** 2 - self.pedestal

        # Gamma bound and reparam

        gamma = lower_bound(self.gamma, self.gamma_bound)
        gamma = gamma ** 2 - self.pedestal
        gamma = gamma.view(ch, ch, 1, 1)

        # Norm pool calc
        norm_ = nn.functional.conv2d(inputs ** 2, gamma, beta)
        norm_ = torch.sqrt(norm_)

        # Apply norm
        outputs = inputs / norm_

        return outputs


class IGDN(nn.Module):
    """Generalized divisive normalization layer.
    y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]))
    """

    def __init__(self,
                 ch,
                 inverse=False,
                 beta_min=1e-6,
                 gamma_init=.1,
                 reparam_offset=2 ** -18):
        super(IGDN, self).__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.reparam_offset_tensor = torch.FloatTensor([reparam_offset])
        self.register_buffer('reparam_offset', self.reparam_offset_tensor)

        self.build(ch)

    def build(self, ch):
        self.pedestal_tensor = self.reparam_offset ** 2
        self.register_buffer('pedestal', self.pedestal_tensor)
        self.beta_bound = (self.beta_min + self.reparam_offset ** 2) ** .5
        self.gamma_bound = self.reparam_offset

        # Create beta param
        beta = torch.sqrt(torch.ones(ch) + self.pedestal)
        self.beta = nn.Parameter(beta)
        self.register_parameter('beta', self.beta)

        # Create gamma param
        eye = torch.eye(ch)
        g = self.gamma_init * eye
        g = g + self.pedestal
        gamma = torch.sqrt(g)

        self.gamma = nn.Parameter(gamma)
        self.register_parameter('gamma', self.gamma)

    def forward(self, inputs):
        _, ch, _, _ = inputs.size()

        # Beta bound and reparam

        beta = lower_bound(self.beta, self.beta_bound)
        beta = beta ** 2 - self.pedestal

        # Gamma bound and reparam

        gamma = lower_bound(self.gamma, self.gamma_bound)
        gamma = gamma ** 2 - self.pedestal
        gamma = gamma.view(ch, ch, 1, 1)

        # Norm pool calc
        norm_ = nn.functional.conv2d(inputs ** 2, gamma, beta)
        norm_ = torch.sqrt(norm_)

        outputs = inputs * norm_

        return outputs


if __name__ == '__main__':
    img = tv.io.read_image("data/RSM20221017T133512_0004_HA.png", tv.io.ImageReadMode.GRAY)
    img = img[:, 78: 1078, 76: 1076]
    img = img[None, :]
    img = img / 255
    input_temp = img.detach().numpy()[0, 0]
    gdn = GDN(1)
    output = gdn.forward(img)
    print(output.shape)
    temp = output.detach().numpy()[0, 0]
    a = input_temp - temp
    plt.imshow(temp)
    plt.show()

