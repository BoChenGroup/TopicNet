import torch.nn as nn
from torch.nn.parameter import Parameter
import torch
import torch.nn.functional as F

real_min = torch.tensor(1e-30)

def log_max(x):
    return torch.log(torch.max(x, real_min.cuda()))

def KL_GamWei(Gam_shape, Gam_scale, Wei_shape, Wei_scale):
    eulergamma = torch.tensor(0.5772, dtype=torch.float32)

    part1 = eulergamma.cuda() * (1 - 1 / Wei_shape) + log_max(
        Wei_scale / Wei_shape) + 1 + Gam_shape * torch.log(Gam_scale)

    part2 = -torch.lgamma(Gam_shape) + (Gam_shape - 1) * (log_max(Wei_scale) - eulergamma.cuda() / Wei_shape)

    part3 = - Gam_scale * Wei_scale * torch.exp(torch.lgamma(1 + 1 / Wei_shape))

    KL = part1 + part2 + part3
    return KL

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

class Conv1D(nn.Module):
    def __init__(self, nf, rf, nx):
        super(Conv1D, self).__init__()
        self.rf = rf
        self.nf = nf
        if rf == 1:  # faster 1x1 conv
            w = torch.empty(nx, nf).cuda()
            nn.init.normal_(w, std=0.02)
            self.w = Parameter(w)
            self.b = Parameter(torch.zeros(nf).cuda())
        else:  # was used to train LM
            raise NotImplementedError

    def forward(self, x):
        if self.rf == 1:
            size_out = x.size()[:-1] + (self.nf,)
            x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w)
            x = x.view(*size_out)
        else:
            raise NotImplementedError
        return x

class DeepConv1D(nn.Module):
    def __init__(self, nf, rf, nx):
        super(DeepConv1D, self).__init__()
        self.rf = rf
        self.nf = nf
        if rf == 1:  # faster 1x1 conv
            w1 = torch.empty(nx, nf).cuda()
            nn.init.normal_(w1, std=0.02)
            self.w1 = Parameter(w1)
            self.b1 = Parameter(torch.zeros(nf).cuda())

            w2 = torch.empty(nf, nf).cuda()
            nn.init.normal_(w2, std=0.02)
            self.w2 = Parameter(w2)
            self.b2 = Parameter(torch.zeros(nf).cuda())

        else:  # was used to train LM
            raise NotImplementedError

    def forward(self, x):
        if self.rf == 1:
            size_out = x.size()[:-1] + (self.nf,)
            x = torch.addmm(self.b1, x.view(-1, x.size(-1)), self.w1)
            rx = x
            x = torch.nn.functional.relu(x)
            x = torch.addmm(self.b2, x.view(-1, x.size(-1)), self.w2)
            x = x.view(*size_out)
            x = x + rx
        else:
            raise NotImplementedError
        return x


class ResConv1D(nn.Module):
    def __init__(self, nf, rf, nx):
        super(ResConv1D, self).__init__()
        self.rf = rf
        self.nf = nf
        if rf == 1:  # faster 1x1 conv
            w1 = torch.empty(nx, nf).cuda()
            nn.init.normal_(w1, std=0.02)
            self.w1 = Parameter(w1)
            self.b1 = Parameter(torch.zeros(nf).cuda())

            w2 = torch.empty(nx, nf).cuda()
            nn.init.normal_(w2, std=0.02)
            self.w2 = Parameter(w2)
            self.b2 = Parameter(torch.zeros(nf).cuda())
        else:  # was used to train LM
            raise NotImplementedError

    def forward(self, x):
        if self.rf == 1:
            rx = x
            size_out = x.size()[:-1] + (self.nf,)
            x = torch.addmm(self.b1, x.view(-1, x.size(-1)), self.w1)
            x = torch.nn.functional.relu(x)
            x = torch.addmm(self.b2, x.view(-1, x.size(-1)), self.w2)
            x = x.view(*size_out)
            x = rx + x
        else:
            raise NotImplementedError
        return x


class Conv1DSoftmax(nn.Module):
    def __init__(self, voc_size, topic_size):
        super(Conv1DSoftmax, self).__init__()

        w = torch.empty(voc_size, topic_size).cuda()
        nn.init.normal_(w, std=0.02)
        self.w = Parameter(w)

    def forward(self, x):
        w = torch.softmax(self.w, dim=0)
        x = torch.mm(w, x.view(-1, x.size(-1)))
        return x

class Conv1DSoftmaxEtm(nn.Module):
    def __init__(self, voc_size, topic_size, emb_size, last_layer=None):
        super(Conv1DSoftmaxEtm, self).__init__()
        self.voc_size = voc_size
        self.topic_size = topic_size
        self.emb_size = emb_size

        if last_layer is None:
            w1 = torch.empty(self.voc_size, self.emb_size).cuda()
            nn.init.normal_(w1, std=0.02)
            self.rho = Parameter(w1)
        else:
            w1 = torch.empty(self.voc_size, self.emb_size).cuda()
            nn.init.normal_(w1, std=0.02)
            # self.rho = last_layer.alphas
            self.rho = Parameter(w1)

        w2 = torch.empty(self.topic_size, self.emb_size).cuda()
        nn.init.normal_(w2, std=0.02)
        self.alphas = Parameter(w2)

    def forward(self, x, t):
        if t == 0:
            w = torch.mm(self.rho, torch.transpose(self.alphas, 0, 1))
        else:
            w = torch.mm(self.rho, torch.transpose(self.alphas, 0, 1))

        w = torch.softmax(w, dim=0)
        x = torch.mm(w, x.view(-1, x.size(-1)))
        return x

import itertools
import math

class GaussSoftmaxV3(nn.Module):
    def __init__(self, voc_size, topic_size, emb_size):
        super(GaussSoftmaxV3, self).__init__()
        self.vocab_size = voc_size
        self.topic_size = topic_size
        self.embed_dim = emb_size
        self.sigma_min = 0.1
        self.sigma_max = 10.0
        self.C = 2.0
        self.lamda = 500.0

        # Model
        w1 = torch.empty(self.vocab_size, self.embed_dim).cuda()
        nn.init.normal_(w1, std=0.02)
        self.mu = Parameter(w1)

        w2 = torch.empty(self.vocab_size, self.embed_dim).cuda()
        nn.init.normal_(w2, std=0.02)
        self.log_sigma = Parameter(w2)

        w3 = torch.empty(self.topic_size, self.embed_dim).cuda()
        nn.init.normal_(w3, std=0.02)
        self.mu_c = Parameter(w3)

        w4 = torch.empty(self.topic_size, self.embed_dim).cuda()
        nn.init.normal_(w4, std=0.02)
        self.log_sigma_c = Parameter(w4)

    def el_energy(self, mu_v, mu_t, sigma_v, sigma_t):

        mu_v = mu_v.unsqueeze(1)  # vocab * 1  * embed
        sigma_v = sigma_v.unsqueeze(1)  # vocab * 1 * embed
        mu_t = mu_t.unsqueeze(0)  # 1 * topic  * embed
        sigma_t = sigma_t.unsqueeze(0)   # 1 * topic  * embed

        det_fac = torch.sum(torch.log(sigma_v + sigma_t), 2)    # V * K
        diff_mu = torch.sum((mu_v - mu_t) ** 2 / (sigma_v + sigma_t), 2)  # V * K
        return -0.5 * (det_fac + diff_mu + self.embed_dim * math.log(2 * math.pi))   # V * K

    def kl_energy(self, mu_v, mu_t, sigma_v, sigma_t):
        mu_v = mu_v.unsqueeze(1)   # vocab * 1  * embed
        sigma_v = torch.exp(sigma_v).unsqueeze(1)   # vocab * 1 * embed
        mu_t = mu_t.unsqueeze(0)   # 1 * topic  * embed
        sigma_t = torch.exp(sigma_t).unsqueeze(0)   # 1 * topic  * embed

        det_fac = torch.sum(torch.log(sigma_t), 2) - torch.sum(torch.log(sigma_v), 2)  # vocab * topic
        trace_fac = torch.sum(sigma_v / sigma_t, 2)  # vocab * * topic
        diff_mu = torch.sum((mu_v - mu_t) ** 2 / sigma_t, 2)  # vocab * * topic
        return 0.5 * (det_fac - self.embed_dim + trace_fac + diff_mu)  # vocab * topic

    def forward(self, x, t):
        for p in itertools.chain(self.log_sigma,
                                 self.log_sigma_c):
            p.data.clamp_(math.log(self.sigma_min), math.log(self.sigma_max))

        for p in itertools.chain(self.mu,
                                 self.mu_c):
            p.data.clamp_(-math.sqrt(self.C), math.sqrt(self.C))

        if t == 0:
            self.w = torch.softmax(self.el_energy(self.mu, self.mu_c, torch.exp((self.log_sigma)), torch.exp(self.log_sigma_c)), dim=0)
            kl_dis = self.kl_energy(self.mu, self.mu_c, self.log_sigma, self.log_sigma_c)
        else:
            self.w = torch.softmax(self.el_energy(self.mu.detach(), self.mu_c, torch.exp((self.log_sigma.detach())), torch.exp(self.log_sigma_c)), dim=0)
            kl_dis = self.kl_energy(self.mu, self.mu_c, self.log_sigma, self.log_sigma_c)

        x = torch.mm(self.w, x.view(-1, x.size(-1)))
        return x, kl_dis


def variable_para(shape, device='cuda'):
    w = torch.empty(shape, device=device)
    nn.init.normal_(w, std=0.02)
    return torch.tensor(w, requires_grad=True)


def save_checkpoint(state, filename, is_best):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print("=> Saving new checkpoint")
        torch.save(state, filename)
    else:
        print("=> Validation Accuracy did not improve")


