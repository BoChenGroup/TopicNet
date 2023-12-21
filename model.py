import torch.nn as nn
from torch.nn.parameter import Parameter
import torch
from utils import *
import numpy as np
import os
# from ../PGBN_tool import PGBN_sampler
import torch.nn.functional as F
import scipy.io as sio # mat

class GBN_model(nn.Module):
    def __init__(self, args):
        super(GBN_model, self).__init__()
        self.real_min = torch.tensor(1e-30)
        self.wei_shape_max = torch.tensor(10.0).float()
        self.wei_shape = torch.tensor(1e-1).float()

        self.vocab_size = args.vocab_size
        self.hidden_size = args.hidden_size

        self.topic_size = args.topic_size
        self.topic_size = [self.vocab_size] + self.topic_size
        self.layer_num = len(self.topic_size) - 1
        self.embed_size = args.embed_size

        self.bn_layer = nn.ModuleList([nn.BatchNorm1d(self.hidden_size) for i in range(self.layer_num)])

        h_encoder = [Conv1D(self.hidden_size, 1, self.vocab_size)]
        for i in range(self.layer_num - 1):
            h_encoder.append(Conv1D(self.hidden_size, 1, self.hidden_size))
        self.h_encoder = nn.ModuleList(h_encoder)

        shape_encoder = [Conv1D(self.topic_size[i + 1], 1, self.topic_size[i + 1] + self.hidden_size) for i in
                         range(self.layer_num - 1)]
        shape_encoder.append(Conv1D(self.topic_size[self.layer_num], 1, self.hidden_size))
        self.shape_encoder = nn.ModuleList(shape_encoder)

        scale_encoder = [Conv1D(self.topic_size[i + 1], 1, self.topic_size[i + 1] + self.hidden_size) for i in range(self.layer_num - 1)]
        scale_encoder.append(Conv1D(self.topic_size[self.layer_num], 1, self.hidden_size))
        self.scale_encoder = nn.ModuleList(scale_encoder)

        decoder = [GaussSoftmaxV3(self.topic_size[i], self.topic_size[i + 1], self.embed_size) for i in
                   range(self.layer_num)]
        self.decoder = nn.ModuleList(decoder)

        for t in range(self.layer_num - 1):
            self.decoder[t + 1].mu = self.decoder[t].mu_c
            self.decoder[t + 1].log_sigma = self.decoder[t].log_sigma_c

        graph_wordnet = sio.loadmat('./dataset/TopicTree_20ng.mat')
        self.graph = []
        for i in range(len(graph_wordnet['graph_topic_adj'][0])):
            self.graph.append(torch.from_numpy(graph_wordnet['graph_topic_adj'][0][i]).cuda())

        self.ob = 1.0

    def log_max(self, x):
        return torch.log(torch.max(x, self.real_min.cuda()))

    def reparameterize(self, Wei_shape_res, Wei_scale, Sample_num=1):
        # sample one
        eps = torch.cuda.FloatTensor(Sample_num, Wei_shape_res.shape[0], Wei_shape_res.shape[1]).uniform_(0, 1)
        theta = torch.unsqueeze(Wei_scale, axis=0).repeat(Sample_num, 1, 1) \
                * torch.pow(-log_max(1 - eps),  torch.unsqueeze(Wei_shape_res, axis=0).repeat(Sample_num, 1, 1))  #
        theta = torch.max(theta, self.real_min.cuda())
        return torch.mean(theta, dim=0, keepdim=False)

    def reparameterize2(self, Wei_shape_res, Wei_scale, Sample_num=50):
        # sample one
        eps = torch.cuda.FloatTensor(Sample_num, Wei_shape_res.shape[0], Wei_shape_res.shape[1]).uniform_(0, 1)
        theta = torch.unsqueeze(Wei_scale, axis=0).repeat(Sample_num, 1, 1) \
                * torch.pow(-log_max(1 - eps),  torch.unsqueeze(Wei_shape_res, axis=0).repeat(Sample_num, 1, 1))  #
        theta = torch.max(theta, self.real_min.cuda())
        return torch.mean(theta, dim=0, keepdim=False)

    def compute_loss(self, x, re_x):
        likelihood = torch.sum(x * self.log_max(re_x) - re_x - torch.lgamma(x + 1))
        return - likelihood / x.shape[1]

    def KL_GamWei(self, Gam_shape, Gam_scale, Wei_shape_res, Wei_scale):
        eulergamma = torch.tensor(0.5772, dtype=torch.float32)
        part1 = Gam_shape * self.log_max(Wei_scale) - eulergamma.cuda() * Gam_shape * Wei_shape_res + self.log_max(Wei_shape_res)
        part2 = - Gam_scale * Wei_scale * torch.exp(torch.lgamma(1 + Wei_shape_res))
        part3 = eulergamma.cuda() + 1 + Gam_shape * self.log_max(Gam_scale) - torch.lgamma(Gam_shape)
        KL = part1 + part2 + part3
        return - torch.sum(KL) / Wei_scale.shape[1]

    def forward(self, x, train_flag=True):

        hidden_list = [0] * self.layer_num
        theta = [0] * self.layer_num
        gam_scale = [0] * self.layer_num
        k_rec = [0] * self.layer_num
        l = [0] * self.layer_num
        l_tmp = [0] * self.layer_num
        phi_theta = [0] * self.layer_num
        loss = [0] * (self.layer_num + 1)
        likelihood = [0] * (self.layer_num + 1)
        KL_dis = [0] * self.layer_num
        graph_kl_loss = [0] * self.layer_num
        throshold = [0] * self.layer_num

        for t in range(self.layer_num):
            if t == 0:
                hidden = F.relu(self.bn_layer[t](self.h_encoder[t](x)))
            else:
                hidden = F.relu(self.bn_layer[t](self.h_encoder[t](hidden_list[t-1])))

            hidden_list[t] = hidden

        for t in range(self.layer_num-1, -1, -1):
            if t == self.layer_num - 1:
                k_rec_temp = torch.max(torch.nn.functional.softplus(self.shape_encoder[t](hidden_list[t])),
                                       self.real_min.cuda())      # k_rec = 1/k
                k_rec[t] = torch.min(k_rec_temp, self.wei_shape_max.cuda())

                l_tmp[t] = torch.max(torch.nn.functional.softplus(self.scale_encoder[t](hidden_list[t])), self.real_min.cuda())

                l[t] = l_tmp[t] / torch.exp(torch.lgamma(1 + k_rec[t]))

                if train_flag:
                    if t == 0:
                        theta[t] = self.reparameterize(k_rec[t].permute(1, 0), l[t].permute(1, 0))
                    else:
                        theta[t] = self.reparameterize2(k_rec[t].permute(1, 0), l[t].permute(1, 0))
                else:
                    theta[t] = l_tmp[t].permute(1, 0)
                phi_theta[t], KL_dis[t] = self.decoder[t](theta[t], t)

            else:
                hidden_phitheta = torch.cat((hidden_list[t], phi_theta[t+1].permute(1, 0).detach()), 1)

                k_rec_temp = torch.max(torch.nn.functional.softplus(self.shape_encoder[t](hidden_phitheta)),
                                       self.real_min.cuda())  # k_rec = 1/k
                k_rec[t] = torch.min(k_rec_temp, self.wei_shape_max.cuda())

                l_tmp[t] = torch.max(torch.nn.functional.softplus(self.scale_encoder[t](hidden_phitheta)), self.real_min.cuda())
                l[t] = l_tmp[t] / torch.exp(torch.lgamma(1 + k_rec[t]))

                if train_flag:
                    if t == 0:
                        theta[t] = self.reparameterize(k_rec[t].permute(1, 0), l[t].permute(1, 0))
                    else:
                        theta[t] = self.reparameterize2(k_rec[t].permute(1, 0), l[t].permute(1, 0))
                else:
                    theta[t] = l_tmp[t].permute(1, 0)
                phi_theta[t], KL_dis[t] = self.decoder[t](theta[t], t)

        for t in range(self.layer_num + 1):
            if t == 0:
                zero = torch.zeros_like(self.graph[t])
                one = torch.ones_like(self.graph[t])
                throshold[t] = torch.min(torch.where(self.graph[t] > 0, zero, one) * KL_dis[t], dim=0)[0] \
                                - torch.max(torch.where(self.graph[t] > 0, one, zero) * KL_dis[t], dim=0)[0]
                graph_kl_loss[t] = 10*torch.mean(torch.relu(10.0 - throshold[t]))

                likelihood[t] = self.compute_loss(x.permute(1, 0), phi_theta[t])
                loss[t] = likelihood[t] + graph_kl_loss[t]

            elif t == self.layer_num:
                loss[t] = self.KL_GamWei(torch.tensor(1.0, dtype=torch.float32).cuda(), torch.tensor(1.0, dtype=torch.float32).cuda(),
                                             k_rec[t - 1].permute(1, 0), l[t - 1].permute(1, 0))
                likelihood[t] = loss[t]

            else:
                zero = torch.zeros_like(self.graph[t])
                one = torch.ones_like(self.graph[t])
                throshold[t] = torch.min(torch.where(self.graph[t] > 0, zero, one) * KL_dis[t], dim=0)[0] \
                               - torch.max(torch.where(self.graph[t] > 0, one, zero) * KL_dis[t], dim=0)[0]
                graph_kl_loss[t] = 10*torch.mean(torch.relu(10.0 - throshold[t]))

                likelihood[t] = self.KL_GamWei(phi_theta[t], torch.tensor(1.0, dtype=torch.float32).cuda(),
                                              k_rec[t - 1].permute(1, 0), l[t - 1].permute(1, 0))

                loss[t] = likelihood[t] + graph_kl_loss[t]

        return phi_theta, theta, loss, likelihood, graph_kl_loss
