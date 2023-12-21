import torch.nn as nn
from torch.nn.parameter import Parameter
import torch
from model import *
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn import preprocessing
from sklearn.cluster import KMeans
import numpy as np
from utils import *

class GBN_trainer:
    def __init__(self, args, voc_path='voc.txt'):
        self.args = args
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.save_path = args.save_path
        self.epochs = args.epochs
        self.voc = self.get_voc(voc_path)
        self.layer_num = len(args.topic_size)

        self.model = GBN_model(args)
        self.optimizer = torch.optim.Adam([{'params': self.model.h_encoder.parameters()},
                                           {'params': self.model.shape_encoder.parameters()},
                                           {'params': self.model.scale_encoder.parameters()}],
                                          lr=self.lr, weight_decay=self.weight_decay)

        self.decoder_optimizer = torch.optim.Adam(self.model.decoder.parameters(),
                                                  lr=self.lr, weight_decay=self.weight_decay)

    def train(self, train_data_loader, test_data_loader):

        for epoch in range(self.epochs):
            for t in range(self.layer_num - 1):
                self.model.decoder[t + 1].mu = self.model.decoder[t].mu_c
                self.model.decoder[t + 1].log_sigma = self.model.decoder[t].log_sigma_c

            self.model.cuda()

            loss_t = [0] * (self.layer_num + 1)
            likelihood_t = [0] * (self.layer_num + 1)
            graph_kl_loss_t = [0] * (self.layer_num + 1)
            num_data = len(train_data_loader)

            for i, (train_data, train_label) in enumerate(train_data_loader):

                self.model.h_encoder.train()
                self.model.shape_encoder.train()
                self.model.scale_encoder.train()
                self.model.decoder.eval()

                train_data = torch.tensor(train_data, dtype=torch.float).cuda()
                train_label = torch.tensor(train_label, dtype=torch.long).cuda()

                re_x, theta, loss_list, likelihood, graph_kl_loss = self.model(train_data)

                for t in range(self.layer_num + 1):
                    if t == 0:
                        loss_list[t].backward(retain_graph=True)
                        loss_t[t] += loss_list[t].item() / num_data
                        likelihood_t[t] += likelihood[t].item() / num_data
                        graph_kl_loss_t[t] += graph_kl_loss[t].item()/num_data

                    elif t < self.layer_num:
                        (1 * loss_list[t]).backward(retain_graph=True)
                        loss_t[t] += loss_list[t].item() / num_data
                        likelihood_t[t] += likelihood[t].item() / num_data
                        graph_kl_loss_t[t] += graph_kl_loss[t].item() / num_data

                    else:
                        (1 * loss_list[t]).backward(retain_graph=True)
                        loss_t[t] += loss_list[t].item() / num_data
                        likelihood_t[t] += likelihood[t].item() / num_data


                for para in self.model.parameters():
                    flag = torch.sum(torch.isnan(para))

                if (flag == 0):
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                self.model.h_encoder.eval()
                self.model.shape_encoder.eval()
                self.model.scale_encoder.eval()
                self.model.decoder.train()

                re_x, theta, loss_list, likelihood, graph_kl_loss = self.model(train_data)

                for t in range(self.layer_num + 1):
                    if t == 0:
                        loss_list[t].backward(retain_graph=True)
                        loss_t[t] += loss_list[t].item() / num_data
                        likelihood_t[t] += likelihood[t].item() / num_data
                        graph_kl_loss_t[t] += graph_kl_loss[t].item() / num_data

                    elif t < self.layer_num:
                        (1 * loss_list[t]).backward(retain_graph=True)
                        loss_t[t] += loss_list[t].item() / num_data
                        likelihood_t[t] += likelihood[t].item() / num_data
                        graph_kl_loss_t[t] += graph_kl_loss[t].item() / num_data
                    else:
                        (1 * loss_list[t]).backward(retain_graph=True)
                        loss_t[t] += loss_list[t].item() / num_data
                        likelihood_t[t] += likelihood[t].item() / num_data

                for para in self.model.parameters():
                    flag = torch.sum(torch.isnan(para))

                if (flag == 0):
                    nn.utils.clip_grad_norm_(self.model.decoder.parameters(), max_norm=20, norm_type=2)
                    self.decoder_optimizer.step()
                    self.decoder_optimizer.zero_grad()

            if epoch % 1 == 0:
                for t in range(self.layer_num + 1):
                    print('epoch {}|{}, layer {}|{}, loss: {}, likelihood: {}, lb: {}, graph_kl_loss: {}'.format(epoch, self.epochs, t,
                                                                                              self.layer_num,
                                                                                              loss_t[t]/2,
                                                                                              likelihood_t[t]/2,
                                                                                              loss_t[t]/2,
                                                                                              graph_kl_loss_t[t]/2))
                self.vis_txt()

    def get_voc(self, voc_path):
        if type(voc_path) == 'str':
            voc = []
            with open(voc_path) as f:
                lines = f.readlines()
            for line in lines:
                voc.append(line.strip())
            return voc
        else:
            return voc_path

    def load_model(self):
        checkpoint = torch.load(self.save_path)
        self.model.load_state_dict(checkpoint['state_dict'])

    def vision_phi(self, Phi, outpath='phi_output', top_n=50):
        if self.voc is not None:
            if not os.path.exists(outpath):
                os.makedirs(outpath)
            phi = 1
            for num, phi_layer in enumerate(Phi):
                phi = np.dot(phi, phi_layer)
                phi_k = phi.shape[1]
                path = os.path.join(outpath, 'phi' + str(num) + '.txt')
                f = open(path, 'w')
                for each in range(phi_k):
                    top_n_words = self.get_top_n(phi[:, each], top_n)
                    f.write(top_n_words)
                    f.write('\n')
                f.close()
        else:
            print('voc need !!')

    def get_top_n(self, phi, top_n):
        top_n_words = ''
        idx = np.argsort(-phi)
        for i in range(top_n):
            index = idx[i]
            top_n_words += self.voc[index]
            top_n_words += ' '
        return top_n_words

    def vis_txt(self):
        phi = []
        for t in range(self.layer_num):
            phi.append(self.model.decoder[t].w.cpu().detach().numpy())

        self.vision_phi(phi)