import torch
import torch.nn as nn

from params import args


"""
Pytorch Implementation of CLUB from
CLUB: A Contrastive Log-ratio Upper Bound of Mutual Information
by Pengyu Cheng et al.
"""
class CLUBSample(nn.Module):
    def __init__(self, behaviors, zero_sim_user_list):
        super(CLUBSample, self).__init__()
        self.behaviors = behaviors

        x_dim = args.emb_size
        y_dim = args.emb_size
        hidden_size = 2 * args.emb_size

        self.zero_sim_user_list = zero_sim_user_list

        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size, y_dim))
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size, y_dim),
                                      nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar


    def forward(self, user_embeds, batch_users):
        mi_list = [None] * (len(self.behaviors) - 1)
        for i in range(len(self.behaviors) - 1):
            zero_sim_users = batch_users[self.zero_sim_user_list[i][batch_users]]

            tar_user_emb = user_embeds[-1][zero_sim_users]
            aux_user_emb = user_embeds[i][zero_sim_users]
            mi_list[i] = self.calc_mi_est(tar_user_emb, aux_user_emb)
        return sum(mi_list) / len(mi_list)
    

    def calc_mi_est(self, x_samples, y_samples):
        if torch.numel(x_samples) == 0 or torch.numel(y_samples) == 0:
            return torch.tensor(0.0, device=x_samples.device)
        
        mu, logvar = self.get_mu_logvar(x_samples)
        sample_size = x_samples.shape[0]
        random_index = torch.randperm(sample_size).long()

        positive = (-(mu - y_samples) ** 2 / logvar.exp() / 2. - logvar / 2.).sum(dim=1)
        negative = (-(mu - y_samples[random_index]) ** 2 / logvar.exp() / 2. - logvar / 2.).sum(dim=1)
        mi = (positive - negative).sum()

        return torch.clamp(mi / 2., min=0.0)


    def learning_loss(self, user_embeds):
        loss_list = [None] * (len(self.behaviors) - 1)
        for i in range(len(self.behaviors) - 1):
            zero_sim_users = self.zero_sim_user_list[i].t()

            tar_user_emb = user_embeds[-1][zero_sim_users]
            aux_user_emb = user_embeds[i][zero_sim_users]
            loss_list[i] = -self.loglikeli(tar_user_emb, aux_user_emb)
        return sum(loss_list) / len(loss_list)
        

    def loglikeli(self, x_samples, y_samples):
        if torch.numel(x_samples) == 0 or torch.numel(y_samples) == 0:
            return torch.tensor(0.0, device=x_samples.device)
        
        mu, logvar = self.get_mu_logvar(x_samples)
        llh = (-(mu - y_samples) ** 2 / logvar.exp() / 2. - logvar / 2.).sum(dim=1).mean()
        return llh
