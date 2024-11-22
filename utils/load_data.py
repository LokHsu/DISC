import os

import torch
import scipy.sparse as sp

from params import args

def get_sim_matrix(dir, dataset, behavior):
    filename = "_".join([behavior, "sim_mat_itemcf.npz"])
    user_sim_mat = sp.load_npz(os.path.join(dir, dataset, filename))
    return torch.tensor(user_sim_mat.todense())


def get_user_sim_list(dataset, behaviors, dir=args.data_dir):
    user_sim_list = [None] * len(behaviors)
    for i, beh in enumerate(behaviors):
        filename = "_".join([beh, "sim_mat_itemcf.npz"])
        user_sim_list[i] = sp.load_npz(os.path.join(dir, dataset, filename))
    return user_sim_list


def get_zero_beh_sim_user_list(dataset, behaviors, device, dir=args.data_dir):
    beh_sim_dict = torch.load(os.path.join(dir, dataset, 'behaviors_sim_dict.pth'))
    zero_sim_user_list = [None] * (len(behaviors) - 1)
    for i, beh in enumerate(behaviors[:-1]):
        zero_sim_users = (beh_sim_dict[beh] <= args.beta)
        zero_sim_user_list[i] = zero_sim_users.to(device)
    return zero_sim_user_list


def get_batch_user_top_sim(user_sim_list, batch_users, device):
    batch_users = batch_users.cpu()
    user_top_sim_list = [None] * len(user_sim_list)
    for i in range(len(user_sim_list)):
        user_sim = user_sim_list[i]
        batch_user_sim = user_sim[batch_users, :][:, batch_users]
        user_top_sim = torch.tensor(batch_user_sim.todense()) > args.alpha
        user_top_sim_list[i] = user_top_sim.to(device)
    return user_top_sim_list
