import random
from datetime import datetime

import pickle
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from models.lightgcn import LightGCN
from models.club import CLUBSample
from utils import graph
from utils.dataloader import *
from utils.load_data import *
from utils.loss import *
from utils.evaluator import *
from utils.wandb_logger import WandbLogger
from params import args

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_random_seed(seed=2024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(train_loader, gcn, club, gcn_opt, club_opt):
    print(f"{datetime.now()}: Negative Sampling start...")
    train_loader.dataset.neg_sample()
    print(f"{datetime.now()}: Negative Sampling ended.")

    epoch_loss = 0
    for u, i, j in tqdm(train_loader):
        user_embs, item_embs = gcn()

        bpr_loss, reg_loss = calc_bpr_loss(u, i, j, user_embs, item_embs, behaviors, device)
        reg_loss = args.l2_reg * reg_loss
        
        batch_users = u.long().to(device)
        top_sim_idx_list = get_batch_user_top_sim(user_sim_list, batch_users, device)
        infonce_loss = calc_infonce_loss(
            user_embs, batch_users, behaviors, zero_sim_user_list, top_sim_idx_list)
        infonce_loss = args.ssl_reg * infonce_loss

        mi = club(user_embs, batch_users)
        mi = args.club_reg * mi

        loss = (bpr_loss + reg_loss + infonce_loss + mi) / args.batch
        epoch_loss += loss.item()

        gcn_opt.zero_grad()
        loss.backward()
        gcn_opt.step()

    user_embs, _ = gcn()
    user_embs = [user_emb.detach() for user_emb in user_embs]

    for _ in range(args.club_train_step):
        learning_loss = club.learning_loss(user_embs)

        club_opt.zero_grad()
        learning_loss.backward()
        club_opt.step()
    
    return epoch_loss


if __name__ == '__main__':
    set_random_seed()

    if args.dataset == 'Tmall':
        behaviors = ['pv', 'fav', 'cart', 'buy']

    elif args.dataset == 'IJCAI_15':
        behaviors = ['click', 'fav', 'cart', 'buy']

    elif args.dataset == 'retailrocket':
        behaviors = ['fav', 'cart', 'buy']

    train_file = args.data_dir + args.dataset + '/trn_'
    test_file = args.data_dir + args.dataset + '/tst_int'

    train_u2i = []
    for i in range(len(behaviors)):
        with open(train_file + behaviors[i], 'rb') as f:
            u2i = pickle.load(f)
            train_u2i.append(u2i)

            if behaviors[i] == args.target:
                user_num = u2i.get_shape()[0]
                item_num = u2i.get_shape()[1]

    train_dataset = TrainDataset(train_u2i, behaviors, item_num)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch, num_workers=8)

    with open(test_file, 'rb') as f:
        test_dataset = TestDataset(pickle.load(f))
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch, num_workers=4, pin_memory=True)

    print(f"{datetime.now()}: Loading user interactation details...")
    adj_matrix = graph.create_adj_mats(train_u2i, user_num, item_num, behaviors, device)
    user_sim_list = get_user_sim_list(args.dataset, behaviors)
    zero_sim_user_list = get_zero_beh_sim_user_list(args.dataset, behaviors, device)
    print(f"{datetime.now()}: Completed.")
    print(f"{user_num = }\n{item_num = }")

    gcn = LightGCN(user_num, item_num, behaviors, adj_matrix).to(device)
    club = CLUBSample(behaviors, zero_sim_user_list).to(device)
    gcn_opt = torch.optim.Adam(gcn.parameters(), lr=args.lr)
    club_opt = torch.optim.Adam(club.parameters(), lr=args.lr)
    if args.wandb:
        wandb_logger = WandbLogger()

    for epoch in range(1, args.epochs + 1):
        print(f"\nTrainning Epoch {epoch}:")
        loss = train(train_loader, gcn, club, gcn_opt, club_opt)
        print(f"Epoch {epoch} Evaluation Metrics:\n{loss = :.6f}")
        with torch.no_grad():
            user_embs, item_embs = gcn()
            test_res = test(
                test_loader,
                train_u2i[-1],
                user_embs[-1].detach(),
                item_embs[-1].detach(),
            )
        if args.wandb:
            wandb_logger.log_metrics(epoch, loss, test_res, gcn)
