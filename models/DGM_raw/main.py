"""
# @time    : 29.10.2025 08:54
# @author  : zhouzy
# @file    : test.py
# @comment : original version of DeepGravity adpated from FAIRLab
"""

import copy
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from data_load import load_data, load_data_by_states
from metrics import *
from model import *

from torch.utils.data import DataLoader, TensorDataset

from pprint import pprint

from utils import ODFlowDataset, my_collate


print("\n  **Loading data...")
# '36' New York
train_states = ['56']
test_states = ['06']
xtrain, ytrain, xvalid, yvalid, xtest, ytest = load_data_by_states(if_shuffle=False, train_state_codes=train_states, test_state_codes=test_states)
# xtrain, ytrain, xvalid, yvalid, xtest, ytest = load_data(if_shuffle=False)
# odmin_, odmax_ = ytrain.min(), ytrain.max()

# feat_scaler = MinMaxScaler((-1, 1)).fit(xtrain)
# od_scaler = OD_normer(ytrain.min(), ytrain.max())

deepgravity = DeepGravity()
# deepgravity = deepgravity.cuda()

# xtrain = torch.FloatTensor(xtrain)
# ytrain = torch.FloatTensor(ytrain)

# ds = TensorDataset(xtrain, ytrain)
ds = ODFlowDataset(xtrain, ytrain)
# dl = DataLoader(ds, batch_size=1000000, shuffle=True)
# dl = DataLoader(ds, batch_size=512, shuffle=True)
dl = DataLoader(ds, batch_size=512, shuffle=True, collate_fn=my_collate)

optimizer = torch.optim.Adam(deepgravity.parameters(), lr=3e-4)
# optimizer = torch.optim.RMSprop(deepgravity.parameters(), lr=5e-6, momentum=0.9)
# optimizer = torch.optim.Adam(deepgravity.parameters(), lr=3e-5)

mode = 'train'

if mode == 'train':
    print('\n  **Start fitting...')
    start = time.time()

    best_valid_loss = np.inf

    best_model = None
    # valid_flag = 100
    valid_flag = 10
    for i in range(100):
        print(f"Epoch {i + 1}:", end=" | ")
        deepgravity.train()

        loss_epoch = []
        for xbatch, ybatch in dl:
            # xbatch, ybatch = xbatch.cuda(), ybatch.cuda()

            optimizer.zero_grad()
            train_losses = []

            for sample_x, sample_y in zip(xbatch, ybatch):
                # yhat = deepgravity(xbatch).squeeze()
                yhat = deepgravity(sample_x).squeeze(dim=-1)
                # loss = torch.mean((yhat - ybatch) ** 2)

                # calculcate the cross entropy loss
                log_probs = F.log_softmax(yhat, dim=1)
                sample_loss = -(sample_y * log_probs).sum(dim=1)

                train_losses.append(sample_loss)

            loss = torch.cat(train_losses, dim=0).mean()
            loss.backward()
            optimizer.step()
            loss_value = loss.item()
            loss_epoch.append(loss_value)
        loss_value = np.mean(loss_epoch)
        print(f"train loss={loss_value:.7g}", end=" | ")

        with torch.no_grad():
            valid_losses = []

            for xvalid_one, yvalid_one in zip(xvalid, yvalid):
                # xvalid_one = feat_scaler.transform(xvalid_one)
                # yvalid_one = od_scaler.normalize(yvalid_one)
                # xvalid_one = torch.FloatTensor(feat_scaler.transform(xvalid_one)).cuda()
                # yvalid_one = torch.FloatTensor(od_scaler.normalize(yvalid_one)).cuda()
                # xvalid_one = torch.FloatTensor(feat_scaler.transform(xvalid_one))
                # yvalid_one = torch.FloatTensor(od_scaler.normalize(yvalid_one))

                yvalid_one = yvalid_one/(yvalid_one.sum(1)[:, None] + 1e-12)

                xvalid_one = torch.FloatTensor(xvalid_one)
                yvalid_one = torch.FloatTensor(yvalid_one)
                deepgravity.eval()
                yhat = deepgravity(xvalid_one).squeeze(dim = -1)
                # yhat = yhat.cpu().detach().numpy()
                # valid_loss = (yhat - yvalid_one.cpu().numpy()) ** 2

                # calculcate the cross entropy loss
                log_probs = F.log_softmax(yhat, dim=1)
                valid_loss = -(yvalid_one * log_probs).sum(dim=1).cpu().detach().numpy()

                valid_losses.append(valid_loss)
            valid_loss = np.concatenate(valid_losses).mean()
            print(f"valid loss={valid_loss:.7g}")
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                # valid_flag = 100
                valid_flag = 10

                best_model = copy.deepcopy(deepgravity)
                # save
                ckpt = {
                    "epoch": i,
                    "model_state": deepgravity.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_val": best_valid_loss,
                }
                torch.save(ckpt, "checkpoints/{}_best_checkpoint.pt".format('+'.join(train_states)))
            else:
                valid_flag -= 1
                if valid_flag == 0:
                    print('Early stopping!')
                    break

    print('Complete!', end=" ")
    print('Consume ', time.time() - start, ' seconds!')
    print("-" * 50)

    print("\n  **Evaluating...")
    with torch.no_grad():
        metrics_all = []
        for x_one, y_one in zip(xtest, ytest):
            # x_one = feat_scaler.transform(x_one)
            # x_one = torch.FloatTensor(x_one).cuda()
            x_one = torch.FloatTensor(x_one)
            # deepgravity.eval()
            # y_one_hat = deepgravity(x_one).squeeze()
            best_model.eval()
            y_one_hat = best_model(x_one).squeeze(dim = -1)
            y_one_hat = F.softmax(y_one_hat, dim=1)
            y_one_hat = y_one_hat.cpu().detach().numpy()

            # y_one_hat = od_scaler.renormalize(y_one_hat)

            # y_one_hat = y_one_hat.reshape([int(np.sqrt(y_one.shape[0])), int(np.sqrt(y_one.shape[0]))])
            # y_one = y_one.reshape([int(np.sqrt(y_one.shape[0])), int(np.sqrt(y_one.shape[0]))])

            y_one_hat = y_one_hat * y_one.sum(1)
            # y_one_hat[y_one_hat < 0] = 0

            metrics = cal_od_metrics(y_one_hat, y_one)
            metrics_all.append(metrics)

        avg_metrics = average_listed_metrics(metrics_all)
        pprint(avg_metrics)

# elif mode == 'test':
#     print('\n  **Loading model...')
#
#     ckpt = torch.load("checkpoints/{}_best_checkpoint.pt".format('+'.join(train_states)), map_location="cpu")
#     deepgravity.load_state_dict(ckpt["model_state"])
#     optimizer.load_state_dict(ckpt["optimizer_state"])
#     start_epoch = ckpt["epoch"] + 1
#     best_val = ckpt["best_val"]
#
#     with torch.no_grad():
#         metrics_all = []
#         for x_one, y_one in zip(xtest, ytest):
#             x_one = feat_scaler.transform(x_one)
#             # x_one = torch.FloatTensor(x_one).cuda()
#             x_one = torch.FloatTensor(x_one)
#             deepgravity.eval()
#             y_one_hat = deepgravity(x_one).squeeze()
#             y_one_hat = y_one_hat.cpu().detach().numpy()
#
#             y_one_hat = od_scaler.renormalize(y_one_hat)
#
#             y_one_hat = y_one_hat.reshape([int(np.sqrt(y_one.shape[0])), int(np.sqrt(y_one.shape[0]))])
#             y_one = y_one.reshape([int(np.sqrt(y_one.shape[0])), int(np.sqrt(y_one.shape[0]))])
#             y_one_hat[y_one_hat < 0] = 0
#
#             metrics = cal_od_metrics(y_one_hat, y_one)
#             metrics_all.append(metrics)
#
#         avg_metrics = average_listed_metrics(metrics_all)
#         pprint(avg_metrics)
#
#         df_metrics = pd.DataFrame(list(avg_metrics.items()), columns=['metric', 'value'])
#         result_file_name =  'tables/evaluations/{}->{}.csv'.format('+'.join(train_states), '+'.join(test_states))
#         df_metrics.to_csv(result_file_name, index=False)