# -*- coding: utf-8 -*-
"""
# @time    : 29.10.2025 08:54
# @author  : zhouzy
# @file    : test.py
"""
import copy
import time

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from data_load import load_data, load_data_by_states, load_state_areas, split_train_valid_test, construct_validtest
from metrics import *
from model import *

from torch.utils.data import DataLoader, TensorDataset

from pprint import pprint

#
# print("\n  **Loading data...")
# train_states = ['30']
# # sample_states = ['06']
# # xtrain, ytrain, xvalid, yvalid, _, _ = load_data_by_states(if_shuffle=False, train_state_codes=train_states, test_state_codes=sample_states)
# # xtrain, ytrain, xvalid, yvalid, xtest, ytest = load_data(if_shuffle=False)
# # odmin_, odmax_ = ytrain.min(), ytrain.max()
#
# # feat_scaler = MinMaxScaler((-1, 1)).fit(xtrain)
# # od_scaler = OD_normer(ytrain.min(), ytrain.max())
#
# deepgravity = DeepGravity()
# # deepgravity = deepgravity.cuda()
#
# # xtrain = torch.FloatTensor(feat_scaler.transform(xtrain))
# # ytrain = torch.FloatTensor(od_scaler.normalize(ytrain))
#
# # ds = TensorDataset(xtrain, ytrain)
# # dl = DataLoader(ds, batch_size=1000000, shuffle=True)
# # dl = DataLoader(ds, batch_size=512, shuffle=True)
#
# # optimizer = torch.optim.Adam(deepgravity.parameters(), lr=3e-4)
# optimizer = torch.optim.Adam(deepgravity.parameters(), lr=3e-5)
#
# print('\n  **Loading model...')
#
# ckpt = torch.load("checkpoints/{}_best_checkpoint.pt".format('+'.join(train_states)), map_location="cpu")
# deepgravity.load_state_dict(ckpt["model_state"])
# optimizer.load_state_dict(ckpt["optimizer_state"])
# start_epoch = ckpt["epoch"] + 1
# best_val = ckpt["best_val"]
#
# print('\n  **Start testing...')
#
# df_us_states = pd.read_csv("../../geo_data/us_states_fips.csv", dtype={"FIPS": str})
# test_states = list(set(df_us_states['FIPS']) - set(train_states))
# # test_metrics = []
# # test_states = ['06']
#
# deepgravity.eval()
# # by state
# # with torch.no_grad():
# #     for test_area in test_states:
# #         print('\n  **Testing {}...'.format(test_area))
# #
# #         _, _, _, _, xtest, ytest = load_data_by_states(if_shuffle=False, train_state_codes=train_states,
# #                                                            test_state_codes=[test_area])
# #         if len(xtest) == 0 or len(ytest) == 0:
# #             continue
# #         metrics_all = []
# #         for x_one, y_one in zip(xtest, ytest):
# #             x_one = feat_scaler.transform(x_one)
# #             # x_one = torch.FloatTensor(x_one).cuda()
# #             x_one = torch.FloatTensor(x_one)
# #             y_one_hat = deepgravity(x_one).squeeze()
# #             y_one_hat = y_one_hat.cpu().detach().numpy()
# #
# #             y_one_hat = od_scaler.renormalize(y_one_hat)
# #
# #             y_one_hat = y_one_hat.reshape([int(np.sqrt(y_one.shape[0])), int(np.sqrt(y_one.shape[0]))])
# #             y_one = y_one.reshape([int(np.sqrt(y_one.shape[0])), int(np.sqrt(y_one.shape[0]))])
# #             y_one_hat[y_one_hat < 0] = 0
# #
# #             metrics = cal_od_metrics(y_one_hat, y_one)
# #             metrics_all.append(metrics)
# #
# #         avg_metrics = average_listed_metrics(metrics_all)
# #         avg_metrics['STATEFP'] = test_area
# #         pprint(avg_metrics)
# #         test_metrics.append(avg_metrics)
# #
# # df_metrics = pd.DataFrame(test_metrics)
# # df_metrics = df_metrics[['STATEFP'] + [c for c in df_metrics.columns if c != 'STATEFP']]
# # # result_file_name = 'evaluations/{}->{}.csv'.format('+'.join(train_states), '+'.join(test_states))
# # result_file_name = 'evaluations/source_{}.csv'.format('+'.join(train_states))
# # df_metrics.to_csv(result_file_name, index=False)
# # df_metrics = pd.DataFrame(test_metrics)
# # df_metrics = df_metrics[['STATEFP'] + [c for c in df_metrics.columns if c != 'STATEFP']]
# # # result_file_name = 'evaluations/{}->{}.csv'.format('+'.join(train_states), '+'.join(test_states))
# # result_file_name = 'evaluations/source_{}.csv'.format('+'.join(train_states))
# # df_metrics.to_csv(result_file_name, index=False)
#
# # by county
# areas = load_state_areas(if_shuffle = False, state_codes = test_states)
# _, _, test_areas = split_train_valid_test(areas, train_ratio=0.0, valid_ratio=0.0, test_ratio=1.0)
# xtest, ytest = construct_validtest(test_areas)
# with torch.no_grad():
#     metrics_all = []
#     for x_one, y_one, test_area in zip(xtest, ytest, test_areas):
#         print('\n  **Testing {}...'.format(test_area))
#         # x_one = feat_scaler.transform(x_one)
#         # x_one = torch.FloatTensor(x_one).cuda()
#         x_one = torch.FloatTensor(x_one)
#         # y_one_hat = deepgravity(x_one).squeeze()
#         # y_one_hat = y_one_hat.cpu().detach().numpy()
#         y_one_hat =  deepgravity(x_one).squeeze(dim=-1)
#         y_one_hat = F.softmax(y_one_hat, dim=1)
#         y_one_hat = y_one_hat.cpu().detach().numpy()
#
#         # y_one_hat = od_scaler.renormalize(y_one_hat)
#
#         # y_one_hat = y_one_hat.reshape([int(np.sqrt(y_one.shape[0])), int(np.sqrt(y_one.shape[0]))])
#         # y_one = y_one.reshape([int(np.sqrt(y_one.shape[0])), int(np.sqrt(y_one.shape[0]))])
#
#         y_one_hat = y_one_hat * y_one.sum(1)
#         # y_one_hat[y_one_hat < 0] = 0
#
#         metrics = cal_od_metrics(y_one_hat, y_one)
#         metrics['GEOID'] = test_area
#         metrics_all.append(metrics)
#     # for test_area in test_areas:
#         # avg_metrics = average_listed_metrics(metrics_all)
#         # avg_metrics['STATEFP'] = test_area
#         # pprint(avg_metrics)
#         # test_metrics.append(avg_metrics)
# df_metrics = pd.DataFrame(metrics_all)
# df_metrics = df_metrics[['GEOID'] + [c for c in df_metrics.columns if c != 'GEOID']]
# # result_file_name = 'evaluations/{}->{}.csv'.format('+'.join(train_states), '+'.join(test_states))
# result_file_name = 'evaluations/tables/test/source_{}_county_level.csv'.format('+'.join(train_states))
# df_metrics.to_csv(result_file_name, index=False)
#
# print('\n  **Finish testing...')

# ['40'],['41'],['42'],['44'],['45'],['46'],['47'],['48'],['49']
train_states = [['50'], ['51'], ['53'], ['54'], ['55'], ['56']]
for train_state in train_states:
    deepgravity = DeepGravity()

    optimizer = torch.optim.Adam(deepgravity.parameters(), lr=3e-5)

    print('\n  **Loading {} model...'.format('+'.join(train_state)))

    ckpt = torch.load("checkpoints/{}_best_checkpoint.pt".format('+'.join(train_state)), map_location="cpu")
    deepgravity.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    start_epoch = ckpt["epoch"] + 1
    best_val = ckpt["best_val"]

    df_us_states = pd.read_csv("../../geo_data/us_states_fips.csv", dtype={"FIPS": str})
    test_states = list(set(df_us_states['FIPS']) - set(train_state))

    deepgravity.eval()

    areas = load_state_areas(if_shuffle=False, state_codes=test_states)
    _, _, test_areas = split_train_valid_test(areas, train_ratio=0.0, valid_ratio=0.0, test_ratio=1.0)
    xtest, ytest = construct_validtest(test_areas)
    with torch.no_grad():
        metrics_all = []
        for x_one, y_one, test_area in zip(xtest, ytest, test_areas):
            print('\n  **Testing {}...'.format(test_area))
            x_one = torch.FloatTensor(x_one)
            y_one_hat = deepgravity(x_one).squeeze(dim=-1)
            y_one_hat = F.softmax(y_one_hat, dim=1)
            y_one_hat = y_one_hat.cpu().detach().numpy()

            y_one_hat = y_one_hat * y_one.sum(1)

            metrics = cal_od_metrics(y_one_hat, y_one)
            metrics['GEOID'] = test_area
            metrics_all.append(metrics)

    df_metrics = pd.DataFrame(metrics_all)
    df_metrics = df_metrics[['GEOID'] + [c for c in df_metrics.columns if c != 'GEOID']]
    result_file_name = 'evaluations/tables/test/source_{}_county_level.csv'.format('+'.join(train_state))
    df_metrics.to_csv(result_file_name, index=False)

print('\n  **Finish testing...')
