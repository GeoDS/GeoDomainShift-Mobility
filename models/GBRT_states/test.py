# -*- coding: utf-8 -*-
"""
# @time    : 26.11.2025 11:41
# @author  : zhouzy
# @file    : test.py
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from data_load import load_data, load_data_by_states, load_state_areas, split_train_valid_test, construct_validtest
from metrics import *
import pickle

feat_scaler = MinMaxScaler((-1, 1))

train_states = ['36']

df_us_states = pd.read_csv("data/us_states_fips.csv", dtype={"FIPS": str})
test_states = list(set(df_us_states['FIPS']) - set(train_states))

areas = load_state_areas(if_shuffle = False, state_codes = test_states)
_, _, test_areas = split_train_valid_test(areas, train_ratio=0.0, valid_ratio=0.0, test_ratio=1.0)
xtest, ytest = construct_validtest(test_areas)

with open("checkpoints/{}_gbr_model.pkl".format('+'.join(train_states)), "rb") as f:
    gbrt = pickle.load(f)

metrics_all = []

for x_one, y_one, test_area in zip(xtest, ytest, test_areas):
    x_one = feat_scaler.fit_transform(x_one)

    y_one_hat = gbrt.predict(x_one)
    y_one, y_one_hat = y_one.reshape([int(np.sqrt(x_one.shape[0])),
                                      int(np.sqrt(x_one.shape[0]))]), y_one_hat.reshape([int(np.sqrt(x_one.shape[0])),
                                                                                         int(np.sqrt(x_one.shape[0]))])
    y_one_hat[y_one_hat < 0] = 0
    metrics = cal_od_metrics(y_one_hat, y_one)
    metrics['GEOID'] = test_area
    metrics_all.append(metrics)

df_metrics = pd.DataFrame(metrics_all)
df_metrics = df_metrics[['GEOID'] + [c for c in df_metrics.columns if c != 'GEOID']]
result_file_name = 'evaluations/tables/source_{}_county_level.csv'.format('+'.join(train_states))
df_metrics.to_csv(result_file_name, index=False)