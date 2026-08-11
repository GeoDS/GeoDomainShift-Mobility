# -*- coding: utf-8 -*-
"""
# @time    : 20.04.2025 10:57
# @author  : zhouzy
# @file    : test_all.py
"""
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from data_load import load_data, load_data_by_states, load_state_areas, split_train_valid_test, construct_validtest
from metrics import *
import pickle
# ['01'], ['04'], ['05'],['06'], ['08'], ['09'], ['10'], ['12'], ['13'],['17'], ['18'], ['19'],
  # ['20'], ['21'], ['22'], ['23'],['24'],['25'],['26'],['27'],['28'],['29'],
  # ['30'],['31'],['32'],['33'],['34'],['35'],['36'],['37'],['38'],['39'],
  # ['40'], ['41'], ['42'], ['44'],['45'],['46'],['47'],['48'],['49'],
  # ['50'], ['51'], ['53'], ['54'], ['55'], ['56']
states = [  ['16']]

for i in range(len(states)):
    train_state = states[i]
    df_us_states = pd.read_csv("../../geo_data/us_states_fips.csv", dtype={"FIPS": str})
    test_states = list(set(df_us_states['FIPS']) - set(train_state))

    areas = load_state_areas(if_shuffle = False, state_codes = test_states)
    _, _, test_areas = split_train_valid_test(areas, train_ratio=0.0, valid_ratio=0.0, test_ratio=1.0)
    xtest, ytest = construct_validtest(test_areas)


    rf = joblib.load("checkpoints/{}_rf_model.joblib".format('+'.join(train_state)))

    metrics_all = []

    for x_one, y_one, test_area in zip(xtest, ytest, test_areas):
    
        y_one_hat = rf.predict(x_one)
        y_one, y_one_hat = y_one.reshape([int(np.sqrt(x_one.shape[0])),
                                          int(np.sqrt(x_one.shape[0]))]), y_one_hat.reshape([int(np.sqrt(x_one.shape[0])),
                                                                                             int(np.sqrt(x_one.shape[0]))])
        y_one_hat[y_one_hat < 0] = 0
        metrics = cal_od_metrics(y_one_hat, y_one)
        metrics['GEOID'] = test_area
        metrics_all.append(metrics)

    df_metrics = pd.DataFrame(metrics_all)
    df_metrics = df_metrics[['GEOID'] + [c for c in df_metrics.columns if c != 'GEOID']]
    result_file_name = 'evaluations/tables/test/source_{}_county_level.csv'.format('+'.join(train_state))
    df_metrics.to_csv(result_file_name, index=False)
    
    print('\n  **Finish {} testing...'.format('+'.join(train_state)))