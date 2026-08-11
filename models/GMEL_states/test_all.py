# -*- coding: utf-8 -*-
"""
# @time    : 20.04.2025 10:57
# @author  : zhouzy
# @file    : test_all.py
"""
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from data_load import build_graph, get_scalers, load_data_by_states, load_state_areas, split_train_valid_test, construct_sample
from metrics import *
from model import *
# ['01'], ['04'], ['05'], 
states = [['06'], ['08'], ['09'], ['10'], ['12'], ['13'], ['16'], ['17'], ['18'], ['19'],
  ['20'], ['21'], ['22'], ['23'],['24'],['25'],['26'],['27'],['28'],['29'],
  ['30'],['31'],['32'],['33'],['34'],['35'],['36'],['37'],['38'],['39'],
  ['40'], ['41'], ['42'], ['44'],['45'],['46'],['47'],['48'],['49'],
  ['50'], ['51'], ['53'], ['54'], ['55'], ['56']]

for i in range(len(states)):
    train_state = states[i]
    df_us_states = pd.read_csv("../../geo_data/us_states_fips.csv", dtype={"FIPS": str})
    test_states = list(set(df_us_states['FIPS']) - set(train_state))

    nfeats_train, adjs_train, dises_train, ods_train, nfeats_valid, adjs_valid, dises_valid, ods_valid, _, _, _, _ = load_data_by_states(train_state_codes=train_state, test_state_codes=test_states)
    nfeat_scaler, dis_scaler, od_scaler = get_scalers(nfeats_train, dises_train, ods_train)

    areas = load_state_areas(if_shuffle = False, state_codes = test_states)
    _, _, test_areas = split_train_valid_test(areas, train_ratio=0.0, valid_ratio=0.0, test_ratio=1.0)
    nfeats_test, adjs_test, dises_test, ods_test = construct_sample(test_areas)
    
    gmel = GMEL()
    print('\n  **Loading model...')
    ckpt = torch.load("checkpoints/{}_best_checkpoint.pt".format('+'.join(train_state)), map_location="cpu")
    gmel.load_state_dict( ckpt["model_state"])

    gbrt = joblib.load("checkpoints/{}_gbr_model.joblib".format('+'.join(train_state)))
    metrics_all = []
    for nfeat, adj, dis, od, test_area in zip(nfeats_test, adjs_test, dises_test, ods_test, test_areas):
        nfeat = nfeat_scaler.transform(nfeat)
        nfeat = torch.FloatTensor(nfeat)
        g = build_graph(adj)
    
        with torch.no_grad():
            _, _, _, h_in, h_out = gmel(g, nfeat)
            h_in = h_in.cpu().detach().numpy()
            h_out = h_out.cpu().detach().numpy()
            feat = np.concatenate([h_in, h_out], axis=1)
    
            feat_o = feat.reshape([feat.shape[0], 1, feat.shape[1]]).repeat(feat.shape[0], axis=1)
            feat_d = feat.reshape([1, feat.shape[0], feat.shape[1]]).repeat(feat.shape[0], axis=0)
            feat = np.concatenate([feat_o, feat_d, dis.reshape([dis.shape[0], dis.shape[0], 1])], axis=2).reshape([-1, feat.shape[1]*2+1])
    
            od_hat = gbrt.predict(feat).reshape([int(np.sqrt(feat.shape[0])), int(np.sqrt(feat.shape[0]))])
            od_hat = np.clip(od_hat, 0, None)
            od = od.reshape([int(np.sqrt(feat.shape[0])), int(np.sqrt(feat.shape[0]))])
            metrics = cal_od_metrics(od_hat, od)
            metrics['GEOID'] = test_area
            metrics_all.append(metrics)

    df_metrics = pd.DataFrame(metrics_all)
    df_metrics = df_metrics[['GEOID'] + [c for c in df_metrics.columns if c != 'GEOID']]
    result_file_name = 'evaluations/tables/test/source_{}_county_level.csv'.format('+'.join(train_state))
    df_metrics.to_csv(result_file_name, index=False)
    
    print('\n  **Finish {} testing...'.format('+'.join(train_state)))