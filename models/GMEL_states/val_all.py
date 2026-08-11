import pandas as pd
import torch

from data_load import build_graph, get_scalers,load_data_by_states
from metrics import *
import joblib

from model import GMEL

# ['01'], ['04'], ['05'], ['06'], ['08'], ['09'], ['10'], ['12'], ['13'], ['16'], ['17'], ['18'], ['19'],
#   ['20'], ['21'], ['22'], ['23'],['24'],['25'],['26'],['27'],['28'],['29'],
#   ['30'],['31'],['32'],['33'],['34'],
states = [['01'], ['04'], ['05'], ['06'], ['08'], ['09'], ['10'], ['12'], ['13'], ['16'], ['17'], ['18'], ['19'],
  ['20'], ['21'], ['22'], ['23'],['24'],['25'],['26'],['27'],['28'],['29'],
  ['30'],['31'],['32'],['33'],['34'],['35'],['36'],['37'],['38'],['39'],
  ['40'], ['41'], ['42'], ['44'],['45'],['46'],['47'],['48'],['49'],
  ['50'], ['51'], ['53'], ['54'], ['55'], ['56']]

df_metrics_all = []
for i in range(len(states)):
    train_state = states[i]
    test_state = states[(i-1) % len(states)]

    nfeats_train, adjs_train, dises_train, ods_train, nfeats_valid, adjs_valid, dises_valid, ods_valid, _, _, _, _ = load_data_by_states(train_state_codes=train_state, test_state_codes=test_state)
    nfeat_scaler, dis_scaler, od_scaler = get_scalers(nfeats_train, dises_train, ods_train)

    gmel = GMEL()
    print('\n  **Loading model...')
    ckpt = torch.load("checkpoints/{}_best_checkpoint.pt".format('+'.join(train_state)), map_location="cpu")
    gmel.load_state_dict( ckpt["model_state"])
    
    gbrt = joblib.load("checkpoints/{}_gbr_model.joblib".format('+'.join(train_state)))

    metrics_all = []
    for nfeat, adj, dis, od in zip(nfeats_valid, adjs_valid, dises_valid, ods_valid):
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
            metrics_all.append(metrics)

    df_metrics = pd.DataFrame(metrics_all)
    df_metrics_all.append(df_metrics)

df_metrics_all = pd.concat(df_metrics_all, ignore_index=True)

result_file_name = 'evaluations/tables/val/val_county_level.csv'
print(df_metrics_all['CPC'].mean())
print(df_metrics_all['CPC'].std())
print(df_metrics_all['RMSE'].mean())
print(df_metrics_all['RMSE'].std())
df_metrics_all.to_csv(result_file_name, index=False)

print('\n  **Finish testing...')