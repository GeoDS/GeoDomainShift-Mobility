
import copy
import time

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from data_load import load_data, load_data_by_states, load_state_areas, split_train_valid_test, construct_validtest
from metrics import *
from model import *

from torch.utils.data import DataLoader, TensorDataset

from pprint import pprint


states = [ ['01'], ['04'], ['05'], ['06'], ['08'], ['09'], ['10'],['12'], ['13'], ['16'], ['17'], ['18'], ['19'],
  ['20'], ['21'], ['22'], ['23'],['24'],['25'],['26'],['27'],['28'],['29'],
  ['30'],['31'],['32'],['33'],['34'],['35'], ['36'],['37'],['38'],['39'],
  ['40'], ['41'], ['42'], ['44'],['45'],['46'],['47'],['48'],['49'],
  ['50'], ['51'], ['53'], ['54'], ['55'], ['56']]

test_states = ['06']

df_metrics_all = []
for train_state in states:
    deepgravity = DeepGravity()

    optimizer = torch.optim.Adam(deepgravity.parameters(), lr=3e-5)

    print('\n  **Loading {} model...'.format('+'.join(train_state)))

    ckpt = torch.load("checkpoints/{}_best_checkpoint.pt".format('+'.join(train_state)), map_location="cpu")
    deepgravity.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    start_epoch = ckpt["epoch"] + 1
    best_val = ckpt["best_val"]

    xtrain, ytrain, xvalid, yvalid, _, _ = load_data_by_states(if_shuffle=False, train_state_codes=train_state, test_state_codes=test_states)

    deepgravity.eval()

    # areas = load_state_areas(if_shuffle=False, state_codes=test_states)
    # _, _, test_areas = split_train_valid_test(areas, train_ratio=0.0, valid_ratio=0.0, test_ratio=1.0)
    # xtest, ytest = construct_validtest(test_areas)
    with torch.no_grad():
        metrics_all = []
        for x_one, y_one in zip(xvalid, yvalid):
            # print('\n  **Testing {}...'.format(test_area))
            x_one = torch.FloatTensor(x_one)
            y_one_hat = deepgravity(x_one).squeeze(dim=-1)
            y_one_hat = F.softmax(y_one_hat, dim=1)
            y_one_hat = y_one_hat.cpu().detach().numpy()

            y_one_hat = y_one_hat * y_one.sum(1)

            metrics = cal_od_metrics(y_one_hat, y_one)
            metrics_all.append(metrics)

    df_metrics = pd.DataFrame(metrics_all)
    # df_metrics = df_metrics[['GEOID'] + [c for c in df_metrics.columns if c != 'GEOID']]
    df_metrics_all.append(df_metrics)

result_file_name = 'evaluations/tables/val/val_county_level.csv'  
df_metrics_all = pd.concat(df_metrics_all, ignore_index=True)
print(df_metrics_all['CPC'].mean())
print(df_metrics_all['CPC'].std())
print(df_metrics_all['RMSE'].mean())
print(df_metrics_all['RMSE'].std())
df_metrics_all.to_csv(result_file_name, index=False)

print('\n  **Finish testing...')
