import pandas as pd

from data_load import load_data, load_data_by_states, load_state_areas, split_train_valid_test, construct_validtest
from metrics import *
import joblib

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
    xtrain, ytrain, xvalid, yvalid, _, _ = load_data_by_states(if_shuffle=False, train_state_codes=train_state, test_state_codes=test_state)
    
    rf = joblib.load("checkpoints/{}_rf_model.joblib".format('+'.join(train_state)))
    

    metrics_all = []

    for x_one, y_one in zip(xvalid, yvalid):
    
        y_one_hat = rf.predict(x_one)
        y_one, y_one_hat = y_one.reshape([int(np.sqrt(x_one.shape[0])),
                                          int(np.sqrt(x_one.shape[0]))]), y_one_hat.reshape([int(np.sqrt(x_one.shape[0])),
                                                                                             int(np.sqrt(x_one.shape[0]))])
        y_one_hat[y_one_hat < 0] = 0
        metrics = cal_od_metrics(y_one_hat, y_one)
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