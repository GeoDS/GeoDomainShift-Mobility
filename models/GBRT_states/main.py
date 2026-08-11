import time

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from data_load import load_data, load_data_by_states
from metrics import *

from pprint import pprint

import pickle

print("\n  **Loading data...")
# '36' New York
train_states = ['04']
test_states = ['06']
xtrain, ytrain, xvalid, yvalid, xtest, ytest = load_data_by_states(if_shuffle=False, train_state_codes=train_states, test_state_codes=test_states)

# feat_scaler = StandardScaler()
# xtrain = feat_scaler.fit_transform(xtrain)

gbrt = GradientBoostingRegressor(n_estimators=20,
                                 min_samples_split=2,
                                 min_samples_leaf=2,
                                 max_depth=None)

print('\n  **Start fitting...')
start = time.time()
gbrt.fit(xtrain, ytrain)

with open("checkpoints/{}_gbr_model.pkl".format('+'.join(train_states)), "wb") as f:
    pickle.dump(gbrt, f)

print('Complete!', end=" ")
print('Consume ', time.time()-start, ' seconds!')
print("-"*50)

print("\n  **Evaluating...")
metrics_all = []
for x_one, y_one in zip(xvalid, yvalid):
    y_one_hat = gbrt.predict(x_one)
    y_one, y_one_hat = y_one.reshape([int(np.sqrt(x_one.shape[0])), 
                                      int(np.sqrt(x_one.shape[0]))]), y_one_hat.reshape([int(np.sqrt(x_one.shape[0])), 
                                                                                         int(np.sqrt(x_one.shape[0]))])
    y_one_hat[y_one_hat < 0] = 0
    metrics = cal_od_metrics(y_one_hat, y_one)
    metrics_all.append(metrics)

avg_metrics = average_listed_metrics(metrics_all)
pprint(avg_metrics)