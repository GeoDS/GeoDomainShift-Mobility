import time

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from data_load import load_data, load_data_by_states
from metrics import *

from pprint import pprint

import joblib


# ['01'], ['04'], ['05'], ['06'], ['08'], ['09'], ['10'], ['12'], ['13'], ['16'], ['17'], ['18'], ['19'],
#   ['20'], ['21'], ['22'], ['23'],['24'],['25'],['26'],['27'],['28'],['29'],
#   ['30'],['31'],['32'],['33'],['34'],['35'],['36'],['37'],['38'],['39'],
#   ['40'], ['41'], ['42'], ['44'],['45'],['46'],['47'],['48'],['49'],
#   ['50'], ['51'], ['53'], ['54'], ['55'], 
states = [['56'], ['55']]
# train_states = ['01']
# test_states = ['06']
for i in range(len(states) - 1):
    train_state = states[i]
    test_state = states[(i+1)]

    print("\n  **Loading data {}...".format('+'.join(train_state)))
    xtrain, ytrain, xvalid, yvalid, _, _ = load_data_by_states(if_shuffle=False, train_state_codes=train_state, test_state_codes=test_state)
    
    # feat_scaler = StandardScaler()
    # xtrain = feat_scaler.fit_transform(xtrain)
    
    # gbrt = GradientBoostingRegressor(n_estimators=100,
    #                                  min_samples_split=2,
    #                                  min_samples_leaf=2,
    #                                  max_depth=None)

    gbrt = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42
    )
    
    print('\n  **Start fitting...')
    start = time.time()
    gbrt.fit(xtrain, ytrain)
    
    # with open("checkpoints/{}_gbr_model.pkl".format('+'.join(train_state)), "wb") as f:
        # pickle.dump(gbrt, f)
    joblib.dump(gbrt, "checkpoints/{}_gbr_model.joblib".format('+'.join(train_state)))
    
    print('Complete!', end=" ")
    print('Consume ', time.time()-start, ' seconds!')
    # print("-"*50)
    
    # print("\n  **Evaluating...")
    # metrics_all = []
    # for x_one, y_one in zip(xvalid, yvalid):
    #     y_one_hat = gbrt.predict(x_one)
    #     y_one, y_one_hat = y_one.reshape([int(np.sqrt(x_one.shape[0])), 
    #                                       int(np.sqrt(x_one.shape[0]))]), y_one_hat.reshape([int(np.sqrt(x_one.shape[0])), 
    #                                                                                          int(np.sqrt(x_one.shape[0]))])
    #     y_one_hat[y_one_hat < 0] = 0
    #     metrics = cal_od_metrics(y_one_hat, y_one)
    #     metrics_all.append(metrics)
    
    # avg_metrics = average_listed_metrics(metrics_all)
    # pprint(avg_metrics)