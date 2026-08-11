import time

from sklearn.ensemble import RandomForestRegressor
from data_load import load_data, load_data_by_states
from metrics import *

from pprint import pprint
import joblib



states = [ ['01'], ['04'], ['05'], ['06'], ['08'], ['09'], ['10'],['12'], ['13'], ['16'], ['17'], ['18'], ['19'],
  ['20'], ['21'], ['22'], ['23'],['24'],['25'],['26'],['27'],['28'],['29'],
  ['30'],['31'],['32'],['33'],['34'],['35'], ['36'],['37'],['38'],['39'],
  ['40'], ['41'], ['42'], ['44'],['45'],['46'],['47'],['48'],['49'],
  ['50'], ['51'], ['53'], ['54'], ['55'], ['56']]

for i in range(len(states) - 1):
    train_state = states[i]
    test_state = states[(i+1)]

    print("\n  **Loading data {}...".format('+'.join(train_state)))
    xtrain, ytrain, xvalid, yvalid, _, _ = load_data_by_states(if_shuffle=False, train_state_codes=train_state, test_state_codes=test_state)
    
    random_forest = RandomForestRegressor(n_estimators = 100,
        oob_score = True,
        max_depth = None,
        min_samples_split = 2,
        min_samples_leaf = 2,
        n_jobs = 2)

    print('\n  **Start fitting...')
    start = time.time()
    random_forest.fit(xtrain, ytrain)

    print('Complete!', end=" ")
    print('Consume ', time.time()-start, ' seconds!')
    print("-"*50)

    # model_bundle = {
    #     "model": random_forest,
    #     "feature_names": list(xtrain.columns)
    # }

    joblib.dump(random_forest , "checkpoints/{}_rf_model.joblib".format('+'.join(train_state)))

# print("\n  **Evaluating...")
# metrics_all = []
# for x_one, y_one in zip(xtest, ytest):
#     y_one_hat = random_forest.predict(x_one)
#     y_one, y_one_hat = y_one.reshape([int(np.sqrt(x_one.shape[0])), 
#                                       int(np.sqrt(x_one.shape[0]))]), y_one_hat.reshape([int(np.sqrt(x_one.shape[0])), 
#                                                                                          int(np.sqrt(x_one.shape[0]))])
#     metrics = cal_od_metrics(y_one_hat, y_one)
#     metrics_all.append(metrics)

# avg_metrics = average_listed_metrics(metrics_all)
# pprint(avg_metrics)