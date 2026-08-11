from copy import copy
import time

import joblib

from data_load import *
from metrics import *
from model import *

from sklearn.ensemble import GradientBoostingRegressor

from pprint import pprint
# ['01'], ['04'], ['05'], ['06'], ['08'], ['09'], ['10'],['12'], ['13'], ['16'], ['17'], ['18'], ['19'],
#   ['20'], ['21'], ['22'], ['23'],['24'],['25'],['26'],['27'],['28'],['29'],
#   ['30'],['31'],['32'],['33'],['34'],['35'], ['36'],['37'],['38'],['39'],
#   ['40'], ['41'], ['42'], ['44'],['45'],['46'],['47'],['48'],['49'],
#   ['50'], ['51'], ['53'], ['54'], ['55'], 
states = [ ['56'], ['55']]


for i in range(len(states) - 1):
    train_state = states[i]
    test_state = states[(i+1)]

    print("\n  **Loading data {}...".format('+'.join(train_state)))
    nfeats_train, adjs_train, dises_train, ods_train, nfeats_valid, adjs_valid, dises_valid, ods_valid, nfeats_test, adjs_test, dises_test, ods_test = load_data_by_states(train_state_codes=train_state, test_state_codes=test_state)

    nfeat_scaler, dis_scaler, od_scaler = get_scalers(nfeats_train, dises_train, ods_train)
    
    gmel = GMEL()
    # gmel = gmel.cuda()
    
    optimizer = torch.optim.Adam(gmel.parameters(), lr=3e-4)
    
    print('\n  **Start fitting...')
    start = time.time()
    
    best_valid_loss = np.inf
    valid_flag = 10
    for i in range(1000):
        start_epoch = time.time()
        print(f"Epoch {i+1}:", end=" | ")
        gmel.train()
    
        loss_epoch = []
        for nfeat, adj, dis, od in zip(nfeats_train, adjs_train, dises_train, ods_train):
            nfeat = nfeat_scaler.transform(nfeat)
            dis = dis_scaler.transform(dis.reshape(-1, 1)).reshape(dis.shape)
            od = od_scaler.transform(od.reshape(-1, 1)).reshape(od.shape)
            
            # nfeat = torch.FloatTensor(nfeat).cuda()
            # g = build_graph(adj).to(torch.device('cuda'))
            # dis = torch.FloatTensor(dis).cuda()
            # od = torch.FloatTensor(od).cuda()
            nfeat = torch.FloatTensor(nfeat)
            g = build_graph(adj)
            dis = torch.FloatTensor(dis)
            od = torch.FloatTensor(od)
    
            optimizer.zero_grad()
    
            flow_in, flow_out, flow, h_in, h_out = gmel(g, nfeat)
            loss = torch.mean((flow_in-od.sum(0))**2) + torch.mean((flow_out-od.sum(1))**2) + torch.mean((flow-od)**2)
            loss.backward()
            optimizer.step()
            loss_value = loss.item()
            loss_epoch.append(loss_value)
        loss_value = np.mean(loss_epoch)
        print(f"train loss={loss_value:.7g}", end=" | ")
    
        with torch.no_grad():
            valid_losses = []
            for nfeat, adj, dis, od in zip(nfeats_valid, adjs_valid, dises_valid, ods_valid):
                nfeat = nfeat_scaler.transform(nfeat)
                dis = dis_scaler.transform(dis.reshape(-1, 1)).reshape(dis.shape)
                od = od_scaler.transform(od.reshape(-1, 1)).reshape(od.shape)
                
                # nfeat = torch.FloatTensor(nfeat).cuda()
                # g = build_graph(adj).to(torch.device('cuda'))
                # dis = torch.FloatTensor(dis).cuda()
                # od = torch.FloatTensor(od).cuda()
                nfeat = torch.FloatTensor(nfeat)
                g = build_graph(adj)
                dis = torch.FloatTensor(dis)
                od = torch.FloatTensor(od)
    
                flow_in, flow_out, flow, h_in, h_out = gmel(g, nfeat)
                loss = torch.mean((flow_in-od.sum(0))**2) + torch.mean((flow_out-od.sum(1))**2) + torch.mean((flow-od)**2)
                valid_losses.append(loss.item())
            valid_loss = np.mean(valid_losses)
            print(f"valid loss={valid_loss:.7g}", end=" | ")
            print(f"consume {time.time()-start_epoch:.2f} seconds")
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                valid_flag = 10

                ckpt = {
                        "epoch": i,
                        "model_state": gmel.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "best_val": best_valid_loss,
                    }
                torch.save(ckpt, "checkpoints/{}_best_checkpoint.pt".format('+'.join(train_state)))
            else:
                valid_flag -= 1
                if valid_flag == 0:
                    break
    
    print('Complete!', end=" ")
    print('Consume ', time.time()-start, ' seconds!')
    print("-"*50)


    gbrt = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            min_samples_split=2,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=42
        )
    
    print("\n  **Training GBRT...")
    start = time.time()
    
    xtrain = []
    ytrain = []
    with torch.no_grad():
        for nfeat, adj, dis, od in zip(nfeats_train, adjs_train, dises_train, ods_train):
            nfeat = nfeat_scaler.transform(nfeat)
            
            nfeat = torch.FloatTensor(nfeat)
            g = build_graph(adj)
    
            _, _, _, h_in, h_out = gmel(g, nfeat)
            h_in = h_in.cpu().detach().numpy()
            h_out = h_out.cpu().detach().numpy()
            feat = np.concatenate([h_in, h_out], axis=1)
    
            feat_o = feat.reshape([feat.shape[0], 1, feat.shape[1]]).repeat(feat.shape[0], axis=1)
            feat_d = feat.reshape([1, feat.shape[0], feat.shape[1]]).repeat(feat.shape[0], axis=0)
            feat = np.concatenate([feat_o, feat_d, dis.reshape([dis.shape[0], dis.shape[0], 1])], axis=2).reshape([-1, feat.shape[1]*2+1])
            xtrain.append(feat)
            ytrain.append(od.reshape(-1))
            
    xtrain = np.concatenate(xtrain, axis=0)
    ytrain = np.concatenate(ytrain, axis=0)
    
    gbrt.fit(xtrain, ytrain)

    joblib.dump(gbrt, "checkpoints/{}_gbr_model.joblib".format('+'.join(train_state)))
    
    print('Complete!', end=" ")
    print('Consume ', time.time()-start, ' seconds!')
    print("-"*50)