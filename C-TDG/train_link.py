import torch

from utils import get_node_sets, scoring, optimizer_to, set_seed, dst_strategies, REGRESSION_SCORES
from torch_geometric.nn.models.tgn import LastNeighborLoader
from torch_geometric.loader import TemporalDataLoader
from datasets import get_dataset
from conf import edgebank
import negative_sampler
import numpy as np
import datetime
import wandb
import time
import ray
import os

def compute_fake_score(conf):
    scores=scoring([0,1,0,1], [1,0,1,0], torch.tensor([1., 0., 1., 0.]), is_regression=conf['link_regression'])
    for k in scores:
        if 'confusion' in k:
            continue
        scores[k] = -np.inf if not conf['link_regression'] else np.inf
    scores['loss'] = -np.inf if not conf['link_regression'] else np.inf
    scores['time'] = datetime.timedelta(seconds=1)
    return scores
    

def train(data, model, optimizer, train_loader, criterion, neighbor_loader, helper, train_neg_sampler=None, requires_grad=True, device='cpu'):
    model.train()
    
    # Start with a fresh memory and an empty graph
    model.reset_memory()
    neighbor_loader.reset_state()

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        if train_neg_sampler is None:
            # NOTE: When the train_neg_sampler is None we are doing link regression
            original_n_id = torch.cat([src, pos_dst]).unique()
        else:
            # Sample negative destination nodes.
            neg_dst = train_neg_sampler.sample(src).to(device)
            original_n_id = torch.cat([src, pos_dst, neg_dst]).unique()
            batch.neg_dst = neg_dst

        n_id = original_n_id
        edge_index = torch.empty(size=(2,0)).long()
        e_id = neighbor_loader.e_id[n_id]
        for _ in range(model.num_gnn_layers):
            n_id, edge_index, e_id = neighbor_loader(n_id)

        helper[n_id] = torch.arange(n_id.size(0), device=device)
        
        # Model forward
        # NOTE: src_emb, pos_dst_emb are the embedding that will be saved in memory
        pos_out, neg_out, src_emb, pos_dst_emb = model(batch=batch, n_id=n_id, msg=data.msg[e_id].to(device),
                                                       t=data.t[e_id].to(device), edge_index=edge_index, id_mapper=helper)
        if requires_grad:
            if train_neg_sampler is None:
                loss = criterion(pos_out, batch.y)
            else:
                loss = criterion(pos_out, torch.ones_like(pos_out))
                loss += criterion(neg_out, torch.zeros_like(neg_out))

        # Update memory and neighbor loader with ground-truth state.
        model.update(src, pos_dst, t, msg, src_emb, pos_dst_emb)
        neighbor_loader.insert(src, pos_dst)

        if requires_grad:
            loss.backward()
            optimizer.step()

        model.detach_memory()


@torch.no_grad()
def eval(data, model, loader, criterion, neighbor_loader, helper, neg_sampler=None, eval_seed=12345,
         return_predictions=False, device='cpu', eval_name='eval', wandb_log=False):
    t0 = time.time()
    model.eval()

    y_pred_list, y_true_list, y_pred_confidence_list = [], [], []
    for batch in loader:
        batch = batch.to(device)
        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        if neg_sampler is None:
            # NOTE: When the neg_sampler is None we are doing link regression
            original_n_id = torch.cat([src, pos_dst]).unique()
        else:
            # Sample negative destination nodes
            neg_dst = neg_sampler.sample(src, eval=True, eval_seed=eval_seed).to(device) # Ensure deterministic sampling across epochs
            original_n_id = torch.cat([src, pos_dst, neg_dst]).unique()
            batch.neg_dst = neg_dst

        n_id = original_n_id
        edge_index = torch.empty(size=(2,0)).long()
        e_id = neighbor_loader.e_id[n_id]
        for _ in range(model.num_gnn_layers):
            n_id, edge_index, e_id = neighbor_loader(n_id)

        helper[n_id] = torch.arange(n_id.size(0), device=device)
        
        # Model forward
        # NOTE: src_emb, pos_dst_emb are the embedding that will be saved in memory
        pos_out, neg_out, src_emb, pos_dst_emb = model(batch=batch, n_id=n_id, msg=data.msg[e_id].to(device),
                                                       t=data.t[e_id].to(device), edge_index=edge_index, id_mapper=helper)

        if neg_sampler is None:
            y_true = batch.y.cpu()
            y_pred = pos_out.detach().cpu()
            y_pred_list.append(y_pred)
        else:
            loss = criterion(pos_out, torch.ones_like(pos_out))
            loss += criterion(neg_out, torch.zeros_like(neg_out))

            y_pred = torch.cat([pos_out, neg_out], dim=0).cpu()
            
            y_true = torch.cat([torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))],
                               dim=0)    
            y_pred_list.append((y_pred.sigmoid() > 0.5).float())

        y_pred_confidence_list.append(y_pred)
        y_true_list.append(y_true)

        # Update memory and neighbor loader with ground-truth state.
        model.update(src, pos_dst, t, msg, src_emb, pos_dst_emb)
        neighbor_loader.insert(src, pos_dst)

    t1 = time.time()

    # Compute scores  
    y_true_list = torch.cat(y_true_list).unsqueeze(1)
    y_pred_list = torch.cat(y_pred_list)
    y_pred_confidence_list = torch.cat(y_pred_confidence_list)
    scores = scoring(y_true_list, y_pred_list, y_pred_confidence_list, is_regression=neg_sampler is None)
    scores['loss'] = criterion(y_pred_confidence_list, y_true_list).item() 
    scores['time'] = datetime.timedelta(seconds=t1 - t0)

    true_values = (y_true_list, y_pred_list, y_pred_confidence_list) if return_predictions else None
    if wandb_log:
        for k, v in scores.items():
            if  k == 'confusion_matrix':
                continue
            else:
                wandb.log({f"{eval_name} {k}, {neg_sampler}":v if k != 'time' else v.total_seconds()}, commit=False)
                
        _cm = wandb.plot.confusion_matrix(preds=y_pred_list.squeeze().numpy(),
                                          y_true=y_true_list.squeeze().numpy(),
                                          class_names=["negative", "positive"])
        wandb.log({f"conf_mat {eval_name}, {neg_sampler}" : _cm}, commit='val' in eval_name or 'test' in eval_name)
        
    return scores, true_values


@ray.remote(num_cpus=1, num_gpus=0.) #int(os.environ.get('NUM_CPUS_PER_TASK', 1)), num_gpus=float(os.environ.get('NUM_GPUS_PER_TASK', 0.)))
def link_prediction(model_instance, conf):
    return link_prediction_single(model_instance, conf)


def link_prediction_single(model_instance, conf):
    if conf['wandb']:
        wandb.init(project=conf['data_name'], group=conf['model'], config=conf)

    # Set the configuration seed
    set_seed(conf['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = get_dataset(root=conf['data_dir'], name=conf['data_name'], seed=conf['exp_seed']).to(device)
    train_data, val_data, test_data = data.train_val_test_split(val_ratio=conf['split'][0], test_ratio=conf['split'][1])

    train_loader = TemporalDataLoader(train_data, batch_size=conf['batch'])
    val_loader = TemporalDataLoader(val_data, batch_size=conf['batch'])
    test_loader = TemporalDataLoader(test_data, batch_size=conf['batch'])

    neighbor_loader = LastNeighborLoader(data.num_nodes, size=conf['sampler']['size'], device=device)
    
    # Helper vector to map global node indices to local ones.
    assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)

    # Define model
    model = model_instance(**conf['model_params']).to(device)

    criterion = torch.nn.BCEWithLogitsLoss() if not conf['link_regression'] else REGRESSION_SCORES[conf['metric']] 
    optimizer = torch.optim.Adam(model.parameters(), lr=conf['optim_params']['lr'], weight_decay=conf['optim_params']['wd'])

    (train_src_nodes, train_dst_nodes, 
     val_src_nodes, val_dst_nodes, 
     test_src_nodes,test_dst_nodes) = get_node_sets(conf['strategy'], train_data, val_data, test_data)

    if conf['link_regression']:
        train_neg_link_sampler = None
        val_neg_link_sampler = None
        test_neg_link_sampler = None
    else:
        neg_sampler_instance = getattr(negative_sampler, conf['neg_sampler'])
        train_neg_link_sampler = neg_sampler_instance(train_src_nodes, train_dst_nodes, name='train', 
                                                      check_link_existence=not conf['no_check_link_existence'],
                                                      seed=conf['exp_seed']+1)
        val_neg_link_sampler = neg_sampler_instance(val_src_nodes, val_dst_nodes, name='val', 
                                                    check_link_existence=not conf['no_check_link_existence'],
                                                    seed=conf['exp_seed']+2)
        test_neg_link_sampler = neg_sampler_instance(test_src_nodes, test_dst_nodes, name='test', 
                                                     check_link_existence=not conf['no_check_link_existence'],
                                                     seed=conf['exp_seed']+3)

    history = []
    best_epoch = 0
    best_score = -np.inf if not conf['link_regression'] else np.inf
    isbest = lambda current, best, regression: current > best if not regression else current < best

    # Load previuos ckpt if exists
    path_save_best = os.path.join(conf['ckpt_path'], f'conf_{conf["conf_id"]}_seed_{conf["seed"]}.pt')
    if os.path.exists(path_save_best) and not conf['overwrite_ckpt']:
        # Load the existing checkpoint
        print(f'Loading {path_save_best}')
        ckpt = torch.load(path_save_best, map_location=device)
        best_epoch = ckpt['epoch']
        best_score = ckpt['best_score']
        history = ckpt['history']
        if ckpt['train_ended']:
            # The model was already trained, then return
            return ckpt['test_score'], ckpt['val_score'], ckpt['train_score'], ckpt['epoch'], conf, ckpt['history']
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        optimizer_to(optimizer, device) # Map the optimizer to the current device    
    model.to(device)
    
    epoch_times = []
    for e in range(best_epoch, conf['epochs']):
        t0 = time.time()
        if conf['debug']: print('Epoch {:d}:'.format(e))

        train(data=data, model=model, optimizer=optimizer, train_loader=train_loader, criterion=criterion, 
              neighbor_loader=neighbor_loader, train_neg_sampler=train_neg_link_sampler, helper=assoc, requires_grad=conf['model']!=edgebank, device=device)
        
        model.reset_memory()
        neighbor_loader.reset_state()

        try:
            tr_scores, _ = eval(data=data, model=model, loader=train_loader, criterion=criterion, 
                                neighbor_loader=neighbor_loader, neg_sampler=train_neg_link_sampler, helper=assoc, 
                                eval_seed=conf['exp_seed'], device=device, eval_name='train_eval', wandb_log=conf['wandb'])
            
            if conf['reset_memory_eval']:
                model.reset_memory()
            vl_scores, vl_true_values = eval(data=data, model=model, loader=val_loader, criterion=criterion, 
                                            neighbor_loader=neighbor_loader, neg_sampler=val_neg_link_sampler, 
                                            helper=assoc, eval_seed=conf['exp_seed'], device=device,
                                            eval_name='val_eval', wandb_log=conf['wandb'])
        except ValueError as err:
            print(err)
            print(f'\n{conf} crashed.. continuing')
            break

        history.append({
            'train': tr_scores,
            'val': vl_scores
        })

        if len(history) == 1 or isbest(vl_scores[conf['metric']], best_score, conf['link_regression']):
            best_score = vl_scores[conf['metric']]
            best_epoch = e
            torch.save({
                'train_ended': False,
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_score': best_score,
                'loss': (tr_scores['loss'], vl_scores['loss'], None),
                'tr_scores': tr_scores,
                'vl_scores': vl_scores,
                'true_values': (None, vl_true_values, None),
                'history': history
            }, path_save_best)

        if conf['debug']: print(f'\ttrain :{tr_scores}\n\tval :{vl_scores}')
        epoch_times.append(time.time()-t0)

        if conf['debug'] or (conf['verbose'] and e % conf['patience'] == 0): 
            print(f'Epoch {e}: {np.mean(epoch_times)} +/- {np.std(epoch_times)} seconds per epoch') 

        if e - best_epoch > conf['patience']:
            break

    # Evaluate on test
    try:
        if conf['debug']: print('Loading model at epoch {}...'.format(best_epoch))
        ckpt = torch.load(path_save_best, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        ckpt['test_score'] = {}
        ckpt['val_score'] = {}
        ckpt['train_score'] = {}
        ckpt['true_values'] = {}
        ckpt['loss'] = {}
    except:
        if conf['use_all_strategies_eval']:
            strategies = dst_strategies
        else:
            strategies = [conf['strategy']]

        ckpt = {
            'test_score': {},
            'val_score': {},
            'train_score': {}
        }
        scores = compute_fake_score(conf)
        for s in strategies:
            ckpt['test_score'][s] = scores
            ckpt['val_score'][s] = scores
            ckpt['train_score'][s] = scores
        history = [{'train':scores, 'val':scores}]
        return ckpt['test_score'], ckpt['val_score'], ckpt['train_score'], e, conf, history


    if conf['use_all_strategies_eval']:
        strategies = dst_strategies
    else:
        strategies = [conf['strategy']]

    for strategy in strategies:
        if conf['link_regression']:
            tmp_train_neg_link_sampler = None
            tmp_val_neg_link_sampler = None
            tmp_test_neg_link_sampler = None
        elif strategies == conf['strategy']:
            tmp_train_neg_link_sampler = train_neg_link_sampler
            tmp_val_neg_link_sampler = val_neg_link_sampler
            tmp_test_neg_link_sampler = test_neg_link_sampler
        else:
            (tmp_train_src_nodes, tmp_train_dst_nodes, 
             tmp_val_src_nodes, tmp_val_dst_nodes, 
             tmp_test_src_nodes, tmp_test_dst_nodes) = get_node_sets(strategy, train_data, val_data, test_data)

            neg_sampler_instance = getattr(negative_sampler, conf['neg_sampler'])
            tmp_train_neg_link_sampler = neg_sampler_instance(tmp_train_src_nodes, tmp_train_dst_nodes,
                                                              check_link_existence=not conf['no_check_link_existence'],
                                                              name='train', seed=conf['exp_seed']+1)
            tmp_val_neg_link_sampler = neg_sampler_instance(tmp_val_src_nodes, tmp_val_dst_nodes,
                                                            check_link_existence=not conf['no_check_link_existence'],
                                                            name='val', seed=conf['exp_seed']+2)
            tmp_test_neg_link_sampler = neg_sampler_instance(tmp_test_src_nodes, tmp_test_dst_nodes,
                                                             check_link_existence=not conf['no_check_link_existence'],
                                                             name='test', seed=conf['exp_seed']+3)

        model.reset_memory()
        neighbor_loader.reset_state()

        fake_tr=False
        try:
            tr_scores, tr_true_values = eval(data=data, model=model, loader=train_loader, criterion=criterion, 
                                            neighbor_loader=neighbor_loader, neg_sampler=tmp_train_neg_link_sampler, 
                                            helper=assoc, eval_seed=conf['exp_seed'], device=device, 
                                            eval_name='train_eval', wandb_log=conf['wandb'])
        except:
            ts_scores = compute_fake_score(conf)
            vl_scores = compute_fake_score(conf)
            tr_scores = compute_fake_score(conf)
            tr_true_values, vl_true_values, ts_true_values = torch.tensor([-1, -1]), torch.tensor([-1, -1]), torch.tensor([-1, -1])
            fake_tr=True
        
        if not fake_tr: 
            fake_vl=False
            try:
                if conf['reset_memory_eval']:
                    model.reset_memory()
                vl_scores, vl_true_values = eval(data=data, model=model, loader=val_loader, criterion=criterion, 
                                                neighbor_loader=neighbor_loader, neg_sampler=tmp_val_neg_link_sampler, 
                                                helper=assoc, eval_seed=conf['exp_seed'], device=device, 
                                                eval_name='val_eval', wandb_log=conf['wandb'])
            except:
                ts_scores = compute_fake_score(conf)
                vl_scores = compute_fake_score(conf)
                vl_true_values, ts_true_values = torch.tensor([-1, -1]), torch.tensor([-1, -1])
                fake_vl=True
            
            if not fake_vl:
                try:
                    if conf['reset_memory_eval']:
                        model.reset_memory()

                    ts_scores, ts_true_values = eval(data=data, model=model, loader=test_loader, criterion=criterion, 
                                                    neighbor_loader=neighbor_loader, neg_sampler=tmp_test_neg_link_sampler, 
                                                    helper=assoc, eval_seed=conf['exp_seed'], device=device, 
                                                    eval_name='test_eval', wandb_log=conf['wandb'])
                except:
                    ts_scores = compute_fake_score(conf)
                    ts_true_values = torch.tensor([-1, -1])

        ckpt['test_score'][strategy] = ts_scores
        ckpt['val_score'][strategy] = vl_scores
        ckpt['train_score'][strategy] = tr_scores
        ckpt['true_values'][strategy] = (tr_true_values, vl_true_values, ts_true_values)
        ckpt['loss'][strategy] = (tr_scores['loss'], vl_scores['loss'], ts_scores['loss'])

    ckpt['train_ended'] = True
    torch.save(ckpt, path_save_best)

    history = ckpt['history'] if conf['log'] else None
    conf['model size'] = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return ckpt['test_score'], ckpt['val_score'], ckpt['train_score'], ckpt['epoch'], conf, history
