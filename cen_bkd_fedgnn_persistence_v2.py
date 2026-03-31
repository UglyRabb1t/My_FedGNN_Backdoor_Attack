"""
Backdoor Persistence Experiment for Federated Learning
=================================================
This script implements the backdoor persistence experiment:
1. Attack phase: Inject backdoor triggers from epoch_backdoor to epoch_stop_backdoor
2. Persistence phase: Stop trigger injection, continue benign training
3. Observe how the backdoor accuracy degrades over time
"""
import argparse

from Common.Node.workerbasev2 import WorkerBaseV2
from gradmask import compute_grad_mask, apply_grad_mask
import torch
from torch import nn
from torch import device
import json
import os

from Common.Utils.options import args_parser
from Common.Utils.gnn_util import inject_global_trigger_test, inject_global_trigger_train, load_pkl, split_dataset
import time
from Common.Utils.evaluate import gnn_evaluate_accuracy_v2
from GNN_common.train.metrics import accuracy_TU as accuracy
import numpy as np
import torch.nn.functional as F
from GNN_common.data.TUs import TUsDataset
from GNN_common.nets.TUs_graph_classification.load_net import gnn_model  # import GNNs
from torch.utils.data import DataLoader
from defense import foolsgold
import copy


def server_robust_agg(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


class ClearDenseClient(WorkerBaseV2):
    def __init__(self, client_id, model, loss_func, train_iter, attack_iter, test_iter, config, optimizer, device, grad_stub, args, scheduler, is_malicious=False):
        super(ClearDenseClient, self).__init__(model=model, loss_func=loss_func, train_iter=train_iter, attack_iter=attack_iter,
                                               test_iter=test_iter, config=config, optimizer=optimizer, device=device)
        self.client_id = client_id
        self.grad_stub = None
        self.args = args
        self.scheduler = scheduler
        self.is_malicious = is_malicious

    def update(self):
        pass


class MaliciousClient(ClearDenseClient):
    """
    Malicious client for backdoor persistence experiment.
    Can switch between attack mode and benign mode.
    """
    def __init__(self, client_id, model, loss_func, train_iter, attack_iter, test_iter, config, optimizer, device, grad_stub, args, scheduler):
        super(MaliciousClient, self).__init__(
            client_id=client_id, model=model, loss_func=loss_func, train_iter=train_iter,
            attack_iter=attack_iter, test_iter=test_iter, config=config, optimizer=optimizer,
            device=device, grad_stub=grad_stub, args=args, scheduler=scheduler, is_malicious=True
        )
        self.mask_grad_list = None
        self.benign_loader = None  # Store benign loader for computing benign gradients

    def set_grad_mask(self, mask_grad_list):
        """Set the gradient mask computed from global model"""
        self.mask_grad_list = mask_grad_list

    def set_attack_mode(self, enable_attack, train_loader=None, attack_loader=None):
        """Switch between attack mode and benign mode"""
        if enable_attack and train_loader is not None:
            self.train_iter = train_loader
        else:
            # Use benign data for training
            pass  # Will be set externally

        if enable_attack and attack_loader is not None:
            self.attack_iter = attack_loader
        else:
            self.attack_iter = None

    def gnn_train_v2(self):
        """Local training with optional GradMask for backdoor attacks"""
        from torch.nn.utils import parameters_to_vector
        initial_model_params = parameters_to_vector(self.model.parameters()).detach()

        self.model.train()
        self.acc_record = [0]
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()

        # Compute gradients on attack data (backdoor data)
        attack_grad_list = []
        self.optimizer.zero_grad()
        for batch_graphs, batch_labels in self.train_iter:
            batch_graphs = batch_graphs.to(self.device)
            batch_x = batch_graphs.ndata['feat'].to(self.device)
            batch_e = batch_graphs.edata['feat'].to(self.device)
            batch_labels = batch_labels.to(torch.long)
            batch_labels = batch_labels.to(self.device)
            batch_scores = self.model.forward(batch_graphs, batch_x, batch_e)
            l = self.model.loss(batch_scores, batch_labels)
            l.backward()
            train_l_sum += l.cpu().item()
            train_acc_sum += accuracy(batch_scores, batch_labels)
            n += batch_labels.size(0)
            batch_count += 1

        # Store attack gradients
        if self.args.gradmask_ratio < 1.0 and self.mask_grad_list is not None:
            for param in self.model.parameters():
                if param.grad is not None:
                    attack_grad_list.append(param.grad.clone())
            self.optimizer.zero_grad()

            # Compute gradients on benign data
            for batch_graphs, batch_labels in self.benign_loader:
                batch_graphs = batch_graphs.to(self.device)
                batch_x = batch_graphs.ndata['feat'].to(self.device)
                batch_e = batch_graphs.edata['feat'].to(self.device)
                batch_labels = batch_labels.to(torch.long).to(self.device)
                batch_scores = self.model.forward(batch_graphs, batch_x, batch_e)
                l = self.model.loss(batch_scores, batch_labels)
                l.backward()

            # Replace masked gradients with benign gradients
            mask_iter = iter(self.mask_grad_list)
            benign_grad_idx = 0
            for param in self.model.parameters():
                if param.grad is not None:
                    mask = next(mask_iter)
                    # Where mask is 0, use benign gradient; where mask is 1, use attack gradient
                    param.grad = attack_grad_list[benign_grad_idx] * mask + param.grad * (1 - mask)
                    benign_grad_idx += 1

        self.optimizer.step()

        self._weights_list = []
        self._level_length = [0]

        for param in self.model.parameters():
            self._level_length.append(param.data.numel() + self._level_length[-1])
            self._weights_list += param.data.view(-1).cpu().numpy().tolist()

        self._weights = self.model.state_dict()

        # Evaluate
        if self.attack_iter is not None:
            test_acc, test_l, att_acc = self.gnn_evaluate()
        else:
            test_acc, test_l = self.gnn_evaluate()

        with torch.no_grad():
            self._update = parameters_to_vector(self.model.parameters()).double() - initial_model_params

        return train_l_sum / batch_count, train_acc_sum / n, test_l, test_acc


class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


if __name__ == '__main__':
    args = args_parser()

    with open(args.config) as f:
        config = json.load(f)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataset = TUsDataset(args)

    collate = dataset.collate
    MODEL_NAME = config['model']
    net_params = config['net_params']
    if MODEL_NAME in ['GCN', 'GAT']:
        if net_params['self_loop']:
            print("[!] Adding graph self-loops for GCN/GAT models (central node trick).")
            dataset._add_self_loops()
    net_params['in_dim'] = dataset.all.graph_lists[0].ndata['feat'][0].shape[0]
    num_classes = torch.max(dataset.all.graph_labels).item() + 1
    net_params['n_classes'] = num_classes
    net_params['dropout'] = args.dropout

    ## set a global model
    global_model = gnn_model(MODEL_NAME, net_params)
    global_model = global_model.to(device)

    client = []
    loss_func = nn.CrossEntropyLoss()
    # Load data
    partition, avg_nodes = split_dataset(args, dataset)
    drop_last = True if MODEL_NAME == 'DiffPool' else False
    filename = "./Data/global_trigger/%d/%s_%s_%d_%d_%d_%.2f_%.2f_%.2f"\
              %(args.seed, MODEL_NAME, config['dataset'], args.num_workers, args.num_mali, args.epoch_backdoor,
                 args.frac_of_avg, args.poisoning_intensity, args.density) + '.pkl'
    global_trigger = load_pkl(filename)
    print("Triggers loaded!")
    args.num_mali = len(global_trigger)

    # Create loaders for attacker client (client[0])
    train_trigger_graphs, final_idx = inject_global_trigger_train(partition[0], avg_nodes, args, global_trigger)
    test_trigger_graphs = inject_global_trigger_test(partition[-1], avg_nodes, args, global_trigger)
    tmp_graphs = [partition[0][idx] for idx in range(len(partition[0])) if idx not in final_idx]

    # Backdoor data loader (for attack phase)
    train_dataset = train_trigger_graphs + tmp_graphs
    backdoor_train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                   drop_last=drop_last,
                                   collate_fn=dataset.collate)
    # Benign data loader (for persistence phase and gradient mask computation)
    benign_train_loader = DataLoader(tmp_graphs, batch_size=args.batch_size, shuffle=True,
                                   drop_last=drop_last,
                                   collate_fn=dataset.collate)
    backdoor_attack_loader = DataLoader(test_trigger_graphs, batch_size=args.batch_size, shuffle=True,
                                   drop_last=drop_last,
                                   collate_fn=dataset.collate)

    # Create local trigger loaders
    test_local_trigger_load = []
    for i in range(len(global_trigger)):
        test_local_trigger = inject_global_trigger_test(partition[-1], avg_nodes, args, [global_trigger[i]])
        tmp_load = DataLoader(test_local_trigger, batch_size=args.batch_size, shuffle=True,
                           drop_last=drop_last,
                           collate_fn=dataset.collate)
        test_local_trigger_load.append(tmp_load)

    # Initialize clients
    for i in range(args.num_workers):
        local_model = copy.deepcopy(global_model)
        local_model = local_model.to(device)
        optimizer = torch.optim.Adam(local_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=args.step_size, gamma=args.gamma)

        print("Client %d training data num: %d"%(i, len(partition[i])))
        print("Client %d testing data num: %d"%(i, len(partition[-1])))
        train_loader = DataLoader(partition[i], batch_size=args.batch_size, shuffle=True,
                                    drop_last=drop_last,
                                    collate_fn=dataset.collate)
        attack_loader = None
        test_loader = DataLoader(partition[-1], batch_size=args.batch_size, shuffle=True,
                                    drop_last=drop_last,
                                    collate_fn=dataset.collate)

        # Use MaliciousClient for client[0], ClearDenseClient for others
        if i == 0:
            client.append(MaliciousClient(client_id=i, model=local_model, loss_func=loss_func, train_iter=train_loader,
                                      attack_iter=attack_loader, test_iter=test_loader, config=config, optimizer=optimizer,
                                      device=device, grad_stub=None, args=args, scheduler=scheduler))
        else:
            client.append(ClearDenseClient(client_id=i, model=local_model, loss_func=loss_func, train_iter=train_loader,
                                      attack_iter=attack_loader, test_iter=test_loader, config=config, optimizer=optimizer,
                                      device=device, grad_stub=None, args=args, scheduler=scheduler))

    print(f"\n{'='*80}")
    print(f"BACKDOOR PERSISTENCE EXPERIMENT")
    print(f"{'='*80}")
    print(f"Attack Phase:      Epoch {args.epoch_backdoor} - {args.epoch_stop_backdoor-1}")
    print(f"Persistence Phase:  Epoch {args.epoch_stop_backdoor} - {args.epochs-1}")
    print(f"{'='*80}\n")

    weight_history = []

    for epoch in range(args.epochs):
        print(f'\nepoch: {epoch}')

        # Determine if we are in attack phase or persistence phase
        in_attack_phase = args.epoch_backdoor <= epoch < args.epoch_stop_backdoor

        # Only attack if in attack phase AND at the attack interval
        should_attack = in_attack_phase and (epoch % args.attack_interval == 0)

        if should_attack:
            print(f"[ATTACK PHASE] Injecting backdoor triggers into client[0] (interval: every {args.attack_interval} epochs)")
            client[0].train_iter = backdoor_train_loader
            client[0].benign_loader = benign_train_loader  # Set benign loader for computing benign gradients
            client[0].attack_iter = backdoor_attack_loader
            # Compute gradient mask using global model on benign data
            if args.gradmask_ratio < 1.0:
                print("Computing gradient mask with global model on benign data...")
                mask_grad_list = compute_grad_mask(
                    model=global_model,
                    benign_data_loader=benign_train_loader,
                    loss_func=loss_func,
                    device=device,
                    ratio=args.gradmask_ratio,
                    aggregate_all_layer=False
                )
                client[0].set_grad_mask(mask_grad_list)
                print(f"Gradient mask computed, retaining {args.gradmask_ratio * 100}% of smallest gradient parameters")
        else:
            # In attack phase but not at attack interval: use benign data
            if in_attack_phase:
                print(f"[ATTACK PHASE] Interval epoch - using benign data (attack every {args.attack_interval} epochs)")
                client[0].train_iter = benign_train_loader
                client[0].benign_loader = None
                client[0].attack_iter = None
                client[0].set_grad_mask(None)  # Clear gradient mask
            # Persistence phase: use benign data only
            elif epoch >= args.epoch_stop_backdoor:
                print(f"[PERSISTENCE PHASE] Stopping backdoor injection, using benign data only")
                client[0].train_iter = benign_train_loader
                client[0].benign_loader = None
                client[0].attack_iter = None
                client[0].set_grad_mask(None)  # Clear gradient mask

        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()

        for i in range(args.num_workers):
            att_list = []
            train_loss, train_acc, test_loss, test_acc = client[i].gnn_train_v2()
            client[i].scheduler.step()

            # Always evaluate attack accuracy on all clients
            global_att = gnn_evaluate_accuracy_v2(backdoor_attack_loader, client[i].model)
            print('Client %d, loss %.4f, train acc %.3f, test loss %.4f, test acc %.3f'
                    % (i, train_loss, train_acc, test_loss, test_acc))
            print('Client %d with global trigger: %.3f'%(i, global_att))

            for j in range(len(global_trigger)):
                tmp_acc = gnn_evaluate_accuracy_v2(test_local_trigger_load[j], client[i].model)
                print('Client %d with local trigger %d: %.3f'%(i, j, tmp_acc))
                att_list.append(tmp_acc)

            if not args.filename == "":
                save_path = os.path.join(args.filename, str(args.seed), config['model'] + '_' + args.dataset + '_%d_%d_%.2f_%.2f_%.2f'\
                          %(args.num_workers, args.num_mali, args.frac_of_avg, args.poisoning_intensity, args.density) + '_%d.txt'%i)
                path = os.path.split(save_path)[0]
                isExist = os.path.exists(path)
                if not isExist:
                    os.makedirs(path)
                with open(save_path, 'a') as f:
                    f.write('%.3f %.3f %.3f %.3f %.3f '%(train_loss, train_acc, test_loss, test_acc, global_att))
                    for j in range(len(global_trigger)):
                        f.write('%.3f'%att_list[j])
                        f.write(' ')
                    f.write('\n')

        weights = []
        for i in range(args.num_workers):
            weights.append(client[i].get_weights())
            weight_history.append(client[i].get_weights_list())

        # Aggregation in the server to get the global model
        if args.defense == 'foolsgold':
            result, weight_history, alpha = foolsgold(args, weight_history, weights, global_model, client[0])
            save_path = os.path.join("./Results/alpha/DBA_Persistence", str(args.seed), MODEL_NAME + '_' + args.dataset + \
                        '_%d_%d_%.2f_%.2f_%.2f'%(args.num_workers, args.num_mali, args.frac_of_avg, args.poisoning_intensity, args.density) + '_alpha.txt')
            path = os.path.split(save_path)[0]
            isExist = os.path.exists(path)
            if not isExist:
                os.makedirs(path)
            with open(save_path, 'a') as f:
                for i in range(args.num_workers):
                    f.write("%.3f" % (alpha[i]))
                    f.write(' ')
                f.write("\n")
        else:
            result = server_robust_agg(weights)

        for i in range(args.num_workers):
            client[i].set_weights(weights=result)
            client[i].upgrade()

        # update global model's weights
        global_model.load_state_dict(result)

        # evaluate the global model: test_acc
        test_acc = gnn_evaluate_accuracy_v2(client[0].test_iter, global_model)
        print("Global Test acc: %.3f"%test_acc)

        if not args.filename == "":
            save_path = os.path.join(args.filename, str(args.seed), MODEL_NAME + '_' + args.dataset + '_%d_%d_%.2f_%.2f_%.2f'\
                       %(args.num_workers, args.num_mali, args.frac_of_avg, args.poisoning_intensity, args.density) + '_global_test.txt')
            path = os.path.split(save_path)[0]
            isExist = os.path.exists(path)
            if not isExist:
                os.makedirs(path)

            with open(save_path, 'a') as f:
                f.write("%.3f" % (test_acc))
                f.write("\n")

        # Always track attack accuracy
        local_att_acc = []
        global_att_acc = gnn_evaluate_accuracy_v2(backdoor_attack_loader, global_model)
        print('Global model with global trigger: %.3f'%global_att_acc)

        for i in range(len(global_trigger)):
            tmp_acc = gnn_evaluate_accuracy_v2(test_local_trigger_load[i], global_model)
            print('Global model with local trigger %d: %.3f'%(i, tmp_acc))
            local_att_acc.append(tmp_acc)

        if not args.filename == "":
            save_path = os.path.join(args.filename, str(args.seed), MODEL_NAME + '_' + args.dataset + '_%d_%d_%.2f_%.2f_%.2f'%(args.num_workers, args.num_mali, args.frac_of_avg, args.poisoning_intensity, args.density) + '_global_attack.txt')
            path = os.path.split(save_path)[0]
            isExist = os.path.exists(path)
            if not isExist:
                os.makedirs(path)
            with open(save_path, 'a') as f:
                f.write("%.3f" % (global_att_acc))
                f.write(' ')
                for i in range(len(global_trigger)):
                    f.write("%.3f" % (local_att_acc[i]))
                    f.write(' ')
                f.write('\n')
