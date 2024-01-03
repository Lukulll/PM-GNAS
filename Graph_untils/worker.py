import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from net import Net
import time
import numpy as np
from ogb.nodeproppred import Evaluator

class Worker:
    def __init__(self, hp, data, device, dname):
        self.device = device
        self.hp = hp
        self.data = data.to(self.device)
        self.in_dim = data.num_features
        self.dname = dname 
        if self.dname == "ogbn-proteins":
            self.eva = "binary"
            self.cri = nn.BCEWithLogitsLoss()
            self.evaluator = Evaluator(name=dname)
            self.out_dim = 112
        else:
            self.eva = "classical"
            self.out_dim = data.num_classes

    def evaluate(self, output, labels, mask):
        _, indices = torch.max(output, dim=1)
        correct = torch.sum(indices[mask] == labels[mask])
        return correct.item() * 1.0 / mask.sum().item()

    def count_parameters_in_MB(self, model):
        return np.sum(np.prod(v.size()) for name, v in model.named_parameters())/1e6

    def run(self, arch, return_all_info = False):
        if type(arch) == tuple:
            ops, link = arch
        elif type(arch) == dict:
            ops, link = arch['op'], arch['link']
        else:
            ops, link = arch.ops, arch.link
        model = Net(ops, link, self.hp, self.in_dim, self.out_dim, self.dname).to(self.device)
        opt = torch.optim.Adam if self.hp.optimizer == "Adam" else torch.optim.SGD
        optimizer = opt(model.parameters(), lr=self.hp.lr, weight_decay=self.hp.wd)

        # Train model
        best_performance = 0
        min_val_loss = float("inf")
        data = self.data
        labels = data.y
        #print(labels.size())
        dur = []
        latency = 0

        for epoch in range(self.hp.num_epochs):
            model.train()
            optimizer.zero_grad()
            #print(epoch)

            logits = model(data)
            if self.eva == "classical":
                logits = F.log_softmax(logits, 1)
                loss = F.nll_loss(logits[data.train_mask], labels[data.train_mask])
            elif self.eva == "binary":
                labels = labels.float()
                loss = self.cri(logits[data.train_mask], labels[data.train_mask])
            loss.backward()
            optimizer.step()

            # evaluate
            model.eval()
            t0 = time.time()
            logits = model(data)
            latency = time.time() - t0

            if self.eva == "classical":
                logits = F.log_softmax(logits, 1)

                train_acc = self.evaluate(logits, labels, data.train_mask)
                val_acc = self.evaluate(logits, labels, data.val_mask)
                test_acc = self.evaluate(logits, labels, data.test_mask)
                train_loss = float(F.nll_loss(logits[data.train_mask], labels[data.train_mask]))
                val_loss = float(F.nll_loss(logits[data.val_mask], labels[data.val_mask]))
                test_loss = float(F.nll_loss(logits[data.test_mask], labels[data.test_mask]))

            elif self.eva == "binary":
                #print(logits)
                train_acc = self.evaluator.eval({'y_true': labels[data.train_mask], 'y_pred': logits[data.train_mask]})['rocauc']
                val_acc = self.evaluator.eval({'y_true': labels[data.val_mask], 'y_pred': logits[data.val_mask]})['rocauc']
                test_acc = self.evaluator.eval({'y_true': labels[data.test_mask], 'y_pred': logits[data.test_mask]})['rocauc']
                train_loss = self.cri(logits[data.train_mask], labels[data.train_mask]).item()
                val_loss = self.cri(logits[data.val_mask], labels[data.val_mask]).item()
                test_loss = self.cri(logits[data.test_mask], labels[data.test_mask]).item()

            if val_loss < min_val_loss:  # and train_loss < min_train_loss
                min_val_loss = val_loss
                best_performance = test_acc

            info = (
                train_acc,
                val_acc,
                test_acc,
                train_loss,
                val_loss,
                test_loss,
                best_performance
            )
            info = tuple(map(lambda x:round(x, 3), info))
            dur.append(info)

        if return_all_info:
            infos = {}
            infos['perf'] = best_performance
            infos['dur'] = dur
            infos['para'] = self.count_parameters_in_MB(model)
            infos['latency'] = latency
            return infos
        else:
            return best_performance