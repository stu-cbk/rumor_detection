import numpy as np
import torch

class EarlyStopping:

    def __init__(self, patience=7, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.accs=0
        self.F1=0
        self.F2 = 0
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, accs,F1,F2,model,modelname,str):

        score = (accs+F1+F2) /3

        if self.best_score is None:
            self.best_score = score
            self.accs = accs
            self.F1 = F1
            self.F2 = F2
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print("BEST Accuracy: {:.3f}|R F1: {:.3f}|NR F1: {:.3f}"
                      .format(self.accs,self.F1,self.F2))
        else:
            self.best_score = score
            self.accs = accs
            self.F1 = F1
            self.F2 = F2
            self.save_checkpoint(val_loss, model,modelname,str)
            self.counter = 0