import numpy as np
import torch
from sklearn.metrics import confusion_matrix



def analyze_weaknesses_produce_weights(pth_path, eps=0.25):
    #with 3e-2 expected max weight = 4.6; 1e-3 => 6.9. for 0.25 - ~2.4

    stats = torch.load(pth_path, weights_only=False)


    targets, predictions = stats['train_targets'], stats['train_predictions']

    cm = confusion_matrix(y_true=targets, y_pred=predictions, normalize="true")
    cm = np.round(cm, 3)

    diagonals = np.diag(cm)

    # log-inverse weighting
    raw_weights = np.maximum( np.log(1.0 / (diagonals + eps)), np.zeros_like(diagonals) )
                         

    # add +1 to avoid zero-weights for perfect classes
    raw_weights = raw_weights + 1.0

    return raw_weights
