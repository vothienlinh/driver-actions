# postprocessing/decision.py

"""
Decision module for temporal models.

Chuyển output của LSTM (logits) thành label cuối cùng.
"""

import torch
import torch.nn.functional as F


# =========================================================
# SOFTMAX ARGMAX DECISION
# =========================================================

def lstm_argmax_decision(logits):
    """
    Quyết định nhãn bằng argmax.

    Args
    ----
    logits : torch.Tensor
        Shape: (1, num_classes)

    Returns
    -------
    int
        class_id
    """

    pred_class = int(torch.argmax(logits, dim=1).item())

    return pred_class


# =========================================================
# SOFTMAX PROBABILITY DECISION
# =========================================================

def lstm_softmax_decision(logits):
    """
    Trả về class_id và probability.

    Useful cho debug hoặc visualization.
    """

    probs = F.softmax(logits, dim=1)

    pred_class = int(torch.argmax(probs, dim=1).item())

    confidence = float(probs[0, pred_class].item())

    return pred_class, confidence


# =========================================================
# MEAN SOFTMAX DECISION (OPTIONAL)
# =========================================================

def mean_softmax_decision(logits_list):
    """
    Mean softmax trên nhiều logits (temporal smoothing nhẹ).

    Args
    ----
    logits_list : list[Tensor]

    Returns
    -------
    int
        predicted class
    """

    if len(logits_list) == 0:
        return 0

    probs = [F.softmax(l, dim=1) for l in logits_list]

    mean_prob = torch.mean(torch.stack(probs), dim=0)

    pred_class = int(torch.argmax(mean_prob, dim=1).item())

    return pred_class
