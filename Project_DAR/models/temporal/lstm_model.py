# models/temporal/lstm_model.py
"""
LSTM temporal model for driver action recognition.

"""
import torch
import torch.nn as nn

from configs.ablation_config import (
    NUM_CLASSES,
    LSTM_HIDDEN_SIZE,
    LSTM_NUM_LAYERS,
    LSTM_MODEL_PATH,
    DEVICE
)


# =========================================================
# LSTM MODEL
# =========================================================

class ActionLSTM(nn.Module):
    """
    LSTM model nhận chuỗi one-hot vector.
    Input shape:
        (batch, sequence_length, num_classes)

    Output:
        logits cho num_classes
    """

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        hidden_size=LSTM_HIDDEN_SIZE,
        num_layers=LSTM_NUM_LAYERS
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=num_classes,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):

        out, _ = self.lstm(x)

        # chỉ lấy frame cuối
        out = out[:, -1, :]

        out = self.fc(out)

        return out


# =========================================================
# LOAD PRETRAINED MODEL
# =========================================================

def load_lstm_model(
    model_path=LSTM_MODEL_PATH,
    device=DEVICE
):
    """
    Load pretrained LSTM weights.
    """

    model = ActionLSTM().to(device)

    state_dict = torch.load(
        model_path,
        map_location=device
    )

    model.load_state_dict(state_dict)

    model.eval()

    #print("LSTM model loaded from:", model_path)

    return model
