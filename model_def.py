# model_def.py
import math
import torch
import torch.nn as nn

class ConvFcCheckpointModel(nn.Module):
    """
    Matches checkpoints with keys like:
      conv_layers.<idx>.weight/.bias   (e.g., idx = 0,3,6)
      fc_layers.1.weight/.bias         (first Linear)
      fc_layers.4.weight/.bias         (final Linear)

    We keep param modules at the SAME indices as in the checkpoint so strict loading works.
    At runtime, we execute Conv -> ReLU -> MaxPool(2) after each conv while preserving key names.
    """
    def __init__(self, conv_specs, present_conv_idxs, max_conv_idx,
                 fc_in_features, fc_hidden_out, fc_out_features):
        super().__init__()

        # Build conv_layers with exact index positions so state_dict keys match strictly.
        mods = []
        last_out_ch = None
        for idx in range(max_conv_idx + 1):
            if idx in present_conv_idxs:
                in_ch, out_ch, kH, kW = conv_specs[idx]
                conv = nn.Conv2d(in_ch, out_ch, kernel_size=(kH, kW),
                                 stride=1, padding=(kH // 2, kW // 2), bias=True)
                mods.append(conv)  # exact index for checkpoint keys
                last_out_ch = out_ch
            else:
                # Fill gaps: after a conv at i, we put relu at i+1 and pool at i+2
                if (idx - 1) in present_conv_idxs:
                    mods.append(nn.ReLU(inplace=True))
                elif (idx - 2) in present_conv_idxs:
                    mods.append(nn.MaxPool2d(kernel_size=2))
                else:
                    mods.append(nn.Identity())
        self.conv_layers = nn.Sequential(*mods)

        if last_out_ch is None:
            raise RuntimeError("No conv layers found in checkpoint.")

        # Choose Hf x Wf so that last_out_ch * Hf * Wf == fc_in_features
        if fc_in_features % last_out_ch != 0:
            raise RuntimeError(f"fc_in_features ({fc_in_features}) not divisible by last conv channels ({last_out_ch}).")
        s = fc_in_features // last_out_ch
        r = int(max(1, int(round(s ** 0.5))))
        while r > 1 and s % r != 0:
            r -= 1
        Hf, Wf = r, max(1, s // r)
        self.adapt_pool = nn.AdaptiveAvgPool2d((Hf, Wf))

        # FC stack with the exact Linear positions (1 and 4)
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.2),                          # 0
            nn.Linear(fc_in_features, fc_hidden_out), # 1
            nn.ReLU(inplace=True),                    # 2
            nn.Dropout(0.2),                          # 3
            nn.Linear(fc_hidden_out, fc_out_features) # 4
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.adapt_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        return x


def build_from_state_dict(sd: dict, num_classes_file: int) -> nn.Module:
    """
    Build a ConvFcCheckpointModel that matches the provided state_dict strictly.
    Reads:
      - conv_layers.<idx>.weight -> [out_ch, in_ch, kH, kW]
      - fc_layers.1.weight/bias, fc_layers.4.weight/bias
    """
    # Which conv indices have params?
    present_conv_idxs = sorted({
        int(k.split('.')[1]) for k in sd.keys()
        if k.startswith("conv_layers.") and k.endswith(".weight")
    })
    if not present_conv_idxs:
        raise RuntimeError("No conv_layers.<idx>.weight keys found in checkpoint.")
    max_conv_idx = max(present_conv_idxs)

    # Shapes for each conv index
    conv_specs = {}
    for idx in present_conv_idxs:
        W = sd[f"conv_layers.{idx}.weight"]
        out_ch, in_ch, kH, kW = W.shape
        conv_specs[idx] = (in_ch, out_ch, kH, kW)

    # FC shapes (strictly required)
    if "fc_layers.1.weight" not in sd or "fc_layers.4.weight" not in sd:
        raise RuntimeError("Expected fc_layers.1 and fc_layers.4 Linear weights in checkpoint.")
    W1 = sd["fc_layers.1.weight"]    # [hidden, in]
    W2 = sd["fc_layers.4.weight"]    # [out, hidden]
    fc_in_features   = W1.shape[1]
    fc_hidden_out    = W1.shape[0]
    fc_out_features  = W2.shape[0]   # usually equals #classes learned

    net = ConvFcCheckpointModel(
        conv_specs=conv_specs,
        present_conv_idxs=set(present_conv_idxs),
        max_conv_idx=max_conv_idx,
        fc_in_features=fc_in_features,
        fc_hidden_out=fc_hidden_out,
        fc_out_features=fc_out_features,
    )

    # STRICT load to ensure exact param mapping
    net.load_state_dict(sd, strict=True)
    return net
