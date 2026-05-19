import torch
from torch.nn.functional import one_hot


def get_valid_predictions(model, model_name, dcur, device):
    """
    Return q_arr, kc_arr, true_arr, pred_arr for one validation batch.

    Currently implemented for DKT only, matching the original diagnosis script.
    Add new model adapters here later for AKT/SAKT/SimpleKT/etc.
    """
    if model_name == "dkt":
        return get_dkt_valid_predictions(model, dcur, device)

    raise NotImplementedError(
        f"Diagnosis prediction adapter is not implemented for model_name={model_name!r}. "
        "Currently only 'dkt' is supported."
    )


def get_dkt_valid_predictions(model, dcur, device):
    required_keys = [
        "cseqs",
        "rseqs",
        "shft_cseqs",
        "shft_rseqs",
        "smasks",
    ]

    for key in required_keys:
        if key not in dcur:
            raise KeyError(f"Missing required key in batch: {key}")

    if "shft_qseqs" not in dcur:
        raise KeyError(
            "Missing 'shft_qseqs' in batch. "
            "You need question-level sequences. "
            "Please check whether your dataset was preprocessed with question IDs."
        )

    c = dcur["cseqs"].to(device)
    r = dcur["rseqs"].to(device)
    cshft = dcur["shft_cseqs"].to(device)
    rshft = dcur["shft_rseqs"].to(device)
    qshft = dcur["shft_qseqs"].to(device)
    sm = dcur["smasks"].to(device)

    # DKT forward:
    # y_full shape usually: [batch, seq_len, num_c]
    y_full = model(c.long(), r.long())

    # Select prediction corresponding to next concept cshft.
    y = (y_full * one_hot(cshft.long(), model.num_c)).sum(-1)

    pred_arr = torch.masked_select(y, sm).detach().cpu().numpy()
    true_arr = torch.masked_select(rshft, sm).detach().cpu().numpy()
    kc_arr = torch.masked_select(cshft, sm).detach().cpu().numpy()
    q_arr = torch.masked_select(qshft, sm).detach().cpu().numpy()

    return q_arr, kc_arr, true_arr, pred_arr
