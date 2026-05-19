import torch
from torch.nn.functional import one_hot


def normalize_model_name(model_name):
    """
    Normalize model name aliases.
    pyKT sometimes uses 'dkt+', while your scripts may use 'dkt_plus'.
    """
    name = str(model_name).lower().strip()

    if name in {"dkt", "dkt_plus", "dkt+"}:
        return name

    return name


def get_valid_predictions(model, model_name, dcur, device):
    """
    Return q_arr, kc_arr, true_arr, pred_arr for one validation batch.

    Supported:
    - dkt
    - dkt_plus / dkt+

    Note:
    DKT+ has the same forward prediction interface as DKT.
    The main difference is in the training loss regularization,
    not in how validation predictions are extracted.
    """
    model_name = normalize_model_name(model_name)

    if model_name == "dkt":
        return get_dkt_valid_predictions(model, dcur, device)

    if model_name in {"dkt_plus", "dkt+"}:
        return get_dkt_plus_valid_predictions(model, dcur, device)

    raise NotImplementedError(
        f"Diagnosis prediction adapter is not implemented for model_name={model_name!r}. "
        "Currently supported: 'dkt', 'dkt_plus', 'dkt+'."
    )


def get_dkt_plus_valid_predictions(model, dcur, device):
    """
    DKT+ prediction adapter.

    DKT+ uses the same prediction extraction logic as DKT:
    - input: concept sequence cseqs and response sequence rseqs
    - output: prediction distribution over next concepts
    - select the prediction corresponding to shft_cseqs
    """
    return get_dkt_like_valid_predictions(model, dcur, device)


def get_dkt_valid_predictions(model, dcur, device):
    """
    DKT prediction adapter.
    """
    return get_dkt_like_valid_predictions(model, dcur, device)


def get_dkt_like_valid_predictions(model, dcur, device):
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
    sm = dcur["smasks"].to(device).bool()

    # DKT / DKT+ forward:
    # y_full shape usually: [batch, seq_len, num_c]
    y_full = model(c.long(), r.long())

    # Select prediction corresponding to next concept cshft.
    # one_hot(cshft): [batch, seq_len, num_c]
    # y: [batch, seq_len]
    y = (y_full * one_hot(cshft.long(), num_classes=model.num_c).float()).sum(-1)

    pred_arr = torch.masked_select(y, sm).detach().cpu().numpy()
    true_arr = torch.masked_select(rshft, sm).detach().cpu().numpy()
    kc_arr = torch.masked_select(cshft, sm).detach().cpu().numpy()
    q_arr = torch.masked_select(qshft, sm).detach().cpu().numpy()

    return q_arr, kc_arr, true_arr, pred_arr