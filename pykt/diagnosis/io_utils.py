import json
from pathlib import Path


def safe_id_sort_key(x):
    s = str(x)
    if s.isdigit():
        return (0, int(s))
    return (1, s)


def join_ids(ids):
    return "|".join(str(x) for x in sorted(ids, key=safe_id_sort_key))


def join_names(ids, name_map):
    names = []
    for x in sorted(ids, key=safe_id_sort_key):
        try:
            names.append(name_map.get(int(x), ""))
        except Exception:
            names.append("")
    return "|".join(names)


def load_id_name_maps(data_config):
    """
    Read keyid2idx.json and invert:
        original_id -> internal_id
    into:
        internal_id -> original_id

    Returns:
        concept_name_map: dict[int, str]
        question_name_map: dict[int, str]
    """
    keyid2idx_path = Path(data_config["dpath"]) / "keyid2idx.json"
    concept_name_map = {}
    question_name_map = {}

    if not keyid2idx_path.exists():
        return concept_name_map, question_name_map

    try:
        with keyid2idx_path.open("r", encoding="utf-8") as f:
            keyid2idx = json.load(f)

        concept_map = keyid2idx.get("concepts", {})
        question_map = keyid2idx.get("questions", {})

        concept_name_map = {int(v): str(k) for k, v in concept_map.items()}
        question_name_map = {int(v): str(k) for k, v in question_map.items()}

    except Exception as e:
        print(f"[warn] failed to parse {keyid2idx_path}: {e}")

    return concept_name_map, question_name_map


def clean_model_config(model_name, cfg):
    model_config = dict(cfg)

    # These are training/logging-only configs, not constructor args for most pyKT models.
    training_only_keys = [
        "use_wandb",
        "learning_rate",
        "add_uuid",
        "l2",
        "tree_pred_decay_lr_mult",
    ]

    for k in training_only_keys:
        model_config.pop(k, None)

    return model_config


def resolve_data_config_path(data_config_path):
    """
    Default is ../configs/data_config.json, matching calls from examples/.
    If not found, fall back to configs/data_config.json from current working directory.
    """
    p = Path(data_config_path)

    if p.exists():
        return p.resolve()

    fallback = Path("configs/data_config.json")
    if fallback.exists():
        return fallback.resolve()

    raise FileNotFoundError(
        "data_config.json not found. Tried: "
        f"{p.resolve()} and {fallback.resolve()}"
    )


def resolve_diagnosis_output_dir(save_dir, output_dir):
    """
    Resolve diagnosis output dir.

    Rules:
    1. If output_dir is empty, use <save_dir>/d.
    2. If output_dir is absolute, use it directly.
    3. If output_dir is relative, create it under <save_dir>.

    Examples:
        --output_dir ""      -> <save_dir>/d
        --output_dir d       -> <save_dir>/d
        --output_dir diag    -> <save_dir>/diag
        --output_dir G:/diag -> G:/diag
    """
    save_dir = Path(save_dir).resolve()

    if output_dir is None or str(output_dir).strip() == "":
        return save_dir / "d"

    output_path = Path(str(output_dir).strip())

    if output_path.is_absolute():
        return output_path

    return save_dir / output_path
