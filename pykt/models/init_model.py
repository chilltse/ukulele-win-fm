import torch
import numpy as np
import os
import csv

from .dkt import DKT
from .dkt_plus import DKTPlus
from .dkvmn import DKVMN
from .deep_irt import DeepIRT
from .sakt import SAKT
from .saint import SAINT
from .kqn import KQN
from .atkt import ATKT
from .dkt_forget import DKTForget
from .akt import AKT
from .gkt import GKT
from .gkt_utils import get_gkt_graph
from .lpkt import LPKT
from .lpkt_utils import generate_qmatrix
from .skvmn import SKVMN
from .hawkes import HawkesKT
from .iekt import IEKT
from .atdkt import ATDKT
from .simplekt import simpleKT
from .datakt import BAKTTime
from .qdkt import QDKT
from .qikt import QIKT
from .dimkt import DIMKT
from .sparsekt import sparseKT
from .rkt import RKT
from .folibikt import folibiKT
from .dtransformer import DTransformer
from .stablekt import stableKT
from .extrakt import extraKT
from .rekt import ReKT
from .cskt import CSKT
from .lefokt_akt import LEFOKT_AKT
from .ukt import UKT
from .hcgkt import HCGKT
from .robustkt import Robustkt
from .aegiskc import AegisKC, load_aegiskc_item_fields

device = "cpu" if not torch.cuda.is_available() else "cuda"


def _build_aegiskc_item_fields_from_sequences(data_config):
    dpath = data_config["dpath"]
    seq_file = data_config.get("train_valid_file", "train_valid_sequences.csv")
    seq_path = os.path.join(dpath, seq_file)
    if not os.path.exists(seq_path):
        raise FileNotFoundError(f"aegiskc auto-build failed, sequence file not found: {seq_path}")

    num_c = int(data_config["num_c"])
    field_dims_cfg = data_config.get("num_c_fmkc", None)
    num_fields = len(field_dims_cfg) if isinstance(field_dims_cfg, list) and field_dims_cfg else None

    item_fields = None
    max_vals = None

    csv.field_size_limit(10**8)
    with open(seq_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if "concepts" not in (reader.fieldnames or []) or "concepts_dense" not in (reader.fieldnames or []):
            raise ValueError(
                "aegiskc auto-build requires both `concepts` and `concepts_dense` columns in "
                f"{seq_path}"
            )

        for row in reader:
            concepts = str(row.get("concepts", "")).split(",")
            dense = str(row.get("concepts_dense", "")).split(",")
            n = min(len(concepts), len(dense))
            for i in range(n):
                d_tok = str(dense[i]).strip()
                c_tok = str(concepts[i]).strip()
                if d_tok in {"", "-1"} or c_tok in {"", "-1"}:
                    continue
                did = int(d_tok)
                parts = [int(x) for x in c_tok.split("^")]
                if num_fields is None:
                    num_fields = len(parts)
                if len(parts) != num_fields:
                    continue
                if item_fields is None:
                    item_fields = [[-1] * num_fields for _ in range(num_c)]
                    max_vals = [0] * num_fields
                if 0 <= did < num_c and item_fields[did][0] == -1:
                    item_fields[did] = parts
                for j, v in enumerate(parts):
                    if v > max_vals[j]:
                        max_vals[j] = v

    if item_fields is None:
        raise ValueError(f"aegiskc auto-build failed: no valid concepts/concepts_dense pairs found in {seq_path}")

    for idx in range(num_c):
        if item_fields[idx][0] == -1:
            item_fields[idx] = [0] * num_fields

    if isinstance(field_dims_cfg, list) and len(field_dims_cfg) == num_fields:
        field_dims = [int(x) for x in field_dims_cfg]
    else:
        field_dims = [int(v) + 1 for v in max_vals]
    field_names = [f"field_{i}" for i in range(num_fields)]
    return field_names, field_dims, torch.tensor(item_fields, dtype=torch.long)

def init_model(model_name, model_config, data_config, emb_type):
    tree_num_c = data_config.get("num_c_tree", data_config["num_c"])
    if model_name == "dkt":
        if emb_type != "qid":
            raise ValueError(
                f"Simple DKT only supports emb_type='qid', but got emb_type='{emb_type}'."
            )
        allowed_dkt_cfg = {"emb_size", "dropout", "pretrain_dim"}
        dkt_model_config = {
            k: v for k, v in model_config.items() if k in allowed_dkt_cfg
        }
        model = DKT(
            data_config["num_c"],
            emb_type="qid",
            emb_path=data_config.get("emb_path", ""),
            **dkt_model_config,
        ).to(device)
    elif model_name == "dkt+":
        dktplus_kw = {"emb_type": emb_type, "emb_path": data_config["emb_path"]}
        if emb_type == "qid_fmkc":
            dktplus_kw["num_c_fmkc"] = data_config["num_c_fmkc"]
        model = DKTPlus(data_config["num_c"], **model_config, **dktplus_kw).to(device)
    elif model_name == "dkvmn":
        dkvmn_use_question = bool(model_config.pop("dkvmn_use_question", False))
        model = DKVMN(
            data_config["num_c"],
            num_q=data_config.get("num_q", data_config["num_c"]),
            use_question_input=dkvmn_use_question,
            **model_config,
            emb_type=emb_type,
            emb_path=data_config["emb_path"],
        ).to(device)
    elif model_name == "deep_irt":
        model = DeepIRT(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "sakt":
        sakt_kw = {"emb_type": emb_type, "emb_path": data_config["emb_path"]}
        if emb_type == "qid_fmkc":
            sakt_kw["num_c_fmkc"] = data_config["num_c_fmkc"]
        model = SAKT(data_config["num_c"],  **model_config, **sakt_kw).to(device)
    elif model_name == "saint":
        model = SAINT(data_config["num_q"], data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "dkt_forget":
        model = DKTForget(data_config["num_c"], data_config["num_rgap"], data_config["num_sgap"], data_config["num_pcount"], **model_config).to(device)
    elif model_name == "akt":
        akt_kw = {
            "emb_type": emb_type,
            "emb_path": data_config["emb_path"],
        }
        if emb_type == "qid_fmkc":
            akt_kw["num_c_fmkc"] = data_config["num_c_fmkc"]
        model = AKT(
            data_config["num_c"],
            data_config["num_q"],
            **model_config,
            **akt_kw,
        ).to(device)
    elif model_name == "lefokt_akt":
        model = LEFOKT_AKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "extrakt":
        model = extraKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "folibikt":
        model = folibiKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "kqn":
        model = KQN(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "atkt":
        model = ATKT(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"], fix=False).to(device)
    elif model_name == "atktfix":
        model = ATKT(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"], fix=True).to(device)
    elif model_name == "gkt":
        graph_type = model_config['graph_type']
        fname = f"gkt_graph_{graph_type}.npz"
        graph_path = os.path.join(data_config["dpath"], fname)
        if os.path.exists(graph_path):
            graph = torch.tensor(np.load(graph_path, allow_pickle=True)['matrix']).float()
        else:
            graph = get_gkt_graph(data_config["num_c"], data_config["dpath"], 
                    data_config["train_valid_original_file"], data_config["test_original_file"], graph_type=graph_type, tofile=fname)
            graph = torch.tensor(graph).float()
        model = GKT(data_config["num_c"], **model_config,graph=graph,emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "lpkt":
        qmatrix_path = os.path.join(data_config["dpath"], "qmatrix.npz")
        if os.path.exists(qmatrix_path):
            q_matrix = np.load(qmatrix_path, allow_pickle=True)['matrix']
        else:
            q_matrix = generate_qmatrix(data_config)
        q_matrix = torch.tensor(q_matrix).float().to(device)
        model = LPKT(data_config["num_at"], data_config["num_it"], data_config["num_q"], data_config["num_c"], **model_config, q_matrix=q_matrix, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "skvmn":
        model = SKVMN(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)   
    elif model_name == "hawkes":
        if data_config["num_q"] == 0 or data_config["num_c"] == 0:
            print(f"model: {model_name} needs questions ans concepts! but the dataset has no both")
            return None
        model = HawkesKT(data_config["num_c"], data_config["num_q"], **model_config)
        model = model.double()
        # print("===before init weights"+"@"*100)
        # model.printparams()
        model.apply(model.init_weights)
        # print("===after init weights")
        # model.printparams()
        model = model.to(device)
    elif model_name == "iekt":
        model = IEKT(num_q=data_config['num_q'], num_c=data_config['num_c'],
                max_concepts=data_config['max_concepts'], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"],device=device).to(device)   
    elif model_name == "qdkt":
        model = QDKT(num_q=data_config['num_q'], num_c=data_config['num_c'],
                max_concepts=data_config['max_concepts'], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"],device=device).to(device)
    elif model_name == "qikt":
        model = QIKT(num_q=data_config['num_q'], num_c=data_config['num_c'],
                max_concepts=data_config['max_concepts'], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"],device=device).to(device)
    elif model_name == "atdkt":
        model = ATDKT(data_config["num_q"], data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "datakt":
        model = BAKTTime(data_config["num_c"], data_config["num_q"], data_config["num_rgap"], data_config["num_sgap"], data_config["num_pcount"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "simplekt":
        simplekt_kw = {"emb_type": emb_type, "emb_path": data_config["emb_path"]}
        if emb_type == "qid_fmkc":
            simplekt_kw["num_c_fmkc"] = data_config["num_c_fmkc"]
        model = simpleKT(data_config["num_c"], data_config["num_q"], **model_config, **simplekt_kw).to(device)
    elif model_name == "rekt":
        model = ReKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type).to(device)
    elif model_name == "stablekt":
        model = stableKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "dimkt":
        model = DIMKT(data_config["num_q"],data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "sparsekt":
        model = sparseKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "rkt":
        model = RKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device) 
    elif model_name == "cskt":
        model = CSKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device) 
    elif model_name == "ukt":
        model = UKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "hcgkt":
        model = HCGKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "robustkt":
        model = Robustkt(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "dtransformer":
        model = DTransformer(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type,
                     emb_path=data_config["emb_path"]).to(device)      
    elif model_name == "aegiskc":
        item_fields_path = model_config.pop("aegiskc_item_fields", "") or data_config.get("aegiskc_item_fields", "")
        if item_fields_path:
            _, field_dims, item_fields = load_aegiskc_item_fields(item_fields_path, data_config["num_c"])
        else:
            _, field_dims, item_fields = _build_aegiskc_item_fields_from_sequences(data_config)
        for k in ["batch_size", "num_epochs", "use_wandb", "add_uuid", "learning_rate"]:
            model_config.pop(k, None)
        model = AegisKC(
            num_c=data_config["num_c"],
            emb_type=emb_type,
            field_dims=field_dims,
            item_fields=item_fields,
            **model_config,
        ).to(device)
    else:
        print("The wrong model name was used...")
        return None
    return model

def load_model(model_name, model_config, data_config, emb_type, ckpt_path):
    model = init_model(model_name, model_config, data_config, emb_type)
    net = torch.load(os.path.join(ckpt_path, emb_type+"_model.ckpt"))
    model.load_state_dict(net)
    return model
