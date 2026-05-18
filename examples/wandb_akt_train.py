import argparse
from wandb_train import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    parser.add_argument("--model_name", type=str, default="akt")
    parser.add_argument("--emb_type", type=str, default="qid")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    # parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.2)
    
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--num_attn_heads", type=int, default=8)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)

    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument("--add_uuid", type=int, default=1)

    # qid_tree controls for AKT (all optional; defaults keep current behavior).
    # parser.add_argument("--tree_embed_mode", type=str, default="independent")
    # parser.add_argument("--tree_aux_loss_weight", type=float, default=0.0)
    # parser.add_argument("--tree_aux_loss_mode", type=str, default="direct")
    # parser.add_argument("--tree_aux_max_depth", type=int, default=1)
    # parser.add_argument("--tree_aux_depths", type=str, default="")
    # parser.add_argument("--tree_pred_fusion_mode", type=str, default="none")
    # parser.add_argument("--tree_pred_fusion_weight", type=float, default=0.5)
    # parser.add_argument("--tree_pred_max_depth", type=int, default=1)
    # parser.add_argument("--tree_pred_depths", type=str, default="")
    # parser.add_argument("--tree_aux_ignore_first", type=int, default=1)
    # parser.add_argument("--tree_max_ancestor_depth", type=int, default=16)
    # parser.add_argument("--tree_label_level_up", type=int, default=0)
   
    args = parser.parse_args()

    params = vars(args)
    # Convert empty-string depth specs into None for AKT parser.
    # if params.get("tree_aux_depths", "") == "":
    #     params["tree_aux_depths"] = None
    # if params.get("tree_pred_depths", "") == "":
    #     params["tree_pred_depths"] = None
    # params["tree_aux_ignore_first"] = bool(int(params["tree_aux_ignore_first"]))
    main(params)
