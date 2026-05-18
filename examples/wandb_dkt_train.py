import argparse
from wandb_train import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="assist2015")
    parser.add_argument("--model_name", type=str, default="dkt")
    parser.add_argument("--emb_type", type=str, default="qid")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    # parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.2)
    
    parser.add_argument("--emb_size", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-3)

    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument("--add_uuid", type=int, default=1)

    # qid_tree controls for DKT (all optional; defaults match model defaults).
    # parser.add_argument("--tree_aux_loss_weight", type=float, default=0.3)
    # parser.add_argument("--tree_aux_decay", type=float, default=1.0)
    # parser.add_argument("--tree_aux_max_depth", type=int, default=1)
    # parser.add_argument("--tree_aux_negative_scale", type=float, default=1.0)

    # parser.add_argument("--tree_pred_fusion_mode", type=str, default="depth_decay")
    # parser.add_argument("--tree_pred_fusion_max_weight", type=float, default=0.5)
    # parser.add_argument("--tree_pred_fusion_fixed_weight", type=float, default=0)
    # parser.add_argument("--tree_pred_fusion_count_tau", type=float, default=50.0)
    # parser.add_argument("--tree_pred_fusion_depth_decay", type=float, default=0.1)
    # parser.add_argument("--tree_pred_fusion_counts_path", type=str, default="")

    # parser.add_argument("--tree_aux_gradient_mode", type=str, default="shared")
    # parser.add_argument("--tree_pred_fusion_source", type=str, default="main")
    # parser.add_argument("--tree_pred_fusion_apply_train", type=int, default=1)
    # parser.add_argument("--tree_pred_decay_lr_mult", type=float, default=20.0)
    
    args = parser.parse_args()

    params = vars(args)
    # params["tree_pred_fusion_apply_train"] = bool(int(params["tree_pred_fusion_apply_train"]))
    main(params)
