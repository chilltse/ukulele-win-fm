import os, sys
import argparse
import glob
import shutil
import json
from pykt.preprocess.split_datasets import main as split_concept
from pykt.preprocess.split_datasets_que import main as split_question
from pykt.preprocess import data_proprocess, process_raw_data

dname2paths = {
    "yousician": "../data/yousician/yousician_ukulele.json",
    "yousician_fmkc": "../data/yousician_fmkc/yousician_ukulele.json",
    "dbe_kt22": "../data/dbe_kt22/2_DBE_KT22_Practice_Sequences_100102_json/Practice_Sequences.json",
    "dbe_kt22_tree": "../data/dbe_kt22_tree/2_DBE_KT22_Practice_Sequences_100102_json/Practice_Sequences.json",
    "assist2009": "../data/assist2009/skill_builder_data_corrected_collapsed.csv",
    "assist2012": "../data/assist2012/2012-2013-data-with-predictions-4-final.csv",
    "assist2015": "../data/assist2015/2015_100_skill_builders_main_problems.csv",
    "algebra2005": "../data/algebra2005/algebra_2005_2006_train.txt",
    "bridge2algebra2006": "../data/bridge2algebra2006/bridge_to_algebra_2006_2007_train.txt",
    "statics2011": "../data/statics2011/AllData_student_step_2011F.csv",
    "nips_task34": "../data/nips_task34/train_task_3_4.csv",
    "poj": "../data/poj/poj_log.csv",
    "slepemapy": "../data/slepemapy/answer.csv",
    "assist2017": "../data/assist2017/anonymized_full_release_competition_dataset.csv",
    "assist2017_tree": "../data/assist2017_tree/anonymized_full_release_competition_dataset.csv",
    "xes3g5m": "../data/xes3g5m/question_level/train_valid_sequences_quelevel.csv",
    "xes3g5m_tree": "../data/xes3g5m_tree/question_level/train_valid_sequences_quelevel.csv",
    "xes3g5m_tree_manual_split": "../data/xes3g5m_tree_manual_split/question_level/train_valid_sequences_quelevel.csv",
    "xes3g5m_tree_split_v1": "../data/xes3g5m_tree_split_v1/question_level/train_valid_sequences_quelevel.csv",
    "xes3g5m_tree_split_v2": "../data/xes3g5m_tree_split_v2/question_level/train_valid_sequences_quelevel.csv",
    "nips_task34_tree": "../data/nips_task34_tree/train_task_3_4.csv",
    "junyi2015": "../data/junyi2015/junyi_ProblemLog_original.csv",
    "ednet": "../data/ednet/",
    "ednet5w": "../data/ednet/",
    "peiyou": "../data/peiyou/grade3_students_b_200.csv"
}
configf = "../configs/data_config.json"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--dataset_name", type=str, default="assist2015")
    parser.add_argument("-f","--file_path", type=str, default="../data/peiyou/grade3_students_b_200.csv")
    parser.add_argument("-m","--min_seq_len", type=int, default=3)
    parser.add_argument("-l","--maxlen", type=int, default=200)
    parser.add_argument("-k","--kfold", type=int, default=5)
    parser.add_argument("--rollup_node_ids", type=str, default=None)
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument(
        "--window",
        action="store_true",
        help="Enable sliding-window sequence generation for test sets.",
    )
    # parser.add_argument("--mode", type=str, default="concept",help="question or concept")
    args = parser.parse_args()

    print(args)

    # process raw data
    if args.dataset_name=="peiyou":
        dname2paths["peiyou"] = args.file_path
        print(f"fpath: {args.file_path}")
    config_dataset_name = args.dataset_name
    dname, writef = "", ""

    if args.dataset_name == "xes3g5m" and args.version:
        version = str(args.version).strip()
        with open(configf, "r", encoding="utf-8") as fin:
            data_config = json.load(fin)
        xes3g5m_config = data_config.get("xes3g5m", {})
        versioning = xes3g5m_config.get("versioning", {})

        versions_root = versioning.get("versions_root")
        output_template = versioning.get("output_dataset_dir_template")
        source_sequence_file = versioning.get("source_sequence_file")

        if not versions_root:
            raise KeyError("Missing xes3g5m.versioning.versions_root in configs/data_config.json")
        if not output_template:
            raise KeyError("Missing xes3g5m.versioning.output_dataset_dir_template in configs/data_config.json")
        if not source_sequence_file:
            raise KeyError("Missing xes3g5m.versioning.source_sequence_file in configs/data_config.json")

        version_root = os.path.normpath(os.path.join(versions_root, version))
        out_root = os.path.normpath(output_template.format(version=version))
        out_metadata_dir = os.path.join(out_root, "metadata")

        questions_src = os.path.join(version_root, "questions.json")
        kc_map_src = os.path.join(version_root, "kc_routes_map.json")
        if not os.path.exists(questions_src):
            raise FileNotFoundError(f"Versioned questions.json not found: {questions_src}")
        if not os.path.exists(kc_map_src):
            raise FileNotFoundError(f"Versioned kc_routes_map.json not found: {kc_map_src}")

        os.makedirs(out_metadata_dir, exist_ok=True)
        questions_dst = os.path.join(out_metadata_dir, "questions.json")
        kc_map_dst = os.path.join(out_metadata_dir, "kc_routes_map.json")
        shutil.copy2(questions_src, questions_dst)
        shutil.copy2(kc_map_src, kc_map_dst)

        from pykt.preprocess.xes3g5m_preprocess import read_data_from_csv
        read_file = source_sequence_file
        writef = os.path.join(out_root, "data.txt")
        dname = out_root
        old_questions_json = os.environ.get("QUESTIONS_JSON")
        old_kc_routes_map_json = os.environ.get("KC_ROUTES_MAP_JSON")
        os.environ["QUESTIONS_JSON"] = questions_dst
        os.environ["KC_ROUTES_MAP_JSON"] = kc_map_dst
        try:
            read_data_from_csv(
                read_file,
                writef,
                rollup_node_ids=args.rollup_node_ids,
            )
        finally:
            if old_questions_json is None:
                os.environ.pop("QUESTIONS_JSON", None)
            else:
                os.environ["QUESTIONS_JSON"] = old_questions_json
            if old_kc_routes_map_json is None:
                os.environ.pop("KC_ROUTES_MAP_JSON", None)
            else:
                os.environ["KC_ROUTES_MAP_JSON"] = old_kc_routes_map_json

        config_dataset_name = f"{args.dataset_name}_{version}"
    else:
        dname, writef = process_raw_data(
            args.dataset_name,
            dname2paths,
            rollup_node_ids=args.rollup_node_ids,
        )

    if args.dataset_name in {"xes3g5m", "xes3g5m_tree"} and args.rollup_node_ids:
        rollup_suffix = "_".join(
            [x.strip() for x in str(args.rollup_node_ids).split(",") if x.strip()]
        )
        if rollup_suffix:
            config_dataset_name = f"{args.dataset_name}_rollup_{rollup_suffix}"
    print("-"*50)
    print(f"dname: {dname}, writef: {writef}, config_dataset_name: {config_dataset_name}")
    # split
    # remove stale cached processed files across platforms
    for pkl_path in glob.glob(os.path.join(dname, "*.pkl")):
        try:
            os.remove(pkl_path)
        except OSError:
            pass

    #for concept level model
    split_concept(
        dname,
        writef,
        config_dataset_name,
        configf,
        args.min_seq_len,
        args.maxlen,
        args.kfold,
        args.window,
    )
    print("="*100)

    #for question level model
    split_question(
        dname,
        writef,
        config_dataset_name,
        configf,
        args.min_seq_len,
        args.maxlen,
        args.kfold,
        args.window,
    )

