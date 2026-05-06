import os, sys

def process_raw_data(dataset_name,dname2paths):
    readf = dname2paths[dataset_name]
    dname = "/".join(readf.split("/")[0:-1])
    writef = os.path.join(dname, "data.txt")
    if dataset_name in ["xes3g5m", "xes3g5m_tree"]:
        # Keep outputs in dataset root directory, consistent with other datasets.
        root_dir = "/".join(readf.split("/")[:-2])
        dname = root_dir
        writef = os.path.join(dname, "data.txt")
    if dataset_name in ["dbe_kt22", "dbe_kt22_tree"]:
        # Practice_Sequences.json is stored in a subdirectory; keep outputs in dataset root.
        root_dir = "/".join(readf.split("/")[:-2])
        dname = root_dir
        writef = os.path.join(dname, "data.txt")
        os.makedirs(dname, exist_ok=True)
    print(f"Start preprocessing data: {dataset_name}")
    if dataset_name == "yousician":
        from .yousician_preprocess import read_data_from_json
    if dataset_name == "yousician_fmkc":
        from .yousician_preprocess_kc_fmkc import read_data_from_json as read_data_from_json_fmkc
    if dataset_name in ["dbe_kt22", "dbe_kt22_tree"]:
        from .dbe_kt22_preprocess import read_data_from_json as read_data_from_json_dbe_kt22
    if dataset_name == "assist2009":
        from .assist2009_preprocess import read_data_from_csv
    elif dataset_name == "assist2012":
        from .assist2012_preprocess import read_data_from_csv
    elif dataset_name == "assist2015":
        from .assist2015_preprocess import read_data_from_csv
    elif dataset_name == "algebra2005":
        from .algebra2005_preprocess import read_data_from_csv
    elif dataset_name == "bridge2algebra2006":
        from .bridge2algebra2006_preprocess import read_data_from_csv
    elif dataset_name == "statics2011":
        from .statics2011_preprocess import read_data_from_csv
    elif dataset_name == "nips_task34":
        from .nips_task34_preprocess import read_data_from_csv
    elif dataset_name == "poj":
        from .poj_preprocess import read_data_from_csv
    elif dataset_name == "slepemapy":
        from .slepemapy_preprocess import read_data_from_csv
    elif dataset_name == "assist2017":
        from .assist2017_preprocess import read_data_from_csv
    elif dataset_name in ["xes3g5m", "xes3g5m_tree"]:
        from .xes3g5m_preprocess import read_data_from_csv
    elif dataset_name == "junyi2015":
        from .junyi2015_preprocess import read_data_from_csv, load_q2c
    elif dataset_name in ["ednet","ednet5w"]:
        from .ednet_preprocess import read_data_from_csv
    elif dataset_name == "peiyou":
        from .aaai2022_competition import read_data_from_csv, load_q2c
    
    if dataset_name == "junyi2015":
        dq2c = load_q2c(readf.replace("junyi_ProblemLog_original.csv","junyi_Exercise_table.csv"))
        read_data_from_csv(readf, writef, dq2c)
    elif dataset_name == "peiyou":
        fname = readf.split("/")[-1]
        dq2c = load_q2c(readf.replace(fname,"questions.json"))
        read_data_from_csv(readf, writef, dq2c)
    elif dataset_name in ["ednet5w","ednet"]:
        dname, writef = read_data_from_csv(readf, writef, dataset_name=dataset_name)
    elif dataset_name == "yousician":
        read_data_from_json(readf, writef)
    elif dataset_name == "yousician_fmkc":
        root = os.path.dirname(os.path.abspath(readf))
        data_root = os.path.dirname(root)
        dname = os.path.join(data_root, "yousician_fmkc")
        os.makedirs(dname, exist_ok=True)
        writef = os.path.join(dname, "data.txt")
        read_data_from_json_fmkc(readf, writef)
    elif dataset_name in ["dbe_kt22", "dbe_kt22_tree"]:
        read_data_from_json_dbe_kt22(readf, writef)
    elif dataset_name != "nips_task34":#default case
        read_data_from_csv(readf, writef)
    else:
        metap = os.path.join(dname, "metadata")
        read_data_from_csv(readf, metap, "task_3_4", writef)
     
    return dname,writef
