import argparse
import os
import json
from fl_bench.medmnist.src.utils import read_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-job", type=str, help="Path to job config.")
    parser.add_argument(
        "-method", type=str, default="fedavg", choices=["central", "fedavg", "fedprox", "fedopt", "scaffold", "fedinit"]
    )
    parser.add_argument("-alpha", type=float, default=1)
    parser.add_argument("-model_name", type=str, default=argparse.SUPPRESS)
    parser.add_argument("-num_rounds", type=int, default=argparse.SUPPRESS)
    parser.add_argument("-local_steps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("-lr", type=float, default=argparse.SUPPRESS)
    parser.add_argument(
        "--opt_args", type=str, default=argparse.SUPPRESS, help="optimizer arguments, pass as 'k:v, k:v'"
    )
    parser.add_argument("-seed", type=int, default=222)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--use_hyperion_dirs", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--train_split_root",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "splits", "combo") + os.sep,
    )

    args = parser.parse_args()

    job_name = os.path.basename(args.job)
    client_config_filename = os.path.join(args.job, job_name, "config", "config_fed_client.json")
    server_config_filename = os.path.join(args.job, job_name, "config", "config_fed_server.json")
    train_config_filename = os.path.join(args.job, job_name, "config", "config_train.json")
    meta_config_filename = os.path.join(args.job, "meta.json")

    client_config = read_json(client_config_filename)
    server_config = read_json(server_config_filename)
    train_config = read_json(train_config_filename)
    meta_config = read_json(meta_config_filename)

    # update model
    print(f"Setting model name = {args.model_name}")
    model_path = f"fl_bench.networks.medmnist_nets.{args.model_name}"
    server_config["model_path"] = model_path
    train_config["model_name"] = args.model_name
    job_name += f"_{args.model_name}"
    # update rounds
    print(f"Setting num_rounds = {args.num_rounds}")
    server_config["num_rounds"] = args.num_rounds
    job_name += f"_{args.num_rounds}"
    # update lr
    print(f"Setting lr = {args.lr}")
    train_config["learning_rate"] = args.lr
    job_name += f"_{args.lr}"
    # update local steps
    print(f"Setting local steps = {args.local_steps}")
    train_config["local_steps"] = args.local_steps
    job_name += f"_{args.local_steps}"

    # update max steps
    print(f"Setting max steps = {args.num_rounds * args.local_steps}")
    train_config["max_steps"] = args.num_rounds * args.local_steps

    # update alpha
    print(f"Setting alpha = {args.alpha}")
    server_config["alpha"] = args.alpha
    job_name = f"{job_name}_alpha{args.alpha}"
    meta_config["name"] = job_name
    split_dir = os.path.join(args.train_split_root, f"alpha={args.alpha}_seed={args.seed}")
    print(f"Set train split root to {split_dir}")
    server_config["TRAIN_SPLIT_ROOT"] = split_dir
    train_config["train_idx_root"] = split_dir
    write_json(client_config, client_config_filename)
    write_json(server_config, server_config_filename)
    write_json(meta_config, meta_config_filename)

    # update method
    # convert central to fedavg for config
    if args.method == "central":
        args.method = "fedavg"

    print(f"Setting method = {args.method}")
    client_config["components"][0]["args"]["method"] = args.method

    # FedOpt requires a different shareable generator
    if args.method == "fedopt":
        server_config["workflows"][0]["args"]["shareable_generator_id"] = "shareable_generator_fedopt"
    else:
        server_config["workflows"][0]["args"]["shareable_generator_id"] = "shareable_generator_base"

    # SCAFFOLD requires:
    # 1. A different trainer (automatic change via config)
    # 2. the addition of task filters for vars that should not have control variates
    # 3. Different aggregator
    # 4. Different scatter and gather workflow

    if args.method == "scaffold":
        server_config["task_result_filters"] = [
            {
                "tasks": ["train", "submit_model", "validate"],
                "filters": [
                    {
                        "name": "ExcludeVars",
                        "args": {"exclude_vars": "(?:num_batches_tracked)|(?:attn.relative_position_index)"},
                    }
                ],
            }
        ]
        server_config["components"][5]["args"]["expected_data_kind"] = {
            "_model_weights_": "WEIGHT_DIFF",
            "scaffold_c_diff": "WEIGHT_DIFF",
        }
        server_config["workflows"][0]["name"] = "ScatterAndGatherScaffold"

    else:
        server_config["task_result_filters"] = []
        for i in [4, 5]:
            if server_config["components"][i]["id"] == "aggregator":
                server_config["components"][i]["args"]["expected_data_kind"] = "WEIGHT_DIFF"
        server_config["workflows"][0]["name"] = "ScatterAndGather"

    job_name += f"_medmnist_{args.method}"
    # update seed
    print(f"Setting seed = {args.seed}")
    client_config["components"][0]["args"]["seed"] = args.seed
    server_config["seed"] = args.seed

    meta_config["name"] = job_name
    write_json(client_config, client_config_filename)
    write_json(server_config, server_config_filename)
    write_json(train_config, train_config_filename)
    write_json(meta_config, meta_config_filename)

    if "opt_args" in args:
        train_config = read_json(train_config_filename)
        # train_config["optimizer_args"] = ast.literal_eval(args.opt_args)
        train_config["optimizer_args"] = json.loads(args.opt_args)
        write_json(train_config, train_config_filename)

    train_config = read_json(train_config_filename)
    train_config["use_half_precision"] = True if args.amp else False
    write_json(train_config, train_config_filename)
    print(f"Setting amp = {args.amp}")

    server_config = read_json(server_config_filename)
    client_config = read_json(client_config_filename)
    server_config["hyperion"] = True if args.use_hyperion_dirs else False
    client_config["components"][0]["args"]["hyperion"] = True if args.use_hyperion_dirs else False
    write_json(server_config, server_config_filename)
    write_json(client_config, client_config_filename)
    print(f"Setting hyperion = {args.use_hyperion_dirs}")

    train_config = read_json(train_config_filename)
    train_config["batch_size"] = args.batch_size
    write_json(train_config, train_config_filename)
    print(f"Setting batch_size = {args.batch_size}")


if __name__ == "__main__":
    main()
