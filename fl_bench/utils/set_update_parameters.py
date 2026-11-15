import argparse
import json
import os
import uuid


def read_json(filename):
    assert os.path.isfile(filename), f"{filename} does not exist!"

    with open(filename, "r") as f:
        return json.load(f)


def write_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", type=str, choices=['cifar10', 'prostate_2d'], default='cifar10')
    parser.add_argument("-job", type=str, help="Path to job config.")
    parser.add_argument(
        "--train_split_root",
        type=str,
        default="splits",
        help="Location where to save data splits.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=argparse.SUPPRESS,
        help="Value controls the degree of heterogeneity. "
        "Lower values of alpha means higher heterogeneity."
        "Values of <= 0. means no data sampling. "
        "Assumes central training.",
    )
    parser.add_argument(
        "-learner",
        type=str,
        default='baseline',
        choices=['baseline', 'adamw', 'savemore', 'scaffold']
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=argparse.SUPPRESS,
        help="name of model",
    )
    parser.add_argument(
        "-save_freq",
        type=int,
        default= 10,
        help="Saving frequency for local models. Creates num_rounds // save_freq * n_clients models",
    )

    parser.add_argument(
        "--num_rounds",
        type=int,
        default=argparse.SUPPRESS,
        help="number of global rounds",
    )
    parser.add_argument(
        "--lr", type=float, default=argparse.SUPPRESS, help="local learning rate"
    )
    parser.add_argument(
        "--epochs", type=int, default=argparse.SUPPRESS, help="local aggregation epochs"
    )

    args = parser.parse_args()

    job_name = os.path.basename(args.job)
    client_config_filename = os.path.join(
        args.job, job_name, "config", "config_fed_client.json"
    )
    server_config_filename = os.path.join(
        args.job, job_name, "config", "config_fed_server.json"
    )
    meta_config_filename = os.path.join(args.job, "meta.json")
    client_config = read_json(client_config_filename)
    server_config = read_json(server_config_filename)
    meta_config = read_json(meta_config_filename)

    if "learner" in args and args.dataset == "cifar10":
        client_config = read_json(client_config_filename)
        dictt = {
            "baseline": "fl_bench.cifar10.pt.learners.cifar10_learner.CIFAR10Learner",
            "adamw": "fl_bench.cifar10.pt.learners.cifar10_learner_adamw.CIFAR10Learner_AdamW",
            "savemore": "fl_bench.cifar10.pt.learners.cifar10_learner_savemore.CIFAR10Learner_savemore",
            "scaffold": "fl_bench.cifar10.pt.learners.cifar10_scaffold_learner.CIFAR10ScaffoldLearner",
        }
        client_config["components"][0]["path"] = dictt[args.learner]
        print(f"Setting learner to {dictt[args.learner]}")
        write_json(client_config, client_config_filename)

    if "num_rounds" in args:
        # update federated rounds
        server_config = read_json(server_config_filename)
        meta_config = read_json(meta_config_filename)
        print(f"Setting num_rounds to num_rounds={args.num_rounds}")
        token = str(uuid.uuid4())
        job_name = f"{job_name}_num_rounds={args.num_rounds}"
        server_config["num_rounds"] = args.num_rounds
        meta_config["name"] = job_name

        write_json(server_config, server_config_filename)
        write_json(meta_config, meta_config_filename)

    if "lr" in args:
        # update federated rounds
        client_config = read_json(client_config_filename)
        meta_config = read_json(meta_config_filename)
        print(f"Setting lr to lr={args.lr}")
        token = str(uuid.uuid4())
        job_name = f"{job_name}_lr={args.lr}"
        client_config["components"][0]["args"]["lr"] = args.lr
        meta_config["name"] = job_name

        write_json(client_config, client_config_filename)
        write_json(meta_config, meta_config_filename)

    if "epochs" in args:
        # update local epochs
        client_config = read_json(client_config_filename)
        meta_config = read_json(meta_config_filename)
        print(f"Setting epochs to epochs={args.epochs}")
        token = str(uuid.uuid4())
        job_name = f"{job_name}_epochs={args.epochs}"
        client_config["components"][0]["args"]["aggregation_epochs"] = args.epochs
        meta_config["name"] = job_name
        write_json(client_config, client_config_filename)
        write_json(meta_config, meta_config_filename)

    if "model_name" in args:
        # update model name
        client_config = read_json(client_config_filename)
        server_config = read_json(server_config_filename)
        meta_config = read_json(meta_config_filename)
        print(f"Setting model name to {args.model_name}")
        job_name = f"{job_name}_model={args.model_name}"
        if args.dataset == 'cifar10':
            client_config["components"][0]["args"]["model_arch"] = args.model_name
            server_config["model_path"] = (
                "fl_bench.networks.cifar10_nets." + args.model_name
            )
        elif args.dataset == 'prostate_2d':
            client_config["components"][0]["args"]["model_arch"] = args.model_name
            server_config["model_path"] = (
                "fl_bench.networks.prostate2d_nets." + args.model_name
            )
        meta_config["name"] = job_name
        write_json(client_config, client_config_filename)
        write_json(server_config, server_config_filename)
        write_json(meta_config, meta_config_filename)

    if "save_freq" in args:
        # update save frequency
        client_config = read_json(client_config_filename)
        meta_config = read_json(meta_config_filename)
        print(f"Setting save_freq to {args.save_freq}")
        client_config["components"][0]["args"]["save_freq"] = args.save_freq
        write_json(client_config, client_config_filename)

    if "alpha" in args:
        # update alpha and split data dir
        print(f"Updating {meta_config_filename} to alpha={args.alpha}")

        if args.alpha > 0.0:
            job_name = f"{job_name}_alpha={args.alpha}"
            server_config["alpha"] = args.alpha
            meta_config["name"] = job_name
            split_dir = os.path.join(args.train_split_root, f"{job_name}")
            print(f"Set train split root to {split_dir}")
            server_config["TRAIN_SPLIT_ROOT"] = split_dir
            client_config["TRAIN_SPLIT_ROOT"] = split_dir
            write_json(client_config, client_config_filename)
            write_json(server_config, server_config_filename)
            write_json(meta_config, meta_config_filename)
        else:
            print("Assuming centralized training.")


if __name__ == "__main__":
    main()
