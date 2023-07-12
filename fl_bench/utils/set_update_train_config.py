import argparse
import json
import os
import ast

from nvflare.app_opt.pt.fedopt import 

def read_json(filename):
    assert os.path.isfile(filename), f"{filename} does not exist!"

    with open(filename, "r") as f:
        return json.load(f)


def write_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def map_func(dataset: str, learner: str) -> str:
    """generates the proper learner path

    Args:
        dataset (str): name of the dataset
        learner (str): name of the learner

    Returns:
        learner_path (str): the path of the learner
    """

    pascal = dataset.replace("_", " ").title().replace(" ", "") + "Learner"
    learner_path = f"fl_bench.{dataset}.learners.{dataset}_{learner}_learner.{pascal}"

    return learner_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", type=str, choices=['fed_isic'], default='fed_isic')
    parser.add_argument("-job", type=str, help="Path to job config.")
    parser.add_argument(
        "-learner",
        type=str,
        default='base',
        choices=['base', 'scaffold', 'ditto', 'moon']
    )
    parser.add_argument(
        "-model_name",
        type=str,
        default=argparse.SUPPRESS,
        help="name of model",
    )

    parser.add_argument(
        "-num_rounds",
        type=int,
        default=argparse.SUPPRESS,
        help="number of global rounds",
    )
    parser.add_argument(
        "-epochs", type=int, default=argparse.SUPPRESS, help="number of local epochs"
    )

    parser.add_argument(
        "-lr", type=float, default=argparse.SUPPRESS, help="local learning rate"
    )
    parser.add_argument(
        "--opt_args",
        type=str,
        default=argparse.SUPPRESS,
        help="optimizer arguments, pass as 'k:v, k:v'"
    )

    parser.add_argument(
        "--save_freq",
        type=int,
        default= 10,
        help="Saving frequency for local models. Creates num_rounds // save_freq * n_clients models",
    )

    args = parser.parse_args()

    job_name = os.path.basename(args.job)
    client_config_filename = os.path.join(
        args.job, job_name, "config", "config_fed_client.json"
    )
    server_config_filename = os.path.join(
        args.job, job_name, "config", "config_fed_server.json"
    )
    train_config_filename = os.path.join(
        args.job, job_name, "config", "config_train.json"
    )
    meta_config_filename = os.path.join(args.job, "meta.json")

    client_config = read_json(client_config_filename)
    server_config = read_json(server_config_filename)
    train_config = read_json(train_config_filename)
    meta_config = read_json(meta_config_filename)

    # update model, rounds, epochs and lr
    print(f"Setting model name = {args.model_name}")
    model_path = f"fl_bench.networks.{args.dataset}_nets.{args.model_name}"
    server_config["model_path"] = model_path
    train_config["model_name"] = args.model_name
    job_name += f'_{args.model_name}'

    print(f"Setting num_rounds = {args.num_rounds}")
    server_config["num_rounds"] = args.num_rounds
    job_name += f'_{args.num_rounds}'

    print(f"Setting lr = {args.lr}")
    train_config["learning_rate"] = args.lr
    job_name += f'_{args.lr}'

    print(f"Setting epochs = {args.epochs}")
    train_config["epochs"] = args.epochs
    job_name += f'_{args.epochs}'

    learner_path = map_func(args.dataset, args.learner)
    print(f"Setting learner_path = {learner_path}")
    client_config["components"][0]["path"] = learner_path
    job_name += f'_{args.dataset}_{args.learner}'

    meta_config["name"] = job_name
    write_json(client_config, client_config_filename)
    write_json(server_config, server_config_filename)
    write_json(train_config, train_config_filename)
    write_json(meta_config, meta_config_filename)

    if "opt_args" in args:
        train_config = read_json(train_config_filename)
        train_config["optimizer_args"] = ast.literal_eval(args.opt_args)
        write_json(train_config, train_config_filename)

    if "save_freq" in args:
        train_config = read_json(train_config_filename)
        print(f"Setting save_freq to {args.save_freq}")
        train_config["save_freq"] = args.save_freq
        write_json(train_config, train_config_filename)


if __name__ == "__main__":
    main()
