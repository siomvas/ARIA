import argparse
import os
import ast
from fl_bench.medmnist.src.utils import read_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-job", type=str, help="Path to job config.")
    parser.add_argument("-method", type=str, default='fedavg', choices=['fedavg', 'fedprox', 'fedopt', 'scaffold'])
    parser.add_argument("-alpha", type=float, default=1)
    parser.add_argument("-model_name", type=str, default=argparse.SUPPRESS)
    parser.add_argument("-num_rounds", type=int, default=argparse.SUPPRESS)
    parser.add_argument("-local_steps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("-lr", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--opt_args", type=str, default=argparse.SUPPRESS, help="optimizer arguments, pass as 'k:v, k:v'")
    parser.add_argument("-seed", type=int, default=222)
    parser.add_argument("-amp", type=bool, default=True)

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

    # update model
    print(f"Setting model name = {args.model_name}")
    model_path = f"fl_bench.networks.medmnist_nets.{args.model_name}"
    server_config["model_path"] = model_path
    train_config["model_name"] = args.model_name
    job_name += f'_{args.model_name}'
    # update rounds
    print(f"Setting num_rounds = {args.num_rounds}")
    server_config["num_rounds"] = args.num_rounds
    job_name += f'_{args.num_rounds}'
    # update lr
    print(f"Setting lr = {args.lr}")
    train_config["learning_rate"] = args.lr
    job_name += f'_{args.lr}'
    # update local steps
    print(f"Setting local steps = {args.local_steps}")
    train_config["local_steps"] = args.local_steps
    job_name += f'_{args.local_steps}'

    # update max steps
    print(f"Setting max steps = {args.num_rounds * args.local_steps}")
    train_config["max_steps"] = args.num_rounds * args.local_steps

    # update alpha
    print(f"Setting alpha = {args.alpha}")
    server_config["alpha"] = args.alpha
    # update method
    # learner_path = map_func(args.dataset, args.learner)
    print(f"Setting method = {args.method}")
    client_config["components"][0]["args"]["method"] = args.method
    if args.method == 'fedopt':
        server_config["workflows"][0]["args"]["shareable_generator_id"] = "shareable_generator_fedopt"
    else:
        server_config["workflows"][0]["args"]["shareable_generator_id"] = "shareable_generator_base"
    job_name += f'_medmnist_{args.method}'
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
        train_config["optimizer_args"] = ast.literal_eval(args.opt_args)
        write_json(train_config, train_config_filename)
    
    if "amp" in args:
        train_config = read_json(train_config_filename)
        train_config["use_half_precision"] = args.amp
        write_json(train_config, train_config_filename)


if __name__ == "__main__":
    main()
