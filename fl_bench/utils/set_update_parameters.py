# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from email.policy import default
import json
import os
from socket import AI_MASK
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
    parser.add_argument("-job", type=str, help="Path to job config.")
    parser.add_argument(
        "--train_split_root", type=str, default=argparse.SUPPRESS, help="Location where to save data splits.",
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
        '--num_rounds',
        type=int,
        default=argparse.SUPPRESS,
        help='number of global rounds'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=argparse.SUPPRESS,
        help='local learning rate'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=argparse.SUPPRESS,
        help='local aggregation epochs'
    )    
    args = parser.parse_args()

    job_name = os.path.basename(args.job)
    client_config_filename = os.path.join(args.job, job_name, "config", "config_fed_client.json")
    server_config_filename = os.path.join(args.job, job_name, "config", "config_fed_server.json")
    meta_config_filename = os.path.join(args.job, "meta.json")
    client_config = read_json(client_config_filename)
    server_config = read_json(server_config_filename)
    meta_config = read_json(meta_config_filename)

    if 'alpha' in args:
        # update alpha and split data dir
        print(f"Updating {meta_config_filename} to alpha={args.alpha}")

        if args.alpha > 0.0:
            token = str(uuid.uuid4())
            job_name = f"{job_name}_alpha{args.alpha}"
            server_config["alpha"] = args.alpha
            meta_config["name"] = job_name
            split_dir = os.path.join(args.train_split_root, f"{job_name}_{token}")
            print(f"Set train split root to {split_dir}")
            server_config["TRAIN_SPLIT_ROOT"] = split_dir
            client_config["TRAIN_SPLIT_ROOT"] = split_dir
            write_json(client_config, client_config_filename)
            write_json(server_config, server_config_filename)
            write_json(meta_config, meta_config_filename)
        else:
            print("Assuming centralized training.")

    if 'num_rounds' in args:
        # update federated rounds
        print(f"Setting num_rounds to num_rounds={args.num_rounds}")
        token = str(uuid.uuid4())
        job_name = f"{job_name}_num_rounds{args.num_rounds}"
        server_config["num_rounds"] = args.num_rounds
        meta_config["name"] = job_name

        write_json(client_config, client_config_filename)
        write_json(server_config, server_config_filename)
        write_json(meta_config, meta_config_filename)

    if 'lr' in args:
        # update federated rounds
        print(f"Setting lr to lr={args.lr}")
        token = str(uuid.uuid4())
        job_name = f"{job_name}_lr{args.lr}"
        client_config["components"][0]["args"]["lr"] = args.lr
        meta_config["name"] = job_name

        write_json(client_config, client_config_filename)
        write_json(meta_config, meta_config_filename)

if __name__ == "__main__":
    main()


# TODO epochs
# TODO re-read json and meta
# Test aim 
# commit