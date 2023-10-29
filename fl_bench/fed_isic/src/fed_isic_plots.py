# Add TB plots for
# 1) Average val accuracy over all clients
# 2) Accuracy of final global model on pooled validation data

import argparse
import glob
import os

import dotenv
import pandas as pd
import tensorflow as tf
import torch
from sklearn.metrics import balanced_accuracy_score
from torch.utils.tensorboard.writer import SummaryWriter

from fl_bench.networks.fed_isic_nets import *
from fl_bench.fed_isic.src.FedIsicDM import FedIsicDataModule


def read_eventfile(filepath, tags=["val_metric_global_model"]):
    data = {}
    for summary in tf.compat.v1.train.summary_iterator(filepath):
        for v in summary.summary.value:
            if v.tag in tags:
                # print(v.tag, summary.step, v.simple_value)
                if v.tag in data.keys():
                    data[v.tag].append({'step':summary.step, 'value': v.simple_value})
                else:
                    data[v.tag] = [{'step':summary.step, 'value': v.simple_value}]
    return data


def main():
    dotenv.load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_folder", 
        type=str, 
        default='/path/to/Results/FedIsic',
        help="base_folder containing all experiments"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="debug",
        help="experiment name"
    )
    parser.add_argument("--model", type=str, default="efficientnet", help="model name")
    parser.add_argument("--device", type=str, default="gpu", help="device for pooled inference")

    args = parser.parse_args()

    # Load global model
    global_model = eval(f"{args.model}()")
    global_model_path = os.path.join(
        args.base_folder, args.experiment,
        "simulate_job",
        "app_server",
        "FL_global_model.pt"
    )
    global_model.load_state_dict(torch.load(global_model_path)['model'])
    device = torch.device(
        "cuda:0"
        if (torch.cuda.is_available() and args.device == "gpu")
        else "cpu"
    )
    global_model.to(device)

    # Load pooled validation data
    dataset_base_dir = os.getenv("ISIC_ROOT")
    data_module = FedIsicDataModule(dataset_base_dir, client_idx="9", central=True)

    y = []
    y_pred = []
    with torch.no_grad():

        global_model.eval()
        total_corrects = 0
        for batch_idx, batch_data in enumerate(data_module.val_dataloader):
            data, target = batch_data['image'].to(device), batch_data['label'].to(device)
            output = global_model(data)
            pred = output.argmax(dim=1, keepdim=True)
            corrects = pred.eq(target.view_as(pred)).sum().item()
            y.append(target)
            y_pred.append(pred)
            total_corrects += corrects
        y = torch.cat(y)
        y_pred = torch.cat(y_pred)
        balanced_acc = balanced_accuracy_score(y.cpu(), y_pred.cpu())

        print(f"Total corrects: {total_corrects}")
        print(f"Accuracy: {total_corrects/len(data_module.val_dataset)}")
        print(f"Balanced accuracy: {balanced_acc}")

    df = None
    for idx in range(6):
        run_path = os.path.join(
            args.base_folder,
            args.experiment,
            "simulate_job",
            f"app_site-{idx+1}",
            "events.out.tfevents.*"
        )
        eventfile = glob.glob(run_path)[0]
        for tag in ["val_metric_global_model"]:

            data = read_eventfile(eventfile)[tag]

        if df is None:
            df = pd.DataFrame(data).set_index('step')
            df['value'] = df[f'value_{idx+1}']
        else:
            df = df.join(pd.DataFrame(data).set_index('step'), how="outer", rsuffix=f'_{idx+1}')

    writer = SummaryWriter(os.path.join(args.base_folder, args.experiment))

    for step, row in df.iterrows():

        writer.add_scalar(step, row.mean(), f'avg_balanced_val_accuracy')
        writer.add_scalar(step, total_corrects/len(data_module.val_dataset), 'global_model_acc_on_pooled')


if __name__ == "__main__":
    main()
