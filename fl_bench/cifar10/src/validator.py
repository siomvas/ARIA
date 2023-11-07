from typing import Any, Dict
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class Validator(object):
    def __init__(self):
        self.metric_score = 0.0

    def validate_loop(self, model, data_loader) -> Dict[str, Any]:
        # Run inference over whole validation set
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                correct, total = 0, 0
                for batch in tqdm(data_loader, desc="Validation DataLoader", dynamic_ncols=True):
                    batch[0] = batch[0].to("cuda:0")
                    batch[1] = batch[1].to("cuda:0")
                    batch.append(model(batch[0]))
                    _, pred_label = torch.max(batch[2].data, 1)
                    total += batch[0].data.size()[0]
                    correct += (pred_label == batch[1].data).sum().item()
                self.metric_score = correct / float(total)

        metrics = {"accuracy": self.metric_score}

        return metrics

    def run(self, model: torch.nn.Module, data_loader: DataLoader) -> Dict[str, Any]:
        model.eval()
        return self.validate_loop(model, data_loader)
