import numpy as np
import torch
import evaluate
from focal_loss.focal_loss import FocalLoss
from transformers import Trainer, TrainerCallback
from copy import deepcopy
import matplotlib.pyplot as plt

""" Define metric for evaluation """
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

""" Collate function to pre-process images into tensors """
def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

""" Generate and save history plots """
def plot_history(ylabel,train_data,val_data,epoch):
    plt.plot(epoch,train_data,"b",label="Train")
    plt.plot(epoch,val_data,"r",label="Val")
    plt.ylabel(ylabel)
    plt.xlabel("Epoch")
    plt.legend()
    name = ylabel + ".png"
    plt.savefig(name)
    plt.clf()

""" Trainer class override to use Focal Loss """

loss_fn = FocalLoss(gamma=5)

class FocalTrainer(Trainer):
    
    def compute_loss(self,
                     model,
                     inputs,
                     return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        m = torch.nn.Softmax(dim=-1)
        loss = loss_fn(m(logits),labels)
        return (loss, outputs) if return_outputs else loss

""" Callback to get train accuracy (or preferred metric) """
class CustomCallback(TrainerCallback):
    
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy
        