from datasets import load_dataset 
from transformers import AutoImageProcessor
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
from utils import compute_metrics, collate_fn, FocalTrainer, CustomCallback, plot_history
import torch
import pandas as pd
import numpy as np


from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
# Load training and validatio data-sets and pre-process the images

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_checkpoint = "MBZUAI/swiftformer-xs"
#model_checkpoint = "MBZUAI/swiftformer-s"
batch_size = 16
n_epochs = 50

train_ds = load_dataset("imagefolder", data_files="C:/Users/User/Documents/Corsi Uni/Tesi/data/train.zip",split="train")
val_ds = load_dataset("imagefolder", data_files="C:/Users/User/Documents/Corsi Uni/Tesi/data/val.zip",split="train")


labels = train_ds.features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

image_processor  = AutoImageProcessor.from_pretrained(model_checkpoint)

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
if "height" in image_processor.size:
    size = (image_processor.size["height"], image_processor.size["width"])
    crop_size = size
    max_size = None
elif "shortest_edge" in image_processor.size:
    size = image_processor.size["shortest_edge"]
    crop_size = (size, size)
    max_size = image_processor.size.get("longest_edge")

train_transforms = Compose(
        [
            RandomResizedCrop(crop_size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(crop_size),
            ToTensor(),
            normalize,
        ]
    )

def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    example_batch["pixel_values"] = [
        train_transforms(image.convert("RGB")) for image in example_batch["image"]
    ]
    return example_batch

def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch

train_ds.set_transform(preprocess_train)
val_ds.set_transform(preprocess_val)

# Model training

model = AutoModelForImageClassification.from_pretrained(
    model_checkpoint, 
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes = True, # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
)
model.to(device)
model_name = model_checkpoint.split("/")[-1]

args = TrainingArguments(
    f"{model_name}-finetuned-HAM10k_nofc",
    remove_unused_columns=False,
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=n_epochs,
    warmup_ratio=0.1,
    metric_for_best_model="accuracy",
    push_to_hub=False,
    logging_steps = 100
)

trainer = FocalTrainer(
    model,
    args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)

trainer.add_callback(CustomCallback(trainer)) 
train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()
train_history = pd.DataFrame(trainer.state.log_history)

epoch = range(1,n_epochs+1)
train_loss = np.array(train_history["train_loss"].dropna())
train_loss = np.delete(train_loss,-1)
train_accuracy = np.array(train_history["train_accuracy"].dropna())
eval_loss = np.array(train_history["eval_loss"].dropna())
eval_accuracy = np.array(train_history["eval_accuracy"].dropna())

plot_history("Accuracy",train_accuracy,eval_accuracy,epoch)
plot_history("Loss",train_loss,eval_loss,epoch) 