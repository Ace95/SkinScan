#https://docs.google.com/spreadsheets/d/1iKYJf324aO133ARG5dR5fex8LxfhERQSTH3BIeDLzB0/edit#gid=376879794 Doc con i risultati di ogni train/test run.

from transformers import AutoImageProcessor
import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor
from datasets import load_dataset 
from torchmetrics.classification import ConfusionMatrix
import numpy as np


from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    Resize,
    ToTensor,
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#model_folder = "C:/Users/User/Documents/Corsi Uni/Tesi/models/HAM10k_ft/convnextv2-base-1k-224-finetuned-HAM10k" # ConvNextV2 
#model_folder = "C:/Users/User/Documents/Corsi Uni/Tesi/models/HAM10k_ft/swiftformer-xs-finetuned-HAM10k_nofc" # Swiftformer-XS
model_folder = "C:/Users/User/Documents/Corsi Uni/Tesi/models/HAM10k_ft/swinv2-base-patch4-window12-192-22k-finetuned-HAM10k_NOCROP" # Swin v2

test_ds = load_dataset("imagefolder", data_files="C:/Users/User/Documents/Corsi Uni/Tesi/data/test.zip",split="train")

image_processor = AutoImageProcessor.from_pretrained(model_folder)
model = AutoModelForImageClassification.from_pretrained(model_folder)

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
if "height" in image_processor.size:
    size = (image_processor.size["height"], image_processor.size["width"])
    crop_size = size
    max_size = None
elif "shortest_edge" in image_processor.size:
    size = image_processor.size["shortest_edge"]
    crop_size = (size, size)
    max_size = image_processor.size.get("longest_edge")

test_transforms = Compose(
        [
            Resize(size),
            CenterCrop(crop_size),
            ToTensor(),
            normalize,
        ]
    )

def preprocess_test(example_batch):
    """Apply test_transforms across a batch."""
    example_batch["pixel_values"] = [test_transforms(image.convert("RGB")) for image in example_batch["image"]]
    del example_batch["image"]
    return example_batch

test_ds.set_transform(preprocess_test)

testloader = torch.utils.data.DataLoader(test_ds, batch_size=1,
                                         shuffle=False)

model.eval()
correct = 0
total = 0
preds = np.array([])
target = np.array([])
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels  = data['pixel_values'], data['label']
        # calculate outputs by running images through the network
        outputs = model(images)
        logits = outputs.logits
        predicted = logits.argmax(-1).item()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print(correct, "   ", predicted, "  ", labels)
        preds = np.append(preds,predicted)
        target = np.append(target,labels)

confmat = ConfusionMatrix(task="multiclass",num_classes=7)

print("--------------------RESULTS--------------------------")
print("TOTAL: ",total)
print("CORRECT: ",correct)
print("ACCURACY: ", (correct/total)*100, "%")
print("--------------------ConfMat--------------------------")
print(confmat(torch.from_numpy(preds),torch.from_numpy(target)))