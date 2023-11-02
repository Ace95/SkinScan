import warnings
warnings.filterwarnings('ignore')
from torchvision import transforms
from pytorch_grad_cam import run_dff_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor
from functools import partial
from xai_utils import category_name_to_index, run_grad_cam_on_image, print_top_categories, swinT_reshape_transform_huggingface

from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    Resize,
    ToTensor,
)

image = Image.open("./imgs/ISIC_0027672.jpg") 
model_path = "C:/Users/User/Documents/Corsi Uni/Tesi/models/HAM10k_ft/swinv2-base-patch4-window12-192-22k-finetuned-HAM10k_NOCROP"
model = AutoModelForImageClassification.from_pretrained(model_path)
image_processor = AutoImageProcessor.from_pretrained(model_path) 

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
if "height" in image_processor.size:
    size = (image_processor.size["height"], image_processor.size["width"])
    crop_size = size
    max_size = None
elif "shortest_edge" in image_processor.size:
    size = image_processor.size["shortest_edge"]
    crop_size = (size, size)
    max_size = image_processor.size.get("longest_edge")

transforms = Compose(
        [
            #Resize(size),
            #CenterCrop(crop_size),
            ToTensor(),
            normalize,
        ]
    )

img_transforms = Compose(
    [
        Resize(size),
        CenterCrop(crop_size),
    ]
)

img_tensor = transforms(image)
#image = img_transforms(image)

label = print_top_categories(model, img_tensor)

#target_layer = model.swinv2.pooler
target_layer = model.swinv2.layernorm
targets_for_gradcam = [ClassifierOutputTarget(category_name_to_index(model, label))]
reshape_transform = partial(swinT_reshape_transform_huggingface,
                            width=19,
                            height=15)

gradcam_img = Image.fromarray(run_grad_cam_on_image(model=model,
                      target_layer=target_layer,
                      targets_for_gradcam=targets_for_gradcam,
                      reshape_transform=reshape_transform,
                      input_tensor=img_tensor,
                      input_image=image ))
gradcam_img.save("./cam_img.png")

