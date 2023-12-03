import torch
from typing import List, Callable, Optional, Dict
from cams import GradCAM
import numpy as np
from scipy.special import softmax
import cv2
from deep_feature_factorization import DeepFeatureFactorization
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

""" Select which output we want to explain """
class ClassifierOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]
    
""" Model wrapper to return a tensor """
class HuggingfaceToTensorModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(HuggingfaceToTensorModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).logits
    
""" Reshape Swin-T v2 tensor """
def swinT_reshape_transform_huggingface(tensor, width, height):
    result = tensor.reshape(tensor.size(0),
                            height,
                            width,
                            tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result
    
""" Get label from id """
def category_name_to_index(model, category_name):
    name_to_index = dict((v, k) for k, v in model.config.id2label.items())
    return name_to_index[category_name]

""" Overlay the heatmap on the original image """
def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET,
                      image_weight: float = 0.5) -> np.ndarray:

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise Exception(
            f"image_weight should be in the range [0, 1].\
                Got: {image_weight}")
    
    cam = (1 - image_weight) * heatmap + image_weight * img
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    img = np.uint8(255 * img)
    result = np.hstack((img, cam))
    return np.uint8(result)

""" Run GradCAM on an image """
def run_grad_cam_on_image(model: torch.nn.Module,
                          target_layer: torch.nn.Module,
                          targets_for_gradcam: List[Callable],
                          reshape_transform: Optional[Callable],
                          input_tensor,
                          input_image,
                          method: Callable=GradCAM):
    with method(model=HuggingfaceToTensorModelWrapper(model),
                 target_layers=[target_layer],
                 reshape_transform=reshape_transform) as cam:

        repeated_tensor = input_tensor[None, :].repeat(len(targets_for_gradcam), 1, 1, 1)

        batch_results = cam(input_tensor=repeated_tensor,
                            targets=targets_for_gradcam)
        results = []
        for grayscale_cam in batch_results:
            visualization = show_cam_on_image(np.float32(input_image)/255,
                                              grayscale_cam,
                                              use_rgb=True)
            results.append(visualization)
        return np.hstack(results)
    
""" Print the Top k predicted categories """
def print_top_categories(model, img_tensor, top_k=7):
    outputs = model(img_tensor.unsqueeze(0))
    logits = outputs.logits
    probabilities = softmax(logits.cpu().detach().numpy(), axis=1)
    indices = logits.cpu()[0, :].detach().numpy().argsort()[-top_k :][::-1]
    for i in indices:
        print(f"Predicted class {i}: {model.config.id2label[i]}, Score: {probabilities[0][i]}")

    pred = logits.argmax(-1).item()
    label = model.config.id2label[pred]

    return label

""" Run Deep Feature Factorization on an image"""
def run_dff_on_image(model: torch.nn.Module,
                     target_layer: torch.nn.Module,
                     classifier: torch.nn.Module,
                     img_pil,
                     img_tensor: torch.Tensor,
                     reshape_transform=Optional[Callable],
                     n_components: int = 3,
                     top_k: int = 2) -> np.ndarray:

    rgb_img_float = np.array(img_pil) / 255
    dff = DeepFeatureFactorization(model=model,
                                   reshape_transform=reshape_transform,
                                   target_layer=target_layer,
                                   computation_on_concepts=classifier)

    concepts, batch_explanations, concept_outputs = dff(
        img_tensor[None, :], n_components)

    concept_outputs = torch.softmax(
        torch.from_numpy(concept_outputs),
        axis=-1).numpy()
    concept_label_strings = create_labels_legend(concept_outputs,
                                                 labels=model.config.id2label,
                                                 top_k=top_k)
    visualization = show_factorization_on_image(
        rgb_img_float,
        batch_explanations[0],
        image_weight=0.4,
        concept_labels=concept_label_strings)

    result = np.hstack((np.array(img_pil), visualization))
    return result

""" Generate dff heatmap and overlay it on the original image"""
def show_factorization_on_image(img: np.ndarray,
                                explanations: np.ndarray,
                                colors: List[np.ndarray] = None,
                                image_weight: float = 0.5,
                                concept_labels: List = None) -> np.ndarray:
    n_components = explanations.shape[0]
    if colors is None:
        _cmap = plt.cm.get_cmap('tab20')
        colors = [
            np.array(
                _cmap(i)) for i in np.arange(
                0,
                1,
                1.0 /
                n_components)]
    concept_per_pixel = explanations.argmax(axis=0)
    masks = []
    for i in range(n_components):
        mask = np.zeros(shape=(img.shape[0], img.shape[1], 3))
        mask[:, :, :] = colors[i][:3]
        explanation = explanations[i]
        explanation[concept_per_pixel != i] = 0
        mask = np.uint8(mask * 255)
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2HSV)
        mask[:, :, 2] = np.uint8(255 * explanation)
        mask = cv2.cvtColor(mask, cv2.COLOR_HSV2RGB)
        mask = np.float32(mask) / 255
        masks.append(mask)

    mask = np.sum(np.float32(masks), axis=0)
    result = img * image_weight + mask * (1 - image_weight)
    result = np.uint8(result * 255)

    if concept_labels is not None:
        px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
        fig = plt.figure(figsize=(result.shape[1] * px, result.shape[0] * px))
        plt.rcParams['legend.fontsize'] = int(
            14 * result.shape[0] / 256 / max(1, n_components / 6))
        lw = 5 * result.shape[0] / 256
        lines = [Line2D([0], [0], color=colors[i], lw=lw)
                 for i in range(n_components)]
        plt.legend(lines,
                   concept_labels,
                   mode="expand",
                   fancybox=True,
                   shadow=True)

        plt.tight_layout(pad=0, w_pad=0, h_pad=0)
        plt.axis('off')
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plt.close(fig=fig)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        data = cv2.resize(data, (result.shape[1], result.shape[0]))
        result = np.hstack((result, data))
    return result

""" Create legends showing which concepts are shown in the image"""
def create_labels_legend(concept_scores: np.ndarray,
                         labels: Dict[int, str],
                         top_k=2):
    concept_categories = np.argsort(concept_scores, axis=1)[:, ::-1][:, :top_k]
    concept_labels_topk = []
    for concept_index in range(concept_categories.shape[0]):
        categories = concept_categories[concept_index, :]
        concept_labels = []
        for category in categories:
            score = concept_scores[concept_index, category]
            label = f"{','.join(labels[category].split(',')[:3])}:{score:.2f}"
            concept_labels.append(label)
        concept_labels_topk.append("\n".join(concept_labels))
    return concept_labels_topk