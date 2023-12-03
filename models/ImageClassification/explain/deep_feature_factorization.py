import numpy as np
import torch
from typing import Callable
from sklearn.decomposition import NMF
import cv2

""" Class for extracting activations and registering gradients from targetted intermediate layers """
class ActivationsAndGradients:

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output

        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            return

        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()

""" Function to scale the heatmaps and resize them to img size """
def scale_cam_image(cam, target_size=None):
    result = []
    for img in cam:
        img = img - np.min(img)
        img = img / (1e-7 + np.max(img))
        if target_size is not None:
            img = cv2.resize(img, target_size)
        result.append(img)
    result = np.float32(result)

    return result

""" Compute dff """
def dff(activations: np.ndarray, n_components: int = 5):

    batch_size, channels, h, w = activations.shape
    reshaped_activations = activations.transpose((1, 0, 2, 3))
    reshaped_activations[np.isnan(reshaped_activations)] = 0
    reshaped_activations = reshaped_activations.reshape(
        reshaped_activations.shape[0], -1)
    offset = reshaped_activations.min(axis=-1)
    reshaped_activations = reshaped_activations - offset[:, None]

    model = NMF(n_components=n_components, init='random', random_state=25)
    W = model.fit_transform(reshaped_activations)
    H = model.components_
    concepts = W + offset[:, None]
    explanations = H.reshape(n_components, batch_size, h, w)
    explanations = explanations.transpose((1, 0, 2, 3))
    return concepts, explanations

""" Deep Feature Factorization (https://arxiv.org/abs/1806.10206) Pytorch implementation """
class DeepFeatureFactorization:

    def __init__(self,
                 model: torch.nn.Module,
                 target_layer: torch.nn.Module,
                 reshape_transform: Callable = None,
                 computation_on_concepts=None
                 ):
        self.model = model
        self.computation_on_concepts = computation_on_concepts
        self.activations_and_grads = ActivationsAndGradients(
            self.model, [target_layer], reshape_transform)

    def __call__(self,
                 input_tensor: torch.Tensor,
                 n_components: int = 16):
        batch_size, channels, h, w = input_tensor.size()
        _ = self.activations_and_grads(input_tensor)

        with torch.no_grad():
            activations = self.activations_and_grads.activations[0].cpu(
            ).numpy()

        concepts, explanations = dff(activations, n_components=n_components)

        processed_explanations = []

        for batch in explanations:
            processed_explanations.append(scale_cam_image(batch, (w, h)))

        if self.computation_on_concepts:
            with torch.no_grad():
                concept_tensors = torch.from_numpy(
                    np.float32(concepts).transpose((1, 0)))
                concept_outputs = self.computation_on_concepts(
                    concept_tensors).cpu().numpy()
            return concepts, processed_explanations, concept_outputs
        else:
            return concepts, processed_explanations

    def __del__(self):
        self.activations_and_grads.release()

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in ActivationSummary with block: {exc_type}. Message: {exc_value}")
            return True