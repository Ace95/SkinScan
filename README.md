# SkinScan - Recognition of pigmented skin lesions with Vision Transformers and Bayesian Networks
Thesis project for the MSc degree in Artificial Intelligence @ UNIBO a.y. 2022/23

# Introduction
SkinScan aims to provide reliable, efficient, and cost-effective tools to assist physicians in identifying pigmented skin lesions. The application harnesses the capabilities of two modules, each handling distinct data types: an image classifier for analyzing lesion images and a Bayesian network for estimating the probability of developing a specific disease. The resulting system is deployed through a user-friendly Web App utilizing Kserve, making the trained models and algorithms accessible on a Kubernetes cluster.

# Models
## Image Classification
In our research, we started by adapting and evaluating three different model architectures for image classification in order to find the optimal solution both in terms of accuracy and latency. After that, we developed two tools that help to explain the classification results by providing important insights about the salient areas of the image.

## Bayesian Network
Bayesian Networks (BN) are an example of probabilistic graphical models in the form of  directed acyclic graphs (DAG) in which each node represents a random variable and the edges indicate a probabilistic relationship between two nodes.In our application, a Bayesian network provides an efficient and reliable approach to handle all the non-visual information about the patient's medical history.
