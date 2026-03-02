# Autoencoder for Image Denoising

A deep learning pipeline designed to remove noise from images using a **Variational Autoencoder (VAE)** architecture built in PyTorch. 

Image noise can significantly degrade image quality and hinder downstream computer vision tasks. By training an autoencoder to map noisy images into a compressed latent space and then reconstruct them back to their clean original form, the model learns to naturally filter out noise artifacts.

## 🌟 Key Features

- **Autoencoder Architecture**: Custom PyTorch model using symmetrical convolutional (encoder) and transpose-convolutional (decoder) blocks.
- **Reproducible Pipeline**: End-to-end training and evaluation orchestration using **DVC (Data Version Control)**.
- **Experiment Tracking**: Automatic logging of hyperparameters, loss curves, and model artifacts using **MLflow**.
- **Real-Time Visualization**: Integrates with **Visdom** to visualize testing and reconstruction results during execution.

## 🛠️ Tech Stack

- **Deep Learning**: PyTorch, TorchVision
- **MLOps**: DVC, MLflow
- **Visualization**: Visdom
- **Language**: Python

## 🚚 Installation

```bash
# Clone the repository
git clone https://github.com/aniketpoojari/Image-Denoising.git
cd Image-Denoising

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

The entire training process is managed via DVC. Before running the pipeline, you need to start the monitoring servers:

1. **Start MLflow Server** (for tracking metrics and models):
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0
```

2. **Start Visdom Server** (for real-time image visualization):
```bash
visdom
```

3. **Run the Pipeline**:
Configure your hyperparameters in `params.yaml`, then execute the pipeline:
```bash
dvc repro
```

## 📈 Evaluation & Results

- **Metric**: The model is evaluated using **Mean Squared Error (MSE)** between the reconstructed image and the clean ground truth.
- **Artifacts**: View metric graphs and model comparisons on your local MLflow server at `http://localhost:5000/`.
- **Deployment Ready**: The script automatically compares the new model against the baseline. If it improves the MSE, the model is saved to the `saved_models` directory for downstream deployment.
