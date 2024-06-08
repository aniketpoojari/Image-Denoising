from data_loader import get_data_loader
from model import get_model, initialize_weights, get_essentials
import argparse
from common import read_params
import visdom
import numpy as np
import torch
import torchvision
import mlflow
from datetime import datetime


def training(config_path):
    config = read_params(config_path)
    training_dir = config["data"]["training_dir"]
    testing_dir = config["data"]["testing_dir"]
    features = config["training"]["features"]
    lr = config["training"]["lr"]
    visualize = config["training"]["visualize"]
    epoches = config["training"]["epoches"]
    batch_size = config["training"]["batch_size"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mlflow_config = config["mlflow_config"]
    experiment_name = mlflow_config["experiment_name"]
    run_name = mlflow_config["run_name"]
    registered_model_name = mlflow_config["registered_model_name"]
    server_uri = mlflow_config["server_uri"]
    mlflow.set_tracking_uri(server_uri)
    mlflow.set_experiment(experiment_name=experiment_name)

    train_loader = get_data_loader(training_dir, batch_size)
    test_loader = get_data_loader(testing_dir, batch_size)

    AutoEncoder = get_model(features).to(device)
    criterion, optmizer = get_essentials(AutoEncoder, lr)
    initialize_weights(AutoEncoder)

    if visualize:
        vis = visdom.Visdom(env="Image Denoising")
        image_grid_plot = vis.image(np.random.rand(3, 32, 32).astype(np.float32))

    with mlflow.start_run(run_name=run_name) as mlflow_run:

        for epoch in range(epoches):

            train_losses = []
            train_l = 0
            AutoEncoder.train()
            for noisy, clean in train_loader:
                encoding = AutoEncoder(noisy.to(device))

                # Calculate the loss
                loss = criterion(encoding, clean.to(device))
                train_losses.append(loss * noisy.shape[0])
                train_l += noisy.shape[0]

                # Zero the gradients, perform a backward pass, and update the weights
                optmizer.zero_grad()
                loss.backward()
                optmizer.step()

            test_losses = []
            test_l = 0
            AutoEncoder.eval()
            with torch.no_grad():
                for noisy, clean in test_loader:
                    encoding = AutoEncoder(noisy.to(device))

                    # Calculate the loss
                    loss = criterion(encoding, clean.to(device))
                    test_losses.append(loss * noisy.shape[0])
                    test_l += noisy.shape[0]

            # Print losses
            print(
                f"Epoch [{epoch}/{epoches}] Train Loss: {sum(train_losses)/train_l} Test Loss: {sum(test_losses)/test_l}"
            )

            if visualize:
                input = torch.empty(1, 3, 224, 224)
                output = torch.empty(1, 3, 224, 224)
                AutoEncoder.eval()
                with torch.no_grad():
                    for noisy, clean in test_loader:
                        input = torch.cat((input, noisy), dim=0)
                        output = torch.cat(
                            (output, AutoEncoder(noisy.to(device)).cpu()), dim=0
                        )

                # Create image grids for visualization
                n = torchvision.utils.make_grid(input[1:6].cpu(), normalize=True)
                c = torchvision.utils.make_grid(output[1:6].cpu(), normalize=True)

                # Concatenate real and fake image grids vertically for visualization
                combined_grid = torch.cat((n, c), dim=1)

                # Send the combined grid to Visdom for visualization
                vis.image(combined_grid.cpu().numpy(), win=image_grid_plot)

        mlflow.log_params(
            {
                "features": features,
                "lr": lr,
                "epoches": epoches,
                "batch_size": batch_size,
            }
        )
        mlflow.log_metric("Train MSE", sum(train_losses) / train_l)
        mlflow.log_metric("Test MSE", sum(test_losses) / test_l)
        mlflow.pytorch.log_model(
            AutoEncoder,
            f"{mlflow_run.info.run_id}",
            registered_model_name=registered_model_name,
        )
        with open("training_completion.txt", "w") as file:
            # Get the current date and time
            current_datetime = datetime.now()
            # Format the date and time as a string
            formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
            file.write("Training Completed at " + formatted_datetime)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    training(config_path=parsed_args.config)
