import os
import json
import yaml

import torch
import torch.nn as nn
import torch.optim as optim

import optuna
import optuna.visualization as vis
import joblib

from torch.optim.lr_scheduler import MultiStepLR

from dataset import dataloader
from model import MyCNN

from train import train_epoch, test_epoch

from evaluate import (
    plot_loss,
    plot_accuracy,
    plot_confusion_matrix
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("results", exist_ok=True)

GLOBAL_BEST_ACC = 0

GLOBAL_BEST_PARAMS = {}

GLOBAL_BEST_LOSS = float("inf")


def load_config():
    with open("config.yml", "r") as f:
        return yaml.safe_load(f)


def create_optimizer(model, optimizer_name, lr, momentum):

    if optimizer_name == "Adam":
        return optim.Adam(
            model.parameters(),
            lr=lr
        )

    elif optimizer_name == "SGD":
        return optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum
        )

    elif optimizer_name == "RMSprop":
        return optim.RMSprop(
            model.parameters(),
            lr=lr,
            momentum=momentum
        )

    raise ValueError(f"Unknown optimizer: {optimizer_name}")


def objective(trial):

    global GLOBAL_BEST_ACC
    global GLOBAL_BEST_PARAMS
    global GLOBAL_BEST_LOSS

    config = load_config()

    lr = trial.suggest_float(
        "lr",
        float(config["optimized"]["lr"]["min"]),
        float(config["optimized"]["lr"]["max"]),
        log=True
    )

    batch_size = trial.suggest_categorical(
        "batch_size",
        config["optimized"]["batch_size"]["values"]
    )

    optimizer_name = trial.suggest_categorical(
        "optimizer",
        config["optimized"]["optimizer"]["values"]
    )

    num_conv_layers = trial.suggest_int(
        "num_conv_layers",
        config["optimized"]["num_conv_layers"]["min"],
        config["optimized"]["num_conv_layers"]["max"]
    )

    num_filters = trial.suggest_categorical(
        "num_filters",
        config["optimized"]["num_filters"]["values"]
    )

    trainloader, testloader = dataloader(batch_size)

    model = MyCNN(
        num_conv_layers=num_conv_layers,
        num_filters=num_filters
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = create_optimizer(
        model=model,
        optimizer_name=optimizer_name,
        lr=lr,
        momentum=config["default"]["momentum"]
    )

    scheduler = MultiStepLR(
        optimizer,
        milestones=[20, 40],
        gamma=0.1
    )

    train_losses = []
    train_accs = []

    test_losses = []
    test_accs = []

    best_trial_acc = 0

    for epoch in range(config["default"]["epochs"]):

        train_loss, train_acc = train_epoch(
            model,
            trainloader,
            criterion,
            optimizer
        )

        test_loss, test_acc, y_true, y_pred = test_epoch(
            model,
            testloader,
            criterion
        )

        scheduler.step()

        train_losses.append(train_loss)
        train_accs.append(train_acc)

        test_losses.append(test_loss)
        test_accs.append(test_acc)

        if test_acc > best_trial_acc:
            best_trial_acc = test_acc


        if test_acc > GLOBAL_BEST_ACC:

            GLOBAL_BEST_ACC = test_acc

            GLOBAL_BEST_PARAMS = {
                "lr": lr,
                "batch_size": batch_size,
                "optimizer": optimizer_name,
                "num_conv_layers": num_conv_layers,
                "num_filters": num_filters
            }

            GLOBAL_BEST_LOSS = test_loss

            torch.save(
                model.state_dict(),
                "results/best_model.pth"
            )

            plot_loss(
                train_losses,
                test_losses,
                "results/loss_curve.png",
                epoch=config["default"]["epochs"] - 1
            )

            plot_accuracy(
                train_accs,
                test_accs,
                "results/accuracy_curve.png",
                epoch=config["default"]["epochs"] - 1
            )

            plot_confusion_matrix(
                y_true,
                y_pred,
                "results/confusion_matrix.png"
            )

    trial.set_user_attr(
    "best_accuracy",
    best_trial_acc
    )
    return test_losses[-1]

def save_study_results(study):

    global GLOBAL_BEST_ACC
    global GLOBAL_BEST_PARAMS
    global GLOBAL_BEST_LOSS

    with open(
        "results/best_params.json",
        "w"
    ) as f:

        json.dump(
            GLOBAL_BEST_PARAMS,
            f,
            indent=4
        )

    with open(
        "results/study_summary.txt",
        "w"
    ) as f:

        f.write(
            "OPTUNA STUDY SUMMARY\n"
        )

        f.write(
            "=" * 40 + "\n\n"
        )

        f.write(
            f"Trials Completed: {len(study.trials)}\n"
        )

        f.write(
            f"Best Accuracy: {GLOBAL_BEST_ACC:.2f}%\n"
        )

        f.write(
            f"Associated Loss: {GLOBAL_BEST_LOSS:.6f}\n\n"
        )

        f.write(
            "Best Parameters\n"
        )

        f.write(
            "-" * 20 + "\n"
        )

        for key, value in GLOBAL_BEST_PARAMS.items():
            f.write(f"{key}: {value}\n")

def save_optuna_plots(study):

    fig = vis.plot_optimization_history(study)

    fig.write_image(
        "results/optimization_history.png"
    )

    fig = vis.plot_param_importances(study)

    fig.write_image(
        "results/parameter_importance.png"
    )

    fig = vis.plot_parallel_coordinate(study)

    fig.write_image(
    "results/parallel_coordinates.png"
    )


def save_model_info(model):

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    with open(
        "results/model_info.txt",
        "w"
    ) as f:

        f.write(
            f"Total Parameters: "
            f"{total_params:,}\n"
        )

        f.write(
            f"Trainable Parameters: "
            f"{trainable_params:,}\n"
        )


def main():

    config = load_config()

    dummy_model = MyCNN(
        num_conv_layers=4,
        num_filters=64
    )

    save_model_info(dummy_model)
    with open(
        "results/model_architecture.txt",
        "w"
    ) as f:

        f.write(str(dummy_model))

    study = optuna.create_study(
        direction="minimize",
        study_name="cifar10_cnn"
    )

    study.optimize(
        objective,
        n_trials=config["default"]["num_trials"]
    )

    joblib.dump(
    study,
    "results/optuna_study.pkl"
    )

    save_study_results(study)
    save_optuna_plots(study)

    print("\nBest Trial")
    print("=" * 50)

    print(
        f"Best Loss: "
        f"{study.best_trial.value:.6f}"
    )

    print(
        f"Best Accuracy: "
        f"{study.best_trial.user_attrs['best_accuracy']:.2f}%"
    )

    print("\nParameters")

    for key, value in study.best_trial.params.items():
        print(f"{key}: {value}")

    print("=" * 50)


if __name__ == "__main__":
    main()
