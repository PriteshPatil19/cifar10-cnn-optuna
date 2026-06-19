import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_epoch(model, trainloader, criterion,
                optimizer):

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for inputs, labels in trainloader:

        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(inputs)

        if outputs.ndim > 2:
            outputs = outputs.view(outputs.size(0), -1)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        predicted = torch.argmax(outputs, dim=1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    loss = running_loss / len(trainloader)
    accuracy = 100 * correct / total

    return loss, accuracy


def test_epoch(model, testloader, criterion):

    model.eval()

    running_loss = 0
    correct = 0
    total = 0

    y_true = []
    y_pred = []

    with torch.no_grad():

        for inputs, labels in testloader:

            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)

            if outputs.ndim > 2:
                outputs = outputs.view(outputs.size(0), -1)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            predicted = torch.argmax(outputs, dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    loss = running_loss / len(testloader)
    accuracy = 100 * correct / total

    return loss, accuracy, y_true, y_pred
