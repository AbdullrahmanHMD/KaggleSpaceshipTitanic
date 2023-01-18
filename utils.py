import torch
from tqdm import tqdm
import os


def evaluate(model, data_loader):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.eval()

    # Moving the model to the selected device:
    model = model.to(device)

    num_correct = 0
    count = 0
    for x, y in data_loader:

        # Moving the data to the selected device:
        x = x.to(device)
        y = y.type(torch.LongTensor)

        y = y.to(device)

        # Generating predictions:
        y_pred = model(x)

        # Getting the label of each entry in the batch:
        _, y_pred = torch.max(y_pred, axis=1)

        # Calculating the number of correct prediction in one epoch:
        num_correct += (y_pred == y).sum().item()

        # Calculating the number of data points. This is useful when the number
        # of data points is not divisible by the batch size.
        count += x.shape[0]

    # Calculating the accuracy:
    accuracy = num_correct / count

    return accuracy


def train(model, optimizer, criterion, train_loader, val_loader, scheduler, epochs, verbose=False):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.train()
    # Moving the model to the selected device:
    model = model.to(device)

    train_accuracies = []
    val_accuracies = []
    lrs = []

    total_loss = []

    # base_lr = optimizer.param_groups[0]['lr']
    # eta_min = scheduler.eta_min

    for epoch in tqdm(range(epochs)):
        epoch_loss = 0.0
        for x, y in train_loader:

            optimizer.zero_grad()

            # Moving the data to the selected device:
            x = x.to(device)
            y = y.type(torch.LongTensor)

            y = y.to(device)

            # Forward pass:
            y_pred = model(x)

            # Calculating the loss:
            loss = criterion(y_pred, y)

            # Accumulating the loss:
            epoch_loss += loss.item()

            # Backpropagation:
            loss.backward()

            lrs.append(optimizer.param_groups[0]['lr'])

            # Stepping the optimizer:
            optimizer.step()

        # optimizer.param_groups[0]['lr'] = base_lr
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_loader.batch_size, eta_min=eta_min)

        scheduler.step()

        # Adding the epoch loss to the total_loss array for plotting the
        # loss later on.
        total_loss.append(epoch_loss)

        print('Evaluating epoch...')

        # The accuracy of the model in the current epoch on the training set:
        train_acc = evaluate(model, train_loader)
        train_accuracies.append(train_acc)

        # The accuracy of the model in the current epoch on the test set:
        val_acc = evaluate(model, val_loader)
        val_accuracies.append(val_acc)

        # scheduler.step()

        if verbose:
            print(f'Epoch: {epoch} | Train_acc: {100 * train_acc:.2f}% | Val_acc: {100 * val_acc:.2f}% \
Loss: {epoch_loss:.2f}')

    if not verbose:
        print(f'Train_acc: {100 * train_accuracies[-1]:.2f}% | Val_acc: {100 * val_accuracies[-1]:.2f}% \
Loss: {total_loss[-1]:.2f}')

    return total_loss, train_accuracies, val_accuracies, lrs



def predict(model, data_loader, as_tensor=True):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    multiple_models = isinstance(model, list)

    if multiple_models:
        for mdl in model:
            mdl.eval()
    else:
        model.eval()

    y_preds = torch.tensor([])

    for x in data_loader:
        x = x.to(device)

        if multiple_models:
            y_pred = torch.zeros_like(y)
            for mdl in model:
                mdl = mdl.to(device)
                y_pred += mdl(x)

            _, y_pred = torch.max(y_pred, axis=1)
            y_preds = torch.cat([y_preds, y_pred])

        else:
            y_pred = model(x)
            _, y_pred = torch.max(y_pred, axis=1)

            y_preds = torch.cat([y_preds, y_pred])

    if as_tensor:
        return y_preds

    return y_preds.cpu().detach().numpy()


def save_models(models):
    for i, model in enumerate(models):
        model_name = f'model_{i}'
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_name)
        torch.save(model.state_dict(), path)


def load_models(num_models=3, main_path=None):
    models = []

    if main_path is None:
        main_path = os.path.dirname(os.path.abspath(__file__))

    for i in range(num_models):
        model_name = f'model_{i}'
        path = os.path.join(main_path, model_name)




