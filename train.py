import os
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms

from utils.config_utils import get_config_from_json
from data_loader import MyDataLoader
from models.ConvLSTM import ConvLSTM
from models.ViViT import ViViT
from models.ResNet50 import ResNet50Custom
from models.ViT import ViT
from models.Forestformer import CustomTransformerModel

# Set the GPU device index to use
opt = {"gpu_ids": []}
# opt['gpu_ids'] = [2, 3]
gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
# gpu_list = str(opt['gpu_ids'])
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list


def train():
    print('[INFO] Loading config...')
    config = get_config_from_json("configs/forestformer.json")
    batch_size = config.batch_size
    learning_rate = config.lr
    decay = config.decay
    num_epochs = config.num_epochs
    cp_dir = config.cp_dir
    tb_dir = config.tb_dir

    print('[INFO] Loading data...')
    dl = MyDataLoader(config=config)
    X_train, y_train = dl.get_train_data()
    mean, std = dl.get_mean_std()
    # Normalize the data
    X_train = transforms.Normalize(mean, std)(X_train)

    # Sequence data   ---> ConvLSTM
    if config.model == 'convlstm':
        X_train = torch.reshape(X_train, (
        X_train.shape[0], config.num_length, config.input_channels, X_train.shape[2], X_train.shape[3]))

    train_set = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    num_train = len(train_set)

    # release memory
    del train_set, X_train, y_train

    X_val, y_val = dl.get_val_data()
    # Normalize the data
    X_val = transforms.Normalize(mean, std)(X_val)

    # Sequence data   ---> ConvLSTM
    if config.model == 'convlstm':
        X_val = torch.reshape(X_val, (
        X_val.shape[0], config.num_length, config.input_channels, X_val.shape[2], X_val.shape[3]))

    val_set = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    num_val = len(val_set)

    # release memory
    del val_set, X_val, y_val

    print('[INFO] Loading Model...')
    # saved_model_path = ('./experiments/resnet50_0314_EuroSAT_1/checkpoints/best_model.pth')
    # saved_state_dict = torch.load(saved_model_path)
    print('[INFO] Training Model...')
    device = torch.device('cuda' if opt['gpu_ids'] else 'cpu')
    print(f"Training on {device}")

    # model = ConvLSTM(config=config)
    # model = ResNet50Custom(config=config)
    # model = ViViT(config=config)
    # model = ViT(config=config, device=device)
    model = CustomTransformerModel(config=config)

    # new_state_dict = model.state_dict()

    # for name, param in saved_state_dict.named_parameters():
    #     if name.startswith('conv1') or name.startswith('fc'):
    #         print("Not load the parameters of conv1 and fc")
    #     else:
    #         if name in new_state_dict:
    #             new_state_dict[name] = param
    # model.load_state_dict(new_state_dict)

    # for param in model.layer1.parameters():
    #     param.requires_grad = False
    # for param in model.layer2.parameters():
    #     param.requires_grad = False
    # for param in model.layer3.parameters():
    #     param.requires_grad = False
    # for param in model.layer4.parameters():
    #     param.requires_grad = False

    model = model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # writer = SummaryWriter(tb_dir)

    best_acc = 0.0
    best_epoch = 0

    # Define your early stopping parameters
    best_loss = float('inf')
    patience = 15
    cnt = 0

    for epoch in range(num_epochs):
        print("Epoch: {}/{}".format(epoch + 1, num_epochs))

        model.train()

        train_loss = 0.0
        train_acc = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            # Move the inputs and labels to the device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs.float())
            loss = loss_fn(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update the running loss and accuracy
            train_loss += loss.item() * inputs.size(0)
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            train_acc += acc.item() * inputs.size(0)

        epoch_loss = train_loss / num_train
        epoch_acc = train_acc / num_train

        print(f"Epoch {epoch+1} - Training loss: {epoch_loss:.4f}, Training accuracy: {epoch_acc:.4f}")

        # Write the training loss and accuracy to TensorBoard
        # writer.add_scalar("Train/Loss", epoch_loss, epoch)
        # writer.add_scalar("Train/Accuracy", epoch_acc, epoch)

        model.eval()
        val_loss = 0.0
        val_acc = 0.0

        with torch.no_grad():
            for j, (inputs, labels) in enumerate(val_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs.float())
                loss = loss_fn(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                ret, predictions = torch.max(outputs.data, 1)

                correct_counts = predictions.eq(labels.data.view_as(predictions))
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
                val_acc += acc.item() * inputs.size(0)

        val_epoch_loss = val_loss / num_val
        val_epoch_acc = val_acc / num_val

        print(f"Epoch {epoch+1} - Validation loss: {val_epoch_loss:.4f}, Validation accuracy: {val_epoch_acc:.4f}")

        # Write the validation loss and accuracy to TensorBoard
        # writer.add_scalar("Validation/Loss", val_epoch_loss, epoch)
        # writer.add_scalar("Validation/Accuracy", val_epoch_acc, epoch)

        if val_epoch_acc >= best_acc:
            best_acc = val_epoch_acc
            best_epoch = epoch + 1
            torch.save(model, cp_dir + "best_model.pth")

        print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))

        if val_epoch_loss <= best_loss:
            best_loss = val_epoch_loss
            cnt = 0
        else:
            cnt += 1
            if cnt >= patience:
                print('Early stopping after', epoch+1, 'epochs.')
                break

    # writer.close()


if __name__ == '__main__':
    train()
