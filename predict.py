from tqdm import tqdm
import torch
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import os

from utils.config_utils import get_config_from_json
from utils.metrics import get_confmat_metrics
from data_loader import MyDataLoader
from models.ResNet50 import ResNet50Custom
from models.ConvLSTM import ConvLSTM
from models.Forestformer import CustomTransformerModel


# Set the GPU device index to use
opt = {"gpu_ids": []}
# opt['gpu_ids'] = [2, 3]
gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
# gpu_list = str(opt['gpu_ids'])
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list

# Load config
cfg = 'forestformer'
config = get_config_from_json('./configs/'+cfg+'.json')
exp_name = config['exp_name']
result_csv_path = './experiments/'+exp_name+'/result_2.csv'
result_conf_path = './experiments/'+exp_name+'/result_confmat_2.csv'
checkpoint_path = './experiments/'+exp_name+'/checkpoints/best_model.pth'
n_classes = config.num_classes
label_names = ['beech', 'douglas fir', 'fir', 'larch',
                         'oak', 'other deciduous', 'pine',
                         'spruce']

# Load model
device = torch.device('cuda' if opt['gpu_ids'] else 'cpu')
print(f"Predict on {device}")

model = CustomTransformerModel(config=config)
model = model.to(device)
new_state_dict = model.state_dict()
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
# checkpoint = torch.load(checkpoint_path)
for name, param in checkpoint.named_parameters():
    if name in new_state_dict:
        new_state_dict[name] = param
model.load_state_dict(new_state_dict)


# Load test data
dl = MyDataLoader(config=config)
mean, std = dl.get_mean_std()
X_test, y_test = dl.get_test_data()
# Normalize the data
X_test = transforms.Normalize(mean, std)(X_test)
# Sequence data   ---> ConvLSTM
if config.model == 'convlstm':
    X_test = torch.reshape(X_test, (
        X_test.shape[0], config.num_length, config.input_channels, X_test.shape[2], X_test.shape[3]))
test_set = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False)


# Predict
def make_predict(csv_path, conf_path, labelnames):
    model.eval()
    correct = 0
    count = 0
    with torch.no_grad():
        confusion_matrix = torch.zeros(n_classes, n_classes)
        for batch_idx, (data, target) in tqdm(enumerate(test_loader)):
            data = data.to(device)
            target = target.to(device)

            output = model(data.float())
            pred = output.data.max(1, keepdim=False)[1]
            count += target.shape[0]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            for t, p in zip(target.view(-1), pred.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
        print("Final result: ")
        print('\nTest set:  Accuracy: {}/{} ({:.0f}%)\n'.format(
            correct, count,
            100. * correct / count))

        confusion_matrix = confusion_matrix.numpy()
        precision, recall, f1 = get_confmat_metrics(confusion_matrix)
        print("precision: ", precision)
        print("recall: ", recall)
        print("f1: ", f1)
        # Save the result
        print("Saving result....")
        result_df = pd.DataFrame(data=[precision, recall, f1], columns=labelnames, index=['precision', 'recall', 'f1'])
        result_df.to_csv(csv_path)
        print("Saving confusion mat...")
        confmat_df = pd.DataFrame(data=confusion_matrix, columns=labelnames, index=labelnames)
        confmat_df.to_csv(conf_path)


if __name__ == "__main__":
    make_predict(result_csv_path, result_conf_path, label_names)

