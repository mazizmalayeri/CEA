from read_data import read_data
from ood_params import get_params_data
from utils import set_all_seeds, split_data, normalization
from predictive_models import get_params
from train import train_predictive_model
from detection_methods_posthoc import detection_method
from ood_score import get_ood_score
import torch
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ood_type', default='multiplication', type=str, choices={'multiplication', 'other_domain', 'feature_split'})
    parser.add_argument('--data_in_name', type=str, choices={'diabetics', 'mimic', 'eicu', 'drybean', 'wine', 'sepsis'})
    parser.add_argument('--architecture', default='MLP', type=str, choices={'MLP', 'ResNet', 'FTTransformer'})
    parser.add_argument('--result_path', default='', type=str)

    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--n_epochs', default=10, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)

    parser.add_argument('--percentile_top', default=99.9, type=float)
    parser.add_argument('--addition_coef', default=10, type=float)
    
    return parser.parse_args()


#get arguments
args = get_args()
ood_type = args.ood_type
data_in_name = args.data_in_name
architecture = args.architecture
result_path = args.result_path

seed=args.seed
batch_size = args.batch_size
n_epochs = args.n_epochs
lr = args.lr
weight_decay = args.weight_decay
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('Device:', device)

addition_coef = args.addition_coef
percentile_top = args.percentile_top


if architecture == 'FTTransformer':
  activation_function = 'default' #'default'
else:
  activation_function = 'ReLU' #ReLU,

if data_in_name == 'drybean':
  d_out = 7
else:
  d_out = 2


#reading preprocessed file
data_in, label_in = read_data(data_name=data_in_name, files_path='datasets/')

data_external = None
if ood_type == 'other_domain':
    if data_in_name == 'mimic':
      data_external, _ = read_data(data_name='eicu', files_path='datasets/')
    elif data_in_name == 'eicu':
      data_external, _ = read_data(data_name='mimic', files_path='datasets/')

#Select features and label for in/out data
in_features_np, ood_features_np, in_label_np, scales, random_sample = get_params_data(in_distribution=data_in_name, in_features_df=data_in, in_label_df=label_in,
                              ood_type=ood_type, ood_features_df=data_external)

print(in_features_np.shape, in_label_np.shape)
if ood_type != 'multiplication':
  print(ood_features_np.shape)

#set random seed
set_all_seeds(seed)

#split and normalize data
X, y = split_data(in_features_np, in_label_np, random_state=seed)
X, y, preprocess = normalization(X, y, device)

report_frequency = len(X['train']) // batch_size // 5
if ood_type  == 'other_domain' or ood_type  == 'feature_split':
    ood_features_tensor = torch.tensor(preprocess.transform(ood_features_np), device=device)
else:
    ood_features_tensor = None

#define and train prediction model for posthoc methods
print('\nPreparing prediction model for the experiment ...')
model, optimizer = get_params(architecture, d_out, lr, weight_decay, X['train'].shape[1], activation_function)
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
train_predictive_model(model, optimizer, criterion, X, y, batch_size, n_epochs, device, report_frequency)

detectors = ['unit', 'MDS', 'RMDS', 'KNN', 'vim', 'she_euclidean', 'klm', 'OpenMax', 'MSP', 'mls', 'temp_scaling', 'ebo', 'gram', 'gradnorm', 'react', 'dice', 'ash']

# OOD performance
for detector in detectors:
  print('Baseline:', detector)
  score_function = detection_method(original=True, detector=detector, model=model, device=device, k_knn=5, x_train=X['train'], y_train=y['train'], batch_size=128, n_classes=d_out, x_val=X['val'], y_val=y['val'], vim_dim=64, lr=lr, n_epochs=n_epochs, react_prob=90)
  get_ood_score(model=model, in_test_features=X['test'], in_test_labels=y['test'], ood_type=ood_type, score_function=score_function, batch_size=batch_size, device=device, preprocess=preprocess, random_sample=random_sample, scales=scales, out_features=ood_features_tensor)
  print('\n')

for detector in detectors:
  print('Ours:', detector)
  score_function = detection_method(original=False, detector=detector, model=model, device=device, percentile_top=percentile_top, addition_coef=addition_coef, k_knn=5, x_train=X['train'], y_train=y['train'], batch_size=128, n_classes=d_out, x_val=X['val'], y_val=y['val'], vim_dim=64, lr=lr, n_epochs=n_epochs, react_prob=90)
  get_ood_score(model=model, in_test_features=X['test'], in_test_labels=y['test'], ood_type=ood_type, score_function=score_function, batch_size=batch_size, device=device, preprocess=preprocess, random_sample=random_sample, scales=scales, out_features=ood_features_tensor)
  print('\n')