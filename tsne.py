from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import torch

from models import *
from utils import *
from dataloaders import *
from optimizers import *


# visualize t-SNE
def get_tsne(model_name, path):
    _, test_dataloader, num_classes = get_dataloader(data_name='cifar100')
    model = model_dict[model_name](num_classes=num_classes)
    model = model.to('cuda')
    
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    all_features, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for i, (data, labels) in tqdm(enumerate(test_dataloader)):
            data = data.to('cuda')
            features, _ = model(data, is_feat=True)
            
            all_features.append(features[-1].data.cpu().numpy())
            all_labels.append(labels.data.cpu().numpy())
    all_features = np.concatenate(all_features, 0)
    all_labels = np.concatenate(all_labels, 0)

    tsne = TSNE()
    all_features = tsne.fit_transform(all_features)
    plot_features(all_features, all_labels, num_classes)

def plot_features(features, labels, num_classes):
    colors = ['C' + str(i) for i in range(num_classes)]
    plt.figure(figsize=(6, 6))
    for l in range(num_classes):
        plt.scatter(
            features[labels == l, 0],
            features[labels == l, 1],
            c=colors[l], s=1, alpha=0.4)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
path = "checkpoint/T_mo=resnet32x4_S_mo=resnet8x4_OPT_op=sgd_lr=0.05_mo=0.9_we=0.0005_DAT_da=cifar100_ba=64_T=4_kl_w=0.9_ce_w=0.1_kd_t=kd_rho=None_seed=42_20250606_0700/ckpt_best.pth"
get_tsne("resnet8x4", path)