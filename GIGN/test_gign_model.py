import os
import pandas as pd
import torch
from GIGN import GIGN
from dataset_GIGN import GraphDataset, PLIDataLoader
import numpy as np
from utils import load_model_dict
from sklearn.metrics import mean_squared_error

def val(model, dataloader, device):
    model.eval()

    pred_list = []
    label_list = []
    for data in dataloader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data)
            label = data.y

            pred_list.append(pred.detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())
            
    pred = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    coff = np.corrcoef(pred, label)[0, 1]
    rmse = np.sqrt(mean_squared_error(label, pred))

    model.train()

    return rmse, coff

def main():
    data_root = './pdbbind_rna_data'
    graph_type = 'Graph_GIGN'
    batch_size = 128

    test_dir = os.path.join(data_root, 'test_set')
    test_df = pd.read_csv(os.path.join(data_root, 'test_labels.csv'))

    test_set = GraphDataset(test_dir, test_df, graph_type=graph_type, create=False)
    test_loader = PLIDataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = GIGN(35, 256).to(device)
    
    load_model_dict(model, './pdbbind_rna_model_fineturn/20241010_085413_GIGN_repeat2/model/last_epoch-113, train_loss-0.0553, train_rmse-0.2351, valid_rmse-1.7327, valid_pr--0.2664.pt')

    test_rmse, test_pr = val(model, test_loader, device)
    
    msg = "test_rmse-%.4f, test_pr-%.4f" % (test_rmse, test_pr)
    print(msg)

if __name__ == "__main__":
    main()
