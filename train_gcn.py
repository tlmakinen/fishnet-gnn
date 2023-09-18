import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from torch.nn import LayerNorm, Linear, ReLU
from tqdm import tqdm

from torch_geometric.loader import RandomNodeLoader
from torch_geometric.nn import DeepGCNLayer, GENConv
from torch_geometric.utils import scatter

import sys,os
import json
import cloudpickle as pickle

from nets import *

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)
        
def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


### SPECIFY WHICH MODEL WE'RE RUNNING
model_size = sys.argv[1]
LOAD_MODEL = bool(int(sys.argv[2]))


### READ IN CONFIGS
config_file_path = './comparison/configs.json'

with open(config_file_path) as f:
        configs = json.load(f)




# model stuff
HIDDEN_CHANNELS = configs["model_params"]["default_gcn"][model_size]["hidden_channels"]
NUM_LAYERS = configs["model_params"]["default_gcn"][model_size]["num_layers"]
MODEL_NAME = configs["model_params"]["default_gcn"][model_size]["name"]

# optimizer schedule
LEARNING_RATE = configs["training_params"]["learning_rate"]
EPOCHS = int(configs["training_params"]["epochs"])

# data + out directories
DATA_DIR = configs["training_params"]["data_dir"]
MODEL_DIR = configs["training_params"]["model_dir"]


### CONSTRUCT MODEL NAME AND OUTPUT PATH
MODEL_NAME += "nc_%d_nlyr_%d"%(HIDDEN_CHANNELS, NUM_LAYERS)

MODEL_PATH = MODEL_DIR + MODEL_NAME

print("training %s, saving to %s"%(MODEL_NAME, MODEL_DIR))
print("training for %d epochs"%(EPOCHS))


### SAVE CONFIGS TO OUTDIR
with open(MODEL_DIR + 'confings_%s.json'%(MODEL_NAME), 'w') as f:
    json.dump(configs, f)


### INITIALISE DATA

dataset = PygNodePropPredDataset('ogbn-proteins', root=DATA_DIR)
splitted_idx = dataset.get_idx_split()
data = dataset[0]
data.node_species = None
data.y = data.y.to(torch.float)

# Initialize features of nodes by aggregating edge features.
row, col = data.edge_index
data.x = scatter(data.edge_attr, col, dim_size=data.num_nodes, reduce='sum')

# Set split indices to masks.
for split in ['train', 'valid', 'test']:
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[splitted_idx[split]] = True
    data[f'{split}_mask'] = mask

train_loader = RandomNodeLoader(data, num_parts=40, shuffle=True,
                                num_workers=5)
test_loader = RandomNodeLoader(data, num_parts=5, num_workers=5)


class DeeperGCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers):
        super().__init__()

        self.node_encoder = Linear(data.x.size(-1), hidden_channels)
        self.edge_encoder = Linear(data.edge_attr.size(-1), hidden_channels)

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_channels, hidden_channels, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1,
                                 ckpt_grad=i % 3)
            self.layers.append(layer)

        self.lin = Linear(hidden_channels, data.y.size(-1))

    def forward(self, x, edge_index, edge_attr):
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        x = self.layers[0].conv(x, edge_index, edge_attr)

        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=0.1, training=self.training)

        return self.lin(x)

# initialise model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DeeperGCN(hidden_channels=HIDDEN_CHANNELS, num_layers=NUM_LAYERS).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.BCEWithLogitsLoss()
evaluator = Evaluator('ogbn-proteins')


if LOAD_MODEL:
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    history = load_obj(MODEL_DIR + MODEL_NAME + "history.pkl")


# print out model complexity
print("number of learnable parameters in model: ", count_parameters(model))

def train(epoch):
    model.train()

    pbar = tqdm(total=len(train_loader))
    pbar.set_description(f'Training epoch: {epoch:04d}')

    total_loss = total_examples = 0
    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * int(data.train_mask.sum())
        total_examples += int(data.train_mask.sum())

        pbar.update(1)

    pbar.close()

    return total_loss / total_examples


@torch.no_grad()
def test():
    model.eval()

    y_true = {'train': [], 'valid': [], 'test': []}
    y_pred = {'train': [], 'valid': [], 'test': []}

    pbar = tqdm(total=len(test_loader))
    pbar.set_description(f'Evaluating epoch: {epoch:04d}')

    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr)

        for split in y_true.keys():
            mask = data[f'{split}_mask']
            y_true[split].append(data.y[mask].cpu())
            y_pred[split].append(out[mask].cpu())

        pbar.update(1)

    pbar.close()

    train_rocauc = evaluator.eval({
        'y_true': torch.cat(y_true['train'], dim=0),
        'y_pred': torch.cat(y_pred['train'], dim=0),
    })['rocauc']

    valid_rocauc = evaluator.eval({
        'y_true': torch.cat(y_true['valid'], dim=0),
        'y_pred': torch.cat(y_pred['valid'], dim=0),
    })['rocauc']

    test_rocauc = evaluator.eval({
        'y_true': torch.cat(y_true['test'], dim=0),
        'y_pred': torch.cat(y_pred['test'], dim=0),
    })['rocauc']

    return train_rocauc, valid_rocauc, test_rocauc

# training history

history = {
    "train_aucs": [],
    "valid_aucs": [],
    "test_aucs": [],
    "losses": []
}

best_rocauc = 0.0

# training loop
for epoch in range(1, EPOCHS + 1):
    loss = train(epoch)
    train_rocauc, valid_rocauc, test_rocauc = test()
    print(f'Loss: {loss:.4f}, Train: {train_rocauc:.4f}, '
          f'Val: {valid_rocauc:.4f}, Test: {test_rocauc:.4f}')

    history["losses"].append(loss)
    history["train_aucs"].append(train_rocauc)
    history["valid_aucs"].append(valid_rocauc)
    history["test_aucs"].append(test_rocauc)

    # save best model
    if test_rocauc > best_rocauc:
        print("saving best model")
        torch.save(model.state_dict(), MODEL_PATH + "_best")
        # save training history
        save_obj(history, MODEL_DIR + MODEL_NAME + "_history")
        best_rocauc = test_rocauc


    # save intermittently
    if epoch % 10 == 0:
        torch.save(model.state_dict(), MODEL_PATH)

        # save training history
        save_obj(history, MODEL_DIR + MODEL_NAME + "_history")

# save everything
torch.save(model.state_dict(), MODEL_PATH)

# save training history
save_obj(history, MODEL_DIR + MODEL_NAME + "_history")





