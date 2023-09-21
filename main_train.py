import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from torch.nn import LayerNorm, Linear, ReLU
from tqdm import tqdm

from torch_geometric.loader import RandomNodeLoader
from torch_geometric.nn import DeepGCNLayer, GENConv
from torch_geometric.utils import scatter
from torch.optim.lr_scheduler import OneCycleLR
from accelerate import Accelerator


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


### READ IN CONFIGS
config_file_path = sys.argv[1] #'./comparison/configs.json'

### SPECIFY WHICH MODEL WE'RE RUNNING
model_type = int(sys.argv[2])
model_size = sys.argv[3]
LOAD_MODEL = bool(int(sys.argv[4]))
BEST = True if sys.argv[5] == "best" else False

if model_type == 0:
    model_type = "fishnet_gcn"
else:
    model_type = "default_gcn"


with open(config_file_path) as f:
        configs = json.load(f)


# FIX RANDOM SEED
seed = configs["training_params"]["seed"]
torch.manual_seed(seed)



# model stuff
HIDDEN_CHANNELS = configs["model_params"][model_type][model_size]["hidden_channels"]
NUM_LAYERS = configs["model_params"][model_type][model_size]["num_layers"]
MODEL_NAME = configs["model_params"][model_type][model_size]["name"]
TEST_BATCHING = configs["model_params"][model_type][model_size]["test_batching"]


# optimizer schedule
LEARNING_RATE = configs["training_params"]["learning_rate"]
EPOCHS = int(configs["training_params"]["epochs"])
DO_SCHEDULER = bool(int(configs["training_params"]["do_lr_scheduler"]))

# data + out directories
DATA_DIR = configs["training_params"]["data_dir"]
MODEL_DIR = configs["training_params"]["model_dir"]
LOAD_DIR = configs["training_params"]["load_dir"]


if not os.path.exists(MODEL_DIR):
   # Create a new directory if it does not exist
   os.makedirs(MODEL_DIR)
   print("created new directory", MODEL_DIR)

### CONSTRUCT MODEL NAME AND OUTPUT PATH
MODEL_NAME += "nc_%d_nlyr_%d"%(HIDDEN_CHANNELS, NUM_LAYERS)
MODEL_PATH = MODEL_DIR + MODEL_NAME
LOAD_PATH = LOAD_DIR + MODEL_NAME


### START THE PROGRAMME

print("training %s, saving to %s"%(MODEL_NAME, MODEL_DIR))
print("training for %d epochs"%(EPOCHS))
if LOAD_MODEL:
    print("loading model from %s"%(LOAD_DIR) )


### SAVE CONFIGS TO OUTDIR
with open(MODEL_DIR + 'configs_%s.json'%(MODEL_NAME), 'w') as f:
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


### TRAIN AND TEST LOADERS ####
train_loader = RandomNodeLoader(data, num_parts=40, shuffle=True,
                                num_workers=5)
test_loader = RandomNodeLoader(data, num_parts=TEST_BATCHING, num_workers=5)





# initialise model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if model_type == "fishnet_gcn":
    FISHNETS_N_P = configs["model_params"][model_type][model_size]["fishnets_n_p"]
    model = FishnetGCN(n_p=FISHNETS_N_P, num_layers=NUM_LAYERS, hidden_channels=HIDDEN_CHANNELS).to(device)

else:
    model = DeeperGCN(hidden_channels=HIDDEN_CHANNELS, num_layers=NUM_LAYERS).to(device)

# start up the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.BCEWithLogitsLoss()
evaluator = Evaluator('ogbn-proteins')

# accelerate code with accelerator
if LOAD_MODEL:
    accel_path = LOAD_PATH
else:
    accel_path = MODEL_PATH

accelerator = Accelerator(project_dir=accel_path)


if DO_SCHEDULER:
    lr_scheduler = OneCycleLR(optimizer, max_lr=1e-2, total_steps=EPOCHS*len(train_loader), pct_start=0.15, final_div_factor=1e3)
    model, optimizer, train_loader, scheduler = accelerator.prepare(
                    model, optimizer, train_loader, lr_scheduler)
    # Register the LR scheduler
    accelerator.register_for_checkpointing(lr_scheduler)
    
else:
    model, optimizer, train_loader = accelerator.prepare(
                    model, optimizer, train_loader)
    

# if we want to load a model we'll do it here AFTER insatiating the model
if LOAD_MODEL:
    print("LOADING MODEL FROM  ", LOAD_PATH)
    #accelerator = Accelerator(project_dir=LOAD_PATH)

    # load best model ?
    if BEST:
        print("loading best model")
        accelerator.load_state(LOAD_PATH + "_best")

    else: 
        accelerator.load_state(LOAD_PATH)
    
    model, optimizer, train_loader = accelerator.prepare(
                model, optimizer, train_loader)

    
    # restart optimizer ?
    history = load_obj(LOAD_DIR + MODEL_NAME + "_history.pkl")
    
# Save the starting state
else:
    accelerator.save_state(MODEL_PATH)




# print out model complexity
print("number of learnable parameters in model: ", count_parameters(model))

def train(epoch):
    #model.train()

    pbar = tqdm(total=len(train_loader), position=0)
    pbar.set_description(f'Training epoch: {epoch:04d}')

    total_loss = total_examples = 0
    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        #loss.backward()
        accelerator.backward(loss)
        optimizer.step() 
        if DO_SCHEDULER:
            lr_scheduler.step()

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

    pbar = tqdm(total=len(test_loader), position=0)
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

    if LOAD_MODEL:
        train_rocauc, valid_rocauc, test_rocauc = test()
        print(f'loaded stats, Train: {train_rocauc:.4f}, '
            f'Val: {valid_rocauc:.4f}, Test: {test_rocauc:.4f}')

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
        #torch.save(model.state_dict(), MODEL_PATH + "_best")
        accelerator.save_model(model, MODEL_PATH + "_best")
        # save training history
        save_obj(history, MODEL_DIR + MODEL_NAME + "_history")
        best_rocauc = test_rocauc


    # save intermittently
    if epoch % 10 == 0:
        print("saving model")
        #torch.save(model.state_dict(), MODEL_PATH)
        accelerator.save_model(model, MODEL_PATH)
        accelerator.save_state(MODEL_PATH)

        # save training history
        save_obj(history, MODEL_DIR + MODEL_NAME + "_history")

# save everything
print("FINISHED TRAINING. SAVING EVERYTHING.")
accelerator.save_model(model, MODEL_PATH)
accelerator.save_state(MODEL_PATH)

# save training history
save_obj(history, MODEL_DIR + MODEL_NAME + "_history")





