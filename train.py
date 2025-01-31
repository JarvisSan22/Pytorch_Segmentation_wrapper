from src.dataset import Dataset 
from torch.utils.data import DataLoader
from src.args import get_augmentation,get_preprocessing

import segmentation_models_pytorch as smp
import pandas as pd
import gc 
import yaml 
import os 
import torch

print("===== cuda =====")
print(torch.__version__)  # Check PyTorch version
print(torch.cuda.is_available())  # Check if CUDA is available
print(torch.version.cuda)  # Check CUDA version
print(torch.backends.cudnn.enabled)  # Check if cuDNN is enabled
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
#DEVICE = "cpu" # use to find real error in code 
print("DEVICE",DEVICE)
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1' #debugging 


with open("config/train_t1.yaml", "r") as f:
    config = yaml.safe_load(f)
#print(config)
print("===== Dataset =====")
if config["run_system"] == "unbuntu":
    base_dir = "/mnt/d/"
else:
    base_dir = "D:/"
x_train_dir =base_dir+ config["train_image_dir"]# os.getcwd() +  "/images/training/"
y_train_dir = base_dir+config["train_mask_dir"] # os.getcwd() +   "/annotations_instance/training/"
x_valid_dir = base_dir+config["val_image_dir"] #os.getcwd() +   "/images/validation/"
y_valid_dir =base_dir+ config["val_mask_dir"] #os.getcwd() +   "/annotations_instance/validation/"
print(x_train_dir,y_train_dir,x_valid_dir,y_valid_dir)
assert os.path.exists(x_train_dir), f"Image directory '{x_train_dir}' does not exist."
assert os.path.exists(y_train_dir), f"Mask directory '{y_train_dir}' does not exist."
assert os.path.exists(x_valid_dir), f"Image directory '{x_valid_dir}' does not exist."
assert os.path.exists(y_valid_dir), f"Mask directory '{y_valid_dir}' does not exist."

ENCODER=config["encoder"]["name"]
ENCODER_WEIGHTS=config["encoder"]["weights"]
BATCH_TRAIN=config["batch_size"]
BATCH_VAL=config["batch_size"]

train_args_dict = config["train_args"]
valid_args_dict = config["validation_args"]
print(train_args_dict)
train_dataset = Dataset(
    x_train_dir,
    y_train_dir,
    augmentation=get_augmentation(train_args_dict),
    preprocessing=get_preprocessing(ENCODER,ENCODER_WEIGHTS),
)

valid_dataset = Dataset(
    x_valid_dir,
    y_valid_dir,
    augmentation=get_augmentation(valid_args_dict),
    preprocessing=get_preprocessing(ENCODER,ENCODER_WEIGHTS),
)
print("Train data :",train_dataset.__len__())
print("Valid data :",valid_dataset.__len__())

train_loader = DataLoader(train_dataset, batch_size=BATCH_TRAIN, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_VAL, shuffle=False)

print("===== Model =====")

from src.models import getmodel
out_model_name: f"{config['decoder']['name']}_{config['encoder']['name']}_{config['class_num']}_{config['patch_size']}_{config['patch_size']}_lr{config['learning_rate']}_ls{config['lr_schedule']}"
print("Model :",config["decoder"]["name"],"Encoder:",config["encoder"]["name"])
model = getmodel(config["decoder"]["name"],
                 config["encoder"]["name"],
                 config["class_num"],
                 config["decoder"]["in_channels"],
                 config["decoder"]["activation"])
model = model.to(DEVICE)

print("===== Optimizer & scheduler =====")
optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['learning_rate']), weight_decay=float(config['weight_decay']))

from src.lr_scheduler import get_lr_scheduler
scheduler = get_lr_scheduler(optimizer,config)
#scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=config['epochs'], power=0.9)

print("===== Loss  & Metrics =====")
from src.losses import get_loss
mode = "binary"
if config["class_num"] > 1:
    mode = "multiclass"
criterion = get_loss(config["loss"],mode)

from segmentation_models_pytorch import utils

metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
    smp.utils.metrics.Accuracy(threshold=0.5),
]

print("===== Training =====")
# Define training epock =====================================
train_epoch = smp.utils.train.TrainEpoch(
    model, 
    loss=criterion,
    metrics= metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

# Define testing epoch =====================================
valid_epoch = smp.utils.train.ValidEpoch(
    model, 
    loss=criterion,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)



# Memory cleanup
gc.collect()
torch.cuda.empty_cache()
        
        
max_score = 0
os.makedirs(config["output_dir"],exist_ok=True)
os.makedirs(config["log_dir"],exist_ok=True)
EPOCH = config["epochs"]
history = []
for i in range(0, EPOCH):

    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)
    # do something (save model, change lr, etc.)
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model, f'{config["output_dir"]}/best_model.pth')
        print('Model saved!')
    savelog= {
        "epoch":i,
    }
    for k,v in train_logs.items():
        savelog[f"train_{k}"]=v
    for k,v in valid_logs.items():
        savelog[f"valid_{k}"]=v
    history.append(savelog)
    if i == 25:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')
df=pd.DataFrame(history)
df.to_csv(f'{config["log_dir"]}/history.csv')