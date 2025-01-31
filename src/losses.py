import torch 
import segmentation_models_pytorch as smp
#https://smp.readthedocs.io/en/latest/losses.html

def get_loss(loss_name,mode='multiclass'):
    if loss_name == 'Dice_loss':
        loss = smp.losses.DiceLoss(mode=mode)
    if loss_name == 'BCE_loss':
        loss = torch.nn.BCEWithLogitsLoss()
    if loss_name == 'Cross_entropy':
        loss = torch.nn.CrossEntropyLoss()
    if loss_name == "Jaccard_loss":
        loss = smp.losses.JaccardLoss(mode=mode)
    if loss_name == "Tvereg_loss":
        loss = smp.losses.TverskyLoss(mode=mode)
    if loss_name =="Focal_loss":
        loss = smp.losses.FocalLoss(mode=mode)
    if loss_name == "Lovasz_loss":
        loss = smp.losses.LovaszLoss(mode=mode)
    if loss_name == "SoftBCE_loss":
        loss = smp.losses.SoftBCEWithLogitsLoss()
    if loss_name == "MCC_loss":
        loss = smp.losses.MCCLoss(mode=mode)
    loss.__name__ = loss_name # add for metric scoring 
    return loss