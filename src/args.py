import albumentations as albu
import segmentation_models_pytorch as smp


def get_augmentation(config: dict):

    train_transform = []

    if config.get("HorizontalFlip", False):
        train_transform.append(albu.HorizontalFlip(p=config["HorizontalFlip"]))

    if config.get("ShiftScaleRotate", False):
        train_transform.append(albu.ShiftScaleRotate(**config["ShiftScaleRotate"]))

    if config.get("PadIfNeeded", False):
        train_transform.append(albu.PadIfNeeded(**config["PadIfNeeded"]))

    if config.get("RandomCrop", False):
        train_transform.append(albu.RandomCrop(**config["RandomCrop"]))

    if config.get("IAAAdditiveGaussianNoise", False):
        train_transform.append(albu.IAAAdditiveGaussianNoise(p=config["IAAAdditiveGaussianNoise"]))

    if config.get("IAAPerspective", False):
        train_transform.append(albu.IAAPerspective(p=config["IAAPerspective"]))

    if "OneOf_CLAHE_Brightness_Gamma" in config:
        train_transform.append(albu.OneOf([
            albu.CLAHE(p=1),
            albu.RandomBrightnessContrast(p=1),
            albu.RandomGamma(p=1),
        ], p=config["OneOf_CLAHE_Brightness_Gamma"]))

    if "OneOf_Sharpen_Blur" in config:
        train_transform.append(albu.OneOf([
            albu.Sharpen(p=1),
            albu.Blur(blur_limit=3, p=1),
        ], p=config["OneOf_Sharpen_Blur"]))

    if "OneOf_HueSaturation" in config:
        train_transform.append(albu.OneOf([
            albu.HueSaturationValue(p=1),
        ], p=config["OneOf_HueSaturation"]))
   # train_transform.append(albu.Normalize())
   # train_transform.append(ToTensorV2())
    return albu.Compose(train_transform)

def to_tensor(x, **kwargs):
    #print(x.shape)
    if len(x.shape)==2: #mask 
        #x = np.expand_dims(x, axis=0)  # Add channel dimension
        return x.astype('float32')
    else:
      return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(ENCODER,ENCODER_WEIGHTS):
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    
    return  albu.Compose(_transform)