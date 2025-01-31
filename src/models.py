
import segmentation_models_pytorch as smp

def getmodel(decoder_name,encoder_name,class_num=4,in_channles=3,activation=None):

    if decoder_name == "unet":
        return smp.Unet(
                encoder_name=encoder_name,
                encoder_weights='imagenet',
                in_channels=in_channles,
                classes=class_num,
                activation=activation
            )
    if decoder_name == "linknet":
        return smp.Linknet(
                encoder_name=encoder_name,
                encoder_weights='imagenet',
                in_channels=in_channles,
                classes=class_num,
                activation=activation
            )
    if decoder_name == "fpn":
        return smp.FPN(
                encoder_name=encoder_name,
                encoder_weights='imagenet',
                in_channels=in_channles,
                classes=class_num,
                activation=activation
            )
    if decoder_name == "pspnet":
        return smp.PSPNet(
                encoder_name=encoder_name,
                encoder_weights='imagenet',
                in_channels=in_channles,
                classes=class_num,
                activation=activation
                )
    if decoder_name == "deeplabv3":
        return smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            decoder_atrous_rates = (12,18,24),
            in_channels=in_channles,
            classes=class_num,
            activation=activation
            )