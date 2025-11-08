from monai.networks.nets import SwinUNETR

def get_model():
    return SwinUNETR(
        in_channels=4,
        out_channels=4,
        feature_size=48,
        use_checkpoint=True
    )
