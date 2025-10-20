from models.rangeMixtureSwinTrans import RangeMixtureSwinTransformerModel
from models.adapters_acc import AccurateM1Adapter, AccurateM2Adapter

def build_model(name, cfg):
    n = str(name).lower()
    if n in ["swin", "swin3d"]:
        return RangeMixtureSwinTransformerModel(cfg)
    if n in ["acc_m1", "accurate_m1", "model1"]:
        return AccurateM1Adapter(cfg)
    if n in ["acc_m2", "accurate_m2", "model2"]:
        return AccurateM2Adapter(cfg)
    raise ValueError(f"Unknown model name: {name}")
