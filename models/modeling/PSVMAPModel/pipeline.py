from .PSVMAPNet import build_PSVMAPNet

_GZSL_META_ARCHITECTURES = {
    "Model": build_PSVMAPNet,
}

def build_gzsl_pipeline(cfg):
    meta_arch = _GZSL_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)