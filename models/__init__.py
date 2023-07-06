from models.vggunet_saff_mult import Vggunet_saff_mult

import torch


def create_model(
    args
) -> torch.nn.Module:
    """Models wrapper. Allows to create any model just with parametes

    """
    archs = [Vggunet_saff_mult]
    archs_dict = {a.__name__.lower(): a for a in archs}
    print("archs_dict", archs_dict)
    arch = args['architecture']
    try:
        model_class = archs_dict[arch.lower()]
    except KeyError:
        raise KeyError("Wrong architecture type `{}`. Available options are: {}".format(
            arch, list(archs_dict.keys()),
        ))

    # make unet like supervise affinity:V ggunet_saff_mult
    if arch.lower() == 'vggunet_saff_mult':
        return model_class(args.backbone, args.in_ch, args.use_fim, args.affinity, args.affinity_supervised,
                           args.up, args.classes, args.steps, reduce_dim=args.reduce_dim)
    else:
        raise RuntimeError('No implementation: ', arch.lower())