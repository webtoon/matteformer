from .resnet_decoder import ResNet_D_Dec, ResShortCut_D_Decoder, BasicBlock

__all__ = ['res_shortcut_decoder']

def res_shortcut_decoder(**kwargs):
    return ResShortCut_D_Decoder(BasicBlock, [2, 3, 3, 2], **kwargs)

