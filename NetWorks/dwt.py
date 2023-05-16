import torch 

def dwt(x):

    """
    DWT (Discrete Wavelet Transform) function implementation according to
    "Multi-level Wavelet Convolutional Neural Networks"
    by Pengju Liu, Hongzhi Zhang, Wei Lian, Wangmeng Zuo
    https://arxiv.org/abs/1907.03128
       x shape - BCHW (channel first)
    """
    x = x.permute(0, 2, 3, 1)

    x1 = x[:, 0::2, 0::2, :] #x(2i−1, 2j−1)
    x2 = x[:, 1::2, 0::2, :] #x(2i, 2j-1)
    x3 = x[:, 0::2, 1::2, :] #x(2i−1, 2j)
    x4 = x[:, 1::2, 1::2, :] #x(2i, 2j)


    x_LL = x1 + x2 + x3 + x4
    x_LH = -x1 - x3 + x2 + x4
    x_HL = -x1 + x3 - x2 + x4
    x_HH = x1 - x3 - x2 + x4

    wavelets = torch.cat([x_LL,x_LH,x_HL,x_HH], dim=-1)
    wavelets = wavelets.permute(0, 3, 1, 2)
    return wavelets


def multi_level_dwt(x, levels):
    lst = []
    decomp = x
    for _ in range(levels):
        decomp = dwt(decomp[:,:3,:,:])
        lst.append(decomp)
    return lst

