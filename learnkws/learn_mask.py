'''
Date: 2022-03-04 18:10:52
LastEditors: Cyan
LastEditTime: 2022-03-07 10:21:34
'''

import torch


def padding_mask(lengths: torch.Tensor) -> torch.Tensor:
    """
    Examples:
        >>> lengths = torch.tensor([2, 2, 3], dtype=torch.int32)
        >>> mask = padding_mask(lengths)
        >>> print(mask)
        tensor([[False, False,  True],
                [False, False,  True],
                [False, False, False]])
    """
    batch_size = lengths.size(0)
    max_len = int(lengths.max().item())
    seq = torch.arange(max_len, dtype=torch.int64, device=lengths.device)
    seq = seq.expand(batch_size, max_len)
    return seq >= lengths.unsqueeze(1)

if __name__ == '__main__':
    lengths = torch.tensor([2, 2, 3], dtype=torch.int32)
    print(lengths.numel())
    mask = padding_mask(lengths)
    print(mask, mask.size())
