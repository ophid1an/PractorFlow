import torch
from typing import Union, Literal, Optional

ResolvedDType = Union[torch.dtype, Literal["auto"], None]

_TORCH_DTYPE_BY_NAME = {
    name: value for name, value in vars(torch).items() if isinstance(value, torch.dtype)
}


def dtype_from_config(dtype: Optional[str]) -> ResolvedDType:
    if dtype is None:
        return None

    if dtype == "auto":
        return "auto"

    try:
        return _TORCH_DTYPE_BY_NAME[dtype]
    except KeyError:
        return None
