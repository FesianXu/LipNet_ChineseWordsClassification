# !/usr/bin/env python
# -*- coding:utf-8 -*-

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url