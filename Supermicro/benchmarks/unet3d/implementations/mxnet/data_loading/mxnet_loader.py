# Copyright (c) 2018-2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from mxnet import gluon, nd

from data_loading.transforms import get_transforms


class KitsDataset(gluon.data.Dataset):
    def __init__(self, flags, image_list: list, label_list: list, mode: str):
        self.image_list = image_list
        self.label_list = label_list
        self.transforms = get_transforms(flags.input_shape if mode == "train" else flags.val_input_shape,
                                         flags.layout, mode=mode, oversampling=flags.oversampling)
        self.layout = flags.layout

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        data = {"image": np.load(self.image_list[idx]), "label": np.load(self.label_list[idx])}
        if self.layout == "NDHWC":
            data["image"] = np.moveaxis(data["image"], 0, -1)
            data["label"] = np.moveaxis(data["label"], 0, -1)
        data = self.transforms(data)
        return data["image"], data["label"]
