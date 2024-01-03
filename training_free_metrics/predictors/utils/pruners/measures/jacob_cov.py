# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import torch
import numpy as np

from . import measure


def get_batch_jacobian(net, inputs, target, device, split_data):
    inputs.x.requires_grad_(True)

    N = inputs.x.shape[0]
    #print(split_data)
    for sp in range(split_data):
        st=sp*N//split_data
        en=(sp+1)*N//split_data
        inputs.x = inputs.x.float()
        y = net(inputs)[inputs.train_mask]
        y.backward(torch.ones_like(y))

    jacob = inputs.x.grad.detach()[inputs.train_mask]
    inputs.x.requires_grad_(False)
    return jacob, target.detach()

def eval_score(jacob, labels=None):
    
    corrs = np.corrcoef(jacob)
    v, _  = np.linalg.eig(corrs)
    k = 1e-5
    return -np.sum(np.log(v + k) + 1./(v + k))

@measure('jacob_cov', bn=True)
def compute_jacob_cov(net, inputs, targets, split_data=1, loss_fn=None):
    device = inputs.x.device
    # Compute gradients (but don't apply them)
    net.zero_grad()
    #print(inputs_x[inputs_train_mask])
    jacobs, labels = get_batch_jacobian(net, inputs, targets, device, split_data=split_data)
    #print(labels.shape)
    jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()
    #print(jacobs.shape)
    try:
        jc = eval_score(jacobs, labels)
    except Exception as e:
        print(e)
        jc = np.nan
    return jc
