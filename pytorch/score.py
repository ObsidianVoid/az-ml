import json
import numpy
import torch
from torch.autograd import Variable
from torch import optim
from azureml.core.model import Model

import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

def init():
    global model
    model_path = Model.get_model_path('aml-model')
    model = torch.load(model_path)
    model.eval()

def run(raw_data):
    data = json.loads(raw_data)['data']
    data = numpy.array(data)
    var = Variable(torch.from_numpy(data).float())
    result = model(var).data
    resultList = numpy.array(result.numpy()).tolist()
    return json.dumps({"result": resultList})