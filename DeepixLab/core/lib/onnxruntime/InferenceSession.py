from io import BytesIO

import onnx
import onnxruntime as rt

from .device import DeviceRef


def InferenceSession_with_device(onnx_model_or_path, device : DeviceRef):
    """
    Construct onnxruntime.InferenceSession with this Device.

        onnx_model_or_path  Path|


        device     DeviceRef

    can raise Exception
    """

    if isinstance(onnx_model_or_path, onnx.ModelProto):
        b = BytesIO()
        onnx.save(onnx_model_or_path, b)
        onnx_model_or_path = b.getvalue()


    device_ep = device.info.execution_provider
    if device_ep not in rt.get_available_providers():
        raise Exception(f'{device_ep} is not avaiable in onnxruntime')

    ep_flags = {}
    if device_ep in ['CUDAExecutionProvider','DmlExecutionProvider']:
        ep_flags['device_id'] = device.info.index

    sess_options = rt.SessionOptions()
    sess_options.log_severity_level = 4
    sess_options.log_verbosity_level = -1
    
    if device_ep == 'DmlExecutionProvider':
        sess_options.enable_mem_pattern = False
    sess = rt.InferenceSession(onnx_model_or_path, providers=[ (device_ep, ep_flags) ], sess_options=sess_options)
    return sess
