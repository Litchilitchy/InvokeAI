import os
import torch
from collections import OrderedDict
from bigdl.nano.pytorch import InferenceOptimizer

def optimize_unet(
        unet,
        accelerator="jit", 
        ipex=True, 
        precision='float32',
        device='CPU',
        samples=None, 
        height=512,
        width=512,
        low_memory=False,
        cache=False, 
        fail_if_no_cache=False, 
        channels_last=False):
        """
        Trace a torch.nn.Module and convert it into an accelerated module for inference.

        For example, this function returns a PytorchOpenVINOModel when accelerator=='openvino'.

        :param low_memory: only valid when accelerator="jit" and ipex=True, model will use less memory during inference
        :cache_dir: the directory to save the converted model
        """
        generator = torch.Generator(device="cpu")
        generator.manual_seed(1)
        loaded = False

        conv_in = OrderedDict()
        conv_in.in_channels = unet.conv_in.in_channels
        name_or_path = unet._name_or_path

        latent_shape = (2, unet.in_channels, height // 8, width // 8)
        image_latents = torch.randn(latent_shape, generator=generator, device="cpu", dtype=torch.float32)
        cross_attention_dim = unet.config.cross_attention_dim
        encoder_hidden_states = torch.randn((2, 77, cross_attention_dim), generator=generator, device="cpu", dtype=torch.float32)
        input_sample = (image_latents, torch.Tensor([980]).long(), encoder_hidden_states)

        unet_input_names = ["sample", "timestep", "encoder_hidden_states"]
        unet_output_names = ["unet_output"]
        unet_dynamic_axes = {"sample": [0], "encoder_hidden_states": [0], "unet_output": [0]}

        if cache:
            cache_path = _get_cache_path(name_or_path, accelerator=accelerator, ipex=ipex, precision=precision, low_memory=low_memory, device=device)
            if precision == "bfloat16" and accelerator != "openvino":
                pass
            elif os.path.exists(cache_path):
                try:
                    print(f"Loading the existing cache from {cache_path}")
                    nano_unet = InferenceOptimizer.load(cache_path, model=unet, device=device)
                    loaded = True
                except Exception as e:
                    loaded = False
                    print(f"The cache path {cache_path} exists, but failed to load. Error message: {str(e)}")


        print("precision is", precision)
        if not loaded:
            if fail_if_no_cache:
                raise Exception("You have to download the model to nano_stable_diffusion folder")
                
            extra_args = {}
            if precision == 'float32':
                if accelerator == "jit":
                    weights_prepack = False if low_memory else None
                    extra_args["weights_prepack"] = weights_prepack
                    extra_args["use_ipex"] = ipex
                    extra_args["jit_strict"] = False
                    extra_args["enable_onednn"] = False
                    extra_args["channels_last"] = channels_last
                elif accelerator is None:
                    if ipex:
                        extra_args["use_ipex"] = ipex
                        extra_args["channels_last"] = channels_last
                    else:
                        raise ValueError("IPEX should be True if accelerator is None and precision is float32.")
                elif accelerator == "openvino":
                    extra_args["input_names"] = unet_input_names
                    extra_args["output_names"] = unet_output_names
                    # # Nano will deal with the GPU/VPU dynamic axes issue
                    # extra_args["dynamic_axes"] = unet_dynamic_axes
                    extra_args["dynamic_axes"] = False
                    extra_args["device"] = device
                else:
                    raise ValueError(f"The accelerator can be one of `None`, `jit`, and `openvino` if the precision is float32, but got {accelerator}")
                nano_unet = InferenceOptimizer.trace(unet,
                                                    accelerator=accelerator,
                                                    input_sample=input_sample,
                                                    **extra_args)
            else:
                precision_map = {
                    'bfloat16': 'bf16',
                    'int8': 'int8',
                    'float16': 'fp16'
                }
                precision_short = precision_map[precision]

                # prepare input samples, calib dataloader and eval functions
                if accelerator == "openvino":
                    extra_args["device"] = device
                    extra_args["input_names"] = unet_input_names
                    extra_args["output_names"] = unet_output_names
                    # Nano will deal with the GPU/VPU dynamic axes issue
                    extra_args["dynamic_axes"] = unet_dynamic_axes

                    if precision_short == "int8":
                        # TODO: openvino int8 here
                        raise ValueError("OpenVINO int8 quantization is not supported.")
                elif accelerator == "onnxruntime":
                    raise ValueError(f"Onnxruntime {precision_short} quantization is not supported.")
                else:
                    # PyTorch bf16
                    if precision_short == "bf16":
                        # Ignore jit & ipex
                        if accelerator == "jit":
                            raise ValueError(f"JIT {precision_short} quantization is not supported.")
                        extra_args["channels_last"] = channels_last
                    elif precision_short == "int8":
                        raise

                # unet
                nano_unet = InferenceOptimizer.quantize(unet,
                                                        accelerator=accelerator,
                                                        precision=precision_short,
                                                        input_sample=input_sample,
                                                        **extra_args)

            # Save model if cache=True
            if cache:
                print(f"Caching the converted unet model to {cache_path}")
                InferenceOptimizer.save(nano_unet, cache_path)

        setattr(nano_unet, "conv_in", conv_in)
        return nano_unet


def _get_cache_path(base_dir, accelerator="jit", ipex=True, precision='float32', low_memory=False, device='CPU'):
    # base_dir = os.path.join(base_dir, f"unet")
    model_dir = [precision]
    if accelerator:
        model_dir.append(accelerator)
    if ipex and accelerator != "openvino":
        model_dir.append("ipex")
        if low_memory:
            model_dir.append('low_memory')
    if device != 'CPU':
        model_dir.append(device)
    model_dir = "_".join(model_dir)
    return os.path.join(base_dir, model_dir)