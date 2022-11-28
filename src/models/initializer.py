import math

from mindspore.common import initializer as weight_init


def trunc_normal_(weight, std):
    weight.set_data(weight_init.initializer(weight_init.TruncatedNormal(sigma=std),
                                            weight.shape,
                                            weight.dtype))


def normal_(weight, std):
    weight.set_data(weight_init.initializer(weight_init.Normal(sigma=std),
                                            weight.shape,
                                            weight.dtype))


def uniform_(weight, scale):
    weight.set_data(weight_init.initializer(weight_init.Uniform(scale=scale),
                                            weight.shape,
                                            weight.dtype))


def zeros_(weight):
    weight.set_data(weight_init.initializer(weight_init.Zero(),
                                            weight.shape,
                                            weight.dtype))


def ones_(weight):
    weight.set_data(weight_init.initializer(weight_init.One(),
                                            weight.shape,
                                            weight.dtype))


def constant_(weight, value):
    weight.set_data(weight_init.initializer(weight_init.Constant(value=value),
                                            weight.shape,
                                            weight.dtype))


def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor.shape[1]
    num_output_fmaps = tensor.shape[0]
    receptive_field_size = 1
    if tensor.dim() > 2:
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    else:
        raise NotImplementedError
    variance = scale / denom

    if distribution == "truncated_normal":
        # constant is stddev of standard normal truncated to (-2, 2)
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        normal_(tensor, std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        uniform_(bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')
