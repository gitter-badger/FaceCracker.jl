###################################################################################################
#
# ResNet50 model for Flux
#   Reference:
#       - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
#
####################################################################################################
using Flux

"""
    The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
Returns Output tensor for the block.
"""
function identity_block(input_tensor, kernel_size, filters)
    filter1, filter2, filter3 = filters
    model = Chain(
        Conv((1, 1), 1=>filter1), 
        BatchNorm(filter1),
        x -> relu.(x),
        Conv(kernel_size, filter1=>filter2, pad=(1, 1)),
        BatchNorm(filter2),
        x -> relu.(x),
        Conv((1, 1), filter2=>filter3),
        BatchNorm(filter3),
        x -> relu.(x)
    ) |> gpu
    model(input_tensor)
end

"""
    conv_block is the block that has a conv layer at shortcut

    # Arguments
        input_tensor: input for the model
        kernel_size: the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
    Note: the shortcut should have strides=(2, 2)
Returns Output tensor for the block.
"""
function conv_block(input_tensor, kernel_size, filters, strides=(2, 2))
    filter1, filter2, filter3 = filters
    model = Chain(
        Conv((1, 1), 1=>filter1, stride=strides),
        BatchNorm(filter1),
        x -> relu.(x),
        Conv(kernel_size, filter1=>filter2, pad=(1, 1)),
        BatchNorm(filter2),
        x -> relu.(x),
        Conv((1, 1), filter2=>filter3),
        BatchNorm(filter3),
        Conv((1, 1), filter3=>filter3, stride=strides),
        BatchNorm(filter3),
        x -> relu.(x)
    ) |> gpu
    model(input_tensor)
end

"""
    Zero-padding layer for input

    # Arguments
        inputs: input tensor
        padding: symmetric padding values for height and width [default=(1, 1)]
Returns add rows and columns of zeros at the top, bottom, left and right side of an input tensor.
"""
function zero_padding(inputs; padding=(1, 1))
    row, col, c, s = size(inputs)
    h, w = padding
    result = vcat(inputs, zeros(h, row, c, s))
    result = hcat(result, zeros(col + h, w, c, s))
    result = hcat(zeros(row + h, w, c, s), result)
    vcat(zeros(h, col + w * 2, c, s), result)
end


ResNet50 = Chain()


