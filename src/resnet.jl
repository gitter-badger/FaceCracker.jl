###################################################################################################
#
# ResNet50 model for Flux
#   Reference:
#       - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
#
####################################################################################################
using Flux

include("config.jl")

"""
    The identity block is the block that has no conv layer at shortcut.

    # Arguments
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
Returns the block model.
"""
function identity_block(kernel_size, filters)
    filter1, filter2, filter3 = filters
    Chain(
        Conv((1, 1), filter1=>filter1), 
        BatchNorm(filter1),
        x -> relu.(x),
        Conv(kernel_size, filter1=>filter2, pad=(1, 1)),
        BatchNorm(filter2),
        x -> relu.(x),
        Conv((1, 1), filter2=>filter3),
        BatchNorm(filter3),
        x -> relu.(x)
    ) |> gpu
end

struct IDBlock
    kernel_size::Tuple
    filters::AbstractArray
    model::Chain
    IDBlock(kernel_size, filters) = new(kernel_size, filters, identity_block(kernel_size, filters))
end

function (ib::IDBlock)(x)
    ib.model(x)
end


"""
    conv_block is the block that has a conv layer at shortcut

    # Arguments
        input_tensor: input for the model
        kernel_size: the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
    Note: the shortcut should have strides=(2, 2)
Returns block model.
"""
function plain_model(kernel_size, filters; strides=(2, 2))
    filter1, filter2, filter3 = filters
    Chain(
        Conv((1, 1), filter1=>filter1, stride=strides),
        BatchNorm(filter1),
        x -> relu.(x),
        Conv(kernel_size, filter1=>filter2, pad=(1, 1)),
        BatchNorm(filter2),
        x -> relu.(x),
        Conv((1, 1), filter2=>filter3),
        BatchNorm(filter3),
    ) |> gpu
end

struct ConvBlock
    kernel_size::Tuple
    filters::AbstractArray
    strides::Tuple
    plain::Chain
    shortcut::Chain
    ConvBlock(kernel_size, filters; strides=(2, 2)) = new(kernel_size, filters, strides, 
        plain_model(kernel_size, filters, strides=strides), 
        Chain(Conv((1, 1), filter[1]=>filters[3], stride=strides), BatchNorm(filters[3])) |> gpu)
end

function (cb::ConvBlock)(x)
    relu.(cb.plain(x) + cb.shortcut(x))
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


ResNetModel = Chain(
    x -> zero_padding(x, padding=(3, 3)),
    Conv((7, 7), 1=>64, stride=(2, 2)),
    BatchNorm(64),
    x -> relu.(x),
    x -> maxpool(x, (3, 3); stride=(2, 2)),
    ConvBlock((3, 3), [64, 64, 256], strides=(1, 1)),
    IDBlock((3, 3), [64, 64, 256]),
    IDBlock((3, 3), [64, 64, 256]),
    ConvBlock((3, 3), [128, 128, 512]),
    IDBlock((3, 3), [128, 128, 512]),
    IDBlock((3, 3), [128, 128, 512]),
    IDBlock((3, 3), [128, 128, 512]),
    ConvBlock((3, 3), [256, 256, 1024]),
    IDBlock((3, 3), [256, 256, 1024]),
    IDBlock((3, 3), [256, 256, 1024]),
    IDBlock((3, 3), [256, 256, 1024]),
    IDBlock((3, 3), [256, 256, 1024]),
    IDBlock((3, 3), [256, 256, 1024]),
    ConvBlock((3, 3), [512, 512, 2048]),
    IDBlock((3, 3), [512, 512, 2048]),
    IDBlock((3, 3), [512, 512, 2048]),
    x -> meanpool(x, (7, 7)),
    x -> reshape(x, :, size(x, 4)),
    Dense(512, NUM_CLASSES, softmax),
) |> gpu


