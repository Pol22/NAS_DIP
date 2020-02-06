from tensorflow.keras.layers import Conv2D, Add, Concatenate, Input, \
                                    MaxPool2D, UpSampling2D, LeakyReLU, \
                                    BatchNormalization, Layer
from tensorflow.compat.v2.nn import depth_to_space, space_to_depth
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import save_model
from random import choice


def res_block(inputs, skip, filter_size, channels):
    # x = BatchNormalization()(inputs)
    # x = Conv2D(channels, filter_size, padding='same')(x)
    # TODO create reflected padding for valid conv
    x = Conv2D(channels, filter_size, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(channels, filter_size, padding='same')(x)
    x = BatchNormalization()(x)
    if skip:
        skip = Conv2D(channels, 1, padding='same')(inputs)
        skip = BatchNormalization()(skip)
        x = Add()([x, skip])
    x = LeakyReLU()(x)
    return x


# class ResBlock(Layer):
#     def __init__(self, skip, filter_size, channels):
#         super(ResBlock, self).__init__()
#         self.conv1 = Conv2D(channels, filter_size, padding='same')
#         self.bn1 = BatchNormalization()
#         self.act1 = LeakyReLU()
#         self.conv2 = Conv2D(channels, filter_size, padding='same')
#         self.bn2 = BatchNormalization()
#         self.skip = skip
#         if self.skip:
#             self.conv_skip = Conv2D(channels, 1, padding='same')
#             self.bn_skip = BatchNormalization()
#             self.add = Add()
#         self.act2 = LeakyReLU()

#     def call(self, inputs):
#         x = self.conv1(inputs)
#         x = self.bn1(x)
#         x = self.act1(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         if self.skip:
#             skip = self.conv_skip(inputs)
#             skip = self.bn_skip(skip)
#             x = self.add([x, skip])
#         return self.act2(x)

#     def get_config(self):
#         return super(ResBlock, self).get_config()


def concat_inputs(x, skip_inputs):
    # print('base', x.shape)
    concat_input = [x]
    for skip_input, skip_channels in skip_inputs:
        # print(skip_input.shape)
        # check height (NHWC)
        up_factor = x.shape[1] // skip_input.shape[1]
        down_factor = skip_input.shape[1] // x.shape[1]
        if up_factor > 1:
            # check channels
            if skip_input.shape[3] // (up_factor * up_factor) > 0:
                skip = depth_to_space(skip_input, up_factor)
            else:
                return None
        elif down_factor > 1:
            # check height and width
            if skip_input.shape[1] // down_factor > 0 and \
               skip_input.shape[2] // down_factor > 0:
                skip = space_to_depth(skip_input, down_factor)
            else:
                return None
        elif up_factor == 1 and down_factor == 1:
            skip = skip_input
        else:
            return None
        # Conv 1x1
        skip = Conv2D(skip_channels, 1, padding='same')(skip)
        # print(up_factor, down_factor, skip.shape)
        # Append to concat input
        concat_input.append(skip)
    # Concat
    return Concatenate()(concat_input)


def units_list_to_model(units_list, input_shape, depth=5):
    assert(len(units_list) == depth)
    # check depth factor (needed for scale2depth and depth2scale)
    assert(input_shape[0] % (2 ** depth) == 0)
    assert(input_shape[1] % (2 ** depth) == 0)
    # check possibility
    assert(input_shape[0] // (2 ** depth) > 0)
    assert(input_shape[1] // (2 ** depth) > 0)

    inputs = Input(shape=input_shape)
    x = inputs
    decoder_inputs = [[] for _ in range(depth)] # list of lists
    # encoders
    for unit in units_list:
        x = res_block(x, unit.enc_skip, unit.enc_filter_size,
                      unit.enc_channels)
        # x = ResBlock(unit.enc_skip, unit.enc_filter_size,
        #              unit.enc_channels)(x)
        x = MaxPool2D(pool_size=(2,2))(x)
        for i, skip_connection in enumerate(unit.skip_connections):
            if skip_connection[0]:
                # (input, channels)
                decoder_inputs[i].append((x, skip_connection[1]))
    # decoders
    decoder_inputs.reverse()
    units_list.reverse()
    for i, unit in enumerate(units_list):
        if decoder_inputs[i]: # skip inputs not empty
            x = concat_inputs(x, decoder_inputs[i])
        if x is None:
            # print('Incompatible model!')
            return None

        x = res_block(x, unit.dec_skip, unit.dec_filter_size,
                      unit.dec_channels)
        # x = ResBlock(unit.dec_skip, unit.dec_filter_size,
        #              unit.dec_channels)(x)
        x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)

    last_unit = choice(units_list)
    x = res_block(x, last_unit.dec_skip, last_unit.dec_filter_size,
                  last_unit.dec_channels)
    # x = ResBlock(last_unit.dec_skip, last_unit.dec_filter_size,
    #              last_unit.dec_channels)(x)
    out = Conv2D(3, 1, padding='same', activation='sigmoid')(x)

    return Model(inputs=inputs, outputs=out)

# from unit import *

# while 1 > 0:
#     val = gen_bin_model()
#     units = bin_model_to_units_list(val)
#     model = units_list_to_model(units, (576, 384, 32))
#     if model is not None:
#         plot_model(model, show_shapes=True)
#         save_model(model, 'test_model.h5', include_optimizer=False)
#         break
