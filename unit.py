import numpy as np


class Unit(object):
    def __init__(self, enc_skip, enc_filter_size, enc_channels,
                       dec_skip, dec_filter_size, dec_channels,
                       skip_connections, depth):
        assert(len(skip_connections) == depth)
        self.enc_skip = enc_skip
        self.enc_filter_size = enc_filter_size
        self.enc_channels = enc_channels
        self.dec_skip = dec_skip
        self.dec_filter_size = dec_filter_size
        self.dec_channels = dec_channels
        self.skip_connections = skip_connections
        self.depth = depth


def gen_bin_model(depth):
    length = depth * (14 + 4 * depth) # NAS-DIP
    return np.random.randint(2, size=length)


def bin_list_to_str(array):
    return ''.join(list(map(str, array)))


def bin_list_to_hex(array, depth):
    length = depth * (14 + 4 * depth) # NAS-DIP
    assert(len(array) == length)
    int_value = int(bin_list_to_str(array), 2)
    hex_value = f'{int_value:x}'
    return hex_value


def hex_to_bin_list(hex_str, depth):
    length = depth * (14 + 4 * depth) # NAS-DIP
    int_value = int(hex_str, 16)
    bin_str = f'{int_value:b}'
    bin_str = '0' * (length - len(bin_str)) + bin_str
    array = list(map(int, list(bin_str)))
    return np.array(array)


def bin_list_to_int(array):
    result = 0
    for i, value in enumerate(reversed(array)):
        result += int(value) * (2 ** i)
    return result


def bin_model_to_units_list(array, depth):
    unit_length = 14 + 4 * depth
    assert(array.size % unit_length == 0)
    result = list()
    for i in range(depth):
        bin_unit = array[i*unit_length:(i+1)*unit_length]
        # encoder params
        enc_skip = bool(bin_unit[0])
        # [3, 5, ..., 15]
        enc_filter_size = 2 * bin_list_to_int(bin_unit[1:4]) + 1
        # [4, 8, ..., 512]
        enc_channels = 2 ** (bin_list_to_int(bin_unit[4:7]) + 2)
        # decoder params
        dec_skip = bool(bin_unit[7])
        # [3, 5, ..., 15]
        dec_filter_size = 2 * bin_list_to_int(bin_unit[8:11]) + 1
        # [4, 8, ..., 512]
        dec_channels = 2 ** (bin_list_to_int(bin_unit[11:14]) + 2)
        # skip_connections
        skip_connections = list()
        for j in range(depth):
            bin_skip = bin_unit[14+j*4:14+(j+1)*4]
            skip_open = not bool(bin_skip[0]) # 0 is open
            skip_channels = 2 ** (bin_list_to_int(bin_skip[1:4]) + 2)
            skip_connections.append((skip_open, skip_channels))

        result.append(
            Unit(enc_skip, enc_filter_size, enc_channels,
                 dec_skip, dec_filter_size, dec_channels,
                 skip_connections, depth)
        )

        # print(enc_skip, enc_filter_size, enc_channels,
        #          dec_skip, dec_filter_size, dec_channels,
        #          skip_connections, depth)

        # print(bin_list_to_str(bin_unit))

    return result
