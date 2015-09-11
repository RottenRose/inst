from common import Dims
from common import Atom


class layer_param(object):
    """
    input_dim
    output_dim
    kernel_dim
    bias_dim

    """

    def __init__(self, string):  # ['mlp', 16]
        self.layer_type = string[0]
        self.dim = tuple([int(x) for x in string[1:len(string)]])
        self.input_dim = Dims()
        self.output_dim = Dims()
        self.weight_dim = Dims()
        self.bias_dim = Dims()
        self.parse_dims()
        # w,b,input data base addr
        self.w_addr_base = 0
        self.b_addr_base = 0
        self.input_addr_base = 0
        self.output_addr_base = 0
        self.batch_ = 0
        self.channel_ = 0
        self.kernel_y = 0
        self.kernel_x = 0
        self.stride_y = 0
        self.stride_x = 0
        self.pad_y = 0
        self.pad_x = 0
        self.y_ = 0
        self.x_ = 0

    def parse_conv(self):
        if len(self.dim) == 7:
            self.channel_ = self.dim[0]
            self.kernel_y = self.dim[1]
            self.kernel_x = self.dim[2]
            self.stride_y = self.dim[3]
            self.stride_x = self.dim[4]
            self.pad_y = self.dim[5]
            self.pad_x = self.dim[6]
            return True
        else:
            return False

    def parse_pool(self):
        if len(self.dim) == 6:
            self.kernel_y = self.dim[0]
            self.kernel_x = self.dim[1]
            self.stride_y = self.dim[2]
            self.stride_x = self.dim[3]
            self.pad_y = self.dim[4]
            self.pad_x = self.dim[5]
            return True
        else:
            return False

    def parse_mlp(self):
        if len(self.dim) == 1:
            self.channel_ = self.dim[0]
            self.y_ = 1
            self.x_ = 1
            return True
        else:
            print 'ERR: parse mlp '
            return False

    def parse_data(self):
        if len(self.dim) == 4:
            self.batch_ = self.dim[0]
            self.channel_ = self.dim[1]
            self.y_ = self.dim[2]
            self.x_ = self.dim[3]
            self.output_dim = Dims(self.batch_, self.channel_, self.y_, self.x_)
            self.input_dim = self.output_dim
            return True
        else:
            print 'ERR: parse data '
            return False

    def parse_dims(self):
        results = {
            'data': lambda: self.parse_data(),
            'mlp': lambda: self.parse_mlp(),
            'conv': lambda: self.parse_conv(),
            'pool': lambda: self.parse_pool(),
        }
        return results[self.layer_type]()

    def compute_mlp(self, layer_in):
        self.batch_ = layer_in.batch_
        in_size = layer_in.channel_ * layer_in.y_ * layer_in.x_
        self.weight_dim = Dims(self.channel_, in_size, 1, 1)
        self.bias_dim = Dims(1, self.channel_, 1, 1)
        self.output_dim = Dims(layer_in.batch_, self.channel_, 1, 1)
        self.input_dim = Dims(layer_in.batch_, layer_in.channel_, 1, 1)

    def compute_conv(self, layer_in):
        self.batch_ = layer_in.batch_
        self.y_ = (layer_in.y_ + self.pad_y * 2 - self.kernel_y) / self.stride_y + 1
        self.x_ = (layer_in.x_ + self.pad_x * 2 - self.kernel_x) / self.stride_x + 1
        self.weight_dim = Dims(self.channel_, layer_in.channel_, self.kernel_y, self.kernel_x)
        self.bias_dim = Dims(1, self.channel_, 1, 1)
        self.input_dim = Dims(layer_in.batch_, layer_in.channel_, layer_in.y_, layer_in.x_)
        self.output_dim = Dims(layer_in.batch_, self.channel_, self.y_, self.x_)

    def compute_pool(self, layer_in):
        self.batch_ = layer_in.batch_
        self.y_ = (layer_in.y_ + layer_in.pad_y * 2 - self.kernel_y) / self.stride_y + 1
        self.x_ = (layer_in.x_ + layer_in.pad_x * 2 - self.kernel_x) / self.stride_x + 1
        self.input_dim = layer_in.output_dim
        self.output_dim = Dims(layer_in.batch_, self.channel_, self.y_, self.x_)

    def compute(self, layer_in):
        results = {
            'mlp': lambda: self.compute_mlp(layer_in),
            'conv': lambda: self.compute_conv(layer_in),
            'pool': lambda: self.compute_pool(layer_in)
        }
        return results[self.layer_type]()

    def print_info(self):
        print 'layer type: %s' % (self.layer_type)
        print 'input dim is\t:',
        self.input_dim.print_info()
        print '\noutput dim is\t:',
        self.output_dim.print_info()
        print '\nweight dim is\t:',
        self.weight_dim.print_info()
        print '\nbias dim is\t:',
        self.bias_dim.print_info()


def compute_params(layers):
    for x in range(1, len(layers)):
        layers[x].compute(layers[x - 1])
    return layers


def load_model(file):
    print 'loading model file %s' % (file)
    layers = [layer_param(x.strip('\n').split(',')) for x in open(file).readlines() if ',' in x]
    re = filter(layer_param.parse_dims, layers)
    layers = compute_params(re)
    return layers
