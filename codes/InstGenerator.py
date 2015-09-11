from InstTemplete import InstTemplate


class InstGenerator(InstTemplate):
    def __init__(self, *layers):
        super(InstTemplate, self).__init__()
        if len(layers) != 1:
            'ERR: wrong InstGen init'
            return
        self.layers = layers[0]
        self.w_space = 0
        self.b_space = 0
        self.input_space = 0
        self.w_ddr_addr_base = 0
        self.b_ddr_addr_base = 0
        self.din_ddr_addr_base = 0
        self.dout_ddr_addr_base = 0
        self.if_last = False

        self.compute_space()
        self.compute_addr_base()


    def compute_space(self):
        # get weight space
        for i,l in enumerate(self.layers):
            self.w_space += l.weight_dim.count_ * 2
            self.b_space += l.bias_dim.count_ * 2
        # print 'weight space: ', self.w_space
        # print 'bias space: ', self.b_space
        # get input data space
        self.input_space += self.layers[0].input_dim.count_ * 2
        # print 'input data space: ', self.input_space

    def compute_addr_base(self):
        self.w_ddr_addr_base = 0
        self.b_ddr_addr_base = self.w_ddr_addr_base + self.w_space
        self.din_ddr_addr_base = self.b_ddr_addr_base + self.b_space
        self.dout_ddr_addr_base = self.din_ddr_addr_base + self.input_space
        # print 'compute for each layer'
        w_base_temp = self.w_ddr_addr_base
        b_base_temp = self.b_ddr_addr_base
        d_base_temp = self.din_ddr_addr_base
        for i,l in enumerate(self.layers):
            if i == len(self.layers)-1:
                l.if_last = True
            else:
                l.if_last = False
            if i == 1:
                # only the first layer(after data) reads data from d base
                l.input_addr_base = d_base_temp
            else:
                # others should read the output of their previous layers
                l.input_addr_base = self.dout_ddr_addr_base
            # all layers should store their output to the same space
            l.output_addr_base = self.dout_ddr_addr_base
            # only layers with weight should be involved
            if l.layer_type == 'mlp' or l.layer_type == 'conv':
                l.w_addr_base = w_base_temp
                w_base_temp += l.weight_dim.count_ * 2
                l.b_addr_base = b_base_temp
                b_base_temp += l.bias_dim.count_ * 2
            else:
                l.w_addr_base = -1
                l.b_addr_base = -1

    def print_info(self):
        print '[base addr of each layer]'
        for i, x in enumerate(self.layers):
            print '-->for layer %d(%s)' % (i, x.layer_type)
            print 'w: %d\nb: %d\ninput: %d\noutput: %d\n' % (
                x.w_addr_base, x.b_addr_base, x.input_addr_base, x.output_addr_base)


class InstGenMlp(InstTemplate):
    """
    generate insts for one mlp layer
    """

    def __init__(self, layer):
        InstTemplate.__init__(self)
        self.config = layer
        self.out_inst_num = InstTemplate.mod(self, layer.output_dim.channel_, self.Tn)
        self.in_inst_num = InstTemplate.mod(self, layer.input_dim.channel_, self.sram_size)
        self.inst_number = self.out_inst_num * self.in_inst_num
        self.input_size_list = InstTemplate.token_size(self, layer.input_dim.channel_ * 2, self.sram_size * 2)
        self.output_size_list = InstTemplate.token_size(self, layer.output_dim.channel_ * 2, self.Tn * 2)
        self.weight_size_list = []
        for y in xrange(self.in_inst_num):
            for x in xrange(self.out_inst_num):
                self.weight_size_list.append(self.input_size_list[y] * self.output_size_list[x] / 2)
        self.bias_size_list = self.output_size_list
        # address
        self.weight_addr_list = self.compute_addrs(layer.w_addr_base, self.weight_size_list)
        self.bias_addr_list = self.compute_addrs(layer.b_addr_base, self.bias_size_list)
        self.input_addr_list = self.compute_addrs(layer.input_addr_base, self.input_size_list)
        self.output_addr_list = self.compute_addrs(layer.output_addr_base, self.output_size_list)
        self.output_sram_addr_list = self.compute_addrs(0, self.output_size_list)
        # self.print_info()

    def generate(self):
        for i in range(self.in_inst_num):
            for o in range(self.out_inst_num):
                # CP
                if i == self.in_inst_num-1 and o == self.out_inst_num-1 and self.config.if_last == True:
                    self.CP_inst = self.CP['end']
                else:
                    self.CP_inst = self.CP['continue']
                # SB inst
                # print i * self.out_inst_num + o, self.weight_addr_list[i * self.out_inst_num + o]
                self.SB_inst['op'] = self.SB['op']['LOAD']
                self.SB_inst['addr'] = self.SB['addr'](self.weight_addr_list[i * self.out_inst_num + o])
                # print self.SB_inst['addr'].code
                self.SB_inst['size'] = self.SB['size'](self.weight_size_list[i * self.out_inst_num + o])
                # NBin
                if o == 0:
                    # load from ddr
                    self.NBin_inst['op'] = self.NBin['op']['LOAD']
                    self.NBin_inst['addr'] = self.NBin['addr'](self.input_addr_list[i])
                    self.NBin_inst['size'] = self.NBin['size'](self.input_size_list[i])
                else:
                    self.NBin_inst['op'] = self.NBin['op']['READ']
                    self.NBin_inst['addr'] = self.NBin['addr'](0)
                    self.NBin_inst['size'] = self.NBin['size'](self.input_size_list[i])
                # NBout
                # size
                self.NBout_inst['size'] = self.NBout['size'](self.output_size_list[o])
                # NBout op1, addr1
                if i == self.in_inst_num - 1:
                    self.NBout_inst['op1'] = self.NBout['op1']['STORE']
                    self.NBout_inst['addr1'] = self.NBout['addr1'](self.output_addr_list[o])
                else:
                    self.NBout_inst['op1'] = self.NBout['op1']['WRITE']
                    self.NBout_inst['addr1'] = self.NBout['addr1'](self.output_sram_addr_list[o])
                # op2, addr2
                if i == 0:
                    self.NBout_inst['op2'] = self.NBout['op2']['LOAD']
                    self.NBout_inst['addr2'] = self.NBout['addr2'](self.bias_addr_list[o])
                else:
                    self.NBout_inst['op2'] = self.NBout['op2']['READ']
                    self.NBout_inst['addr2'] = self.NBout['addr2'](self.output_sram_addr_list[o])
                # NFU
                # op1: all must need mul and add
                self.NFU_inst['op1'] = self.NFU['op1']['MUL']
                self.NFU_inst['op2'] = self.NFU['op2']['ADD']
                # iter number: input/Tn
                itern = self.input_size_list[i] / 2 / self.Tn - 1
                self.NFU_inst['iter'] = self.NFU['iter'](itern)
                if i == self.in_inst_num - 1:
                    # last one, should take sigmoid action
                    self.NFU_inst['op3'] = self.NFU['op3']['SIGMOID']
                else:
                    self.NFU_inst['op3'] = self.NFU['op3']['NOP']

                # push one inst into list
                self.assemble()

    def is_last_inst(self, i, o):
        return i == self.in_inst_num - 1 & o == self.out_inst_num - 1

    def compute_addrs(self, base, sizes):
        re = [base]
        for x in range(len(sizes) - 1):
            re.append(re[x] + sizes[x])
        assert len(re) == len(sizes)
        return re

    def print_info(self):
        print 'out inst num: %d\nin inst num: %d\n' % (self.out_inst_num, self.in_inst_num)
        print 'out size list: ', self.output_size_list
        print 'in size list: ', self.input_size_list
        print 'weight size list: ', self.weight_size_list
        print 'bias size list: ', self.bias_size_list
        print 'out addr: ', self.output_addr_list
        print 'in addr: ', self.input_addr_list
        print 'weight addr: ', self.weight_addr_list
        print 'bias addr: ', self.bias_addr_list

    def print_inst(self):
        print 'net parameters: \ntype: mlp'
        print 'scales:',
        self.config.input_dim.print_info()
        print '-->',
        self.config.output_dim.print_info()
        print '\nw:',
        self.config.weight_dim.print_info()
        print '\nb:',
        self.config.bias_dim.print_info()
        print '\n'
        InstTemplate.printInstInfo(self)
