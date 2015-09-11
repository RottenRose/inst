from common import Dims
from common import Atom
import copy


class InstTemplate(object):
    def __init__(self):
        self.InstComponent = ['CP', 'SB', 'NBin', 'NBout', 'NFU']
        self.InstComponent_SB = ['op', 'addr', 'size']
        self.InstComponent_NBin = ['op', 'addr', 'size']
        self.InstComponent_NBout = ['op1', 'op2', 'addr1', 'addr2', 'size']
        self.InstComponent_NFU = ['op1', 'op2', 'op3', 'iter']
        CP = {}
        SB = {}
        NBin = {}
        NBout = {}
        NFU = {}
        # CP
        CP['continue'] = Atom('0', 'CONTINUE:\t')
        CP['end'] = Atom('1', 'END:\t')
        # SB
        SB['op'] = {}
        SB['op']['NOP'] = Atom('000', 'NOP:\t')
        SB['op']['LOAD'] = Atom('001', 'LOAD:\t')
        SB['op']['READ'] = Atom('010', 'READ:\t')
        SB['addr'] = lambda x: Atom(self.int2bin(x, 32), 'ADDR:\t' + str(x))
        SB['size'] = lambda x: Atom(self.int2bin(x, 32), 'SIZE:\t' + str(x))
        # NBin
        NBin['op'] = {}
        NBin['op']['NOP'] = Atom('000', 'NOP:\t')
        NBin['op']['LOAD'] = Atom('001', 'LOAD:\t')
        NBin['op']['READ'] = Atom('010', 'READ:\t')
        NBin['addr'] = lambda x: Atom(self.int2bin(x, 32), 'ADDR:\t' + str(x))
        NBin['size'] = lambda x: Atom(self.int2bin(x, 32), 'SIZE:\t' + str(x))
        # NBout
        NBout['op1'] = {}
        NBout['op1']['NOP'] = Atom('000', 'NOP:\t')
        NBout['op1']['STORE'] = Atom('001', 'STORE:\t')
        NBout['op1']['WRITE'] = Atom('010', 'WRITE:\t')
        NBout['op2'] = {}
        NBout['op2']['NOP'] = Atom('000', 'NOP:\t')
        NBout['op2']['LOAD'] = Atom('001', 'LOAD:\t')
        NBout['op2']['READ'] = Atom('010', 'READ:\t')
        NBout['addr1'] = lambda x: Atom(self.int2bin(x, 32), 'ADDR1:\t' + str(x))
        NBout['addr2'] = lambda x: Atom(self.int2bin(x, 32), 'ADDR2:\t' + str(x))
        NBout['size'] = lambda x: Atom(self.int2bin(x, 32), 'SIZE:\t' + str(x))
        # NFU
        NFU['op1'] = {}
        NFU['op1']['NOP'] = Atom('0000', 'NOP:\t')
        NFU['op1']['MUL'] = Atom('0001', 'MUL:\t')
        NFU['op1']['ADD'] = Atom('0010', 'ADD:\t')
        NFU['op1']['CMP'] = Atom('0011', 'CMP:\t')
        NFU['op2'] = {}
        NFU['op2']['NOP'] = Atom('0000', 'NOP:\t')
        NFU['op2']['ADD'] = Atom('0001', 'ADD:\t')
        NFU['op3'] = {}
        NFU['op3']['NOP'] = Atom('0000', 'NOP:\t')
        NFU['op3']['SIGMOID'] = Atom('0001', 'SIG:\t')
        NFU['op3']['RELU'] = Atom('0010', 'RELU:\t')
        NFU['iter'] = lambda x: Atom(self.int2bin(x, 6), 'ITER:\t' + str(x))
        self.CP = CP
        self.SB = SB
        self.NBin = NBin
        self.NBout = NBout
        self.NFU = NFU
        # all below are temp use, only generate one instruction, which should be push into inst list
        self.CP_inst = Atom('0', 'init')
        self.SB_inst = {}
        self.SB_inst['op'] = Atom('000', 'init')
        self.SB_inst['addr'] = Atom('0' * 32, 'init')
        self.SB_inst['size'] = Atom('0' * 32, 'init')
        self.NBin_inst = {}
        self.NBin_inst['op'] = Atom('000', 'init')
        self.NBin_inst['addr'] = Atom('0' * 32, 'init')
        self.NBin_inst['size'] = Atom('0' * 32, 'init')
        self.NBout_inst = {}
        self.NBout_inst['op1'] = Atom('000', 'init')
        self.NBout_inst['op2'] = Atom('000', 'init')
        self.NBout_inst['addr1'] = Atom('0' * 32, 'init')
        self.NBout_inst['addr2'] = Atom('0' * 32, 'init')
        self.NBout_inst['size'] = Atom('0' * 32, 'init')
        self.NFU_inst = {}
        self.NFU_inst['op1'] = Atom('0' * 4, 'init')
        self.NFU_inst['op2'] = Atom('0' * 4, 'init')
        self.NFU_inst['op3'] = Atom('0' * 4, 'init')
        self.NFU_inst['iter'] = Atom('0' * 6, 'init')
        self.INST = ''
        self.INSTs = []
        self.INSTs_detail = []

        # ralative parameter about tiling and sram
        self.Tn = 16
        self.sram_size = 1024

    def assemble(self):
        tempi = copy.deepcopy({'CP': self.CP_inst, 'SB': self.SB_inst, 'NBin': self.NBin_inst, 'NBout': self.NBout_inst,
                               'NFU': self.NFU_inst})
        self.INSTs_detail.append(tempi)
        sb = ''
        nbin = ''
        nbout = ''
        nfu = ''
        for x in self.InstComponent_SB:
            sb += self.SB_inst[x].code
        for x in self.InstComponent_NBin:
            nbin += self.NBin_inst[x].code
        for x in self.InstComponent_NBout:
            nbout += self.NBout_inst[x].code
        for x in self.InstComponent_NFU:
            nfu += self.NFU_inst[x].code
        self.INST = self.CP_inst.code + sb + nbin + nbout + nfu
        self.INST = "".join(self.INST)
        self.INSTs.append(self.INST)
        # print 'INST: %s\npush into INSTs' % (self.INST)

    def int2bin(self, a, size):
        #         translate int a to binery code which is as long as size
        assert type(a) == int
        b = bin(a).split('b')[-1]
        assert len(b) <= size
        padz = size - len(b)
        ba = '0' * padz + b
        return ba

    def mod(self, inp, n):
        f = lambda x, y: x / y + int(x % y != 0)
        re = f(inp, n)
        return re

    def token_size(self, total, seglen):
        re = []
        i = total / seglen
        re = [seglen for x in range(i)]
        left = total - i * seglen
        if left != 0:
            re.append(left)
        return re

    def printInstInfo(self):
        for i, v in enumerate(self.INSTs_detail):
            print '[%d] %s' % (i + 1, self.INSTs[i])
            print '\n---CP:'
            print '\t%s\t%s' % (v['CP'].meaning, v['CP'].code)

            print '\n---SB:'
            for y in self.InstComponent_SB:
                print '\t[%s]\t%s\t%s' % (y, v['SB'][y].meaning, v['SB'][y].code)
            print '\n---NBin:'
            for y in self.InstComponent_NBin:
                print '\t[%s]\t%s\t%s' % (y, v['NBin'][y].meaning, v['NBin'][y].code)
            print '\n---NBout:'
            for y in self.InstComponent_NBout:
                print '\t[%s]\t%s\t%s' % (y, v['NBout'][y].meaning, v['NBout'][y].code)
            print '\n---NFU:'
            for y in self.InstComponent_NFU:
                print '\t[%s]\t%s\t%s' % (y, v['NFU'][y].meaning, v['NFU'][y].code)
