import sys
import os

import LoadModel
import InstGenerator

global config_file
global inst_file


def main():
    if len(sys.argv) != 2:
        print 'wrong input argvs'
        print('example: python main.py ../configs/mlp.cfg')
        exit(0)
    else:
        config_file = sys.argv[1]
        if(not os.path.isfile(config_file)):
            print 'config file %s does not exist!' %(config_file)
            exit(0)
    output_file = config_file.split('.')[0]+'.inst'
    layers = LoadModel.load_model(config_file)
    insgen = InstGenerator.InstGenerator(layers)
    mlp1 = InstGenerator.InstGenMlp(insgen.layers[1])
    mlp1.generate()
    mlp1.print_inst()


    f = open(output_file, 'w')
    for i in mlp1.INSTs:
        f.write(str(i) + '\n')
    f.close()

    i = 1



main()

import copy
#
# def test():
#     d1 = {'a': 1, 'b':  2}
#     d2 = {'a': 3, 'b':  5}
#     l1= []
#     l1.append(d1)
#     l1.append(d2)
#     l3 = copy.deepcopy(l1)
#     all = []
#     all.append(l3)
#     print all
#     d1['a'] = 6
#     d1['b'] = 9
#     l1=[]
#     l1.append(d1)
#     l1.append(d2)
#     all.append(l1)
#     print all
#
# test()