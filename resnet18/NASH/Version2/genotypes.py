from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

Genotype_res18 = namedtuple('Genotype_res18', 'normal0 normal1 normal2 normal3')


PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'skip_connect',
    'conv_1x1',
    'conv_3x3',
    'conv_5x5'
]


binary = Genotype_res18(normal0=[('skip_connect', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 1), ('max_pool_3x3', 1)], normal1=[('conv_3x3', 0), ('conv_5x5', 0), ('skip_connect', 2), ('max_pool_3x3', 2)], normal2=[('conv_5x5', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 1), ('max_pool_3x3', 1)], normal3=[('conv_5x5', 0), ('conv_5x5', 1), ('max_pool_3x3', 2), ('skip_connect', 2)]) #V2


DARTS_V1 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])

DARTS_V2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])

DARTS = binary

