
def encode_nb201(arch_str):
    # turn arch_str into encoding
    OPS = ['avg_pool_3x3', 'nor_conv_1x1', 'nor_conv_3x3', 'none', 'skip_connect']
    tokens = arch_str.split('|')
    ops = [t.split('~')[0] for i,t in enumerate(tokens) if i not in [0,2,5,9]]
    encoding = []
    for op in ops:
        encoding.append(OPS.index(op))
    return encoding