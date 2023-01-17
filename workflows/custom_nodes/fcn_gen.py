from nipype import Node, Function

def fcn_generate(time_series, window_size=2):
    import numpy as np
    
    return np.corrcoef(time_series.T)

fcn_gen = Node(Function(input_names=['time_series'], output_names=['fcn_matrix'], function=fcn_generate), name='fcn_gen')