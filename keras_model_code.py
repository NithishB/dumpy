from CodeGen import keras_build_simple_sequential, keras_build_simple_parallel

# Sequential model dictionary format
layer_dict1 = {
    'input':{
        'num_layers':1,
        'shape': (10,1)
    },
    'conv1d':{
        'num_layers':2,
        'num_units': [[32,3],[32,5]]
    },
    'lstm':{
        'num_layers':1, 
        'num_units': [32]
    },
    'dense':{
        'num_layers':4, 
        'num_units': [128, 64, 32, 1]
    }
}


# parallel model dictionary format
# layer names are important to build the DAG
layer_dict2 = {
    'num_branches' : 11,
    'branches' : {
        'branch1': {
            'inputs': ['input_1'],
            'output': ['conv_1d_1']
        },
        'branch2': {
            'inputs': ['conv_1d_1'],
            'output': ['lstm_1']
        },
        'branch3': {
            'inputs': ['conv_1d_1','lstm_1'],
            'output': ['concat_1']
        },
        'branch4': {
            'inputs': ['conv_1d_1','lstm_1'],
            'output': ['add_1']
        },
        'branch5': {
            'inputs': ['conv_1d_1','lstm_1'],
            'output': ['multiply_1']
        },
        'branch6': {
            'inputs': ['concat_1','add_1', 'multiply_1'],
            'output': ['concat_2']
        },
        'branch7':{
            'inputs': ['concat_2'],
            'output': ['flatten_1']
        },
        'branch8': {
            'inputs': ['flatten_1'],
            'output': ['dense_1']
        },
        'branch9': {
            'inputs': ['dense_1'],
            'output': ['dense_2']
        },
        'branch10': {
            'inputs': ['dense_2'],
            'output': ['dense_3']
        },
        'branch11': {
            'inputs': ['dense_3'],
            'output': ['output_1']
        }
    },
    'layer_units' : {
        'input_1' : (10,1),
        'conv_1d_1' : [32, 3],
        'lstm_1' : [32],
        'dense_1' : [128],
        'dense_2' : [128],
        'dense_3' : [128],
        'output_1' : [1],
    },
    'layer_activations' : {
        'dense_1' : 'relu',
        'dense_2' : 'relu',
        'dense_3' : 'relu'
    }
}

code1, model1 = keras_build_simple_sequential(layer_dict1)
code2, op, model2 = keras_build_simple_parallel(layer_dict2)

print(code1)
model1.summary()

print()

print(code2)
print(op)
model2.summary()
