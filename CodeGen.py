import os
import numpy as np
import pandas as pd
import tensorflow as tf

def get_rnn_layer(l_type, units, prev_layer, prev_layer_string, usel_type=False):
    string = ""
    x = None
    if 'lstm' in l_type:
        if usel_type:
            string = "\n"+l_type+" = tf.keras.layers.LSTM(units="+str(units)+",return_sequences=True)"+prev_layer_string
        else:
            string = "\nx = tf.keras.layers.LSTM(units="+str(units)+")"+prev_layer_string
        x = tf.keras.layers.LSTM(units=units,return_sequences=True)(prev_layer)
    elif 'gru' in l_type:
        if usel_type:
            string = "\n"+l_type+" = tf.keras.layers.GRU(units="+str(units)+")"+prev_layer_string
        else:
            string = "\nx = tf.keras.layers.GRU(units="+str(units)+",return_sequences=True)"+prev_layer_string
        x = tf.keras.layers.GRU(units=units,return_sequences=True)(prev_layer)
    
    return string, x

def get_cnn_layer(l_type, units, prev_layer, prev_layer_string, usel_type=False):
    string = ""
    x = None
    if '1d' in l_type:
        if usel_type:
            string = "\n"+l_type+" = tf.keras.layers.Conv1D(filters=" + str(units[0]) + ",kernel_size=" + str(units[1]) + ")" + prev_layer_string
        else:
            string = "\nx = tf.keras.layers.Conv1D(filters=" + str(units[0]) + ",kernel_size=" + str(units[1]) + ")" + prev_layer_string
        x = tf.keras.layers.Conv1D(filters=units[0], kernel_size=units[1])(prev_layer)

    elif '2d' in l_type:
        if usel_type:
            string = "\n"+l_type+" = tf.keras.layers.Conv2D(filters=" + str(units[0]) + ",kernel_size=" + str(units[1]) + ")" + prev_layer_string
        else:
            string = "\nx = tf.keras.layers.Conv2D(filters=" + str(units[0]) + ",kernel_size=" + str(units[1]) + ")" + prev_layer_string
        x = tf.keras.layers.Conv2D(filters=n_units[i][0], kernel_size=n_units[i][1])(prev_layer)

    return string, x

def get_dense_layer(l_type, units, prev_layer, prev_layer_string, activ=None, usel_type=False):
    string = ""
    x = None
    if usel_type:
        string += "\n"+l_type+" = tf.keras.layers.Dense(units="+str(units)
    else:
        string += "\nx = tf.keras.layers.Dense(units="+str(units)
    if activ:
        string += ", activation="+str(activ)
        x = tf.keras.layers.Dense(units=units, activation=activ)(prev_layer)
    else:
        x = tf.keras.layers.Dense(units=units)(prev_layer)
    string += ")"+prev_layer_string
    
    return string, x

def get_concat_layer(l_type, input_tensors, input_layer_names, usel_type=False):
    string = ""
    x = None
    if usel_type:
        string += "\n"+l_type+" = tf.keras.layers.Concatenate(axis=1)"+input_layer_names
    else:
        string += "\nx = tf.keras.layers.Concatenate(axis=1)"+input_layer_names
    x = tf.keras.layers.Concatenate(axis=1)(input_tensors)
    
    return string, x

def get_add_layer(l_type, input_tensors, input_layer_names, usel_type=False):
    string = ""
    x = None
    if usel_type:
        string += "\n"+l_type+" = tf.keras.layers.Add()"+input_layer_names
    else:
        string += "\nx = tf.keras.layers.Add()"+input_layer_names
    x = tf.keras.layers.Add()(input_tensors)
    
    return string, x

def get_multiply_layer(l_type, input_tensors, input_layer_names, usel_type=False):
    string = ""
    x = None
    if usel_type:
        string += "\n"+l_type+" = tf.keras.layers.Multiply()"+input_layer_names
    else:
        string += "\nx = tf.keras.layers.Multiply()"+input_layer_names
    x = tf.keras.layers.Multiply()(input_tensors)
    
    return string, x

def get_flatten_layer(l_type, input_tensors, input_layer_names, usel_type=False):
    string = ""
    x = None
    if usel_type:
        string += "\n"+l_type+" = tf.keras.layers.Flatten()"+input_layer_names
    else:
        string += "\nx = tf.keras.layers.Flatten()"+input_layer_names
    x = tf.keras.layers.Flatten()(input_tensors)
    
    return string, x

def keras_build_simple_sequential(layer_dict):
    
    code_string = ""
    prev_layer = None
    
    for l_type in layer_dict.keys():
        
        if l_type == 'input':
            shape = layer_dict[l_type]['shape']
            code_string += "inp = tf.keras.layers.Input(shape="+str(shape)+")"
            inp = tf.keras.layers.Input(shape=shape)
            prev_layer = '(inp)'
        
        else:
            
            n_layers = layer_dict[l_type]['num_layers']
            n_units = layer_dict[l_type]['num_units']
            
            for i in range(n_layers):
                
                if l_type in ['lstm', 'gru']:
                    if prev_layer == '(inp)':
                        string, x = get_rnn_layer(l_type, n_units[i], inp, prev_layer)
                    else:
                        string, x = get_rnn_layer(l_type, n_units[i], x, prev_layer)
                    code_string += string
                
                elif l_type == 'dense':
                    code_string += "\nx = tf.keras.layers.Dense(units="+str(n_units[i])+")"+prev_layer
                    
                    if prev_layer == '(inp)':
                        x = tf.keras.layers.Dense(units=n_units[i])(inp)
                    
                    else:
                        x = tf.keras.layers.Dense(units=n_units[i])(x)
                
                elif 'conv' in l_type:
                    if prev_layer == '(inp)':
                        string, x = get_cnn_layer(l_type, n_units[i], inp, prev_layer)
                    else:
                        string, x = get_cnn_layer(l_type, n_units[i], x, prev_layer)
                    code_string += string
                                            
                prev_layer = '(x)'
        
    code_string += "\nmodel = tf.keras.models.Model(inputs=[inp], outputs=[x], name='Base')"
    model = tf.keras.models.Model(inputs=[inp], outputs=[x], name='Base')
    
    return code_string, model

def keras_build_simple_parallel(layer_dict):
    n_branches = layer_dict['num_branches']
    branches = layer_dict['branches']
    units = layer_dict['layer_units']
    acts = layer_dict['layer_activations']

    ip = [x for x in units.keys() if 'input' in x]
    num_inputs = len(ip)
    op = [x for x in units.keys() if 'output' in x]
    num_outputs = len(op)
    
    code_string = ""
    prev_layer = None
    tensors = {}
    for i in range(num_inputs):
        var_name = "input_"+str(i+1)
        code_string += var_name+" = tf.keras.layers.Input(shape="+str(units[var_name])+")"
        tensors[var_name] = tf.keras.layers.Input(shape=units[var_name])

    for branch in branches.keys():
        l_type = branches[branch]['output'][0]
        try:
            n_units = units[l_type]
        except:
            n_units = None
        try:    
            activs = acts[l_type]
        except:
            activs = None

        if 'conv' in l_type:
            string, tensors[l_type] = get_cnn_layer(l_type, n_units, tensors[branches[branch]['inputs'][0]], "("+branches[branch]['inputs'][0]+")", True)
            code_string += string
        elif any(lyrs in l_type for lyrs in ['lstm','gru']):
            string, tensors[l_type] = get_rnn_layer(l_type, n_units[0], tensors[branches[branch]['inputs'][0]], "("+branches[branch]['inputs'][0]+")", True)
            code_string += string
        elif 'dense' in l_type or 'output' in l_type:
            string, tensors[l_type] = get_dense_layer(l_type, n_units[0], tensors[branches[branch]['inputs'][0]], "("+branches[branch]['inputs'][0]+")", activs, True)
            code_string += string
        elif 'concat' in l_type:
            string, tensors[l_type] = get_concat_layer(l_type, [tensors[k] for k in branches[branch]['inputs']], "("+str(branches[branch]['inputs'])+")", True)
            code_string += string
        elif 'add' in l_type:
            string, tensors[l_type] = get_add_layer(l_type, [tensors[k] for k in branches[branch]['inputs']], "("+str(branches[branch]['inputs'])+")", True)
            code_string += string
        elif 'multiply' in l_type:
            string, tensors[l_type] = get_multiply_layer(l_type, [tensors[k] for k in branches[branch]['inputs']], "("+str(branches[branch]['inputs'])+")", True)
            code_string += string
        elif 'flatten' in l_type:
            string, tensors[l_type] = get_flatten_layer(l_type, tensors[branches[branch]['inputs'][0]], "("+str(branches[branch]['inputs'])+")", True)
            code_string += string

    code_string += "\nmodel = tf.keras.models.Model(inputs="+str([tensors.get(key) for key in ip])+", outputs="+str([tensors.get(key) for key in op])+", name='Parallel_Base')"
    model = tf.keras.models.Model(inputs=[tensors.get(key) for key in ip], outputs=[tensors.get(key) for key in op], name="Parallel_Base")
    
    return code_string, tensors, model

if __name__ == '_main__':
    pass