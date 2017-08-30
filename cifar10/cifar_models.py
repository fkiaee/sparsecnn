#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import os
import sys
# Add the ADMMutils module to the import path
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '../')))
import ADMMutils
def model_load(model_id,X,phase):
    variables_dict = {}; dictTrain = {}; dictTest = {}; placeholders_dict = {};  
    if model_id in [0]: 
    ## ====================nin===================
        layer_list = [1,2,3];layer_type = ['c','c','c']; 
        # define mlpconv layers which are not subjected to sparsity
        shape_records_non_sparse = [(1,1,192,160),(1,1,160,96),(1,1,192,192),(1,1,192,192),(1,1,192,192)];
        non_sparse_weights =  ['cccp1W','cccp2W','cccp3W','cccp4W','cccp5W'];
        non_sparse_biases =  ['cccp1b','cccp2b','cccp3b','cccp4b','cccp5b'];
        for num_layer in range(len(non_sparse_weights)): 
            variables_dict[non_sparse_weights[num_layer]] = tf.Variable(tf.random_normal(list(shape_records_non_sparse[num_layer]), stddev=0.05))
            variables_dict[non_sparse_biases[num_layer]] = tf.Variable(tf.zeros([list(shape_records_non_sparse[num_layer])[-1]]))
            tf.add_to_collection('weight_decay', tf.nn.l2_loss(variables_dict[non_sparse_weights[num_layer]]))   
            tf.add_to_collection('lr_w1', variables_dict[non_sparse_weights[num_layer]])                            
            tf.add_to_collection('lr_b1', variables_dict[non_sparse_weights[num_layer]])                                         
        variables_dict['cccp6W'] = tf.Variable(tf.random_normal([1,1,192,10], stddev=0.05))
        variables_dict['cccp6b'] = tf.Variable(tf.zeros([10]))
        tf.add_to_collection('weight_decay', tf.nn.l2_loss(variables_dict['cccp6W']))   
        tf.add_to_collection('lr_w2',  variables_dict['cccp6W'])                            
        tf.add_to_collection('lr_b2',  variables_dict['cccp6b'])  
        # define conv layers which are subjected to sparsity          
        shape_records_sparse = [(5,5,3,192),(5,5,96,192),(3,3,192,192)]; 
        for num_layer,id_layer in enumerate(layer_list): 
            variables_dict['W_%s%s'%(layer_type[num_layer],str(id_layer))] = tf.Variable(tf.random_normal(list(shape_records_sparse[num_layer]), stddev=0.05))
            variables_dict['b_%s%s'%(layer_type[num_layer],str(id_layer))] = tf.Variable(tf.zeros([list(shape_records_sparse[num_layer])[-1]]))
            tf.add_to_collection('weight_decay', tf.nn.l2_loss(variables_dict['W_%s%s'%(layer_type[num_layer],str(id_layer))]))
            tf.add_to_collection('lr_w1', variables_dict['W_%s%s'%(layer_type[num_layer],str(id_layer))])                            
            tf.add_to_collection('lr_b1', variables_dict['b_%s%s'%(layer_type[num_layer],str(id_layer))])                               
        for num_layer,id_layer in enumerate(layer_list): 
            placeholders_dict['zero_map%s'%str(num_layer)] = tf.placeholder("float", list(shape_records_sparse[num_layer]))
        drop_dict = {"dropout_rate_conv":tf.placeholder("float")}
        dictTrain[drop_dict["dropout_rate_conv"]] = 0.5; 
        dictTest[drop_dict["dropout_rate_conv"]] = 0.0; 
        # define the graph ('forced_zero' nodes are defined to fix sparsity pattern identified using ADMM in fine-tuning step )          
        #conv1 -> conv(5, 5, 192, 1, 1, name='conv1')
        k_h = 5; k_w = 5; c_o = 192; s_h = 1; s_w = 1
        forced_zero1 = tf.mul(variables_dict['W_c1'],placeholders_dict['zero_map0'])
        conv1_in = ADMMutils.conv(X, forced_zero1, variables_dict['b_c1'], k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
        conv1 = tf.nn.relu(conv1_in)
        #cccp1 -> conv(1, 1, 160, 1, 1, name='cccp1')
        k_h = 1; k_w = 1; c_o = 160; s_h = 1; s_w = 1
        cccp1_in = ADMMutils.conv(conv1, variables_dict['cccp1W'], variables_dict['cccp1b'], k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
        cccp1 = tf.nn.relu(cccp1_in)
        #cccp2 -> conv(1, 1, 96, 1, 1, name='cccp2')
        k_h = 1; k_w = 1; c_o = 96; s_h = 1; s_w = 1
        cccp2_in = ADMMutils.conv(cccp1, variables_dict['cccp2W'], variables_dict['cccp2b'], k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
        cccp2 = tf.nn.relu(cccp2_in)
        #maxpool1 -> max_pool(3, 3, 2, 2, name='pool1')
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'SAME'
        maxpool1 = tf.nn.max_pool(cccp2 , ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
        maxpool1 = tf.nn.dropout(maxpool1, 1.-drop_dict["dropout_rate_conv"])
        #conv2 -> conv(5, 5, 192, 1, 1, name='conv2')
        k_h = 5; k_w = 5; c_o = 192; s_h = 1; s_w = 1; group = 1
        forced_zero2 = tf.mul(variables_dict['W_c2'],placeholders_dict['zero_map1'])
        conv2_in = ADMMutils.conv(maxpool1, forced_zero2, variables_dict['b_c2'], k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv2 = tf.nn.relu(conv2_in)
        #cccp3 -> conv(1, 1, 192, 1, 1, name='cccp3')
        k_h = 1; k_w = 1; c_o = 192; s_h = 1; s_w = 1
        cccp3_in = ADMMutils.conv(conv2, variables_dict['cccp3W'], variables_dict['cccp3b'], k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
        cccp3 = tf.nn.relu(cccp3_in)
        #cccp4 -> conv(1, 1, 192, 1, 1, name='cccp4')
        k_h = 1; k_w = 1; c_o = 192; s_h = 1; s_w = 1
        cccp4_in = ADMMutils.conv(cccp3, variables_dict['cccp4W'], variables_dict['cccp4b'], k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
        cccp4 = tf.nn.relu(cccp4_in)
        #avgpool2 -> avg_pool(3, 3, 2, 2, name='pool2')                                        
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'SAME'
        maxpool2 = tf.nn.avg_pool(cccp4, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
        maxpool2 = tf.nn.dropout(maxpool2, 1.-drop_dict["dropout_rate_conv"])
        #conv3 -> conv(3, 3, 192, 1, 1, name='conv3')
        k_h = 3; k_w = 3; c_o = 192; s_h = 1; s_w = 1; group = 1
        forced_zero3 = tf.mul(variables_dict['W_c3'],placeholders_dict['zero_map2'])
        conv3_in = ADMMutils.conv(maxpool2, forced_zero3, variables_dict['b_c3'], k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv3 = tf.nn.relu(conv3_in)
        #cccp5 -> conv(1, 1, 192, 1, 1, name='cccp5')
        k_h = 1; k_w = 1; c_o = 192; s_h = 1; s_w = 1
        cccp5_in = ADMMutils.conv(conv3, variables_dict['cccp5W'], variables_dict['cccp5b'], k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
        cccp5 = tf.nn.relu(cccp5_in)
        #cccp6 -> conv(1, 1, 10, 1, 1, name='cccp6')
        k_h = 1; k_w = 1; c_o = 10; s_h = 1; s_w = 1
        cccp6_in = ADMMutils.conv(cccp5, variables_dict['cccp6W'], variables_dict['cccp6b'], k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
        cccp6 = tf.nn.relu(cccp6_in)
        #avgpool3 -> avg_pool(8, 8, 1, 1, padding='VALID', name='pool3')
        k_h = 8; k_w = 8; s_h = 1; s_w = 1; padding = 'VALID'
        maxpool5 = tf.nn.avg_pool(cccp6, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
        pred = tf.squeeze(maxpool5)        
    if model_id in [1]: 
    ## ====================nin_c3===================
        layer_list = [1,2,3,4];layer_type = ['c','c','c','c']; 
        # define mlpconv layers which are not subjected to sparsity
        shape_records_non_sparse = [(1,1,192,160),(1,1,160,96),(1,1,192,192),(1,1,192,192),(1,1,192,192)];
        non_sparse_weights =  ['cccp1W','cccp2W','cccp3W','cccp4W','cccp5W'];
        non_sparse_biases =  ['cccp1b','cccp2b','cccp3b','cccp4b','cccp5b'];
        for num_layer in range(len(non_sparse_weights)): 
            variables_dict[non_sparse_weights[num_layer]] = tf.Variable(tf.random_normal(list(shape_records_non_sparse[num_layer]), stddev=0.05))
            variables_dict[non_sparse_biases[num_layer]] = tf.Variable(tf.zeros([list(shape_records_non_sparse[num_layer])[-1]]))
            tf.add_to_collection('weight_decay', tf.nn.l2_loss(variables_dict[non_sparse_weights[num_layer]]))   
            tf.add_to_collection('lr_w1', variables_dict[non_sparse_weights[num_layer]])                            
            tf.add_to_collection('lr_b1', variables_dict[non_sparse_weights[num_layer]])                                         
        variables_dict['cccp6W'] = tf.Variable(tf.random_normal([1,1,192,10], stddev=0.05))
        variables_dict['cccp6b'] = tf.Variable(tf.zeros([10]))
        tf.add_to_collection('weight_decay', tf.nn.l2_loss(variables_dict['cccp6W']))   
        tf.add_to_collection('lr_w2',  variables_dict['cccp6W'])                            
        tf.add_to_collection('lr_b2',  variables_dict['cccp6b'])  
        # define conv layers which are subjected to sparsity          
        shape_records_sparse = [(3,3,3,192),(3,3,96,192),(3,1,192,192),(3,3,192,192)]; 
        for num_layer,id_layer in enumerate(layer_list): 
            variables_dict['W_%s%s'%(layer_type[num_layer],str(id_layer))] = tf.Variable(tf.random_normal(list(shape_records_sparse[num_layer]), stddev=0.05))
            variables_dict['b_%s%s'%(layer_type[num_layer],str(id_layer))] = tf.Variable(tf.zeros([list(shape_records_sparse[num_layer])[-1]]))
            tf.add_to_collection('weight_decay', tf.nn.l2_loss(variables_dict['W_%s%s'%(layer_type[num_layer],str(id_layer))]))
            tf.add_to_collection('lr_w1', variables_dict['W_%s%s'%(layer_type[num_layer],str(id_layer))])                            
            tf.add_to_collection('lr_b1', variables_dict['b_%s%s'%(layer_type[num_layer],str(id_layer))])                               
        for num_layer,id_layer in enumerate(layer_list): 
            placeholders_dict['zero_map%s'%str(num_layer)] = tf.placeholder("float", list(shape_records_sparse[num_layer]))
        drop_dict = {"dropout_rate_conv":tf.placeholder("float")}
        dictTrain[drop_dict["dropout_rate_conv"]] = 0.5; 
        dictTest[drop_dict["dropout_rate_conv"]] = 0.0; 
        # define the graph ('forced_zero' nodes are defined to fix sparsity pattern identified using ADMM in fine-tuning step )          
        #conv1 -> conv(5, 5, 192, 1, 1, name='conv1')
        k_h = 3; k_w = 3; c_o = 192; s_h = 1; s_w = 1
        forced_zero1 = tf.mul(variables_dict['W_c1'],placeholders_dict['zero_map0'])
        conv1_in = ADMMutils.conv(X, forced_zero1, variables_dict['b_c1'], k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
        conv1 = tf.nn.relu(conv1_in)
        #cccp1 -> conv(1, 1, 160, 1, 1, name='cccp1')
        k_h = 1; k_w = 1; c_o = 160; s_h = 1; s_w = 1
        cccp1_in = ADMMutils.conv(conv1, variables_dict['cccp1W'], variables_dict['cccp1b'], k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
        cccp1 = tf.nn.relu(cccp1_in)
        #cccp2 -> conv(1, 1, 96, 1, 1, name='cccp2')
        k_h = 1; k_w = 1; c_o = 96; s_h = 1; s_w = 1
        cccp2_in = ADMMutils.conv(cccp1, variables_dict['cccp2W'], variables_dict['cccp2b'], k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
        cccp2 = tf.nn.relu(cccp2_in)
        #maxpool1 -> max_pool(3, 3, 2, 2, name='pool1')
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'SAME'
        maxpool1 = tf.nn.max_pool(cccp2 , ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
        maxpool1 = tf.nn.dropout(maxpool1, 1.-drop_dict["dropout_rate_conv"])
        #conv2 -> conv(3, 3, 192, 1, 1, name='conv2_1')
        k_h = 3; k_w = 3; c_o = 192; s_h = 1; s_w = 1; group = 1
        forced_zero2 = tf.mul(variables_dict['W_c2'],placeholders_dict['zero_map1'])
        conv2_in = ADMMutils.conv(maxpool1, forced_zero2, variables_dict['b_c2'], k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv2 = tf.nn.relu(conv2_in)
        #conv3 -> conv(3,1, 192, 1, 1, name='conv2_2')
        k_h = 3; k_w = 1; c_o = 192; s_h = 1; s_w = 1; group = 1
        forced_zero3 = tf.mul(variables_dict['W_c3'],placeholders_dict['zero_map2'])
        conv3_in = ADMMutils.conv(conv2, forced_zero3, variables_dict['b_c3'], k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv3 = tf.nn.relu(conv3_in)
        #cccp3 -> conv(1, 1, 192, 1, 1, name='cccp3')
        k_h = 1; k_w = 1; c_o = 192; s_h = 1; s_w = 1
        cccp3_in = ADMMutils.conv(conv3, variables_dict['cccp3W'], variables_dict['cccp3b'], k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
        cccp3 = tf.nn.relu(cccp3_in)
        #cccp4 -> conv(1, 1, 192, 1, 1, name='cccp4')
        k_h = 1; k_w = 1; c_o = 192; s_h = 1; s_w = 1
        cccp4_in = ADMMutils.conv(cccp3, variables_dict['cccp4W'], variables_dict['cccp4b'], k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
        cccp4 = tf.nn.relu(cccp4_in)
        #avgpool2 -> avg_pool(3, 3, 2, 2, name='pool2')                                        
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'SAME'
        maxpool2 = tf.nn.avg_pool(cccp4, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
        maxpool2 = tf.nn.dropout(maxpool2, 1.-drop_dict["dropout_rate_conv"])
        #conv4 -> conv(3, 3, 192, 1, 1, name='conv3')
        k_h = 3; k_w = 3; c_o = 192; s_h = 1; s_w = 1; group = 1
        forced_zero4 = tf.mul(variables_dict['W_c4'],placeholders_dict['zero_map3'])
        conv4_in = ADMMutils.conv(maxpool2, forced_zero4, variables_dict['b_c4'], k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv4 = tf.nn.relu(conv4_in)
        #cccp5 -> conv(1, 1, 192, 1, 1, name='cccp5')
        k_h = 1; k_w = 1; c_o = 192; s_h = 1; s_w = 1
        cccp5_in = ADMMutils.conv(conv4, variables_dict['cccp5W'], variables_dict['cccp5b'], k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
        cccp5 = tf.nn.relu(cccp5_in)
        #cccp6 -> conv(1, 1, 10, 1, 1, name='cccp6')
        k_h = 1; k_w = 1; c_o = 10; s_h = 1; s_w = 1
        cccp6_in = ADMMutils.conv(cccp5, variables_dict['cccp6W'], variables_dict['cccp6b'], k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
        cccp6 = tf.nn.relu(cccp6_in)
        #avgpool3 -> avg_pool(8, 8, 1, 1, padding='VALID', name='pool3')
        k_h = 8; k_w = 8; s_h = 1; s_w = 1; padding = 'VALID'
        maxpool5 = tf.nn.avg_pool(cccp6, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
        pred = tf.squeeze(maxpool5)   
    if model_id in [2]: 
    ## ====================nin_c3_lr===================
        layer_list = [1,2,3,4,5,6,7,8];layer_type = ['c','c','c','c','c','c','c','c']; 
        # define mlpconv layers which are not subjected to sparsity
        shape_records_non_sparse = [(1,1,192,160),(1,1,160,96),(1,1,192,192),(1,1,192,192),(1,1,192,192)];
        non_sparse_weights =  ['cccp1W','cccp2W','cccp3W','cccp4W','cccp5W'];
        non_sparse_biases =  ['cccp1b','cccp2b','cccp3b','cccp4b','cccp5b'];
        for num_layer in range(len(non_sparse_weights)): 
            variables_dict[non_sparse_weights[num_layer]] = tf.Variable(tf.random_normal(list(shape_records_non_sparse[num_layer]), stddev=0.05))
            variables_dict[non_sparse_biases[num_layer]] = tf.Variable(tf.zeros([list(shape_records_non_sparse[num_layer])[-1]]))
            tf.add_to_collection('weight_decay', tf.nn.l2_loss(variables_dict[non_sparse_weights[num_layer]]))   
            tf.add_to_collection('lr_w1', variables_dict[non_sparse_weights[num_layer]])                            
            tf.add_to_collection('lr_b1', variables_dict[non_sparse_weights[num_layer]])                                         
        variables_dict['cccp6W'] = tf.Variable(tf.random_normal([1,1,192,10], stddev=0.05))
        variables_dict['cccp6b'] = tf.Variable(tf.zeros([10]))
        tf.add_to_collection('weight_decay', tf.nn.l2_loss(variables_dict['cccp6W']))   
        tf.add_to_collection('lr_w2',  variables_dict['cccp6W'])                            
        tf.add_to_collection('lr_b2',  variables_dict['cccp6b'])  
        # define conv layers which are subjected to sparsity    
        shape_records_sparse = [(1,3,3,96),(3,1,3,96),(1,3,96,96),(3,1,96,96),(1,3,192,96),(3,1,192,96),(1,3,192,96),(3,1,192,96)]; 
        for num_layer,id_layer in enumerate(layer_list): 
            variables_dict['W_%s%s'%(layer_type[num_layer],str(id_layer))] = tf.Variable(tf.random_normal(list(shape_records_sparse[num_layer]), stddev=0.05))
            variables_dict['b_%s%s'%(layer_type[num_layer],str(id_layer))] = tf.Variable(tf.zeros([list(shape_records_sparse[num_layer])[-1]]))
            tf.add_to_collection('weight_decay', tf.nn.l2_loss(variables_dict['W_%s%s'%(layer_type[num_layer],str(id_layer))]))
            tf.add_to_collection('lr_w1', variables_dict['W_%s%s'%(layer_type[num_layer],str(id_layer))])                            
            tf.add_to_collection('lr_b1', variables_dict['b_%s%s'%(layer_type[num_layer],str(id_layer))])                               
        for num_layer,id_layer in enumerate(layer_list): 
            placeholders_dict['zero_map%s'%str(num_layer)] = tf.placeholder("float", list(shape_records_sparse[num_layer]))
        drop_dict = {"dropout_rate_conv":tf.placeholder("float")}
        dictTrain[drop_dict["dropout_rate_conv"]] = 0.5; 
        dictTest[drop_dict["dropout_rate_conv"]] = 0.0; 
        # define the graph ('forced_zero' nodes are defined to fix sparsity pattern identified using ADMM in fine-tuning step )          
        #conv1x -> conv(1, 3, 96, 1, 1, relu=False, name='conv1x')
        k_h = 1; k_w = 3; c_o = 96; s_h = 1; s_w = 1
        forced_zero1 = tf.mul(variables_dict['W_c1'],placeholders_dict['zero_map0'])
        conv1_in = ADMMutils.conv(X, forced_zero1, variables_dict['b_c1'], k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
        #conv1y -> conv(3, 1, 96, 1, 1, relu=False, name='conv1y')
        k_h = 3; k_w = 1; c_o = 96; s_h = 1; s_w = 1; group = 1
        forced_zero2 = tf.mul(variables_dict['W_c2'],placeholders_dict['zero_map1'])
        conv2_in = ADMMutils.conv(X, forced_zero2, variables_dict['b_c2'], k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv_cat1 = tf.concat(3, values=[conv1_in,conv2_in])
        conv_cat1 = tf.nn.relu(conv_cat1)
        #cccp1 -> conv(1, 1, 160, 1, 1, name='cccp1')
        k_h = 1; k_w = 1; c_o = 160; s_h = 1; s_w = 1
        cccp1_in = ADMMutils.conv(conv_cat1, variables_dict['cccp1W'], variables_dict['cccp1b'], k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
        cccp1 = tf.nn.relu(cccp1_in)
        #cccp2 -> conv(1, 1, 96, 1, 1, name='cccp2')
        k_h = 1; k_w = 1; c_o = 96; s_h = 1; s_w = 1
        cccp2_in = ADMMutils.conv(cccp1, variables_dict['cccp2W'], variables_dict['cccp2b'], k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
        cccp2 = tf.nn.relu(cccp2_in)
        #maxpool1 -> max_pool(3, 3, 2, 2, name='pool1')
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'SAME'
        maxpool1 = tf.nn.max_pool(cccp2 , ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
        maxpool1 = tf.nn.dropout(maxpool1, 1.-drop_dict["dropout_rate_conv"])
        #conv3 -> conv(3, 3, 192, 1, 1, name='conv2_1')
        k_h = 1; k_w = 3; c_o = 96; s_h = 1; s_w = 1; group = 1
        forced_zero3 = tf.mul(variables_dict['W_c3'],placeholders_dict['zero_map2'])
        conv3_in = ADMMutils.conv(maxpool1, forced_zero3, variables_dict['b_c3'], k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        #conv4 -> conv(3,1, 192, 1, 1, name='conv2_2')
        k_h = 3; k_w = 1; c_o = 96; s_h = 1; s_w = 1; group = 1
        forced_zero4 = tf.mul(variables_dict['W_c4'],placeholders_dict['zero_map3'])
        conv4_in = ADMMutils.conv(maxpool1, forced_zero4, variables_dict['b_c4'], k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv_cat2 = tf.concat(3, values=[conv3_in,conv4_in])
        conv_cat2 = tf.nn.relu(conv_cat2)
        #conv5 -> conv(1, 3, 96, 1, 1, name='conv3')
        k_h = 1; k_w = 3; c_o = 96; s_h = 1; s_w = 1; group = 1
        forced_zero5 = tf.mul(variables_dict['W_c5'],placeholders_dict['zero_map4'])
        conv5_in = ADMMutils.conv(conv_cat2, forced_zero5, variables_dict['b_c5'], k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        #conv6 -> conv(3, 1, 96, 1, 1, name='conv3')
        k_h = 3; k_w = 1; c_o = 96; s_h = 1; s_w = 1; group = 1
        forced_zero6 = tf.mul(variables_dict['W_c6'],placeholders_dict['zero_map5'])
        conv6_in = ADMMutils.conv(conv_cat2, forced_zero6, variables_dict['b_c6'], k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv_cat3 = tf.concat(3, values=[conv5_in,conv6_in])
        conv_cat3 = tf.nn.relu(conv_cat3)
        #cccp3 -> conv(1, 1, 192, 1, 1, name='cccp3')
        k_h = 1; k_w = 1; c_o = 192; s_h = 1; s_w = 1
        cccp3_in = ADMMutils.conv(conv_cat3, variables_dict['cccp3W'], variables_dict['cccp3b'], k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
        cccp3 = tf.nn.relu(cccp3_in)
        #cccp4 -> conv(1, 1, 192, 1, 1, name='cccp4')
        k_h = 1; k_w = 1; c_o = 192; s_h = 1; s_w = 1
        cccp4_in = ADMMutils.conv(cccp3, variables_dict['cccp4W'], variables_dict['cccp4b'], k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
        cccp4 = tf.nn.relu(cccp4_in)
        #avgpool2 -> avg_pool(3, 3, 2, 2, name='pool2')                                        
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'SAME'
        maxpool2 = tf.nn.avg_pool(cccp4, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
        maxpool2 = tf.nn.dropout(maxpool2, 1.-drop_dict["dropout_rate_conv"])
        #conv5 -> conv(1, 3, 96, 1, 1, name='conv3')
        k_h = 1; k_w = 3; c_o = 96; s_h = 1; s_w = 1; group = 1
        forced_zero7 = tf.mul(variables_dict['W_c7'],placeholders_dict['zero_map6'])
        conv7_in = ADMMutils.conv(maxpool2, forced_zero7, variables_dict['b_c7'], k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        #conv6 -> conv(3, 1, 96, 1, 1, name='conv3')
        k_h = 3; k_w = 1; c_o = 96; s_h = 1; s_w = 1; group = 1
        forced_zero8 = tf.mul(variables_dict['W_c8'],placeholders_dict['zero_map7'])
        conv8_in = ADMMutils.conv(maxpool2, forced_zero8, variables_dict['b_c8'], k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv_cat4 = tf.concat(3, values=[conv7_in,conv8_in])
        conv_cat4 = tf.nn.relu(conv_cat4)
        #cccp5 -> conv(1, 1, 192, 1, 1, name='cccp5')
        k_h = 1; k_w = 1; c_o = 192; s_h = 1; s_w = 1
        cccp5_in = ADMMutils.conv(conv_cat4, variables_dict['cccp5W'], variables_dict['cccp5b'], k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
        cccp5 = tf.nn.relu(cccp5_in)
        #cccp6 -> conv(1, 1, 10, 1, 1, name='cccp6')
        k_h = 1; k_w = 1; c_o = 10; s_h = 1; s_w = 1
        cccp6_in = ADMMutils.conv(cccp5, variables_dict['cccp6W'], variables_dict['cccp6b'], k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
        cccp6 = tf.nn.relu(cccp6_in)
        #avgpool3 -> avg_pool(8, 8, 1, 1, padding='VALID', name='pool3')
        k_h = 8; k_w = 8; s_h = 1; s_w = 1; padding = 'VALID'
        maxpool5 = tf.nn.avg_pool(cccp6, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
        pred = tf.squeeze(maxpool5)            
    for num_layer,id_layer in enumerate(layer_list): 
        #Gamma is the dual variable (i.e., the Lagrange multiplier)
        variables_dict['Gamma%s'%str(num_layer)] = tf.Variable(tf.zeros(list(shape_records_sparse[num_layer])),trainable = False)
        #F is an additional variable to introduce an additional constraint W - F = 0 giving rise to decoupling the objective function
        variables_dict['F%s'%str(num_layer)] = tf.Variable(variables_dict['W_%s%s'%(layer_type[num_layer],str(id_layer))].initialized_value(), trainable = False)
        # sparsity pattern identified using ADMM
        dictTrain[placeholders_dict['zero_map%s'%str(num_layer)]] = np.ones(tuple(shape_records_sparse[num_layer]))# at first round 
    weight_decay_sum = tf.add_n(tf.get_collection('weight_decay'))
    lr_w1_vars = tf.get_collection('lr_w1')
    lr_b1_vars = tf.get_collection('lr_b1')
    lr_w2_vars = tf.get_collection('lr_w2')
    lr_b2_vars = tf.get_collection('lr_b2')
    if phase=='train' or phase=='scratch':
        return pred, weight_decay_sum, lr_w1_vars, lr_b1_vars, lr_w2_vars, lr_b2_vars, variables_dict, placeholders_dict, dictTrain, layer_list, layer_type, shape_records_sparse
    elif phase=='test':
        return pred, weight_decay_sum, lr_w1_vars, lr_b1_vars, lr_w2_vars, lr_b2_vars, variables_dict, placeholders_dict, dictTest, layer_list, layer_type, shape_records_sparse