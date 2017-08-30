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
    ## ====================Caffenet (modified Alexnet)===================
        layer_list = [1,2,3,4,5,1,2,3];layer_type = ['c','c','c','c','c','f','f','f']; layer_list1 = [1,2,3,4,5];
        # define conv and fully connected layers          
        shape_records_sparse = [(11,11,3,96),(5,5,48,256),(3,3,256,384),(3,3,192,384),(3,3,192,256),(9216,4096),(4096,4096),(4096,1000)]; 
        for num_layer,id_layer in enumerate(layer_list): 
            variables_dict['W_%s%s'%(layer_type[num_layer],str(id_layer))] = tf.Variable(tf.random_normal(list(shape_records_sparse[num_layer]), stddev=0.05))
            variables_dict['b_%s%s'%(layer_type[num_layer],str(id_layer))] = tf.Variable(tf.zeros([list(shape_records_sparse[num_layer])[-1]]))
            tf.add_to_collection('weight_decay', tf.nn.l2_loss(variables_dict['W_%s%s'%(layer_type[num_layer],str(id_layer))]))
            tf.add_to_collection('lr_w1', variables_dict['W_%s%s'%(layer_type[num_layer],str(id_layer))])                            
            tf.add_to_collection('lr_b1', variables_dict['b_%s%s'%(layer_type[num_layer],str(id_layer))])                               
        for num_layer,id_layer in enumerate(layer_list1): 
            placeholders_dict['zero_map%s'%str(num_layer)] = tf.placeholder("float", list(shape_records_sparse[num_layer]))
        drop_dict = {"dropout_rate_full":tf.placeholder("float")}
        dictTrain[drop_dict["dropout_rate_full"]] = 0.5;
        dictTest[drop_dict["dropout_rate_full"]] = 0.0;
        #conv1--> conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
        k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4#(11, 11, 3, 96)#(96,)
        forced_zero1 = tf.mul(variables_dict['W_c1'],placeholders_dict['zero_map0'])
        conv1_in = ADMMutils.conv(X, forced_zero1, variables_dict['b_c1'], k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
        conv1 = tf.nn.relu(conv1_in)
        #maxpool1--> max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
        maxpool1 = tf.nn.max_pool(conv1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
        #lrn1--> lrn(2, 2e-05, 0.75, name='norm1')
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        lrn1 = tf.nn.local_response_normalization(maxpool1,depth_radius=radius,alpha=alpha,beta=beta,bias=bias)
        #conv2--> conv(5, 5, 256, 1, 1, group=2, name='conv2')
        k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2#(5, 5, 48, 256)#(256,)
        forced_zero2 = tf.mul(variables_dict['W_c2'],placeholders_dict['zero_map1'])
        conv2_in = ADMMutils.conv(lrn1,forced_zero2,variables_dict['b_c2'], k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv2 = tf.nn.relu(conv2_in)
        #maxpool2--> max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
        maxpool2 = tf.nn.max_pool(conv2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
        #lrn2--> lrn(2, 2e-05, 0.75, name='norm2')
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        lrn2 = tf.nn.local_response_normalization(maxpool2,depth_radius=radius,alpha=alpha,beta=beta,bias=bias)
        #conv3--> conv(3, 3, 384, 1, 1, name='conv3')
        k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1#(3, 3, 256, 384)#(384,)
        forced_zero3 = tf.mul(variables_dict['W_c3'],placeholders_dict['zero_map2'])
        conv3_in = ADMMutils.conv(lrn2, forced_zero3,variables_dict['b_c3'], k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv3 = tf.nn.relu(conv3_in)
        #conv4--> conv(3, 3, 384, 1, 1, group=2, name='conv4')
        k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2#(3, 3, 192, 384)#(384,)
        forced_zero4 = tf.mul(variables_dict['W_c4'],placeholders_dict['zero_map3'])
        conv4_in = ADMMutils.conv(conv3, forced_zero4, variables_dict['b_c4'], k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv4 = tf.nn.relu(conv4_in)
        #conv5--> conv(3, 3, 256, 1, 1, group=2, name='conv5')
        k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2#(3, 3, 192, 256)#(256,)
        forced_zero5 = tf.mul(variables_dict['W_c5'],placeholders_dict['zero_map4'])
        conv5_in = ADMMutils.conv(conv4, forced_zero5, variables_dict['b_c5'], k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv5 = tf.nn.relu(conv5_in)
        #maxpool5--> max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
        maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
        #forced_zero6 = tf.mul(variables_dict['W_f1'],placeholders_dict['zero_map5'])
        fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(np.prod(maxpool5.get_shape()[1:]))]), variables_dict['W_f1'], variables_dict['b_f1'])
        fc6 = tf.nn.dropout(fc6, 1.-drop_dict["dropout_rate_full"])
        #fc7--> fc(4096, name='fc7')#(4096, 4096)#(4096,)
        #forced_zero7 = tf.mul(variables_dict['W_f2'],placeholders_dict['zero_map6'])
        fc7 = tf.nn.relu_layer(fc6, variables_dict['W_f2'], variables_dict['b_f2'])
        fc7 = tf.nn.dropout(fc7, 1.-drop_dict["dropout_rate_full"])
        #fc8--> fc(1000, relu=False, name='fc8')#(4096, 1000)#(1000,)
        #forced_zero8 = tf.mul(variables_dict['W_f3'],placeholders_dict['zero_map7'])
        pred = tf.nn.xw_plus_b(fc7, variables_dict['W_f3'], variables_dict['b_f3'])  

#        #fc6--> fc(4096, name='fc6')#(9216, 4096)#(4096,)
#        try:
#            forced_zero6 = tf.mul(variables_dict['w_f1'],placeholders_dict['zero_map5'])
#            fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(np.prod(maxpool5.get_shape()[1:]))]), forced_zero6, variables_dict['b_f1'])
#        except Exception as e:
#            fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(np.prod(maxpool5.get_shape()[1:]))]), variables_dict['w_f1'], variables_dict['b_f1'])
#            print(str(e))
#        fc6 = tf.nn.dropout(fc6, 1.-drop_dict["dropout_rate_full"])
#        #fc7--> fc(4096, name='fc7')#(4096, 4096)#(4096,)
#        try: 
#            forced_zero7 = tf.mul(variables_dict['w_f2'],placeholders_dict['zero_map6'])
#            fc7 = tf.nn.relu_layer(fc6, forced_zero7, variables_dict['b_f2'])
#        except Exception as e:           
#            fc7 = tf.nn.relu_layer(fc6, variables_dict['w_f2'] , variables_dict['b_f2'])
#            print(str(e))
#        fc7 = tf.nn.dropout(fc7, 1.-drop_dict["dropout_rate_full"])
#        #fc8--> fc(1000, relu=False, name='fc8')#(4096, 1000)#(1000,)
#        try: 
#            forced_zero8 = tf.mul(variables_dict['w_f3'],placeholders_dict['zero_map7'])
#            pred = tf.nn.xw_plus_b(fc7, forced_zero8, variables_dict['b_f3']) 
#        except Exception as e:           
#            pred = tf.nn.xw_plus_b(fc7, variables_dict['w_f3'], variables_dict['b_f3'])       
#            print(str(e))   
    for num_layer,id_layer in enumerate(layer_list1): 
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
        return pred, weight_decay_sum, lr_w1_vars, lr_b1_vars, lr_w2_vars, lr_b2_vars, variables_dict, placeholders_dict, dictTrain, layer_list1, layer_type, shape_records_sparse
    elif phase=='test':
        return pred, weight_decay_sum, lr_w1_vars, lr_b1_vars, lr_w2_vars, lr_b2_vars, variables_dict, placeholders_dict, dictTest, layer_list1, layer_type, shape_records_sparse