#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import tensorflow as tf
import cPickle as pickle
import cifar_models
import os
import sys
import argparse
import math
from fnmatch import fnmatch
# Add the ADMMutils module to the import path
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '../')))
import ADMMutils 
##=============================ADMM options===================
np.random.seed(0);tf.set_random_seed(0);
# Get arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_id', default='0', type=int, help='The id number of the model to evaluate: i.e. 0 -->nin model')
parser.add_argument('--data_path', default=os.path.realpath(os.path.join(os.path.dirname(__file__), '../')), help='Validation data path')#'/media/disk/marc/cifar-10'
parser.add_argument('--ckpt_path', default= 'None', help='Validation data path')# If default value is returned, my favorite path (dependant on the name of the model) is assigned later. 
#task=0: validate original non-sparse network (mu=0) 
#task=1: validate gradually pruned versions of the network (by increasing regularization mu at consecutive steps)  
parser.add_argument('--task', default='0', type=int, help='Select to validate pretrained model or ADMM-based sparse model: i.e. 0 -->pretrained model or 1 --> ADMM sparse CNN')
args = parser.parse_args()
task = args.task; task_set = ['pretrained','sparse_results']
model_id = args.model_id
model_set = ['nin','nin_c3','nin_c3_lr']; 
model_name = model_set[model_id]
print(model_name)
data_path = args.data_path;
if (args.ckpt_path is 'None'):
    if task == 0:
        ckpt_path = os.path.join(data_path,task_set[task],model_name);
    elif task == 1:
        ckpt_path = os.path.join(data_path,task_set[task],model_name);
else:
     ckpt_path = args.ckpt_path;  
st_p = 0.0001; en_p = 1; num_p = 20
mu_log_vals = np.logspace(np.log10(st_p), np.log10(en_p), num=num_p); 
muval = np.concatenate(([0],mu_log_vals),0)
ckp_write_period = 2000; batch_size = 128
filenames = [os.path.join(data_path, 'test_batch.bin')]
num_samples = 10000;
# Create a queue that produces the filenames to read.
filename_queue = tf.train.string_input_producer(filenames)
# Start the image processing workers
example, label = ADMMutils.ImageProducer(filename_queue);
min_fraction_of_examples_in_queue = 0.4; min_queue_examples = int(num_samples * min_fraction_of_examples_in_queue)
# Read 'batch_size' images + labels from the example queue.(without shuffling)
example_batch, label_batch = tf.train.batch([example, label], batch_size=batch_size)
label_batch = tf.reshape(label_batch, [batch_size])
# leatning_rate variable is just defined to have consistency with variables in training phase 
#NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000; NUM_EPOCHS_PER_DECAY = 350;
#num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / batch_size
#decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY); LEARNING_RATE_DECAY_FACTOR = 0.1 
#global_step = tf.Variable(0, trainable=False);starter_learning_rate = 0.1
#learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,decay_steps , LEARNING_RATE_DECAY_FACTOR , staircase=True)
if task==0:
    max_mu_iteration = 1  #task=0 for validating unpruned original network (mu_id = 0)
else:
    max_mu_iteration = 20# task=1 for validating gradually pruned versions of the network (by increasing regularization mu at consecutive steps)
newest = []
for number in range(max_mu_iteration):# get the last saved results for each mu value
    try:
        pattern = '*%s-%s-*.ckpt'%(model_name,str(number))
        files = [os.path.join(path,name) for path, subdirs, files in os.walk(ckpt_path) for name in files if fnmatch(name, pattern)]
        newest.append(max(files, key=os.path.getctime))
    except Exception as e:
        print(str(e))
        continue
model_checkpoint = [(ckpt_meta.split('.')[0]).split('/')[-1] for ckpt_meta in newest]
solpath = {}; solpath['model_ckpts'] = model_checkpoint
label_batch = tf.cast(label_batch, tf.int64)
mu_placeholder = tf.placeholder("float",[1])
# ==========================================================
phase = 'test';
pred, weight_decay_sum, lr_w1_vars, lr_b1_vars, lr_w2_vars, lr_b2_vars, variables_dict, placeholders_dict, dictTest, layer_list, layer_type, shape_records = cifar_models.model_load(model_id,example_batch,phase)
dictTest.update({mu_placeholder:[0]})
#=======================================================
dictTest3 = dictTest.copy(); 
for num_layer,id_layer in enumerate(layer_list): # dual variable, Gamma, and sparsity promoting variable, F, are updated manually and are not trained. 
    tf.add_to_collection('non_trainable_variables',variables_dict['Gamma%s'%str(num_layer)])
    tf.add_to_collection('non_trainable_variables',variables_dict['F%s'%str(num_layer)])
cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(pred, label_batch))
predict_op = tf.argmax(pred, 1)
probs =  tf.nn.softmax(pred)
ispredcorrect = tf.equal(predict_op, label_batch)
accuracy = tf.reduce_mean(tf.cast(ispredcorrect, 'float'))
zero_ind_dict= {};
solpath['nnz']  = [max_mu_iteration*[0] for _ in range(len(layer_list))]; solpath['nz']  =  [max_mu_iteration*[0] for _ in range(len(layer_list))];
solpath['mu']  = max_mu_iteration*[0]
solpath['ACC_test'] = max_mu_iteration*[0];  
prod_val = 0; inputmaps = []; outputmaps = []
for num_layer,id_layer in enumerate(layer_list):
    if layer_type[num_layer]=='c':
        inputmaps.append(shape_records[num_layer][2]); outputmaps.append(shape_records[num_layer][3])
    else:
        outputmaps.append(shape_records[num_layer][1]); inputmaps.append(1) 
for chp_num , model_chp in enumerate(model_checkpoint): 
    sess = tf.Session();
    init = tf.initialize_all_variables();sess.run(init);
    saver = tf.train.Saver(tf.get_collection('non_trainable_variables')+tf.trainable_variables(),max_to_keep=None);
    ckpt = tf.train.get_checkpoint_state(ckpt_path) 
    ckpt.model_checkpoint_path = os.path.join(ckpt_path,'%s.ckpt'%(model_checkpoint[chp_num]))  
    saver.restore(sess,ckpt.model_checkpoint_path )    
    parameters = ckpt.model_checkpoint_path.split('/')[-1].split('-')
    mu_id = int(parameters[1]) 
    print('Succesfully loaded model from %s.' %(model_checkpoint[chp_num]))
    ##=========begin training======================================
    print ('mu_id = %s\t'%(str(mu_id)), end="")
    mu = muval[mu_id];solpath['mu'][mu_id] = mu
    solpath['correct'] = 0.; #solpath['sum_entropy'] = 0;
    dictTest3[mu_placeholder] = [mu]; 
    for num_layer,id_layer in enumerate(layer_list):
        kernel_bunch = sess.run(variables_dict['F%s'%str(num_layer)]);
        if layer_type[num_layer] == 'c':
            kernel_bunch = np.transpose(kernel_bunch,(2,3,0,1))
            kernel_bunch = np.reshape(kernel_bunch,(shape_records[num_layer][2]*shape_records[num_layer][3],shape_records[num_layer][0]*shape_records[num_layer][1]))
            zero_ind_dict[num_layer] = ~np.any( kernel_bunch, axis=1)
        else:
            zero_ind_dict[num_layer] = ~np.any( kernel_bunch, axis=1)
    for num_layer,id_layer in enumerate(layer_list):
        kernel_bunch_sh = shape_records[num_layer];          
        if layer_type[num_layer] == 'c':
            zero_map = np.ones((kernel_bunch_sh[2]*kernel_bunch_sh[3],kernel_bunch_sh[0]*kernel_bunch_sh[1]))
            zero_map[zero_ind_dict[num_layer],:] = 0
            zero_map= np.reshape(zero_map,(kernel_bunch_sh[2],kernel_bunch_sh[3],kernel_bunch_sh[0],kernel_bunch_sh[1]))
            zero_map = np.transpose(zero_map,(2,3,0,1))
        else:
            zero_map = np.ones(kernel_bunch_sh)
            zero_map[zero_ind_dict[num_layer],:] = 0 
        dictTest3[placeholders_dict['zero_map%s'%str(num_layer)]] = zero_map                          
        solpath['nz'][num_layer][mu_id] = float(np.sum(zero_ind_dict[num_layer])) 
        solpath['nnz'][num_layer][mu_id] = (inputmaps[num_layer]*outputmaps[num_layer]-float(np.sum(zero_ind_dict[num_layer])))
        print('zero-state of layer %d is %d out of %d conections'%(num_layer,float(np.sum(zero_ind_dict[num_layer])),inputmaps[num_layer]*outputmaps[num_layer]))   
    # ========================================================
    coord = tf.train.Coordinator()
    try:
        threads = []
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
            threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
        max_steps = int(math.ceil(num_samples / batch_size))
        step = 0
        while step < max_steps: 
            solpath['start_point'] = step+1 
            solpath['correct'] += np.sum(sess.run(accuracy,feed_dict=dictTest3))
            testPerf = float(solpath['correct'])/(solpath['start_point']);   
            print("test acuracy at each step",testPerf)
            if ((step + 1) == max_steps or step % ckp_write_period == 0) and step != 0:
                solpath['ACC_test'][mu_id] = testPerf 
                filename = os.path.join(ckpt_path,'%s_%s_validate.pkl'%(model_name,task_set[task]))
                file1 = open(filename, 'wb')
                pickle.dump(solpath,file1)
                print('results sucessfully saved for mu',mu_id)
                file1.close() 
            step = step + 1
    except Exception as e:
        coord.request_stop(e)
    
    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)
    sess.close()
