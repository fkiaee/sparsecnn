#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import tensorflow as tf
import os
import sys
import argparse
# Add the ADMMutils module to the import path
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '../')))
import ADMMutils
import cifar_models
##==============================ADMM options==============================
np.random.seed(0);tf.set_random_seed(0);
# Get arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_id', default='0', type=int, help='The id number of the model to evaluate: i.e. 0 -->nin model')
parser.add_argument('--sparsity_function_id', default='0', type=int, help='The id number of the sparsity penalty function: i.e. 0 -->l0-norm function and 1 --> l1-norm function')
parser.add_argument('--data_path', default=os.path.realpath(os.path.join(os.path.dirname(__file__), '../')), help='Training data path')#'/media/disk/marc/cifar-10'
parser.add_argument('--ckpt_path_pretrained', default= os.path.realpath(os.path.join(os.path.dirname(__file__), '../pretrained')), help='path for loading pretrained network') 
parser.add_argument('--ckpt_path_ADMM', default= os.path.realpath(os.path.join(os.path.dirname(__file__), '../ADMM_results')), help=' path for saving ADMM ckpt output results') 
args = parser.parse_args()
model_id = args.model_id; sparsity_function_id = args.sparsity_function_id;
model_set = ['nin','nin_c3','nin_c3_lr']
data_path = args.data_path;
model_name = model_set[model_id]
ckpt_path_pretrained = os.path.join(args.ckpt_path_pretrained,model_name) # path for saving ADMM ckpt output results
ckpt_path_ADMM = os.path.join(args.ckpt_path_ADMM,model_name) # path for saving ADMM ckpt output results
if not os.path.exists(ckpt_path_ADMM):
    os.makedirs(ckpt_path_ADMM)
learning_rate = [0.001,0.001,0.001]; quiet = 0
st_p = 0.0001; en_p = 1; num_p = 20
mu_log_vals = np.logspace(np.log10(st_p), np.log10(en_p), num=num_p) # Set values of sparsity-promoting parameter mu
options = {'muval':mu_log_vals,'rho':100.0,'maxiter':10}
if sparsity_function_id == 0:
    options['method'] ='blkcard'
else:
    options['method'] ='blkl1' 
rho = options['rho']; muval = options['muval']
eps_abs = 1.e-4; eps_rel = 1.e-2 # absolute and relative tolerances for the stopping criterion of ADMM
##=========================== Get images and labels for CIFAR-10. ===================
filenames = [os.path.join(data_path, 'data_batch_%d.bin' % i) for i in range(1, 6)]
num_samples = 50000; num_total_samples = 50000; ckpt_write_period_max = 20000; batch_size = 128;    
filename_queue = tf.train.string_input_producer(filenames) #Create a queue that produces the filenames to read.
example, label = ADMMutils.ImageProducer(filename_queue);
min_fraction_of_examples_in_queue = 0.4; min_queue_examples = int(num_total_samples * min_fraction_of_examples_in_queue)
example_batch, label_batch = tf.train.shuffle_batch([example, label], batch_size=batch_size, num_threads=16, capacity=min_queue_examples + 3 * batch_size, min_after_dequeue=min_queue_examples)
label_batch = tf.reshape(label_batch, [batch_size])
label_batch = tf.cast(label_batch, tf.int64)
phase = 'train';
#============================== loading original model graph ========================
pred, weight_decay_sum, lr_w1_vars, lr_b1_vars, lr_w2_vars, lr_b2_vars, variables_dict, placeholders_dict, dictADMM, layer_list, layer_type, shape_records = cifar_models.model_load(model_id,example_batch,phase)
#===================== Constructing the part of graph corresponding to sparsity-promoting step =============================
mu_placeholder = tf.placeholder("float",[1])
assign_Gammas = []; assign_F = []; operation_dict = {};
resF = tf.Variable(0.0,trainable = False); resWF = tf.Variable(0.0,trainable = False)# primal and dual residuals
norm_W_total = tf.Variable(0.0,trainable = False); norm_F_total = tf.Variable(0.0,trainable = False);
norm_Gamma_total = tf.Variable(0.0,trainable = False);
for num_layer,id_layer in enumerate(layer_list):# dual variable, Gamma, and sparsity promoting variable, F, are updated manually and are not trained.
    tf.add_to_collection('non_trainable_variables',variables_dict['Gamma%s'%str(num_layer)])#
    tf.add_to_collection('non_trainable_variables',variables_dict['F%s'%str(num_layer)])
    V = tf.add(variables_dict['W_%s%s'%(layer_type[num_layer],str(id_layer))],tf.mul(variables_dict['Gamma%s'%str(num_layer)],1./rho))   
    if options['method'] == 'blkcard':
        if layer_type[num_layer] == 'c': 
            operation_dict['F_new%s'%str(num_layer)],operation_dict['zero_ind%s'%str(num_layer)] = ADMMutils.block_truncate_conv(V,mu_placeholder,rho)
        else:
            operation_dict['F_new%s'%str(num_layer)],operation_dict['zero_ind%s'%str(num_layer)] = ADMMutils.block_truncate_fc(V,mu_placeholder,rho)
    elif options['method'] == 'blkl1':
        if layer_type[num_layer] == 'c': 
            operation_dict['F_new%s'%str(num_layer)],operation_dict['zero_ind%s'%str(num_layer)] = ADMMutils.block_shrinkage_conv(V,mu_placeholder,rho)
        else:
            operation_dict['F_new%s'%str(num_layer)],operation_dict['zero_ind%s'%str(num_layer)] = ADMMutils.block_shrinkage_fc(V,mu_placeholder,rho)
    resF = tf.add(resF,ADMMutils.frobenius_norm(tf.sub(variables_dict['F%s'%str(num_layer)],operation_dict['F_new%s'%str(num_layer)])))
    resWF = tf.add(resWF,ADMMutils.frobenius_norm(tf.sub(variables_dict['W_%s%s'%(layer_type[num_layer],str(id_layer))],operation_dict['F_new%s'%str(num_layer)])))
    norm_W_total = tf.add(norm_W_total,ADMMutils.frobenius_norm(variables_dict['W_%s%s'%(layer_type[num_layer],str(id_layer))]))
    norm_F_total = tf.add(norm_F_total,ADMMutils.frobenius_norm(operation_dict['F_new%s'%str(num_layer)]))
    Gamma_assign = variables_dict['Gamma%s'%str(num_layer)].assign(tf.add(variables_dict['Gamma%s'%str(num_layer)] ,tf.mul(rho ,tf.sub(variables_dict['W_%s%s'%(layer_type[num_layer],str(id_layer))], operation_dict['F_new%s'%str(num_layer)]))))
    assign_Gammas.append(Gamma_assign.op)  
    norm_Gamma_total = tf.add(norm_Gamma_total,ADMMutils.frobenius_norm(variables_dict['Gamma%s'%str(num_layer)]))  
    F_assign = variables_dict['F%s'%str(num_layer)].assign(operation_dict['F_new%s'%str(num_layer)])
    assign_F.append(F_assign.op)
#==============================defining objectie function, accuracy and training operations ========================
cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(pred, label_batch)) 
penalize_term = ADMMutils.frobenius_norm_square(tf.sub(tf.sub(variables_dict['F0'],tf.mul(variables_dict['Gamma0'],1./rho)),variables_dict['W_c1']))
for num_layer in range(1,len(layer_list)):
    U_temp = tf.sub(variables_dict['F%s'%str(num_layer)],tf.mul(variables_dict['Gamma%s'%str(num_layer)],1./rho))
    penalize_term = tf.add(penalize_term,ADMMutils.frobenius_norm_square(tf.sub(U_temp,variables_dict['W_%s%s'%(layer_type[num_layer],str(layer_list[num_layer]))])))                
penalized_cross_entropy = tf.add(cross_entropy,tf.mul(rho,penalize_term)) 
predict_op = tf.argmax(pred, 1)
ispredcorrect = tf.equal(predict_op, label_batch)
accuracy = tf.reduce_mean(tf.cast(ispredcorrect, 'float'))
train_op_ADMM = tf.train.GradientDescentOptimizer(learning_rate[model_id]).minimize(penalized_cross_entropy) 
train_op_Finetuning = tf.train.GradientDescentOptimizer(learning_rate[model_id]).minimize(cross_entropy)
sess = tf.Session();
##================== Create a saver, start ADMM algorithm (from a pretrained net) or continue ADMM algorithm ===========
saver = tf.train.Saver(tf.get_collection('non_trainable_variables')+tf.trainable_variables(),max_to_keep=None);
init = tf.initialize_all_variables();sess.run(init); 
ckpt = tf.train.get_checkpoint_state(ckpt_path_ADMM) 
if ckpt and ckpt.model_checkpoint_path:    # If already saved output ckpt files exist, continue ADMM algorithm, else use pretrained ckpt files to initialize ADMM algorithm 
  if os.path.isabs(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
  else:
    saver.restore(sess, os.path.join(ckpt_path_ADMM,ckpt.model_checkpoint_path))      
  parameters = ckpt.model_checkpoint_path.split('/')[-1].split('-')
  mu_id = int(parameters[1]); ADMM_step_start = int(parameters[2])
  ADMM_iter= int(parameters[3]); Fine_tune_iter = int(parameters[4].split('.')[0]); 
  NUM_EPOCHS = min(mu_id+1,20); max_steps = int(np.floor(num_samples*NUM_EPOCHS/batch_size)); 
  if (Fine_tune_iter==max_steps and ADMM_iter==max_steps):
      mu_id += 1;  ADMM_step_start = 0; ADMM_iter = 0; Fine_tune_iter = 0 
  print('Succesfully loaded previous step of ADMM algorithm from %s at mu_step=%s ADMM_step=%s ADMM_iter=%s fine_tune_iter=%s.' %
        (ckpt.model_checkpoint_path, mu_id, ADMM_step_start, ADMM_iter, Fine_tune_iter))
else:    # Use the pretrained network as the initial weights for ADMM
    mu_id = 1; ADMM_step_start = 0; ADMM_iter = 0; Fine_tune_iter = 0 
    try:
        ckpt = tf.train.get_checkpoint_state(ckpt_path_pretrained) 
        saver.restore(sess, os.path.join(ckpt_path_pretrained,ckpt.model_checkpoint_path))
        print('Pretrained model succesfully loaded')
    except Exception as e:
        print("There is no such pretrained network: %s!"%(model_name))
##======== Initialize the solution path for different mu values ===============
solpath = {}; zero_ind_dict= {}
solpath['nnz']  = []; solpath['nz']  = []; solpath['mu']  = []
prod_val = 0; inputmaps = []; outputmaps = []
for num_layer,id_layer in enumerate(layer_list):
    prod_val += np.prod(shape_records[num_layer])        
    solpath['nz'].append([]);solpath['nnz'].append([]);
    if layer_type[num_layer]=='c':
        inputmaps.append(shape_records[num_layer][2]); outputmaps.append(shape_records[num_layer][3])
    else:
        outputmaps.append(shape_records[num_layer][1]); inputmaps.append(1)    
dictFinetuning = dictADMM.copy() # Define a separate finetuning dict for having a track of zero_map pleceholders in order to fix saprsity pattern  
##=========begin training======================================
threads = tf.train.start_queue_runners(sess) # Start the image processing workers
while(True):
    mu = muval[mu_id]#1.5 * mu
    print ('mu_id = %s and mu_value = %s\n'%(str(mu_id),str(mu)), end="")
    dictADMM[mu_placeholder] = [mu]
    ADMM_Max_Iter = min(mu_id,10)#options['maxiter']
    NUM_EPOCHS = min(mu_id,20)#250
    max_steps = int(np.floor(num_samples*NUM_EPOCHS/batch_size))
    ckpt_write_period = min(np.floor(max_steps/4),ckpt_write_period_max)
    #Solve the minimization problem using ADMM         
    for ADMMstep in range(ADMM_step_start,ADMM_Max_Iter): 
        if not(mu_id==1 and ADMMstep==0):          
            # ========================================================
            # Performancde promoting step
            # ========================================================
            correct = 0.; sum_entropy = 0.
            count = 0.; sum_penalize_term = 0.;
            for step in range(ADMM_iter,max_steps):                 
                sess.run(train_op_ADMM, feed_dict=dictADMM)
                #sum_entropy += sess.run(cross_entropy, feed_dict=dictADMM1)
                count += 1
                correct += np.sum(sess.run(accuracy,feed_dict=dictADMM))
                trainPerf = float(correct)/ (count); 
                if not(quiet):
                    print("acuracy at ADMM step %d is %2.5f."%(ADMMstep,trainPerf))
                if (step + 1) == max_steps:
                    checkpoint_path = os.path.join(ckpt_path_ADMM, "%s-%s-%s-%s-%s.ckpt"%(model_name,mu_id,ADMMstep,max_steps,str(0)))
                    saver.save(sess, checkpoint_path)
                elif step % ckpt_write_period == 0:
                    checkpoint_path = os.path.join(ckpt_path_ADMM, "%s-%s-%s-%s-%s.ckpt"%(model_name,mu_id,ADMMstep,step,str(0)))
                    saver.save(sess, checkpoint_path)
            ADMM_iter = 0
        # ========================================================
        # Sparse promoting step
        # ========================================================
        for num_layer,id_layer in enumerate(layer_list):
            zero_ind_dict[num_layer] = sess.run(operation_dict['zero_ind%s'%str(num_layer)],feed_dict=dictADMM)
        # =============================================================
        # stopping criterion for ADMM
        # =============================================================
        [resF_val,resWF_val,norm_F_total_val,norm_W_total_val] = sess.run([resF,resWF,norm_F_total,norm_W_total],feed_dict=dictADMM)
        sess.run(assign_Gammas,feed_dict=dictADMM); 
        sess.run(assign_F,feed_dict=dictADMM);
        norm_Gamma_total_val = sess.run(norm_Gamma_total,feed_dict=dictADMM)
        # evaluate the primal epsilon and the dual epsilon
        eps_pri  = np.sqrt(prod_val) * eps_abs + eps_rel * max(norm_W_total_val, norm_F_total_val)
        eps_dual = np.sqrt(prod_val) * eps_abs + eps_rel * norm_Gamma_total_val
        if  (resWF_val < eps_pri)  and  (rho*resF_val < eps_dual):
            print('resWF=%4.4f is lower than eps_pri=%4.4f,stopping criterion of ADMM is satisfied'%(resWF_val,eps_pri))
            break
    # stopping criterion is satisfied or Maximum number of ADMM steps reached!
    for num_layer,id_layer in enumerate(layer_list): 
        sess.run(variables_dict['W_%s%s'%(layer_type[num_layer],str(id_layer))].assign(variables_dict['F%s'%str(num_layer)]))## connect tensorflow and sparsity promoting part
    ADMM_step_start = 0
    # =============================================================
    # Update zero_map placeholders to fix sparsity patterns of weights during finetuning
    # =============================================================
    for num_layer,id_layer in enumerate(layer_list):
        solpath['nz'].append([]);solpath['nnz'].append([]);
        kernel_bunch_sh = shape_records[num_layer];          
        if layer_type[num_layer] == 'c':
            zero_map = np.ones((kernel_bunch_sh[2]*kernel_bunch_sh[3],kernel_bunch_sh[0]*kernel_bunch_sh[1]))
            zero_map[zero_ind_dict[num_layer],:] = 0
            zero_map= np.reshape(zero_map,(kernel_bunch_sh[2],kernel_bunch_sh[3],kernel_bunch_sh[0],kernel_bunch_sh[1]))
            zero_map = np.transpose(zero_map,(2,3,0,1))
        else:
            zero_map = np.ones(kernel_bunch_sh)
            zero_map[:,zero_ind_dict[num_layer]] = 0 
        dictFinetuning[placeholders_dict['zero_map%s'%str(num_layer)]] = zero_map
        solpath['nz'][num_layer].append(float(np.sum(zero_ind_dict[num_layer]))) #np.count_nonzero(bunch_Kernel) / ( blksize[0] * blksize[1] )
        solpath['nnz'][num_layer].append(inputmaps[num_layer]*outputmaps[num_layer]-float(np.sum(zero_ind_dict[num_layer])))
        print('zero-state of layer %d is %d out of %d conections'%(num_layer,float(np.sum(zero_ind_dict[num_layer])),inputmaps[num_layer]*outputmaps[num_layer]))   
    # ========================================================
    # Fine Tunning step
    # ========================================================
    correct = 0.; correct1 = 0.; sum_entropy = 0.;  count = 0.
    for fstep in range(Fine_tune_iter,max_steps):
        sess.run(train_op_Finetuning, feed_dict=dictFinetuning)
        count += 1
        correct += np.sum(sess.run(accuracy,feed_dict=dictFinetuning))
        trainPerf = float(correct)/ (count); 
        if not(quiet):
            print("fine-tuning accuracy at each step",trainPerf)
        if (fstep + 1) == max_steps:       # Save the model checkpoint periodically.
            checkpoint_path = os.path.join(ckpt_path_ADMM, "%s-%s-%s-%s-%s.ckpt"%(model_name,mu_id,ADMMstep,max_steps,max_steps))
            saver.save(sess, checkpoint_path)
        elif fstep % ckpt_write_period == 0:
            checkpoint_path = os.path.join(ckpt_path_ADMM, "%s-%s-%s-%s-%s.ckpt"%(model_name,mu_id,ADMMstep,max_steps,fstep+1))
            saver.save(sess, checkpoint_path)
    Fine_tune_iter = 0 
    solpath['mu'].append(mu)
    mu_id = mu_id + 1
