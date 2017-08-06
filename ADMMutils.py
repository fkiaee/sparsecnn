#!/usr/bin/env python
from __future__ import print_function
import tensorflow as tf
import cPickle as pickle
import os
import numpy as np
from scipy.misc import imread
from scipy.ndimage import zoom 

def frobenius_norm_square(tensor):
    squareroot_tensor = tf.square(tensor)
    frobenius_norm2 = tf.reduce_sum(squareroot_tensor)
    return frobenius_norm2
    
def frobenius_norm(tensor):
    frobenius_norm = tf.sqrt(frobenius_norm_square(tensor))
    return frobenius_norm

def frobenius_norm_block(tensor,dim):
    squareroot_tensor = tf.square(tensor)
    tensor_sum = tf.reduce_sum(squareroot_tensor,dim)
    frobenius_norm = tf.sqrt(tensor_sum)
    return frobenius_norm
    
def ImageProducer(filename_queue): 
    filename = os.path.realpath(os.path.join(os.path.dirname(__file__), 'cifar10/zca.pkl'))
    file = open(filename, 'rb')
    data = pickle.load(file)
    file.close() 
    Wzca= tf.constant(data['zca'],tf.float32)
    label_bytes = 1; 
    height = 32; width = 32; depth = 3
    image_bytes = height * width * depth
    record_bytes = label_bytes + image_bytes
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    key, value = reader.read(filename_queue)
    record_bytes = tf.decode_raw(value, tf.uint8)
    label_byte_slices = tf.slice(record_bytes, [0], [label_bytes]);
    label = tf.cast(label_byte_slices, tf.int32)
    image = tf.slice(record_bytes, [label_bytes], [image_bytes])#tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),[depth,height,width])
    image = tf.cast(image, tf.float32)   
    image = tf.reshape(image,[1,image_bytes])  
    image = tf.sub(image,tf.reduce_mean(image))
    scale = tf.constant(55.); thresh = tf.constant(1.)
    std_val  = tf.div(tf.sqrt(tf.reduce_sum(tf.square(image))),scale); 
    f4 = lambda: std_val
    f5 = lambda: thresh
    normalizer = tf.cond(tf.less(std_val,1e-8),f5,f4)
    image = tf.div(image,normalizer)
    image = tf.sub(image,tf.reduce_mean(image))
    img_RGB = tf.matmul(image,Wzca)
    depth_major = tf.reshape(img_RGB,[depth,height,width])
    image = tf.transpose(depth_major, [1, 2, 0])  
    return image, label    
def read_image(path):
    img = imread(path,mode="RGB")
    h, w, c = np.shape(img)
    scale_size = 256
    crop_size = 224
    assert c == 3
    img = zoom(img, (scale_size/h, scale_size/w,1))
    img = img.astype(np.float32)
    img -= np.array([104., 117., 124.])
    h, w, c = img.shape
    ho, wo = ((h - crop_size) / 2, (w - crop_size) / 2)
    img = img[ho:ho + crop_size, wo:wo + crop_size, :]
    #print(np.shape(img))
    img = img[None, ...]
    return img      
def process_image(img,isotropic):
    '''Crops, scales, and normalizes the given image.
    scale : The image wil be first scaled to this size.
    crop  : After scaling, a central crop of this size is taken.
    mean  : Subtracted from the image    '''
    scale = 256
    crop = 224
    mean = [124., 117., 104.] #RGB;[104., 117., 124.] #BGR
    # Rescale
    if isotropic:
        img_shape = tf.to_float(tf.shape(img)[:2])
        min_length = tf.minimum(img_shape[0], img_shape[1])
        new_shape = tf.to_int32((scale / min_length) * img_shape)
    else:
        new_shape = tf.pack([scale, scale])
    img = tf.image.resize_images(img, new_shape[0], new_shape[1])
    offset = (new_shape - crop) / 2
    img = tf.slice(img, begin=tf.pack([offset[0], offset[1], 0]), size=tf.pack([crop, crop, -1]))
    # Mean subtraction
    return tf.to_float(img) - mean    
def ImageProducer_imagenet(filename_queue,isotropic):
    line_reader = tf.TextLineReader()
    key, line = line_reader.read(filename_queue)
     # line_batch or line (depending if you want to batch)
    filename, label = tf.decode_csv(line,record_defaults=[tf.constant([],dtype=tf.string),tf.constant([],dtype=tf.int32)],field_delim=' ')
    file_contents = tf.read_file(filename)
    example = tf.image.decode_jpeg(file_contents)
    processed_img = process_image(example,isotropic)
    # Convert from RGB channel ordering to BGR This matches, for instance, how OpenCV orders the channels.
    processed_img = tf.reverse(processed_img, [False, False, True])  
    #processed_img.set_shape([224, 224, 3])
    return processed_img, label
       
def conv(inputx, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding) 
    if group==1:
        conv = convolve(inputx, kernel)
    else:
        input_groups = tf.split(3, group, inputx)
        kernel_groups = tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())    
    #return  tf.nn.bias_add(conv, biases) 
 #####################################################   
def block_shrinkage_conv(V,mu,rho):
    coef = 0.5
    V_shape = tf.shape(V); one_val = tf.constant(1.0) 
    b = tf.div(mu,rho)
    V_shape1 = tf.concat(0,[tf.mul(tf.slice(V_shape,[2],[1]),tf.slice(V_shape,[3],[1])),tf.mul(tf.slice(V_shape,[0],[1]),tf.slice(V_shape,[1],[1]))])
    V = tf.reshape(tf.transpose(V,perm=[2,3,0,1]),V_shape1)
    norm_V = frobenius_norm_block(V,1)  
    norm_V_per_dimension = tf.div(norm_V,tf.cast(tf.slice(V_shape1,[1],[1]),'float'))
    zero_part = tf.zeros(V_shape1)
    zero_ind = tf.greater_equal(b,norm_V_per_dimension)
    num_zero = tf.reduce_sum(tf.cast(zero_ind,'float'))
#    f4 = lambda: tf.greater_equal(tf.truediv(tf.add(tf.reduce_min(fro),tf.reduce_mean(fro)),2.0),fro)
    f4 = lambda: tf.greater_equal(tf.reduce_mean(norm_V),norm_V)
    f5 = lambda: zero_ind
    zero_ind = tf.cond(tf.greater(num_zero,tf.mul(coef,tf.cast(V_shape1[0],'float'))),f4,f5)
    G = tf.select(zero_ind,zero_part,tf.mul(tf.sub(one_val,tf.div(b,tf.reshape(norm_V,[-1,1]))),V)) 
    G_shape = tf.concat(0,[tf.slice(V_shape,[2],[1]),tf.slice(V_shape,[3],[1]),tf.slice(V_shape,[0],[1]),tf.slice(V_shape,[1],[1])])
    G = tf.transpose(tf.reshape(G,G_shape),perm=[2,3,0,1])
    return G,zero_ind
    
def block_shrinkage_fc(V,mu,rho):
    coef = 0.5
    V_shape = tf.shape(V); one_val = tf.constant(1.0) 
    b = tf.div(mu,rho)
    norm_V = frobenius_norm_block(V,0)  
    norm_V_per_dimension = tf.div(norm_V,tf.cast(tf.slice(V_shape,[0],[1]),'float'))
    zero_part = tf.zeros(V_shape)
    zero_ind = tf.greater_equal(b,norm_V_per_dimension)
    num_zero = tf.reduce_sum(tf.cast(zero_ind,'float'))
    f4 = lambda: tf.greater_equal(tf.reduce_mean(norm_V),norm_V)
    f5 = lambda: zero_ind
    zero_ind = tf.cond(tf.greater(num_zero,tf.mul(coef,tf.reshape(tf.cast(tf.slice(V_shape,[1],[1]),'float'),[]))),f4,f5)
    G = tf.transpose(tf.select(zero_ind,tf.transpose(zero_part),tf.transpose(tf.mul(V,tf.transpose(tf.sub(one_val,tf.div(b,tf.reshape(norm_V,[-1,1])))))))) 
    return G,zero_ind
def block_truncate_conv(V,mu,rho):
    coef = 0.5
    V_shape = tf.shape(V) 
    b = tf.sqrt(tf.div(tf.mul(2.,mu),rho)) #threshold 
    # Reshape the 4D tensor of weights to a 2D matrix with rows containing the conv filters in vectorized form.
    V_shape1 = tf.concat(0,[tf.mul(tf.slice(V_shape,[2],[1]),tf.slice(V_shape,[3],[1])),tf.mul(tf.slice(V_shape,[0],[1]),tf.slice(V_shape,[1],[1]))])
    V = tf.reshape(tf.transpose(V,perm=[2,3,0,1]),V_shape1)
    norm_V = frobenius_norm_block(V,1)  
    norm_V_per_dimension = tf.div(norm_V,tf.cast(tf.slice(V_shape1,[1],[1]),'float'))
    # Implementation of Eq.10 in the paper using if condition inside the TensorFlow graph with tf.cond
    zero_part = tf.zeros(V_shape1)
    zero_ind = tf.greater_equal(b,norm_V_per_dimension)
    num_zero = tf.reduce_sum(tf.cast(zero_ind,'float'))
    # You can pass parameters to the functions in tf.cond() using lambda
    f4 = lambda: tf.greater_equal(tf.reduce_mean(norm_V),norm_V)
    f5 = lambda: zero_ind
    zero_ind = tf.cond(tf.greater(num_zero,tf.mul(coef,tf.cast(V_shape1[0],'float'))),f4,f5)
    G = tf.select(zero_ind,zero_part,V) 
    G_shape = tf.concat(0,[tf.slice(V_shape,[2],[1]),tf.slice(V_shape,[3],[1]),tf.slice(V_shape,[0],[1]),tf.slice(V_shape,[1],[1])])
    G = tf.transpose(tf.reshape(G,G_shape),perm=[2,3,0,1])
    return G,zero_ind
    
def block_truncate_fc(V,mu,rho):
    coef = 0.5
    V_shape = tf.shape(V) 
    b = tf.sqrt(tf.div(tf.mul(2.,mu),rho)) #threshold 
    norm_V = frobenius_norm_block(V,0)  
    norm_V_per_dimension = tf.div(norm_V,tf.cast(tf.slice(V_shape,[0],[1]),'float'))
    zero_part = tf.zeros(V_shape)
    zero_ind = tf.greater_equal(b,norm_V_per_dimension)
    num_zero = tf.reduce_sum(tf.cast(zero_ind,'float'))
    f4 = lambda: tf.greater_equal(tf.reduce_mean(norm_V),norm_V)
    f5 = lambda: zero_ind
    zero_ind = tf.cond(tf.greater(num_zero,tf.mul(coef,tf.reshape(tf.cast(tf.slice(V_shape,[1],[1]),'float'),[]))),f4,f5)
    G = tf.transpose(tf.select(zero_ind,tf.transpose(zero_part),tf.transpose(V))) 
    return G,zero_ind    