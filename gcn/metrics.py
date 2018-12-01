import tensorflow as tf


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)

def masked_accuracy(preds, labels, mask, target_mean, target_stdev):
    """Accuracy with masking."""
    mask=tf.transpose(mask)
    mask = tf.cast(mask, dtype=tf.float32)
    mask = tf.expand_dims(mask,-1)
    mask = tf.tile(mask,[1,labels.shape[1].value])
    labels = tf.boolean_mask(labels,mask)
    preds = tf.boolean_mask(preds,mask)

    labels = tf.abs(labels + target_mean/target_stdev)
    preds = tf.abs(preds + target_mean/target_stdev)
    diff = tf.abs(tf.subtract(labels,preds))
    loss = tf.divide(diff,labels)
    return tf.reduce_mean(loss,0)

def mean_absolute_error(preds,labels,mask):
    """Mean absolute error with masking."""
    mask = tf.cast(mask,dtype=tf.float32)
    mask = tf.expand_dims(mask,-1)
    mask = tf.tile(mask,[1,labels.shape[1].value])
    mask /= tf.reduce_mean(mask)
    loss = tf.abs(tf.subtract(labels,preds))
    loss = tf.multiply(loss,mask)
    return tf.reduce_mean(loss,0)    

def mean_absolute_error_unnormalized(preds,labels,mask,target_mean,target_stdev):
    """Mean absolute error of unnormalized data"""
    mask = tf.cast(mask,dtype=tf.float32)
    mask = tf.expand_dims(mask,-1)
    mask = tf.tile(mask,[1,labels.shape[1].value])
    mask /= tf.reduce_mean(mask)
    labels = target_stdev*labels+target_mean
    loss = tf.abs(tf.subtract(labels,preds))
    loss = tf.multiply(loss,mask)
    return tf.reduce_mean(loss,0)  
    
def square_error(preds, labels, mask):
    """L2 loss refactored to incorporate masks"""
    mask = tf.cast(mask,dtype=tf.float32)
    mask = tf.expand_dims(mask,-1)
    mask = tf.tile(mask,[1,labels.shape[1].value])

    loss = tf.losses.mean_squared_error(labels,preds,reduction=tf.losses.Reduction.NONE)
    loss = tf.boolean_mask(loss,mask)
    return tf.reduce_mean(loss)






'''


def square_error_old(preds, labels, mask):
    """L2 loss refactored to incorporate masks"""
    mask = tf.cast(mask,dtype=tf.float32)
    mask = tf.expand_dims(mask,-1)
    mask = tf.tile(mask,[1,labels.shape[1].value])
    mask /= tf.reduce_mean(mask)
    loss = tf.losses.mean_squared_error(labels,preds,reduction=tf.losses.Reduction.NONE)
    loss = tf.multiply(loss,mask)
    return tf.reduce_mean(loss)


def square_error_3(preds, labels, mask):
    """L2 loss refactored to incorporate masks"""
    # should be equivalent to the other square error function
    mask = tf.cast(mask,dtype=tf.float32)
    mask = tf.expand_dims(mask,-1)
    mask = tf.tile(mask,[1,labels.shape[1].value])

    loss = tf.losses.mean_squared_error(labels,preds,mask)
    return loss

def masked_accuracy_old(preds, labels, mask, target_mean, target_stdev):
    """Accuracy with masking."""
    mask=tf.transpose(mask)
    mask = tf.cast(mask, dtype=tf.float32)
    mask = tf.expand_dims(mask,-1)
    mask = tf.tile(mask,[1,labels.shape[1].value])
    mask /= tf.reduce_mean(mask)
    #mnabserr = tf.metrics.mean_absolute_error(labels,preds)
    #accuracy_all = tf.multiply(mnabserr,mask)
    #accuracy_all *= mask

    labels *= mask #fixes an annoying divide by 0 problem

    labels = labels + target_mean/target_stdev
    preds = preds + target_mean/target_stdev
    diff = tf.abs(tf.subtract(labels,preds))
    loss = tf.divide(diff,tf.abs(labels))
    loss = tf.multiply(loss,mask)
    return tf.reduce_mean(loss,0)
'''