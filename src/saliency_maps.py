import numpy as np
import tensorflow as tf

def saliencymap(model, input_model, order_output): 
    """Compute gradients for each pixel w/r to the predicted class

    Args:
        model (tf.keras.Model): trained CNN model
        input_model (array or tensor): input for the model

    Returns:
        (array): gradients for each pixel normalized in [0,1]
    """ 
    # FIXME: add last channel needed for this  
    if input_model.shape[-1] != 1:
        input_model = np.expand_dims(input_model,axis=-1)

    input_ = tf.Variable(input_model, dtype=float)

    with tf.GradientTape() as tape:
        pred = model(input_, training=False)
        class_idxs_sorted = np.argsort(pred.numpy().flatten())[::-1]
        loss = pred[0][class_idxs_sorted[0]]
    prob = loss
    pred_class = order_output[class_idxs_sorted[0]]

    grads = tape.gradient(loss, input_)
    dgrad_abs = tf.math.abs(grads)
    dgrad_max_ = np.max(dgrad_abs, axis=3)[0] # this is not necessary for 1-channel

    ## normalize to range between 0 and 1
    arr_min, arr_max  = np.min(dgrad_max_), np.max(dgrad_max_)
    smap = (dgrad_max_ - arr_min) / (arr_max - arr_min + 1e-18)

    return smap, prob, pred_class
