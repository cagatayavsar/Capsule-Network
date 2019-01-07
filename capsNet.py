import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def squash(s, axis=-1, epsilon=1e-7, name=None):
    with tf.name_scope(name, default_name="squash"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keepdims=True)
        safe_norm = tf.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = s / safe_norm
        return squash_factor * unit_vector

def compute_safe_norm(s, axis=-1, epsilon=1e-7, keepdims=False, name=None):
    with tf.name_scope(name, default_name="safe_norm"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keepdims=keepdims)
        return tf.sqrt(squared_norm + epsilon)

def compute_margin_loss(input, y):
    
    T = tf.one_hot(y, depth=10, name="T")
    
    m_plus = 0.9
    m_minus = 0.1
    lambda_val = 0.5
    
    present_error_raw = tf.square(tf.maximum(0., m_plus - input), name="present_error_raw")
    present_error = tf.reshape(present_error_raw, shape=(-1, 10), name="present_error")
    
    absent_error_raw = tf.square(tf.maximum(0., input - m_minus), name="absent_error_raw")
    absent_error = tf.reshape(absent_error_raw, shape=(-1, 10), name="absent_error")
    
    L = tf.add(T * present_error, lambda_val * (1.0 - T) * absent_error, name="L")
    
    margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")
    
    return margin_loss


def dynamic_routing(input, batch_size):
    """  
    Args:
        input: Example shape is [batch_size, 1152, 8].
    
    Return:
        Example shape is [batch_size, 1, 10, 16, 1]
    """

    routing_iteration = 3

    init_sigma = 0.1

    W_init = tf.random_normal(
        shape=(1, 1152, 10, 16, 8),
        stddev=init_sigma, dtype=tf.float32, name="W_init")
    W = tf.Variable(W_init, name="W")
    
    # Shape [batch_size, 1152, 10, 16, 8]
    W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name="W_tiled")

    # Shape [batch_size, 1152, 8, 1]
    input_expanded = tf.expand_dims(input, -1, name="input_expanded")
    
    # Shape [batch_size, 1152, 1, 8, 1]
    input_tile = tf.expand_dims(input_expanded, 2, name="input_tile")
    
    # Shape [batch_size, 1152, 10, 8, 1]
    input_tiled = tf.tile(input_tile, [1, 1, 10, 1, 1], name="input_tiled")
    
    # [batch_size, 1152, 10, 16, 8] * [batch_size, 1152, 10, 8, 1] -> [batch_size, 1152, 10, 16, 1] -> 
    u_ij = tf.matmul(W_tiled, input_tiled, name="u_ij")

    # Shape is [batch_size, 1152, 10, 1, 1]
    b_ij = tf.zeros([batch_size, 1152, 10, 1, 1], dtype=np.float32, name="b_ij")
    
    for i in range(routing_iteration):

        # [batch_size, 1152, 10, 1, 1] -> [batch_size, 1152, 10, 1, 1]
        c_ij = tf.nn.softmax(b_ij, dim=2, name="c_ij")

        # [batch_size, 1152, 10, 1, 1] * [batch_size, 1152, 10, 16, 1] -> [batch_size, 1152, 10, 16, 1]
        s_j = tf.multiply(c_ij, u_ij, name="s_j")
        
        # [batch_size, 1152, 10, 16, 1] -> [batch_size, 1, 10, 16, 1]
        s_j = tf.reduce_sum(s_j, axis=1, keepdims=True)

        # [batch_size, 1, 10, 16, 1] -> [batch_size, 1, 10, 16, 1]
        v_j = squash(s_j, axis=-2, name="v_j")
        
        # [batch_size, 1, 10, 16, 1] -> [batch_size, 1152, 10, 16, 1]
        v_j_tile = tf.tile(v_j, [1, 1152, 1, 1, 1], name="v_j_tile")

        # [batch_size, 1152, 10, 16, 1] * [batch_size, 1152, 10, 16, 1] -> [batch_size, 1152, 10, 1, 1]
        b_increment = tf.matmul(u_ij, v_j_tile, transpose_a=True, name="b_increment")

        # [batch_size, 1152, 10, 1, 1] +  [batch_size, 1152, 10, 1, 1] -> [batch_size, 1152, 10, 1, 1]
        b_ij = tf.add(b_ij, b_increment)

    return v_j

def train():
    
    tf.reset_default_graph()
    
    np.random.seed(42)
    tf.set_random_seed(42)
    
    mnist = input_data.read_data_sets("/tmp/data/")
    
    batch_size = 50
    
    X = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32, name="X")
    y = tf.placeholder(shape=[None], dtype=tf.int64, name="y")
    
    # [28, 28, 1] -> [20, 20, 256]
    conv1_params = {
        "filters": 256,
        "kernel_size": 9,
        "strides": 1,
        "padding": "valid",
        "activation": tf.nn.relu,
    }

    # [20, 20, 256] -> [6, 6, 256]
    conv2_params = {
        "filters": 256,
        "kernel_size": 9,
        "strides": 2,
        "padding": "valid",
        "activation": tf.nn.relu
    }
    
    conv1 = tf.layers.conv2d(X, name="conv1", **conv1_params)
    conv2 = tf.layers.conv2d(conv1, name="conv2", **conv2_params)
    
    s_j = tf.reshape(conv2, [-1, 1152, 8], name="s_j")
    
    v_j = squash(s_j, name="v_j")
    
    v_j = dynamic_routing(v_j, batch_size=batch_size)
    
    # [batch_size, 1, 10, 16, 1] -> [batch_size, 1, 10, 1]
    y_proba = compute_safe_norm(v_j, axis=-2, name="y_proba")
    
    # [batch_size, 1, 10, 1] -> [batch_size, 1, 1]
    y_proba_argmax = tf.argmax(y_proba, axis=2, name="y_proba")
    
    # [batch_size, 1, 1] -> [batch_size,]
    y_pred = tf.squeeze(y_proba_argmax, axis=[1,2], name="y_pred")
    
    margin_loss = compute_margin_loss(y_proba,y)
    
    mask_with_labels = tf.placeholder_with_default(False, shape=(), name="mask_with_labels")
    
    reconstruction_targets = tf.cond(mask_with_labels, # condition
                                lambda: y,        # if True
                                lambda: y_pred,   # if False
                                name="reconstruction_targets")
    
    reconstruction_mask = tf.one_hot(reconstruction_targets,
                                depth=10,
                                name="reconstruction_mask")
    
    reconstruction_mask = tf.reshape(reconstruction_mask, [-1, 1, 10, 1, 1])
   
    v_j_masked = tf.multiply(v_j, reconstruction_mask, name="v_j_masked")
    
    decoder_input = tf.reshape(v_j_masked, [-1, 10*16], name="decoder_input")
    
    #Decoder
    n_hidden1 = 512
    n_hidden2 = 1024
    n_output = 28 * 28
    
    with tf.name_scope("decoder"):
        hidden1 = tf.layers.dense(decoder_input, n_hidden1,
                              activation=tf.nn.relu,
                              name="hidden1")
        hidden2 = tf.layers.dense(hidden1, n_hidden2,
                              activation=tf.nn.relu,
                              name="hidden2")
        decoder_output = tf.layers.dense(hidden2, n_output,
                                     activation=tf.nn.sigmoid,
                                     name="decoder_output")
        
    #Reconstruction Loss
    X_flat = tf.reshape(X, [-1, n_output], name="X_flat")
    squared_difference = tf.square(X_flat - decoder_output, name="squared_difference")
    reconstruction_loss = tf.reduce_mean(squared_difference, name="reconstruction_loss")
    
    alpha = 0.0005

    loss = tf.add(margin_loss, alpha * reconstruction_loss, name="loss")
    
    correct = tf.equal(y, y_pred, name="correct")
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
    
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss, name="training_op")
    
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    
    n_epochs = 1
    batch_size = 50
    restore_checkpoint = True

    n_iterations_per_epoch = mnist.train.num_examples // batch_size
    n_iterations_validation = mnist.validation.num_examples // batch_size
    best_loss_val = np.infty
    checkpoint_path = "./my_capsule_network"
    
    with tf.Session() as sess:
        if restore_checkpoint and tf.train.checkpoint_exists(checkpoint_path):
            saver.restore(sess, checkpoint_path)
        else:
            init.run()

        for epoch in range(n_epochs):
            for iteration in range(1, n_iterations_per_epoch + 1):
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                # Run the training operation and measure the loss:
                _, loss_train = sess.run(
                    [training_op, loss],
                    feed_dict={X: X_batch.reshape([-1, 28, 28, 1]),
                               y: y_batch,
                               mask_with_labels: True})
                print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f}".format(
                          iteration, n_iterations_per_epoch,
                          iteration * 100 / n_iterations_per_epoch,
                          loss_train),
                      end="")

            # At the end of each epoch,
            # measure the validation loss and accuracy:
            loss_vals = []
            acc_vals = []
            for iteration in range(1, n_iterations_validation + 1):
                X_batch, y_batch = mnist.validation.next_batch(batch_size)
                loss_val, acc_val = sess.run(
                        [loss, accuracy],
                        feed_dict={X: X_batch.reshape([-1, 28, 28, 1]),
                                   y: y_batch})
                loss_vals.append(loss_val)
                acc_vals.append(acc_val)
                print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
                          iteration, n_iterations_validation,
                          iteration * 100 / n_iterations_validation),
                      end=" " * 10)
            loss_val = np.mean(loss_vals)
            acc_val = np.mean(acc_vals)
            print("\rEpoch: {}  Val accuracy: {:.4f}%  Loss: {:.6f}{}".format(
                epoch + 1, acc_val * 100, loss_val,
                " (improved)" if loss_val < best_loss_val else ""))

            # And save the model if it improved:
            if loss_val < best_loss_val:
                save_path = saver.save(sess, checkpoint_path)
                best_loss_val = loss_val

if __name__ == "__main__":
	train()