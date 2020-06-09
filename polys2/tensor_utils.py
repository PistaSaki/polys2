import numpy as np
import tensorflow as tf
from tensorflow.keras.backend import ndim


def batched_gather_nd(a, indices, batch_ndim):
    """Return tensor `t` s.t. `t[i] == gather_nd(a[i], indices[i])` for any multiindex `i` of length `batch_ndim`.

    Args:
        a: tensor (np or tf)
        indices: integer-valued tensor (np or tf)
        batch_ndim: int

    Notes:
        The first `batch_ndim` axes of `a` and `indices` should be broadcast-compatible.

    Returns:
        tensor `t` .
    """
    a = tf.convert_to_tensor(a)
    indices = tf.convert_to_tensor(indices, dtype_hint=tf.int32)
    a_shape = tf.shape(a)
    indices_ndim = ndim(indices)
    assert indices_ndim >= batch_ndim + 1, "indices_ndim = {} should be greater than batch_ndim = {}".format(
        indices_ndim, batch_ndim)

    batch_shape = tf.broadcast_dynamic_shape(a_shape[:batch_ndim], tf.shape(indices)[:batch_ndim])

    # we force `indices` to have this batch_shape (needed for broadcasting in `indices` to work)
    indices = indices * tf.ones(
        shape=tf.concat([batch_shape, [1] * (indices_ndim - batch_ndim)], axis=0),
        dtype=indices.dtype
    )

    ###
    sh = tf.concat([batch_shape, tf.shape(indices)[batch_ndim: -1], [1]], axis=0)

    # we want to apply gather_nd, for this we must add indices
    # that traverse the first `batch_ndim` dimensions of our tensors
    ranges = [tf.cast(tf.range(d), dtype=indices.dtype) for d in tf.unstack(sh)]

    # in order to broadcast in `a` we must slightly modify `ranges`:
    ranges = [
                 tf.minimum(r, d - 1)
                 for r, d in zip(ranges[:batch_ndim], tf.unstack(tf.cast(a_shape, indices.dtype))[:batch_ndim])
             ] + ranges[batch_ndim:]
    augmented_indices = tf.concat(tf.meshgrid(*ranges, indexing="ij")[:batch_ndim] + [indices], axis=-1)

    # infer the static shape in tensorflow
    static_batch_shape = tf.broadcast_static_shape(
        a.shape[:batch_ndim],
        indices.shape[:batch_ndim]
    )
    stat_aug_ind_shape = static_batch_shape.as_list() + indices.shape.as_list()[batch_ndim:]
    stat_aug_ind_shape[-1] += batch_ndim
    augmented_indices.set_shape(stat_aug_ind_shape)

    return tf.gather_nd(a, augmented_indices)


def right_apply_map_along_batch(X, A, batch_inds, contract_inds, added_inds,):
    """
    Applies a batch of tensor transformations `A` to a batch of tensors `X`.
    
    Parameters
    ----------
    batch_inds: list of ints
        batch indices of X, these should have same shape as a first group of indices of A
    contract_inds: list of ints
        indices of X that are contracted with the second group of indices of A
    added_indices: list of ints
        after the contraction, the remaining indices of A in the result will be put to these places
    
    Example
    -------
    batch_inds = [0, 1], contract_inds = [2], added_inds = [2] 
    Y[i, j, m, l] = sum_{k} X_{i, j, k, l} * A[i, j, k, m ] 
    """
    assert len(A.shape) == len(batch_inds) + len(contract_inds) + len(added_inds)

    
    #print("A.shape =", A.shape)
    #print("X.shape =", X.shape)
    
    X_ndim = len(X.shape)
    free_inds = [i for i in range(X_ndim) if i not in (batch_inds + contract_inds)]
    ## indices of X = dissjoint union of (batch_inds, free_inds, contracted_inds)
    assert len(X.shape) == len(batch_inds) + len(free_inds) + len(contract_inds), (
        "Indices of X should be dissjoint union of (batch_inds, free_inds, contracted_inds) "+
        " X.ndim = " + str(len(X.shape)) + 
        ", batch_inds = " + str(batch_inds) +
        ", free_inds = " + str(free_inds) +
        ", contract_inds = " + str(contract_inds) + "."
    )
    ## we reindex (transpose) X to put them exactly in this order
    Xt = tf.transpose(X, batch_inds + free_inds + contract_inds)
    #print("X permutation = ", batch_inds + free_inds + contract_inds)
    #print("Xt =", Xt)
    
    ## reshape Xt so that there is only one free index and one contracted index
    Xt_shape = tf.shape(Xt)
    X_batch_shape = Xt_shape[:len(batch_inds)]
    X_free_shape = Xt_shape[len(batch_inds): len(batch_inds) + len(free_inds) ]
    X_contract_shape = Xt_shape[len(batch_inds) + len(free_inds):]
    
    Xtr_shape = tf.concat([
        X_batch_shape,
        [tf.reduce_prod(X_free_shape), tf.reduce_prod(X_contract_shape)]
    ], axis=0)
    Xtr = tf.reshape(Xt, Xtr_shape)
    #print("Xtr.shape =", Xtr.shape)
    
    
    ## reshape A so that it also contains one contract index and one added index
    A_shape = tf.shape(A)
    A_batch_shape = A_shape[:len(batch_inds)]
    A_contract_shape = A_shape[len(batch_inds): len(batch_inds) + len(contract_inds)]
    A_added_shape = A_shape[len(batch_inds) + len(contract_inds):]
    
    Ar_shape = tf.concat([
            A_batch_shape,
            [tf.reduce_prod(A_contract_shape), tf.reduce_prod(A_added_shape)]
    ], axis=0)
    Ar = tf.reshape(A, Ar_shape)
    #print("Ar.shape =", Ar.shape)
    #print("Ar =", Ar)
    
    
    ## check that A, X are compatible for the contraction
    #assert A_contract_shape == X_contract_shape
    
    #print("Ar.shape =", Ar.shape)
    #print("Xtr.shape =", Xtr.shape)

    ## multiply 
    Ytr = tf.matmul(Xtr, Ar)
    
    
    ## unravel the indices of Y
    Ytr_shape = tf.shape(Ytr)
    Ytr_batch_shape = Ytr_shape[:len(batch_inds)]
    Yt_shape =  tf.concat([Ytr_batch_shape, X_free_shape, A_added_shape], axis=0)
    Yt = tf.reshape(Ytr, Yt_shape)
    
    ################
    ## reindexing Y
    
    # First construct permutation p that puts batch_indices and free_indices into right order
    p = list(np.argsort(batch_inds + free_inds))
    # Then insert the added_indices into the right places
    n = len(p)
    for i, j in enumerate(added_inds):
        p.insert(j, i + n)
        
    # reindex
    Y = tf.transpose(Yt, p)
    
    return Y
    
def apply_tensor_product_of_maps(matrices, x, start_index = 0):
    """
    Assume $x\in U \otimes V \otimes ...$ and linear maps
    $f:U \to U'$, $g:V \to V'$ ...
    We return $ (f \otimes g \otimes ...)(x)$.
    """
    n = len(matrices)
    for M in reversed(matrices):
        x = tf.tensordot(M, x, axes = [[1], [start_index + n-1]])
    return x

def flatten_left(values, ndims=None):
    old_shape = tf.shape(values)
    if ndims == None:
        ndims = len(old_shape)
    
    new_shape = tf.concat(
        values = [
            tf.reduce_prod(old_shape[:ndims], keepdims=True),
            old_shape[ndims:]
        ], 
        axis = 0
    )
    
    return tf.reshape(values, new_shape)