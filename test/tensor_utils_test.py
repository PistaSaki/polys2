import numpy as np
import tensorflow as tf

from polys2.tensor_utils import right_apply_map_along_batch

def test_right_apply_along_batch():
    floatx = np.float32
    
    X = np.array([[0, 0],[1, 0]], floatx)
    A = np.array([[ 1., -1.],[ 0.,  1.]], floatx)
    
    right_apply_map_along_batch(
            X,A,
            batch_inds = [], contract_inds = [0], added_inds = [0]
        )
    
    Y = right_apply_map_along_batch(
        X = tf.constant([
            [1, 2, 3],
            [0, 1, 2],
            [-1, 0, 1]
        ], floatx),
        A = tf.constant([
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 1],
        ], floatx), 
        batch_inds=[0],
        contract_inds = [1],
        added_inds = []
    )
    print(Y)
    
if __name__ == "__main__":
    test_right_apply_along_batch()