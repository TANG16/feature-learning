
import numpy as np


def find(query_integer, integers, n_results, closest=True):

    if n_results < 1:
        raise ValueError("n_results must be greater than zero: " + n_results)
    if n_results > len(integers):
        raise ValueError("n_results must be less than or equal to the number of integers: {} < {}".format(len(integers), n_results))

    def count_set_bits(n):
        n = (n & 0x5555555555555555) + ((n & 0xAAAAAAAAAAAAAAAA) >> 1)
        n = (n & 0x3333333333333333) + ((n & 0xCCCCCCCCCCCCCCCC) >> 2)
        n = (n & 0x0F0F0F0F0F0F0F0F) + ((n & 0xF0F0F0F0F0F0F0F0) >> 4)
        n = (n & 0x00FF00FF00FF00FF) + ((n & 0xFF00FF00FF00FF00) >> 8)
        n = (n & 0x0000FFFF0000FFFF) + ((n & 0xFFFF0000FFFF0000) >> 16)
        n = (n & 0x00000000FFFFFFFF) + ((n & 0xFFFFFFFF00000000) >> 32)
        return n

    def get(vector, n):

        unsorted_result = np.argpartition(vector, n)[:n] if closest else \
                          np.argpartition(vector, -n)[-n:]

        def minimum(i1, i2):
            return \
                1 if vector[i1] > vector[i2] else\
                -1 if vector[i1] < vector[i2] else \
                0

        def maximum(i1, i2):
            return \
                -1 if vector[i1] > vector[i2] else\
                1 if vector[i1] < vector[i2] else \
                0

        sorted_result = sorted(unsorted_result, cmp=minimum if closest else maximum)

        print "Hamming distance", vector[sorted_result]

        return sorted_result

    integers_xor = [long(i) for i in np.bitwise_xor(integers, query_integer)]

    hamming_distance = np.array([count_set_bits(i) for i in integers_xor])

    return get(hamming_distance, n_results)


def prepare_integers(feature_data):

    n_inputs, n_features = feature_data.shape

    mean_ = np.mean(feature_data, axis=0)
    bit_states = (feature_data > mean_).astype(int)

    basis = np.array([2 ** i for i in range(0, n_features)])

    return np.array((np.matrix(bit_states) * np.matrix(basis).transpose()).astype(long))

