
import numpy as np


def equalize(matrix, equalizers):

        num_samples, num_features = matrix.shape

        output = np.matrix(np.zeros(matrix.shape))
        for feature in range(num_features):
            output[:, feature] = np.matrix(map(equalizers[feature], np.array(matrix[:, feature]).flatten())).transpose()

        return output


def equalizers(matrix):
    return [equalization_function(np.array(vector)) for vector in matrix]


def equalization_function(values):

        norm_cdf, bin_edges = norm_cdf_with_edges(values)

        def equalize_value(value):
            index = np.digitize(value, bin_edges)
            return 1.0 if index >= len(norm_cdf) else norm_cdf[index]

        return equalize_value


def norm_cdf_with_edges(values, num_bins=500):

    hist, bin_edges = np.histogram(values, bins=num_bins)

    cum_sum = np.cumsum(hist)

    norm_cdf = [float(v)/cum_sum[-1] for v in cum_sum]

    return norm_cdf, bin_edges