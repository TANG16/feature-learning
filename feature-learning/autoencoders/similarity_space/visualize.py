
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from feature_learning.autoencoders.similarity_space.visualize import visualize


def visualize(Wh, bh, bo, h, test_data):

    num_input_units, num_features = Wh.shape

    def configure_axis(ax, title="", x_label="", y_label=""):
        ax.set_axis_bgcolor("black")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        return ax

    input_matrix = np.matrix(test_data)
    activation_matrix = np.matrix(h)
    fig = plt.figure()
    df_input_data = pd.DataFrame(input_matrix)
    input_correlation = np.matrix(df_input_data.corr().values)
    visualization.intensity_grid(
        configure_axis(
            fig.add_subplot(111),
            title="Input absolute correlation",
            x_label="Input unit",
            y_label="Input unit"
        ),
        np.abs(input_correlation),
        value_range=(0, 1)
    )

    plt.show(block=False)
    raw_input("Continue:")
    plt.close()

    fig = plt.figure()
    df_activation_data = pd.DataFrame(activation_matrix)
    activation_correlation = np.matrix(df_activation_data.corr().values)
    visualization.intensity_grid(
        configure_axis(
            fig.add_subplot(111),
            title="Activation absolute correlation",
            x_label="Feature",
            y_label="Feature"
        ),
        np.abs(activation_correlation),
        value_range=(0, 1)
    )

    plt.show(block=False)
    raw_input("Continue:")
    plt.close()

    fig = plt.figure()
    input_activation_correlation = np.matrix(np.zeros((num_input_units, num_features)))
    for input in range(0, num_input_units):
        for feature in range(0, num_features):
            input_activation_correlation[input, feature] = np.corrcoef(
                np.array(input_matrix[:, input]).flatten(),
                np.array(activation_matrix[:, feature]).flatten()
            )[0][1]
    visualization.intensity_grid(
        configure_axis(
            fig.add_subplot(111),
            title="Input - activation absolute correlation",
            x_label="Feature",
            y_label="Input unit"
        ),
        np.abs(input_activation_correlation),
        value_range=(0, 1)
    )

    plt.show(block=False)
    raw_input("Continue:")
    plt.close()

    fig = plt.figure()
    norm_weights_sorted = np.concatenate(sorted(
        Wh.transpose(), key=lambda row: np.mean(np.sum(row)))
    )
    visualization.intensity_grid(
        configure_axis(
            fig.add_subplot(121),
            title="Weights (sorted on mean weight)",
            x_label="Input unit",
            y_label="Feature"
        ),
        norm_weights_sorted
    )
    visualization.intensity_grid(
        configure_axis(
            fig.add_subplot(122),
            title="Weight inverses(sorted on mean weight)",
            x_label="Input unit",
            y_label="Feature"
        ),
        -norm_weights_sorted
    )

    plt.show(block=False)
    raw_input("Continue:")
    plt.close()

    fig = plt.figure()
    visualization.intensity_grid(
        configure_axis(
            fig.add_subplot(111),
            title="Wh'",
            x_label="Input unit",
            y_label="Feature"
        ),
        Wh.transpose()
    )

    plt.show(block=False)
    raw_input("Continue:")
    plt.close()

    fig = plt.figure()
    visualization.parallel_coordinates(
        configure_axis(
            fig.add_subplot(211),
            title="bh",
            x_label="Feature",
            y_label="Bias"
        ),
        bh,
    )
    visualization.parallel_coordinates(
        configure_axis(
            fig.add_subplot(212),
            title="bo",
            x_label="Input unit",
            y_label="Bias"
        ),
        bo,
    )

    plt.show(block=False)
    raw_input("Continue:")
    plt.close()
