#in this file we will generate the data for the model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


def get_data(samples, sedd):

    np.random.seed(sedd)
    x_values = np.random.uniform(0, 2*math.pi, samples)
    y_values = np.sin(x_values)
    y_values += 0.1 * np.random.randn(*y_values.shape)
    plt.plot(x_values, y_values, 'r.')
    plt.show()
    df = pd.DataFrame({"x_values": x_values, "y_values": y_values})
    df.to_csv("sin_data.csv", index=False)


if __name__ == "__main__":
    get_data(1000, 1337)