import numpy as np
import matplotlib.pyplot as plt

def cdfmake(file_name):
    file_np = file_name + '.npy'
    arr1 = np.load(file_np)
    arr2 = np.zeros_like(y)
    value = 0
    for i in range(100):
        value += y[i]
        z[i] = value 
    z = z/value
    return z

def cdfsave(file_name):
    new_name = 'cdf_' + file_name
    np.save(new_name, cdf(file_name))

def quick(file_name):
    file_np = file_name + '.npy'
    y_axis = np.load(file_np)
    num_spacing = len(y_axis)
    x_axis = np.linspace(0, 1, num_spacing)
    plt.plot(x_axis, y_axis, 'ro')
    plt.save()
    plt.show()

def compare(file_name1, file_name2):
    file_np1 = file_name1 + '.npy'
    file_np2 = file_name2 + '.npy'
    x_axis = np.linspace(0, 1, num_spacing)
    y1_axis = np.load(file_np)
    y2_axis = np.load(file_np)
    plt.plot(x_axis, y1_axis, 'ro')
    plt.plot(x_axis, y2_axis, 'bd')
    plot.save()
    plt.show()

if __name__ == "__main__":
    pass