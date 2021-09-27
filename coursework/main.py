import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from tolsolvty import tolsolvty


def read_lines(file_name: str):
    count = 100
    data = []

    for index in range(count):
        data.append(read_file(file_name + str(index) + ".txt"))
    return data


def read_file(file_name: str):
    rows = 1024
    columns = 10

    file = open(file_name, 'r')
    number = int(file.readline().replace("STOP Position =  ", ""))

    result = {}
    for index in range(rows):
        line = file.readline().replace("\n", " ")
        array = {}
        for j in range(columns):
            string = line[: line.find(" ")]
            if j != 0:
                array[j - 1] = float(string)
            line = line.replace(str(string) + '    ', "", 1)
        result[(rows + index - number) % rows] = array

    return result


def read_signal(file_name: str):
    data = []
    file = open(file_name, 'r')
    for line in file.readlines():
        data.append(float(line.replace("[", "").replace("]", "").split(", ")))
    return data


def average_element(data: list, size: int):
    for index in range(size):
        temp = 0
        for number in range(len(data[index])):
            temp += data[index][number]
        data[index] = temp / len(data[index])
    return data


def average_level(data_set: list, size: int):
    for index in range(size):
        data_set[index] = average_element(data_set[index], len(data_set[index]))

    result = []
    for index in range(len(data_set[0])):
        temp = 0
        for number in range(size):
            temp += data_set[number][index]
        result.append(temp / size)
    return result


def print_data(data: list, size: int, color: str):
    x = []
    y = []

    for index in range(size):
        y.append(index)
        x.append(data[index])

    plt.xlabel("Измерения")
    plt.ylabel("Значения")
    plt.plot(y, x, color)


def get_index(value: float, level: list, size: int):
    if value < level[0]:
        return 0
    elif value > level[size - 1]:
        return size - 2

    for index in range(size):
        if (value > level[index]) & (value < level[index + 1]):
            return index


def func_interpolation(x: list, y: list, value: float):
    return y[0] + (value - x[0]) / (x[1] - x[0]) * (y[1] - y[0])


def interpolation(signal: list, dc: list, levels, size: int, level_count: int):
    result = []
    for index in range(size):
        id = get_index(signal[index], levels[:, index], level_count)
        result.append(func_interpolation([levels[id][index], levels[id + 1][index]], [dc[id], dc[id + 1]], signal[index]))
    return result


def one_scale(data: list):
    temp = data.copy()
    temp.sort()
    max_pos = temp[len(temp) - 1]
    max_neg = temp[0]

    result = []
    for element in data:
        if element > 0.0:
            result.append(element / max_pos)
        else:
            result.append(-element / max_neg)
    return result


def get_points(data: list, size: int):
    pos_points, neg_points = [], []
    if data[0] < data[1]:
        pos_points.append(0)
    else:
        neg_points.append(0)

    for index in range(1, size):
        if data[index - 1] <= data[index]:
            pos_points.append(index)
        else:
            neg_points.append(index)
    return [pos_points, neg_points]


def get_matrix(signal: list, data: list, size: int, flag: bool):
    dy = 0.015
    di = 0.5
    count = 0
    if flag:
        offset = 1.0
    else:
        offset = 0.0

    A_bot = np.zeros((size, 3))
    A_top = np.zeros((size, 3))
    B_bot = np.zeros((size, 1))
    B_top = np.zeros((size, 1))

    for index in range(size):
        if index != 0 and data[index] - data[index - 1] > 2:
            count += 1

        A_bot[index][0] = data[index] - di + offset
        A_bot[index][1] = 1
        A_bot[index][2] = count
        B_bot[index][0] = signal[data[index]] - dy * abs(signal[data[index]])

        A_top[index][0] = data[index] + di + offset
        A_top[index][1] = 1
        A_top[index][2] = count
        B_top[index][0] = signal[data[index]] + dy * abs(signal[data[index]])
    return [A_top, A_bot, B_top, B_bot]


def get_amplitude(signal: list, points: list):
    pos_len = len(points[0])
    neg_len = len(points[1])

    [A1_top, A1_bot, B1_top, B1_bot] = get_matrix(signal, points[0], pos_len, True)
    [A2_top, A2_bot, B2_top, B2_bot] = get_matrix(signal, points[1], neg_len, False)

    [tolmax, argmax, envs, ccode] = tolsolvty(A1_bot, A1_top, B1_bot, B1_top)
    a1 = argmax[0]
    b1 = argmax[1]
    [tolmax, argmax, envs, ccode] = tolsolvty(A2_bot, A2_top, B2_bot, B2_top)
    a2 = argmax[0]
    b2 = argmax[1]
    y = abs((b1 * a2 - b2 * a1) / (a2 - a1))

    return [y, a1, b1, a2, b2]


def print_lines(data: list, a1: float, b1: float, a2: float, b2: float):
    width = 40
    x, y1, y2, time = [], [], [], []

    for index in range(width):
        time.append(data[index])
        x.append(index)
        y1.append(a1 * (x[index] - 2) + b1)
        y2.append(a2 * (x[index] - 2) + b2)

    plt.xlabel("Измерения")
    plt.ylabel("Амплитуда сигнала")
    plt.plot(x[: width], data[: width], 'blue')
    plt.plot(x[: width], y1, 'red')
    plt.plot(x[: width], y2, 'green')


def pi_scale(data: list, amplitude: float):
    result = []
    coeff = math.pi / (2 * amplitude)
    for element in data:
        result.append(element * coeff)
    return result


def print_phases(data):
    x, y = [], []
    for element in data:
        x.append(1)
        y.append(element / (2 * math.pi))
    plt.xlabel('Временная шкала')
    plt.scatter(y, x)
    plt.show()
    sns.distplot(y)
    plt.show()


def build_regression(data_set: list, levels: list):
    a, b = [], []
    for index in range(len(data_set[0])):
        y = []
        for data in data_set:
            y.append(data[index])
        z = np.polyfit(np.array(levels), np.array(y), 1)
        a.append(z[0])
        b.append(z[1])

    y = []
    for data in data_set:
        y.append(data[0])
    p = np.poly1d([a[0], b[0]])
    xp = np.linspace(-0.5, 0.5, 100)
    plt.plot(levels, y, '.', xp, p(xp))
    plt.xlabel("Константы")
    plt.ylabel("Значения")
    plt.show()

    build_hist(a)
    build_hist(b)


def build_hist(coeff: list):
    plt.xlabel("Значения")
    plt.ylabel("Общее количество")
    plt.hist(coeff)
    plt.show()


folder = "data"
colors = ['green', 'cyan', 'yellow', 'magenta', 'orange', 'blue']
levels = ["\-0_5V\-0_5V_", "\-0_25V\-0_25V_", "\ZeroLine\ZeroLine_", "\+0_25V\+0_25V_", "\+0_5V\+0_5V_"]
filename = "\Sin_100MHz\sin_100MHz_"

# print level lines
data_set = []
for level in levels:
    data_set.append(read_lines(folder + level))
for i in range(len(data_set)):
    data_set[i] = average_level(data_set[i], len(data_set[i]))
    print_data(data_set[i], len(data_set[i]), colors[i])
plt.show()

# print original signal
signal_number = "2"
signal = read_file(folder + filename + signal_number + ".txt")
signal = average_element(signal, len(signal))
for i in range(len(data_set)):
    print_data(data_set[i], len(data_set[i]), colors[i])
print_data(signal, len(signal), colors[len(colors) - 1])
plt.show()

# signal interpolation
dc = [-0.5, -0.25, 0.0, 0.25, 0.5]
signal = interpolation(signal, dc, np.array(data_set), len(signal), len(data_set))
print_data(signal, len(signal), colors[len(colors) - 1])
plt.show()

# new signal scale
signal = one_scale(signal)
print_data(signal, len(signal), colors[len(colors) - 1])
plt.show()

# harmonic signal parameters
points = get_points(signal, len(signal))
[amplitude, a1, b1, a2, b2] = get_amplitude(signal, points)
print_lines(signal, a1, b1, a2, b2)
plt.show()

# new signal scale
signal = pi_scale(signal, amplitude)
print_data(signal, len(signal), colors[len(colors) - 1])
plt.show()

# sampling phases
print_phases(signal)

# regression
build_regression(data_set, dc)
