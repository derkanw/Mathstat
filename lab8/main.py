import numpy as np
import matplotlib.pyplot as plt


def read_signal(filename):
    data = []
    file = open("wave_ampl.txt", 'r')

    for line in file.readlines():
        data.append([float(element) for element in line.replace('[', '').replace(']', '').split(", ")])
    data = np.asarray(data)
    data = np.reshape(data, (data.shape[1] // 1024, 1024))

    file.close()
    return data[1]

def get_data(signal, zones):
    signal_data = []
    for borders in zones:
        data_part = []
        for j in range(borders[0], borders[1]):
            data_part.append(signal[j])
        signal_data.append(data_part)
    return signal_data


def get_areas(signal):
    bin = int(1.72 * (len(signal) ** (1 / 3)))
    hist = plt.hist(signal, bins=bin)

    x, start_y, finish_y = [], [], []
    types = [0] * bin

    for i in range(bin):
        x.append(hist[0][i])
        start_y.append(hist[1][i])
        finish_y.append(hist[1][i + 1])

    sort_x = sorted(x)
    for i in range(bin):
        if x[i] == sort_x[len(x) - 1]:
            types[i] = "фон"
        elif x[i] == sort_x[len(x) - 2]:
            types[i] = "сигнал"
        else:
            types[i] = "переход"

    return start_y, finish_y, types


def get_zones(signal, start, finish, types):
    signal_types = [0] * len(signal)
    zones, zones_type = [], []

    for i in range(len(signal)):
        for j in range(len(types)):
            if (signal[i] >= start[j]) and (signal[i] <= finish[j]):
                signal_types[i] = types[j]

    currentType = signal_types[0]
    start = 0

    for i in range(len(signal_types)):
        if currentType != signal_types[i]:
            zones_type.append(currentType)
            zones.append([start, i])
            start = i
            currentType = signal_types[i]

    if currentType != zones_type[len(zones_type) - 1]:
        zones_type.append(currentType)
        zones.append([start, len(signal) - 1])

    return zones, zones_type, get_data(signal, zones)


def get_inter(signal):
    sum = 0.0
    mean_signal = np.empty(signal.shape[0])
    for i in range(len(signal)):
        mean_signal[i] = np.mean(signal[i])
    mean = np.mean(mean_signal)

    for i in range(len(mean_signal)):
        sum += (mean_signal[i] - mean) ** 2
    sum /= signal.shape[0]

    return len(signal) * sum


def get_intra(signal):
    result = 0.0
    for i in range(signal.shape[0]):
        mean = np.mean(signal[i])
        sum = 0.0
        for j in range(signal.shape[1]):
            sum += (signal[i][j] - mean) ** 2
        sum /= signal.shape[0]
        result += sum

    return result / signal.shape[0]


def get_fisher(signal, k):
    data = np.reshape(signal, (k, int(signal.size / k)))
    f = get_inter(data) / get_intra(data)
    print("k = " + str(k))
    print("F = " + str(f))
    return f


def get_k(num):
    i = 4
    while num % i != 0:
        i += 1
    return i


def get_params(signal, area_data):
    fishers = []
    for i in range(len(area_data)):
        start = area_data[i][0]
        finish = area_data[i][1]
        k = get_k(finish - start)

        while k == finish - start:
            finish += 1
            k = get_k(finish - start)

        fishers.append(get_fisher(signal[start:finish], k))
    return fishers

def draw_signal(signal):
    plt.title("Входной сигнал")
    plt.plot(range(len(signal)), signal, "blue")
    plt.grid()
    plt.show()


def draw_hist(signal):
    plt.hist(signal, bins=int(1.72 * (len(signal) ** (1 / 3))), color="blue")
    plt.grid()
    plt.title("Гистограмма входного сигнала")
    plt.show()


def draw_areas(signal_data, area_data, types):
    plt.title("График типов областей")
    plt.ylim([-0.5, 0])

    for i in range(len(area_data)):
        if types[i] == "фон":
            color = "yellow"
        elif types[i] == "сигнал":
            color = "red"
        else:
            color = "green"
        plt.plot([element for element in range(area_data[i][0], area_data[i][1], 1)], signal_data[i], color=color,
                 label=types[i])
    plt.grid()
    plt.legend()
    plt.show()


signal = read_signal("wave_ampl.txt")
draw_signal(signal)
draw_hist(signal)
start, finish, types = get_areas(signal)
zones, zones_types, signal_data = get_zones(signal, start, finish, types)
draw_areas(signal_data, zones, zones_types)
get_params(signal, zones)
print(zones)
