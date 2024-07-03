import math
from prettytable import PrettyTable
from tkinter import ttk
import tkinter as tk
from tkinter import *
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from mpmath import *

root = Tk()
root.title("Моделирование случайных величин, вариант 5")
root.geometry("800x300")
root.configure(bg="#F0F0F0")

label_style = {
    "font": ("Helvetica", 12),
    "fg": "#333333",
    "bg": "#F0F0F0",
    "padx": 10,
    "pady": 5
}
entry_style = {
    "font": ("Helvetica", 12),
    "bg": "#FFFFFF",
    "fg": "#333333",
    "bd": 1,
    "relief": "solid"
}
lbl1 = Label(
    root, text="На автоматическую телефонную станцию поступает поток вызовов с интенсивностью λ.", **label_style)
lbl1.grid(column=0, row=0, sticky=W)
lbl2 = Label(
    root, text="С.в. η — число вызовов за t минут, имеет распределение Пуассона со средним λt.", **label_style)
lbl2.grid(column=0, row=1, sticky=W)
lbl3 = Label(root, text="Интенсивность λ:", **label_style)
lbl3.grid(column=0, row=2, sticky=W)
lbl4 = Label(root, text="Время t(с):", **label_style)
lbl4.grid(column=0, row=3, sticky=W)
lbl5 = Label(root, text="Количество экспериментов:", **label_style)
lbl5.grid(column=0, row=4, sticky=W)
lambda_string = StringVar()
time_string = StringVar()
n_string = StringVar()
entry_lambda = Entry(root, width=20, textvariable=lambda_string, **entry_style)
entry_lambda.grid(column=1, row=2, padx=10)
entry_time = Entry(root, width=20, textvariable=time_string, **entry_style)
entry_time.grid(column=1, row=3, padx=10)
entry_n = Entry(root, width=20, textvariable=n_string, **entry_style)
entry_n.grid(column=1, row=4, padx=10)


def gen():
    return np.random.uniform(0, 1)


def puasson(k, param):
    return math.exp(-1*param)*((param)**k / math.factorial(k))

# проводим 1 эксперимент


def experiment(param, lambdaa):
    flag = 1
    k = 0
    summa = 0.0
    value = 0
    rand = gen()
    while (flag != 0):
        summa += puasson(k, param)
        if (rand < summa):
            value = k
            flag = 0
        else:
            k = k + 1
    return value


def getMathExpectation(lambdaa):  # значение мат ожидания
    return lambdaa


def getSelectiveAverage(Y, w_abs, N):  # выборочное среднее
    average = 0
    for i in range(0, len(Y)):
        average += Y[i]*w_abs[i]
    return average / N


# модуль разности мат ожидания и выборочного среднего
def getAbsBetweenExpectations(Y, w_abs, N, lambdaa):
    average = getSelectiveAverage(Y, w_abs, N)
    return abs(lambdaa - average)


def GetDispersion(lambdaa):   # дисперсия
    return lambdaa


def getSelectiveDispersion(Y, w_abs, N):  # выборочная дисперсия
    dispersion = 0
    average = getSelectiveAverage(Y, w_abs, N)
    for i in range(0, len(Y)):
        dispersion += ((Y[i] - average)**2) * w_abs[i]
    return dispersion / N


# модуль разности между дисперсией и выборочной дисперсией
def getAbsBetweenDispersions(Y, w_abs, N, lambdaa):
    return abs(lambdaa - getSelectiveDispersion(Y, w_abs, N))


def getMedian(Y, N):      # медиана
    if (N % 2 == 0):
        return ((Y[int(N / 2)] + Y[int(N / 2) - 1]) / 2)
    else:
        return (Y[int((N) / 2)])


def getRange(Y):  # размах выборки
    return (Y[len(Y) - 1] - Y[0])


def getF(Y, param):      # теоритическая функция распределения
    F = []
    for i in range(0, Y[- 1] + 50):
        summ = 0
        for j in range(0, i):
            summ += puasson(j, param)
        F.append(summ)
    return F


def getF2(Y, param, w_otnos):      # выборочная функция распределения
    F = getF(Y, param)
    F2 = []
    FF2 = getSelectiveF(Y, w_otnos, param)
    i = -1
    while (i != Y[0]):
        F2.append(0)
        i += 1
    j = 0
    while (i != Y[len(Y) - 1]):
        if (i == Y[j]):
            while (i != Y[j+1]):
                F2.append(FF2[j+1])
                i += 1
            j += 1
    for i in range(len(F2), len(F)):
        F2.append(1)
    return F2


def getSelectiveF(Y, w_otnos, param):
    F2 = []
    for i in range(0, len(Y) + 1):
        res = 0
        for j in range(0, i):
            res += w_otnos[j]
        F2.append(res)
    return F2


def getD(N, param, Y, YY, w_otnos):
    D = 0
    F = getF(Y, param)
    F2 = getF2(Y, param, w_otnos)
    maxD = 0
    for i in range(0, len(F)):
        D = abs(F[i] - F2[i])
        if (D > maxD):
            maxD = D
    return maxD


def getMaxPdiff(Y, param, w_otnos):
    maxDiff = 0
    diff = 0
    for i in range(0, len(Y)):
        diff = abs(puasson(Y[i], param) - w_otnos[i])
        if (diff > maxDiff):
            maxDiff = diff
    return maxDiff


def clicked():     # основная фцнкция, в которой все происходит
    # получаем начальные параметры
    lambdaa = float(lambda_string.get())
    time = float(time_string.get())
    N = int(n_string.get())
    param = lambdaa * time
    Y = []   # массив случайных величин - количества подъехавших машин
    YY = []   # YY = Y, тк в У будут вноситься изменения
    table = PrettyTable()
    i = 0
    while (i < N):   # проводим эксперименты, формируем первую строку таблицы случайных величин
        val = experiment(param, lambdaa)
        Y.append(val)
        i = i + 1
    Y.sort()

    w_abs = []
    w_otnos = []
    for i in range(0, len(Y)):
        w_abs.append(0)
    for i in range(0, len(Y)):
        YY.append(Y[i])

    i = 0
    j = 0
    p = 0
    while (j < len(Y)):  # формирование второй строки таблицы
        if (Y[i] == Y[j]):
            w_abs[p] = w_abs[p] + 1
            j = j + 1
        else:
            i = j
            p = p + 1

    i = 0
    j = i + 1
    while (j < len(Y)):  # сортировка массива, удаление всех одинаковых элементов
        if (Y[i] == Y[j]):
            Y.remove(Y[j])
        else:
            i = j
            j = j + 1

    sum_elem = 0  # сумма всех элементов массива w_abs
    for j in range(0, len(w_abs)):
        sum_elem = sum_elem + w_abs[j]

    for i in range(0, len(Y)):  # формируется третья строка матрицы
        w_otnos.append(w_abs[i] / sum_elem)

    for i in range(1, len(Y) + 1):
        table.add_column("", [Y[i - 1], w_abs[i - 1], w_otnos[i - 1]])
    print(table)  # печать таблицы

    F = getF(Y, param)    # массив значений теоретической функции
    F2 = getF2(Y, param, w_otnos)  # массив значений выборочной функции

    properties = PrettyTable()  # формируется таблица характеристик для 2 лабы
    properties.field_names = ["Eη", "x̅", "|Eη -x̅ |", "Dη",
                              "S^2", "|Dη - S^2|", "Me", "R", "D", "|P - nj/n|"]
    properties.add_row([lambdaa, '%.3f' % getSelectiveAverage(Y, w_abs, N),
                       '%.3f' % getAbsBetweenExpectations(
                           Y, w_abs, N, lambdaa),
                        lambdaa,  '%.3f' % getSelectiveDispersion(Y, w_abs, N),
                        '%.3f' % getAbsBetweenDispersions(
                            Y, w_abs, N, lambdaa),
                        getMedian(YY, N), getRange(Y),  '%.3f' % getD(N, param, Y, YY, w_otnos),  '%.3f' % getMaxPdiff(Y, param, w_otnos)])

    # таблица значений результатов экспериментов из 1 лабы и посчитанных теоретических вероятностей
    probabilities = PrettyTable()
    for i in range(1, len(Y) + 1):
        probabilities.add_column(
            "", [Y[i - 1],  '%.3f' % puasson(Y[i-1], param), w_otnos[i - 1]])

    print(properties)
    print(probabilities)

    print('\n')
    print('\n')
    print('\n')

    # строим графики

    """  fig1, ax1 = plt.subplots()
    for i in range(0, len(F)): 
         ax1.hlines(F[i], i-1, i, color = 'red')
    ax1.set_title('График теоретической функции распределения')
    ax1.grid()      

    fig2, ax2 = plt.subplots()
    for i in range(0, len(F2)): 
      ax2.hlines(F2[i], i-1, i, color = 'blue')
    ax2.set_title('График выборочной функции распределения')
    ax2.grid()
    ax2.hlines(1, Y[len(Y) - 1], Y[len(Y) - 1] + 1, color = 'blue')  """

    fig3, ax3 = plt.subplots()
    for i in range(0, len(F)):
        ax3.hlines(F[i], i-1, i, color='red', label="Fтеор")
    for i in range(0, len(F2)):
        ax3.hlines(F2[i], i-1, i, color='blue', label="Fвыб")
    ax3.set_title('Графики  функций распределения')
    ax3.grid()

    plt.show()

    k = int(input("Число интервалов k: "))
    alpha = float(input("Уровень значимости а: "))
    intervals = [float(el) for el in input("Интервалы: ").split()]

    # функция - проверка гипотезы для 3 лабы, реализована ниже
    hypothesis(Y, w_abs, param, N, k, alpha, intervals)
    res = OneHundredSelections(lambdaa, time, N, k, alpha, intervals)
    print("\n\n\n", sum(res), "гипотез принято,",
          (100 - sum(res)), "гипотез отклонено")


btn = Button(root, text="Разыграть", command=clicked)
btn.place(relx=.0, rely=.6)

# третья часть лабораторной работы


def getNj(Y, w_abs, intervals):  # число значений случ вел попавших в интервал j
    Nj = []
    for i in range(0, len(intervals) + 1):
        count = 0
        for j in range(0, len(Y)):
            if (i == 0 and Y[j] >= i and Y[j] < intervals[i]):
                count += w_abs[j]
            elif (i != len(intervals) and Y[j] >= intervals[i - 1] and Y[j] < intervals[i]):
                count += w_abs[j]
            elif (i == len(intervals) and Y[j] >= intervals[len(intervals) - 1]):
                count += w_abs[j]
        Nj.append(count)
    return Nj


def getQj(Y, intervals, param):  # вероятность попадания в интервал
    Qj = []
    for i in range(0, len(intervals)):
        res = 0
        for j in range(0, len(Y)):
            if (i == 0 and Y[j] >= i and Y[j] < intervals[i]):
                res += puasson(Y[j], param)
            elif (i != len(intervals) and Y[j] >= intervals[i - 1] and Y[j] < intervals[i]):
                res += puasson(Y[j], param)
        Qj.append(res)
    summ = 0
    for i in range(0, len(Qj)):
        summ += Qj[i]
    for j in range(0, len(Y)):
        if (Y[j] >= intervals[len(intervals) - 1]):
            res = 1 - summ
    Qj.append(res)
    return Qj


def getR0(k, Nj, Qj, N):
    R0 = 0
    for i in range(k):
        if Qj[i] != 0:
            R0 += ((Nj[i] - N*Qj[i])**2) / (N*Qj[i])
        else:
            print(f"Внимание: Qj[{i}] равно нулю. Пропуск деления.")
    return R0


def f(k, x):  # плотность распределения хи квадрат
    r = k - 1
    # print(r)
    temp = r / 2 - 1
    if (x > 0):
        return math.pow(2, -r / 2) * (1 / math.gamma(r / 2)) * (x**temp) * math.exp(-x / 2)
    else:
        return 0
# R0 while


def integrate(k, R0):
    result = 0
    for i in range(1, 10001):
        result += (f(k, R0 * (i - 1) / 10000.0) +
                   f(k, R0 * i / 10000.0)) * R0 / (2 * 10000.0)
    return result


def checkHypothesis(alpha, FR0):  # решение о принятии гипотезы
    if (FR0 >= alpha):
        return True
    else:
        return False


def hypothesis(Y, w_abs, param, N, k, alpha, intervals):
    Nj = getNj(Y, w_abs, intervals)
    Qj = getQj(Y, intervals, param)
    R0 = getR0(k, Nj, Qj, N)
    FR0 = 1 - integrate(k, R0)
    hyp = PrettyTable()
    print('\n')
    print("Отображение гипотезы в виде теоретических вероятностей q_i:")
    print(Qj)
    print('\n')
    print("F(R0):")
    print(FR0)
    print('\n')
    if (checkHypothesis(alpha, FR0)):
        print("Гипотеза принята")
        return 1
    else:
        print("Гипотеза отклонена")
        return 0


def OneHundredSelections(lambdaa, time, N, k, alpha, intervals):
    param = lambdaa * time
    res = []
    print(k)
    for l in range(100):
        Y = []  # массив случайных величин - количества подъехавших машин
        YY = []  # YY = Y, тк в У будут вноситься изменения
        i = 0
        while (i < N):  # проводим эксперименты, формируем первую строку таблицы случайных величин
            val = experiment(param, lambdaa)
            Y.append(val)
            i = i + 1
        Y.sort()
        w_abs = []
        w_otnos = []
        for i in range(0, len(Y)):
            w_abs.append(0)
        for i in range(0, len(Y)):
            YY.append(Y[i])
        i = 0
        j = 0
        p = 0
        while (j < len(Y)):  # формирование второй строки таблицы
            if (Y[i] == Y[j]):
                w_abs[p] = w_abs[p] + 1
                j = j + 1
            else:
                i = j
                p = p + 1
        i = 0
        j = i + 1
        while (j < len(Y)):  # сортировка массива, удаление всех одинаковых элементов
            if (Y[i] == Y[j]):
                Y.remove(Y[j])
            else:
                i = j
                j = j + 1
        sum_elem = 0  # сумма всех элементов массива w_abs
        for j in range(0, len(w_abs)):
            sum_elem = sum_elem + w_abs[j]
        for i in range(0, len(Y)):  # формируется третья строка матрицы
            w_otnos.append(w_abs[i] / sum_elem)
        print("before 0")
        r = hypothesis(Y, w_abs, param, N, k, alpha, intervals)
        res.append(r)
    return res


root.mainloop()
