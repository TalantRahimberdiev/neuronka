
import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt
import numpy as np


def retrieve(txt):
    opening = open(txt, 'r')
    lst = []
    for x in opening.read():
        if x == '1':
            lst.append(1)
        elif x == '0':
            lst.append(0)
    return lst


# Q G U V
q = retrieve('q.txt')
g = retrieve('g.txt')
u = retrieve('u.txt')
v = retrieve('v.txt')

# Создание меток
y = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
]


plt.imshow(np.array(g).reshape(8, 8))
plt.show()


# преобразование данных и меток в массив

"""
конвертация матрицы 0 и 1 в единый горячий вектор, 
чтобы напрямую передать в сеть, после в list x.
"""

x = [np.array(q).reshape(1, 64), np.array(g).reshape(1, 64),
     np.array(u).reshape(1, 64), np.array(v).reshape(1, 64)]


# преобразование меток в массив
y = np.array(y)
print(x, "\n\n", y)


# функция активации

def sigmoid(x):
    return(1/(1 + np.exp(-x)))


def f_forward(x, w1, w2):
    # hidden
    z1 = x.dot(w1)  # входной слоя 1
    a1 = sigmoid(z1)  # выходной слоя 2

    # выходной слой
    z2 = a1.dot(w2)  # входной выходного слоля
    a2 = sigmoid(z2)  # выходной выходного слоя
    return(a2)

# инициализация весов рандомно


def generate_wt(x, y):
    l = []
    for i in range(x * y):
        l.append(np.random.randn())
    return(np.array(l).reshape(x, y))


def loss(out, Y):
    s = (np.square(out-Y))
    s = np.sum(s)/len(y)
    return(s)

# обратное распределение ошибки


def back_prop(x, y, w1, w2, alpha):

    # hidden layer
    z1 = x.dot(w1)  # входной слоя 1
    a1 = sigmoid(z1)  # выходной слоя 2

    # Output layer
    z2 = a1.dot(w2)  # входной выходного слоя
    a2 = sigmoid(z2)  # выходной выходного слоя
    # ошибка в выходном слое
    d2 = (a2-y)
    d1 = np.multiply((w2.dot((d2.transpose()))).transpose(),
                     (np.multiply(a1, 1-a1)))

    # градиент для w1 и w2
    w1_adj = x.transpose().dot(d1)
    w2_adj = a1.transpose().dot(d2)

    # обновление параметров
    w1 = w1-(alpha*(w1_adj))
    w2 = w2-(alpha*(w2_adj))

    return(w1, w2)


def train(x, Y, w1, w2, alpha=0.01, epoch=10):
    acc = []
    losss = []
    for j in range(epoch):
        l = []
        for i in range(len(x)):
            out = f_forward(x[i], w1, w2)
            l.append((loss(out, Y[i])))
            w1, w2 = back_prop(x[i], y[i], w1, w2, alpha)
        print("эпохи:", j + 1, "======== точность:", (1-(sum(l)/len(x)))*100)
        acc.append((1-(sum(l)/len(x)))*100)
        losss.append(sum(l)/len(x))
    return(acc, losss, w1, w2)


def predict(x, w1, w2):
    Out = f_forward(x, w1, w2)
    maxm = 0
    k = 0
    for i in range(len(Out[0])):
        if(maxm < Out[0][i]):
            maxm = Out[0][i]
            k = i
    if(k == 0):
        print("изображение буквы Q.")
    elif k == 1:
        print("изображение буквы G.")
    elif k == 2:
        print("изображение буквы U.")
    else:
        print('изображение буквы V.')
    plt.imshow(x.reshape(8, 8))
    plt.show()


w1 = generate_wt(64, 8)
w2 = generate_wt(8, 4)
print(w1, "\n\n", w2)

acc, losss, w1, w2 = train(x, y, w1, w2, 0.1, 100)

plt1.plot(acc)
plt1.ylabel('точность:')
plt1.xlabel("эпохи:")
plt1.show()

plt1.plot(losss)
plt1.ylabel('потери:')
plt1.xlabel('эпохи:')
plt1.show()

# веса
print(w1, "\n", w2)

"""
функция предсказания принимает следующие аргументы
    1) матрицу изображения
    2) w1 trained weights
    3) w2 trained weights
"""

predict(x[1], w1, w2)
