import pandas as pd                         
data = pd.read_csv("Credit_N400_p9.csv")               
print(data.head())  

names = ['income','credit_limit','credit_rating','number_of_cards', 'age', 'education level', 'gender', 'student status', 'marriage status']
dataframe = pd.read_csv("Credit_N400_p9.csv", names=names)
print(names)
array = dataframe.values


X = data['Income']; Y = data['Balance']
X = X.tolist()
Y = Y.tolist()

import matplotlib.pyplot as plt
plt.scatter(X, Y, s = 5)
plt.grid()
plt.xlabel("Income")
plt.ylabel("Balance")
plt.show()


#alpha - learning rate
def gradient_descent(X, Y, w, b, alpha):
 
    dl_dw = 0.0
    dl_db = 0.0
    N = len(X)

    for i in range(N):
        dl_dw += -1*X[i] * (Y[i] - (w*X[i] + b))
        dl_db += -1*(Y[i] - (w*X[i] + b))

    w = w - (1/float(N)) * dl_dw * alpha
    b = b - (1/float(N)) * dl_db * alpha

    return w, b


def cost_function (X, Y, w, b):

    N = len(X)
    total_error = 0.0
    for i in range(N):
        total_error += (Y[i] - (w*X[i] - b))**2

    return total_error / (2*float(N))
    


def train_credit(X, Y, w, b, alpha, n_iter):

    for i in range(n_iter):
        w, b = gradient_descent(X, Y, w, b, alpha)

        if i % 400 == 0:
            print ("iteration:", i, "Balance: ", cost_function(X, Y, w, b))


    return w, b


def BalPredict(x, w, b):
    return x*w + b


w, b = train_credit(X, Y, 0.0, 0.0, 0.0001, 7000)
x_new = 50.0
y_new = BalPredict(x_new, w, b)
print(y_new)