import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Global variables
phase = "train"  # phase can be set to either "train" or "eval"

""" 
You are allowed to change the names of function "arguments" as per your convenience, 
but it should be meaningful.

E.g. y, y_train, y_test, output_var, target, output_label, ... are acceptable
but abc, a, b, etc. are not.

"""


def get_features(file_path):
    # Given a file path , return feature matrix and target labels
    car = pd.read_csv(file_path)
    samplecount = car.shape[0]
    car1 = pd.read_csv('df_train.csv')
    car2 = pd.read_csv('df_test.csv')
    car = pd.concat([car, car1, car2], axis=0, ignore_index=True)
    car = car.drop('Index', axis=1)
    car['company'] = car['name'].apply(lambda x: x.split(" ")[0])
    car = car.drop('name', axis=1)
    car['mileage'] = car['mileage'].astype('category')
    car['mileage'] = car['mileage'].apply(lambda x: x.split(" ")[0])
    car['mileage'] = car['mileage'].astype('float')
    count = 0
    for i in car['fuel']:
        if i == 'LPG':
            car.loc[count, 'mileage'] *= 0.001898
            # density of LPG=0.001898 kg/L from google
        elif i == 'CNG':
            car.loc[count, 'mileage'] *= 0.1282
            # density of CNG=0.1282 kg/L from google
        count += 1
    car['torque'] = car.torque.str.split('([!--:-~ ])', expand=True)
    car['torque'] = car['torque'].astype('float')
    count = 0
    for i in car['fuel']:
        if i == 'LPG':
            car.loc[count, 'torque'] *= 9.8
            # gravitational acceleration = 9.8 m/s2
        count += 1
    car = car.drop('torque', axis=1)
    car['year'] = car['year'].astype('object')
    car['engine'] = car['engine'].astype('category')
    car['engine'] = car['engine'].apply(lambda x: x.split(" ")[0])
    car['engine'] = car['engine'].astype('float')
    car_object = car.select_dtypes(include=['object'])
    car_object1 = pd.get_dummies(car_object, drop_first=True)
    car = pd.concat([car, car_object1], axis=1)
    car = car.drop(list(car_object), axis=1)
    car = car.fillna(value=car.mean())
    x, y = car.drop('selling_price', axis=1), car['selling_price']
    for i in x:
        x[i] = (x[i] - x[i].mean()) / x[i].std()
    x = x.iloc[:samplecount, :]
    y = y.iloc[:samplecount]
    phi = x.to_numpy()
    y = y.to_numpy()
    phi = np.c_[np.ones((len(y), 1)), phi]

    return phi, y


def get_features_basis(file_path):
    # Given a file path , return feature matrix and target labels
    phi, y = get_features(file_path)
    phi[1] = np.square(abs(phi[1]))
    phi[4] = np.square(abs(phi[4]))
    phi[5] = np.log(1+(abs(phi[5])))
    phi[:, -30:] = np.log(1+abs(phi[:, -30:]))
    return phi, y


def compute_RMSE(phi, w, y):
    # Root Mean Squared Error
    error = (np.sum(((phi.dot(w)) - y) ** 2)/len(y))**0.5
    return error


def generate_output(phi_test, w):
    # writes a file (output.csv) containing target variables in required format for Submission.
    target = phi_test.dot(w)
    Id = np.arange(0, len(target))
    target = pd.DataFrame(abs(target))
    Id = pd.DataFrame(Id)
    output = pd.concat((Id, target), axis=1)
    output.columns = ['Id', 'Expected']
    return output.to_csv('output.csv', index=False)


def closed_soln(phi, y):
    # Function returns the solution w for Xw=y.
    return np.linalg.pinv(phi).dot(y)


def gradient_descent(phi, y, phi_dev, y_dev):
    # Implement gradient_descent using Mean Squared Error Loss
    # You may choose to use the dev set to determine point of convergence
    RMSE_min = 1e+10
    flag = 0
    learning_rate = 0.46
    w_save = w = np.random.randn(len(phi[0]), )
    while True:
        gradient = phi.T.dot((phi.dot(w)) - y) / len(y)
        w = w - learning_rate * gradient
        flag += 1
        new_RMSE = compute_RMSE(phi_dev, w, y_dev)
        if RMSE_min > new_RMSE:
            RMSE_min = new_RMSE
            w_save = w
            flag = 0
        if flag > 500:
            w = w_save
            break
    return w


def sgd(phi, y, phi_dev, y_dev):
    RMSE_min = 1e+10
    flag = 0
    learning_rate = 0.167
    w_save = w = gradient = np.zeros(phi.shape[1])
    while True:
        flag += 1
        for i in range(phi.shape[0]):
            gradient = phi[i]*((phi[i].dot(w)) - y[i])/phi.shape[0]
            w = w - learning_rate * gradient
        new_RMSE = compute_RMSE(phi_dev, w, y_dev)
        if RMSE_min > new_RMSE:
            RMSE_min = new_RMSE
            w_save = w
            flag = 0
        if flag > 500:
            w = w_save
            break

    return w


def pnorm(phi, y, phi_dev, y_dev, p):
    # Implement gradient_descent with p-norm regularisation using Mean Squared Error Loss
    # You may choose to use the dev set to determine point of convergence
    RMSE_min = 1e+10
    flag = 0
    learning_rate = 0.46
    w_save = w = np.random.randn(len(phi[0]), )
    Lambda = 1e-16
    while True:
        gradient = Lambda * p * \
            ((w.T.dot(w)) ** (p / 2 - 1)) * w / 2 + \
            phi.T.dot((phi.dot(w)) - y) / len(y)
        w = w - learning_rate * gradient
        flag += 1
        new_RMSE = compute_RMSE(phi_dev, w, y_dev)
        if RMSE_min > new_RMSE:
            RMSE_min = new_RMSE
            w_save = w
            flag = 0
        if flag > 500:
            w = w_save
            break
    return w


def plotRMSEdata(phi, y, phi_dev, y_dev):
    x_axis = [2000, 2500, 3000, 4500]
    w1 = gradient_descent(phi[:2000, :], y[:2000],
                          phi_dev[:2000, :], y_dev[:2000])
    w2 = gradient_descent(phi[:2500, :], y[:2500],
                          phi_dev[:2500, :], y_dev[:2500])
    w3 = gradient_descent(phi[:3000, :], y[:3000],
                          phi_dev[:3000, :], y_dev[:3000])
    w = gradient_descent(phi, y, phi_dev, y_dev)
    y_axis = [compute_RMSE(phi_dev, w1, y_dev), compute_RMSE(
        phi_dev, w2, y_dev), compute_RMSE(phi_dev, w3, y_dev), compute_RMSE(phi_dev, w, y_dev)]
    plt.plot(x_axis, y_axis)
    plt.xlabel('Size of the training set')
    plt.ylabel('RMSE on development set')
    return


def main():
    """
    The following steps will be run in sequence by the autograder.
    """
    ######## Task 1 #########
    phase = "train"
    phi, y = get_features('df_train.csv')
    phase = "eval"
    phi_dev, y_dev = get_features('df_val.csv')
    w1 = closed_soln(phi, y)
    w2 = gradient_descent(phi, y, phi_dev, y_dev)
    r1 = compute_RMSE(phi_dev, w1, y_dev)
    r2 = compute_RMSE(phi_dev, w2, y_dev)
    print('1a: ')
    print(abs(r1-r2))
    w3 = sgd(phi, y, phi_dev, y_dev)
    r3 = compute_RMSE(phi_dev, w3, y_dev)
    print('1c: ')
    print(abs(r2-r3))

    ######## Task 2 #########
    w_p2 = pnorm(phi, y, phi_dev, y_dev, 2)
    w_p4 = pnorm(phi, y, phi_dev, y_dev, 4)
    r_p2 = compute_RMSE(phi_dev, w_p2, y_dev)
    r_p4 = compute_RMSE(phi_dev, w_p4, y_dev)
    print('2: pnorm2')
    print(r_p2)
    print('2: pnorm4')
    print(r_p4)

    ######## Task 3 #########
    phase = "train"
    phi_basis, y = get_features_basis('train.csv')
    phase = "eval"
    phi_dev, y_dev = get_features_basis('val.csv')
    w_basis = pnorm(phi_basis, y, phi_dev, y_dev, 2)
    rmse_basis = compute_RMSE(phi_dev, w_basis, y_dev)
    print('Task 3: basis')
    print(rmse_basis)

    phi_test, y_test = get_features('test.csv')
    generate_output(phi_test, w3)

    # print('plot')
    #plotRMSEdata(phi, y, phi_dev, y_dev)


main()
