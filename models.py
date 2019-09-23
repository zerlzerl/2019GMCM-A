from keras.layers import *
from keras.optimizers import Adam
from keras import Sequential


def baseline_model1(input_dim):
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(input_dim,)))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation='sigmoid'))
    model.add(Dense(1))

    opt = Adam(lr=0.002, beta_1=0.9, beta_2=0.99, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=opt, loss='mse', metrics=['mae'])

    return model


def baseline_model2(input_dim):
    # create model
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
    model.add(Dense(32, kernel_initializer='normal', activation='relu'))
    model.add(Dense(16, kernel_initializer='normal', activation='sigmoid'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    opt = Adam(lr=0.002, beta_1=0.9, beta_2=0.99, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=opt, loss='mse', metrics=['mae'])
    return model