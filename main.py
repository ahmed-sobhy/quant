from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import warnings
warnings.filterwarnings("ignore")

import datetime  # For datetime objects
import os.path  # To manage paths
import sys  # To find out the script name (in argv[0])


# Import the backtrader platform
import backtrader as bt
from collections import deque
import numpy as np
import random
from pathlib import Path




from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization
from keras.optimizers import Adam
from keras.models import load_model


EPSILON = 1.0


# Deep Q-learning Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory_profit = deque(maxlen=5000)
        self.memory_loss = deque(maxlen=5000)
        self.memory_nothing = deque(maxlen=5000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.a = []

        my_file = Path("my_model.h5")
        if my_file.is_file():
            print("Load")
            self.model = load_model('my_model.h5')
        else:
            print("Create")
            self.model = self._build_model()


    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(BatchNormalization())
        model.add(LSTM(self.state_size ,activation='relu', input_shape = (1, self.state_size),return_sequences=True))
        model.add(BatchNormalization())
        model.add(LSTM(self.state_size, activation='relu', return_sequences=True))
        model.add(BatchNormalization())
        model.add(Dense(self.state_size, activation='relu'))
        model.add(Dense(self.action_size, activation='relu'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state):
        if action == 1:
            if reward >= 0:
                self.memory_profit.append((state, action, reward, next_state))
            else:
                self.memory_loss.append((state, action, reward, next_state))
        else:
            self.memory_nothing.append((state, action, reward, next_state))



    def act(self, state):
        global EPSILON
        if np.random.rand() <= EPSILON:
            if np.random.rand() < 0.05:
                return 1
            else:
                return 0

        act_values = self.model.predict(state)
        print(act_values)
        self.a.append(np.argmax(act_values[0]))
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = []
        for i in range(int(batch_size/2)):
            if(len(self.memory_profit) > 0 and i % 2 == 0):
                minibatch.append(random.sample(self.memory_profit, 1)[0])
            if (len(self.memory_loss) > 0  and i % 2 != 0):
                minibatch.append(random.sample(self.memory_loss, 1)[0])
            if(len(self.memory_nothing)>0):
                minibatch.append(random.sample(self.memory_nothing, 1)[0])

        if len(minibatch) == 0:
            return



        for state, action, reward, next_state in minibatch:
            target = reward + self.gamma * \
                       np.amax(self.model.predict(next_state)[0][0])
            target_f = self.model.predict(state)
            target_f[0][0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        global EPSILON
        if EPSILON > self.epsilon_min :
            EPSILON *= self.epsilon_decay


# Create a Stratey
class TestStrategy(bt.Strategy):
    def log(self, txt, dt=None, doprint=False):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None

        self.profit_trades = 0
        self.loss_trades = 0


        self.dqn = DQNAgent(5, 2)

        self.BUY = 1
        self.NOTHING = 0

        self.state = None
        self.next_state = None
        self.action = None
        self.reward = None

        self.tp = 0.0010


    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:
                # update variable
                self.next_state = self.get_state()
                self.reward = (order.executed.price - self.buyprice) * 1000
                self.dqn.remember(self.state, self.action, self.reward, self.next_state)

                if(order.executed.price - self.buyprice > 0):
                    self.profit_trades += 1
                else:
                    self.loss_trades += 1

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            pass

        # Write down: no pending order
        self.order = None

    def notify_trade(self, trade):

        if not trade.isclosed:
            return

    def get_state(self):
        l = []
        for i in range(0, self.dqn.state_size):
            l.append(self.dataclose[-i])
        l = (np.array(l) / l[0]) - 1
        s = np.array(l).reshape((1, 1, self.dqn.state_size)) * 1000
        return s


    def next(self):

        if self.order:
            return

        # Check if we are in the market
        if not self.position:
            self.state = self.get_state()
            self.action = self.dqn.act(self.state)
            if self.action == self.BUY:
                self.order = self.buy()
            else:
                self.dqn.remember(self.state, self.action, -self.tp*1000, self.state)
        else:
            price_diff = self.dataclose[0] - self.buyprice
            if abs(price_diff) >= self.tp:
                self.order = self.sell()

    def stop(self):
        self.dqn.replay(min(1024,len(self.dqn.memory_profit + self.dqn.memory_nothing + self.dqn.memory_loss)))

        self.log('Ending Value %.2f Profit Trades %.2f Loss Trades %0.2f Total Trades %.2f Mean %.2f' %
                 (self.broker.getvalue(),self.profit_trades, self.loss_trades, self.profit_trades+self.loss_trades, np.mean(self.dqn.a)), doprint=True)


        self.dqn.model.save('my_model.h5')
        print(EPSILON)

if __name__ == '__main__':
    # Create a cerebro entity
    cerebro = bt.Cerebro()
    cerebro.addstrategy(TestStrategy)

    # Create a Data Feed
    data = bt.feeds.GenericCSVData(dataname="/Users/ahmed/Desktop/forex_data/DAT_MT_EURUSD_M1_2011.csv",
                                   dtformat=('%Y.%m.%d'),
                                   tmformat=("%H:%M"),
                                   datetime=0,
                                   time=1,
                                   high=2,
                                   low=3,
                                   open=4,
                                   close=5,
                                   volume=6,
                                   openinterest=-1,
                                   timeframe=bt.TimeFrame.Minutes)

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    for i in range(100):


        # Set our desired cash start
        cerebro.broker.setcash(1000.0)

        # Add a FixedSize sizer according to the stake
        cerebro.addsizer(bt.sizers.FixedSize, stake=100)

        # Set the commission
        cerebro.broker.setcommission(commission=0.0)

        # Run over everything
        cerebro.run()








