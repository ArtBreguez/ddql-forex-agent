import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam

class DDQN_Agent:
    def __init__(self, state_size, is_eval=False, model_name=""):
        self.state_size = state_size
        self.action_size = 3 # hold, buy, sell
        self.memory = deque(maxlen=2000)
        self.buy_inventory = []
        self.sell_inventory = []
        self.bankroll = 10000
        self.buffer_size = 32
        self.is_eval = is_eval
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model_name = model_name
        if is_eval:
            self.model = load_model(model_name)
            self.target_model = load_model(model_name  + "_target")
        else:
            if model_name:
                self.model = self.load_model(model_name)
                self.target_model = self.load_model(model_name  + "_target")
            else:
                self.model = self._build_model()
                self.target_model = self._build_model()

    def load_model(self, model_name):
        try:
            return load_model(model_name)
        except Exception as e:
            print("Error loading model:", str(e))
            return None


    def _build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam())
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if not self.is_eval and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self):
        minibatch = random.sample(self.memory, self.buffer_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
            self.target_train()
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.target_model.set_weights(target_weights)

    def save(self, name):
        self.model.save_weights(name)
