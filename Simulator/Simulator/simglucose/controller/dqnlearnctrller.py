from .base import Controller
from .base import Action
import numpy as np
import random
from collections import defaultdict
import logging
import pkg_resources
import pandas as pd
from collections import deque
from keras.layers import Dense, GRU, Conv1D, MaxPooling1D, Dropout, Flatten
from keras.optimizers import Adam
from keras.models import Sequential


logger = logging.getLogger(__name__)
CONTROL_QUEST = pkg_resources.resource_filename(
    'simglucose', 'params/Quest.csv')
PATIENT_PARA_FILE = pkg_resources.resource_filename(
    'simglucose', 'params/vpatient_params.csv')


class DQNController(Controller):
    def __init__(self, state_size, action_size, episode, previous_time, model):
        self.state_size = state_size
        self.action_size = action_size
        self.episode = episode
        self.quest = pd.read_csv(CONTROL_QUEST)
        self.patient_params = pd.read_csv(PATIENT_PARA_FILE)

        self.actions = 0
        self.learning_rate = 0.001
        self.discount_factor = 0.99
        self.epsilon = 0.2
        self.epsilon_min = 0.01
        self.batch_size = 128
        self.train_start = 128
        self.basal = 0
        self.epsilon_decay = 0.9992
        self.previous_time = previous_time * 12

        self.time = 0

        self.name = 'dqn'
        self.modelName = ''


        # 리플레이 메모리

        self.bgInsulinMemory = deque(maxlen=self.previous_time)
        self.memory = deque(maxlen=30000)

        self.load_model = False
        # 모델과 타깃 모델 생성
        if model == 'g':
            self.model = self.build_gru_model()
            self.target_model = self.build_gru_model()
            self.epsilon = 0.05
            self.learning_rate = 0.00001
            self.epsilon_min = 1
            self.modelName = 'g'
            if self.load_model:
                self.model.load_weights("g.h5")
                print("load")

        elif model == 'c':
            self.model = self.build_cnn_model()
            self.target_model = self.build_cnn_model()
            self.epsilon = 0.05
            self.learning_rate = 0.001
            self.epsilon_min = 1
            self.modelName = 'c'
            if self.load_model:
                self.model.load_weights("c.h5")
                print("load")
        else:
            self.model = self.build_model()
            self.target_model = self.build_model()


        # 타깃 모델 초기화
        self.update_target_model()

        # 환자 정보
        self.quest = pd.read_csv(CONTROL_QUEST)
        self.patient_params = pd.read_csv(PATIENT_PARA_FILE)

        # self.target = target
        self.onetimetest = 1





    def reset(self, obs, reward, done, info):
        name = info.get('patient_name')

        if any(self.quest.Name.str.match(name)):
            params = self.patient_params[self.patient_params.Name.str.match(
                name)]
            u2ss = params.u2ss.values.item()  # unit: pmol/(L*kg)
            BW = params.BW.values.item()      # unit: kg
        else:
            u2ss = 1.43   # unit: pmol/(L*kg)
            BW = 57.0     # unit: kg

        self.basal = u2ss * BW / 6000  # unit: U/min


    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.previous_time, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(24, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='huber', optimizer=Adam(lr=self.learning_rate))
        return model

    def build_gru_model(self):
        model = Sequential()
        model.add(Dense(128, input_shape=(self.previous_time*self.state_size, 2),
                        activation='relu', kernel_initializer='he_uniform'))
        model.add(GRU(128, return_sequences=True))
        # model.add(GRU(128, return_sequences=True))
        model.add(Flatten())
        model.add(Dense(self.action_size, activation='softmax', kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def build_cnn_model(self):
        model = Sequential()
        print(self.previous_time)
        model.add(Conv1D(32, kernel_size=3, activation='relu',
                         input_shape=(48, 2)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(32, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dense(512, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(self.action_size, activation='softmax',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def prepolicy(self):
        r = random.randrange(self.action_size)
        return r

    def policy(self, state):
        r = 0
        if np.random.rand() <= self.epsilon:
            r = random.randrange(self.action_size)
        else:
            # print("h ", np.reshape(list(self.BGmemory),[1,self.previous_time]))
            # np.reshape(states, (4, 48, 1))
            q_value = self.model.predict(np.reshape(list(self.bgInsulinMemory), (1, 48, 2)))
            # print("q: ", q_value, "shape", q_value.shape)
            r = np.argmax(q_value[0])
            # print("R ", r)
        return r

    def get_basal(self, r):
        if r == 0:
            return Action(basal=0, bolus=0)
        elif r == 1:
            return Action(basal=self.basal, bolus=0)
        elif r == 2:
            return Action(basal=self.basal * 5, bolus=0)

    def bg_insulin_append_sample(self, bg, insulin):
        self.bgInsulinMemory.append([bg, insulin])

    def append_sample(self, bg, action, reward, next_bg, next_action, done):
        self.bg_insulin_append_sample(bg, action)
        self.memory.append((list(self.bgInsulinMemory), action, reward, [next_bg, next_action], done))
        # print("next_state ", next_state, "state ", (list(self.BGmemory)))

    def train_model(self):
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay

        if self.onetimetest == 1:
            self.onetimetest = 0
            print("new episode's self.epsilon", self.epsilon)

        mini_batch = random.sample(self.memory, self.batch_size)
        states = np.zeros((self.batch_size, self.previous_time, 2))
        # print("self.memory :", self.memory)
        next_states = np.zeros((self.batch_size, self.previous_time, 2))
        actions, rewards, dones = [], [], []
        # print(mini_batch[0][3])
        for i in range(self.batch_size):
            states[i] = mini_batch[i][0][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])

        if self.onetimetest == 1:
            self.onetimetest = 0
            # print(states)
            # print(states.shape)

        # # 현재 상태에 대한 모델의 큐함수
        # target = self.model.predict(states)
        # # 다음 상태에 대한 타깃 모델의 큐함수
        # target_val = self.target_model.predict(next_states)

        # 현재 상태에 대한 모델의 큐함수
        target = self.model.predict(np.reshape(states, (self.batch_size, 48, 2)))
        # 다음 상태에 대한 타깃 모델의 큐함수
        target_val = self.target_model.predict(np.reshape(next_states, (self.batch_size, 48, 2)))

        # 벨만 최적 방정식을 이용한 업데이트 타깃
        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.discount_factor * (np.amax(target_val[i]))

        self.model.fit(np.reshape(states, (self.batch_size, 48, 2)), target, batch_size=self.batch_size,
                       epochs=1, verbose=0
                       )


class DqnPredController(Controller):
    def __init__(self, state_size, action_size, episode, previous_time, model):
        self.state_size = state_size
        self.action_size = action_size
        self.episode = episode
        self.quest = pd.read_csv(CONTROL_QUEST)
        self.patient_params = pd.read_csv(PATIENT_PARA_FILE)

        self.actions = 0
        self.learning_rate = 0.001
        self.discount_factor = 0.99
        self.epsilon = 0.2
        self.epsilon_min = 0.01
        self.batch_size = 128
        self.train_start = 128
        self.basal = 0
        self.epsilon_decay = 0.9992
        self.previous_time = previous_time * 12

        self.time = 0

        self.name = 'dqn'


        # 리플레이 메모리

        self.bgInsulinMemory = deque(maxlen=self.previous_time)
        self.memory = deque(maxlen=30000)

        # 모델과 타깃 모델 생성
        if model == 'g':
            self.model = self.build_gru_model()
            self.target_model = self.build_gru_model()
            self.epsilon = 0.05
            self.learning_rate = 0.00001
            self.epsilon_min = 1
        elif model == 'c':
            self.model = self.build_cnn_model()
            self.target_model = self.build_cnn_model()
            self.epsilon = 0.05
            self.learning_rate = 0.001
            self.epsilon_min = 1
        else:
            self.model = self.build_model()
            self.target_model = self.build_model()


        # 타깃 모델 초기화
        self.update_target_model()

        # 환자 정보
        self.quest = pd.read_csv(CONTROL_QUEST)
        self.patient_params = pd.read_csv(PATIENT_PARA_FILE)

        # self.target = target
        self.onetimetest = 1

        self.load_model = True

        if self.load_model:
            self.model.load_weights("teest.h5")
            print("load")

    def reset(self, obs, reward, done, info):
        name = info.get('patient_name')

        if any(self.quest.Name.str.match(name)):
            params = self.patient_params[self.patient_params.Name.str.match(
                name)]
            u2ss = params.u2ss.values.item()  # unit: pmol/(L*kg)
            BW = params.BW.values.item()      # unit: kg
        else:
            u2ss = 1.43   # unit: pmol/(L*kg)
            BW = 57.0     # unit: kg

        self.basal = u2ss * BW / 6000  # unit: U/min


    def build_model(self):

        model = Sequential()
        model.add(Dense(24, input_dim=self.previous_time, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(24, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='huber', optimizer=Adam(lr=self.learning_rate))
        return model

    def build_gru_model(self):

        model = Sequential()
        model.add(Dense(128, input_shape=(self.previous_time*self.state_size, 2),
                        activation='relu', kernel_initializer='he_uniform'))
        model.add(GRU(128, return_sequences=True))
        # model.add(GRU(128, return_sequences=True))
        model.add(Flatten())
        model.add(Dense(self.action_size, activation='softmax', kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def build_cnn_model(self):

        model = Sequential()
        print(self.previous_time)
        model.add(Conv1D(32, kernel_size=3, activation='relu',
                         input_shape=(48, 2)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(32, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dense(512, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(self.action_size, activation='softmax',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def policy(self, state):
        r = 0
        if np.random.rand() <= self.epsilon:
            r = random.randrange(self.action_size)
        else:
            # print("h ", np.reshape(list(self.BGmemory),[1,self.previous_time]))
            # np.reshape(states, (4, 48, 1))
            q_value = self.model.predict(np.reshape(list(self.bgInsulinMemory), (1, 48, 2)))
            # print("q: ", q_value, "shape", q_value.shape)
            r = np.argmax(q_value[0])
            # print("R ", r)
        return r

    def get_basal(self, r):
        if r == 0:
            return Action(basal=0, bolus=0)
        elif r == 1:
            return Action(basal=self.basal, bolus=0)
        elif r == 2:
            return Action(basal=self.basal * 5, bolus=0)

