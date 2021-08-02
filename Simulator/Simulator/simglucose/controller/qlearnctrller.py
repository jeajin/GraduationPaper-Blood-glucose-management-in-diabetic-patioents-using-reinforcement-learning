from .base import Controller
from .base import Action
import numpy as np
import random
from collections import defaultdict
import logging
import pkg_resources
import pandas as pd


logger = logging.getLogger(__name__)
CONTROL_QUEST = pkg_resources.resource_filename(
    'simglucose', 'params/Quest.csv')
PATIENT_PARA_FILE = pkg_resources.resource_filename(
    'simglucose', 'params/vpatient_params.csv')


class QLController(Controller):
    def __init__(self, actions):
        self.quest = pd.read_csv(CONTROL_QUEST)
        self.patient_params = pd.read_csv(PATIENT_PARA_FILE)
        self.actions = actions
        self.learning_rate = 0.01
        self.discount_factor = 0.9
        self.epsilon = 1.1
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0])
        self.basal = -1
        self.tempaction = 0



    def policy(self, observation, reward, done, **kwargs):
        pname = kwargs.get('patient_name')
        if self.basal == -1:
            self.basal = self._bb_policy(pname)
        state = observation.CGM
        if np.random.rand() < self.epsilon:
            # 무작위 행동 반환

            action = np.random.choice(self.actions)
        else:
            # 큐함수에 따른 행동 반환
            state_action = self.q_table[state]
            action = self.arg_max(state_action)
        self.tempaction = action
        if action == 2:
            action = 5
        action = Action(basal=action * self.basal, bolus=0)
        return action


    def arg_max(state_action):
        max_index_list = []
        max_value = state_action[0]
        for index, value in enumerate(state_action):
            if value > max_value:
                max_index_list.clear()
                max_value = vcalue
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return random.choice(max_index_list)

    def _bb_policy(self, name):

        if any(self.quest.Name.str.match(name)):
            quest = self.quest[self.quest.Name.str.match(name)]
            params = self.patient_params[self.patient_params.Name.str.match(
                name)]
            u2ss = params.u2ss.values.item()  # unit: pmol/(L*kg)
            BW = params.BW.values.item()  # unit: kg
        else:
            quest = pd.DataFrame([['Average', 1 / 15, 1 / 50, 50, 30]],
                                 columns=['Name', 'CR', 'CF', 'TDI', 'Age'])
            u2ss = 1.43  # unit: pmol/(L*kg)
            BW = 57.0  # unit: kg

        basal = u2ss * BW / 6000  # unit: U/min

        return basal

    def learn(self, state, action, reward, next_state):
        q_1 = self.q_table[state][self.tempaction]
        # 벨만 최적 방정식을 사용한 큐함수의 업데이트
        q_2 = reward + self.discount_factor * max(self.q_table[next_state])
        self.q_table[state][self.tempaction] += self.learning_rate * (q_2 - q_1)

