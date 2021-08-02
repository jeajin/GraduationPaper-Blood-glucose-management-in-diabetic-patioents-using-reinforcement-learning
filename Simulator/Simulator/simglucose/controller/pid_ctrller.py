from .base import Controller
from .base import Action
import logging

logger = logging.getLogger(__name__)


class PIDController(Controller):
    def __init__(self, P = 1.58E-04, I = 1.00E-07, D = 1.00E-02, target=140):
        self.P = P
        self.I = I
        self.D = D
        self.target = target
        self.integrated_state = 0
        self.prev_state = 0
        self.total_pid = {"child#001": {"p": 3.49E-05, "i": 1.00E-07, "d": 1.00E-03},
        'child#002':  {"p": 3.98E-05, "i":  2.87E-08, "d": 3.98E-03},
        'child#003':  {"p": 6.31E-05, "i":  1.74E-08, "d": 1.00E-03},
        'child#004':  {"p": 6.31E-05, "i":  1.00E-07, "d": 1.00E-03},
        'child#005':  {"p": 1.00E-04, "i":  2.87E-08, "d": 6.31E-03},
        'child#006':  {"p": 3.49E-05, "i":  1.00E-07, "d": 1.00E-03},
        'child#007':  {"p": 3.98E-05, "i":  6.07E-08, "d": 2.51E-03},
        'child#008':  {"p": 3.49E-05, "i":  3.68E-08, "d": 1.00E-03},
        'child#009':  {"p": 3.49E-05, "i":  1.00E-07, "d": 1.00E-03},
        'child#010':  {"p": 4.54E-06, "i":  3.68E-08, "d": 2.51E-03},
        'adolescent#001':  {"p": 1.74E-04, "i":  1.00E-07, "d": 1.00E-02},
        'adolescent#002':  {"p": 1.00E-04, "i":  1.00E-07, "d": 6.31E-03},
        'adolescent#003':  {"p": 1.00E-04, "i":  1.00E-07, "d": 3.98E-03},
        'adolescent#004':  {"p": 1.00E-04, "i":  1.00E-07, "d": 4.79E-03},
        'adolescent#005':  {"p": 6.31E-05, "i":  1.00E-07, "d": 6.31E-03},
        'adolescent#006':  {"p": 4.54E-10, "i":  1.58E-11, "d": 1.00E-02},
        'adolescent#007':  {"p": 1.07E-07, "i":  6.07E-08, "d": 6.31E-03},
        'adolescent#008':  {"p": 4.54E-10, "i":  4.54E-12, "d": 1.00E-02},
        'adolescent#009':  {"p": 6.31E-05, "i":  1.00E-07, "d": 3.98E-03},
        'adolescent#010':  {"p": 4.54E-10, "i":  4.54E-12, "d": 1.00E-02},
        'adult#001':  {"p": 1.58E-04, "i":  1.00E-07, "d": 1.00E-02},
        'adult#002':  {"p": 3.98E-04, "i":  1.00E-07, "d": 1.00E-02},
        'adult#003':  {"p": 4.54E-10, "i":  1.00E-07, "d": 1.00E-02},
        'adult#004':  {"p": 1.00E-04, "i":  1.00E-07, "d": 3.98E-03},
        'adult#005':  {"p": 3.02E-04, "i":  1.00E-07, "d": 1.00E-02},
        'adult#006':  {"p": 2.51E-04, "i":  2.51E-07, "d": 1.00E-02},
        'adult#007':  {"p": 1.22E-04, "i":  3.49E-07, "d": 2.87E-03},
        'adult#008':  {"p": 1.00E-04, "i":  1.00E-07, "d": 1.00E-02},
        'adult#009':  {"p": 1.00E-04, "i":  1.00E-07, "d": 1.00E-02},
        'adult#010':  {"p": 1.00E-04, "i":  1.00E-07, "d": 1.00E-02}}
        self.name = 'pid'

    def set_parameter(self, patient_name):
        self.P = self.total_pid[patient_name]['p']
        self.I = self.total_pid[patient_name]['i']
        self.D = self.total_pid[patient_name]['d']
        self.integrated_state = 0
        self.prev_state = 0
        print("he start patient name : ", patient_name, "p", self.P, "i",
              self.I, "d", self.D, " =?", self.total_pid[patient_name])


    def policy(self, observation, reward, done, **kwargs):

        sample_time = kwargs.get('sample_time')

        # BG is the only state for this PID controller
        bg = observation.CGM
        # print("bg ", bg)
        control_input = self.P * max([(bg - self.target), 0]) + \
            self.I * self.integrated_state + \
            self.D * abs(bg - self.prev_state) / sample_time

        # logger.info('Control input: {}'.format(control_input))
        # print("control_input :", control_input," max", max([(bg - self.target), 0]), "self.integrated_state", self.integrated_state, "bg - self.prev_state", (bg - self.prev_state))
        # update the states
        self.prev_state = bg
        self.integrated_state += (bg - self.target) * sample_time
        # logger.info('prev state: {}'.format(self.prev_state))
        # logger.info('integrated state: {}'.format(self.integrated_state))

        # return the action
        action = Action(basal=control_input, bolus=0)
        return action



    def reset(self):
        self.integrated_state = 0
        self.prev_state = 0
