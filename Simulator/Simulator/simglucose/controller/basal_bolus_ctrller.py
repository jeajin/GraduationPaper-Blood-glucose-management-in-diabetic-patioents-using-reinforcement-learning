from .base import Controller
from .base import Action
import numpy as np
import pandas as pd
import pkg_resources
import logging

logger = logging.getLogger(__name__)
CONTROL_QUEST = pkg_resources.resource_filename(
    'simglucose', 'params/Quest.csv')
PATIENT_PARA_FILE = pkg_resources.resource_filename(
    'simglucose', 'params/vpatient_params.csv')


class BBController(Controller):
    """
    This is a Basal-Bolus Controller that is typically practiced by a Type-1
    Diabetes patient. The performance of this controller can serve as a
    baseline when developing a more advanced controller.
    """
    def __init__(self, target=140):
        self.quest = pd.read_csv(CONTROL_QUEST)
        self.patient_params = pd.read_csv(
            PATIENT_PARA_FILE)
        self.target = target
        self.patient_info = {"child#001": { "CR": 28.62, "CF": 103.02, "Age": 9, "TDI":17.47},
                            "child#002": { "CR": 27.51, "CF": 99.02, "Age": 9, "TDI": 18.18},
                            "child#003": { "CR": 31.21, "CF": 112.35, "Age": 8, "TDI": 16.02},
                            "child#004": { "CR": 25.23, "CF": 90.84, "Age": 12, "TDI": 19.82},
                            "child#005": { "CR": 12.21, "CF": 43.97, "Age": 10, "TDI": 40.93},
                            "child#006": { "CR": 24.72, "CF": 89.00, "Age": 8, "TDI": 20.22},
                            "child#007": { "CR": 13.81, "CF": 49.71, "Age": 9, "TDI": 36.21},
                            "child#008": { "CR": 23.26, "CF": 83.74, "Age": 10, "TDI": 21.49},
                            "child#009": { "CR": 28.75, "CF": 103.48, "Age": 7, "TDI": 17.39},
                            "child#010": { "CR": 24.21, "CF": 87.16, "Age": 12, "TDI": 20.65},
                            "adolescent#001": { "CR": 13.61, "CF": 49.00, "Age": 18, "TDI": 36.73},
                            "adolescent#002": { "CR": 8.06, "CF": 29.02, "Age": 19, "TDI": 62.03},
                            "adolescent#003": { "CR": 20.62, "CF": 74.25, "Age": 15, "TDI": 24.24},
                            "adolescent#004": { "CR": 14.18, "CF": 51.06, "Age": 17, "TDI": 35.25},
                            "adolescent#005": { "CR": 14.70, "CF": 52.93, "Age": 16, "TDI": 34.00},
                            "adolescent#006": { "CR": 10.08, "CF": 36.30, "Age": 14, "TDI": 49.58},
                            "adolescent#007": { "CR": 11.46, "CF": 41.25, "Age": 16, "TDI": 43.64},
                            "adolescent#008": { "CR": 7.89, "CF": 28.40, "Age": 14, "TDI": 63.39},
                            "adolescent#009": { "CR": 20.77, "CF": 74.76, "Age": 19, "TDI": 24.08},
                            "adolescent#010": { "CR": 15.07, "CF": 54.26, "Age": 17, "TDI": 33.17},
                            "adult#001": { "CR": 9.92, "CF": 35.70, "Age": 61, "TDI": 50.42},
                            "adult#002": { "CR": 8.64, "CF": 31.10, "Age": 65, "TDI": 57.87},
                            "adult#003": { "CR": 8.86, "CF": 31.90, "Age": 27, "TDI": 56.43},
                            "adult#004": { "CR": 14.79, "CF": 53.24, "Age": 66, "TDI": 33.81},
                            "adult#005": { "CR": 7.32, "CF": 26.35, "Age": 52, "TDI": 68.32},
                            "adult#006": { "CR": 8.14, "CF": 29.32, "Age": 26, "TDI": 61.39},
                            "adult#007": { "CR": 11.90, "CF": 42.85, "Age": 35, "TDI": 42.01},
                            "adult#008": { "CR": 11.69, "CF": 42.08, "Age": 48, "TDI": 42.78},
                            "adult#009": { "CR": 7.44, "CF": 26.78, "Age": 68, "TDI": 67.21},
                            "adult#010": { "CR": 7.76, "CF": 27.93, "Age": 68, "TDI": 64.45}}
        self.CR = 1/15
        self.CF = 1/50
        self.Age = 30
        self.TDI = 50

        self.params = 0
        self.name = 'bb'

        self.bbquest = pd.DataFrame([['Average', self.CR , self.CF, self.TDI, self.Age]], columns=['Name', 'CR', 'CF', 'TDI', 'Age'])
        self.u2ss = 1.43  # unit: pmol/(L*kg)
        self.BW = 57.0  # unit: kg

        self.basal = self.u2ss * self.BW / 6000

        self.env_sample_time = 0.5

    def set_parameter(self, patient_name):
        print("start patient: ", patient_name)
        self.CR = self.patient_info[patient_name]['CR']
        self.CF = self.patient_info[patient_name]['CF']
        self.Age = self.patient_info[patient_name]['Age']
        self.TDI = self.patient_info[patient_name]['TDI']

        self.bbquest = pd.DataFrame([['Average', self.CR , self.CF, self.TDI, self.Age]], columns=['Name', 'CR', 'CF', 'TDI', 'Age'])
        self.params = self.patient_params[self.patient_params.Name.str.match(
            patient_name)]

        self.u2ss = self.params.u2ss.values.item()  # unit: pmol/(L*kg)
        self.BW = self.params.BW.values.item()
        self.basal = self.u2ss * self.BW / 6000

        print("self.basal ", self.basal)

    def policy(self, observation, reward, done, **kwargs):
        # sample_time = kwargs.get('sample_time', 1) 0.5
        # print(sample_time)
        # pname = kwargs.get('patient_name')
        meal = kwargs.get('meal')
        # action = self._bb_policy(
        #     meal,
        #     observation.CGM)
        return Action(basal=self.basal, bolus=0)
        # return action

    def _bb_policy(self, meal, glucose):
        # if any(self.quest.Name.str.match(name)):
        #     bbquest = self.quest[self.quest.Name.str.match(name)]
        #     params = self.patient_params[self.patient_params.Name.str.match(
        #         name)]
        #     u2ss = params.u2ss.values.item()  # unit: pmol/(L*kg)
        #     BW = params.BW.values.item()      # unit: kg
        # else:
        #     bbquest = pd.DataFrame([['Average', 1 / 15, 1 / 50, 50, 30]],
        #                      columns=['Name', 'CR', 'CF', 'TDI', 'Age'])
        #     u2ss = 1.43   # unit: pmol/(L*kg)
        #     BW = 57.0     # unit: kg

        if meal > 0:
            # logger.info('Calculating bolus ...')
            # logger.debug('glucose = {}'.format(glucose))
            bolus = (meal / self.bbquest.CR.values + (glucose > 150)
                    * (glucose - self.target) / self.bbquest.CF.values).item()  # unit: U
            bolus = bolus / self.env_sample_time  # unit: U/min
        else:
            bolus = 0 # unit: U
        # This is to convert bolus in total amount (U) to insulin rate (U/min). 
        # The simulation environment does not treat basal and bolus
        # differently. The unit of Action.basal and Action.bolus are the same
        # (U/min).
        # bolus = bolus / env_sample_time  # unit: U/min
        return Action(basal=self.basal, bolus=bolus)
        """
        Helper function to compute the basal and bolus amount.

        The basal insulin is based on the insulin amount to keep the blood
        glucose in the steady state when there is no (meal) disturbance. 
               basal = u2ss (pmol/(L*kg)) * body_weight (kg) / 6000 (U/min)

        The bolus amount is computed based on the current glucose level, the
        target glucose level, the patient's correction factor and the patient's
        carbohydrate ratio.
               bolus = ((carbohydrate / carbohydrate_ratio) + 
                       (current_glucose - target_glucose) / correction_factor)
                       / sample_time
        NOTE the bolus computed from the above formula is in unit U. The
        simulator only accepts insulin rate. Hence the bolus is converted to
        insulin rate.
        """
    def reset(self):
        pass
