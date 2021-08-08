from simglucose.controller.pid_ctrller import PIDController
from simglucose.controller.qlearnctrller import QLController
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.controller.dqnlearnctrller import DQNController
from simglucose.simulation.user_interface import simulate

# dqn controller
DQNcontroller = DQNController(state_size=1, action_size=3, episode=3, previous_time=4, model='c')
s = simulate(controller=DQNcontroller, sim_time=48, animate=True, parallel=False, name='', selection=1, seed=0, start_time=0, cgm_selection=2, pump_selection=2)

# basal bolus controller
# controller = BBController()
# s = simulate(controller=controller, sim_time=240, animate=True, parallel=False, name='', selection=1, seed=0, start_time=0, cgm_selection=2, pump_selection=2)

# pid controller
# controller = PIDController()
# s = simulate(controller=controller, sim_time=24, animate=True, parallel=False, name='', selection=1, seed=0,
#             start_time=0, cgm_selection=2, pump_selection=2)


# Bcontroller = BBController()
# s = simulate(controller=Bcontroller, sim_time=24, animate=True, parallel=False, name='', selection=1, seed=0, start_time=0, cgm_selection=2, pump_selection=2)


# Qcontroller = QLController(actions=3)
# s = simulate(controller=Qcontroller)

