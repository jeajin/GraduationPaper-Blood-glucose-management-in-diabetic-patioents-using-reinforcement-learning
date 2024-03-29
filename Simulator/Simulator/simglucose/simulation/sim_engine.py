import logging
import time
import os
import numpy as np

pathos = True
try:
    from pathos.multiprocessing import ProcessPool as Pool
except ImportError:
    print('You could install pathos to enable parallel simulation.')
    pathos = False

logger = logging.getLogger(__name__)


class SimObj(object):
    def __init__(self,
                 env,
                 controller,
                 sim_time,
                 animate=True,
                 path=None):
        self.env = env
        self.controller = controller
        self.sim_time = sim_time
        self.animate = animate
        self._ctrller_kwargs = None
        self.path = path

    def predSimulate(self):
        ttic = time.time()
        for episode in range(1):
            self.controller.onetimetest = 1
            tic = time.time()
            done = False
            score = 0
            # print('episode {} is start'.format(episode))
            obs, reward, done, info = self.env.reset()
            self.controller.reset(obs, reward, done, info)

            # state = np.reshape(obs.CGM, [1, self.controller.state_size])
            state = obs.CGM
            t = 0
            action = 0
            for pre in range(self.controller.previous_time):
                next_action = self.controller.prepolicy()
                next_obs, reward, done, info = self.env.step(self.controller.get_basal(action))
                self.controller.bg_insulin_append_sample(obs.CGM, action)
                obs = next_obs
                action = next_action

            while self.env.time < self.env.scenario.start_time + self.sim_time and not done:
                if self.animate:
                    self.env.render()

                # get_action
                next_action = self.controller.policy(np.reshape(state, [1, self.controller.state_size]))

                # step
                next_obs, reward, done, info = self.env.step(self.controller.get_basal(action))
                next_state = np.reshape(next_obs.CGM, [1, self.controller.state_size])
                # print("reward", reward)
                # 리플레이 메모리에 샘플 <s, a, r, s'> 저장
                self.controller.append_sample(obs.CGM, action, reward, next_obs.CGM, next_action, done)


                # score += reward
                state = next_state
                obs = next_obs
                action = next_action
                t += 1

                if state < 40:
                    print("Hypoglycemia")
                    done = True
                elif state > 400:
                    print("Hyperglycemia")
                    done = True



            print("time: ", t, "  insulin:", action, "  episode:", episode, "  score:", score, "memory length:",
                  len(self.controller.memory), " BG:", state)
            ttoc = time.time()
        print('Simulation took total {} seconds.'.format(ttoc - ttic))
        print()

    def simulate(self):
        t = 0
        # 초기화
        obs, reward, done, info = self.env.reset()
        tic = time.time()
        # 추출 시작, action(Insulin)을 제공하면 state(Blood Glucose)가 반환
        while self.env.time < self.env.scenario.start_time + self.sim_time and not done:
            t = t+1
            if self.animate:
                self.env.render()
            # 컨트롤러에 따른 Action Pick
            action = self.controller.policy(obs, reward, done, **info)

            # Action 에 따른 환경으로 부터 BG 등을 반환
            obs, reward, done, info = self.env.step(action)
        toc = time.time()
        logger.info('Simulation took {} seconds.'.format(toc - tic))
        print("                                   time is :", t)




    def dqsimulate(self):
        ttic = time.time()
        print("start")
        epi = self.controller.episode
        # scores, episodes = [], []
        for episode in range(epi):
            self.controller.onetimetest = 1
            tic = time.time()
            done = False
            score = 0
            # print('episode {} is start'.format(episode))
            obs, reward, done, info = self.env.reset()
            self.controller.reset(obs, reward, done, info)

            # state = np.reshape(obs.CGM, [1, self.controller.state_size])
            state = obs.CGM
            t = 0
            action = 0
            for pre in range(self.controller.previous_time):
                next_action = self.controller.prepolicy()
                next_obs, reward, done, info = self.env.step(self.controller.get_basal(action))
                self.controller.bg_insulin_append_sample(obs.CGM, action)
                obs = next_obs
                action = next_action

            while self.env.time < self.env.scenario.start_time + self.sim_time and not done:
                if self.animate:
                    self.env.render()

                # get_action
                next_action = self.controller.policy(np.reshape(state, [1, self.controller.state_size]))

                # step
                next_obs, reward, done, info = self.env.step(self.controller.get_basal(action))
                next_state = np.reshape(next_obs.CGM, [1, self.controller.state_size])
                # print("reward", reward)
                # 리플레이 메모리에 샘플 <s, a, r, s'> 저장
                self.controller.append_sample(obs.CGM, action, reward, next_obs.CGM, next_action, done)

                # 매 타임스텝마다 학습
                if len(self.controller.memory) >= self.controller.train_start:
                    self.controller.train_model()

                # score += reward
                state = next_state
                obs = next_obs
                action = next_action
                t += 1

                if state < 40:
                    # print("Hypoglycemia")
                    # score -= 5
                    done = True
                elif state > 400:
                    # print("Hyperglycemia")
                    # score -= 3
                    done = True

                if done:
                    # 각 에피소드마다 타깃 모델을 모델의 가중치로 업데이트
                    self.controller.update_target_model()
                    # 에피소드마다 학습 결과 출력
                    # scores.append(score)
                    # episodes.append(episode)
                    toc = time.time()
                    # print(self.controller.modelName, " is model name")
                    if self.controller.modelName == 'c':
                        self.controller.model.save_weights("c800.h5")
                        print("save model weight CNN")
                    elif self.controller.modelName == 'g':
                        self.controller.model.save_weights("g800.h5")
                        print("save model weight GRU")
                    # print('Simulation took episode {} seconds.'.format(toc - tic))


            print("time: ", t, "  insulin:", action, "  episode:", episode, "  score:", score, "memory length:",
                  len(self.controller.memory), " BG:", state)
            ttoc = time.time()
        print('Simulation took total {} seconds.'.format(ttoc - ttic))
        print()

    def results(self):
        return self.env.show_history()

    def save_results(self):
        df = self.results()
        if not os.path.isdir(self.path):
            os.makedirs(self.path)
        filename = os.path.join(self.path, str(self.env.patient.name) + '.csv')
        df.to_csv(filename)

    def reset(self):
        self.env.reset()
        self.controller.reset()


def sim(sim_object):
    # print("Process ID: {}".format(os.getpid()))
    # print('Simulation starts ...')
    if sim_object.controller.name == 'dqn':
        # print("start dqn train simulator")
        sim_object.dqsimulate()
    elif sim_object.controller.name == 'dqnpred':
        print("start dqn predict simulator")
        sim_object.predSimulate()
    else:
        print("start basic simulator")
        sim_object.simulate()
    sim_object.save_results()
    # print('Simulation Completed!')
    return sim_object.results()


def batch_sim(sim_instances, parallel=False):
    tic = time.time()
    if parallel and pathos:
        with Pool() as p:
            results = p.map(sim, sim_instances)
    else:
        if parallel and not pathos:
            print('Simulation is using single process even though parallel=True.')
        results = [sim(s) for s in sim_instances]
    toc = time.time()
    print('Simulation took {} sec.'.format(toc - tic))
    return results
