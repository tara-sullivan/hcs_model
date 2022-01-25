import numpy as np
import pandas as pd
import time

from agent_behavior import Params
from agent_behavior import AgentParams
from agent_behavior import Agent

# %% Default values for the simulation
sim_num_default = 10000

n_j_default, n_g_default, = 2, 2
group_idx = 0

ab_0_all_default = np.ones((n_g_default, n_j_default, 2), dtype=np.float64)
ab_0_default = ab_0_all_default[0]

w_default = np.ones((n_j_default,), dtype=np.float64)
v_default =np.ones((n_j_default,), dtype=np.float64)

h_0_default = ab_0_default[:, 0] * v_default
ability_default = 0.5 * np.ones((n_j_default,), dtype=np.float64)


# %%
# if __name__ == '__main__':


# %%

class AgentSimulation:
    """
    Simulate a course history for an agent.
    """
    def __init__(self,
                 agent: Agent,
                 sim_num: int=10):
        # make notation easier
        self.agent=agent
        self.sim_num=sim_num

        sim_results = self.run_sim()

        self.specialize_idx = sim_results['specialize_idx']
        self.chosen_field = sim_results['chosen_field']
        self.chosen_field_state = sim_results['chosen_field_state']
        self.course_history = sim_results['course_history']

    def run_sim(self):
        # Create arrays
        specialize_idx = np.empty(self.sim_num)
        chosen_field = np.empty(self.sim_num)
        chosen_field_state = np.empty((self.sim_num, 2))
        course_history_list = []

        print('Simulating agents...')
        start = time.time()
        for i in range(self.sim_num):
            history = self.agent.find_agent_history()

            # Fill in result arrays
            chosen_field[i] = history['chosen_field']
            chosen_field_state[i] = history['chosen_field_state']
            specialize_idx[i] = history['specialize_idx']
            # Create a list of dataframes; quicker to concatenate later
            course_history_list.append(
                pd.DataFrame(list(zip(history['course_tl'], history['outcome_tl'])),
                             columns=['subject', 'outcome']))
        # concatenate dataframes for full course history
        course_history = pd.concat(course_history_list,
                                   keys=range(self.sim_num),
                                   names=['student', 't'])
        end = time.time()
        print(f'Time elapsed: {end - start:,.2f} s')

        return {
            'chosen_field': chosen_field,
            'course_history': course_history,
            'chosen_field_state': chosen_field_state,
            'specialize_idx': specialize_idx
        }

# %%

if __name__ == '__main__':
    # Global and group parameters
    params = Params(
        ab_0=ab_0_all_default,
        delta=0.96,
        v_all=v_default,
        wage=w_default,
    )

    agent_params_default = AgentParams(
        params=params,
        group_idx=group_idx,
        h_0=h_0_default,
        theta=ability_default
    )

    agent = Agent(agent_params_default)

    sim = AgentSimulation(agent, sim_num=100)

    def print_i(idx):
        print('True ability: ' + str(sim.agent.theta))
        print('Course History: ')
        print(sim.course_history.loc[idx])
        count = np.unique(sim.course_history.loc[idx, 'subject'],
                          return_counts=True)
        print('Subject {j} courses: {c}'.format(j=count[0], c=count[1]))
        print('Final state: ' + str(sim.chosen_field_state[idx]))
        print('Specialize index: ' + str(sim.specialize_idx[idx]))

    print_i(0)
