import matplotlib.pyplot as plt
import numpy as np

from agent_behavior import Params

from sim_agents_t import find_alpha_beta
from sim_agents_t import DynamicAgentSimulation

from img_tools import plot_line_labels

plot_df = plot_line_labels.plot_df

# %%

np.random.seed(10)
np.random.seed(11)


n_j_default, n_g_default, = 2, 2
group_idx = 0

ab_0_ones_all = np.ones((n_g_default, n_j_default, 2), dtype=np.float64)
ab_0_ones = ab_0_ones_all[group_idx]

w_default = np.ones((n_j_default,), dtype=np.float64)
v_default = np.ones((n_j_default,), dtype=np.float64)

h_0_default = ab_0_ones[:, 0] * v_default
ability_default = 0.5 * np.ones((n_j_default,), dtype=np.float64)

# Global and group parameters
params_all_t0 = Params(
    ab_0=ab_0_ones_all,
    delta=0.96,
    v_all=v_default,
    wage=w_default,
)

# number of each agent type in each new period
num_agent_t_g = np.array([100, 100])

sim = DynamicAgentSimulation(
    params_t0=params_all_t0,
    n_agent_t_g=num_agent_t_g,
    belief_update_rule=find_alpha_beta
)

# %%

fig, ax = plt.subplots()

agent_choose_field_0 = sim.n_agent_gj[0].unstack()

plot_df(df=agent_choose_field_0, ax=ax)

plt.show()
