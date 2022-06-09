import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agent_behavior import Params

from sim_agents_t import find_alpha_beta
from sim_agents_t import DynamicAgentSimulation

from img_tools import plot_line_labels
from img_tools import tikzplotlib_functions as tplf
import tikzplotlib as tpl

# Set image path
import os

dirname = os.path.dirname(__file__)
tex_img_path = os.path.join(dirname, 'fig_tex_code')

plot_df = plot_line_labels.plot_df
# %%
np.random.seed(10)

# Dictionary of fields
field_dict_all = {0: 'Field X',
                  1: 'Field Y'}
group_dict_all = {0: 'Women', 1: 'Men'}

n_j_default = len(field_dict_all.keys())
n_g_default = len(field_dict_all.keys())

ab_0_ones_all = np.ones((n_g_default, n_j_default, 2), dtype=np.float64)
# ab_0_ones_all[0, 0, :] = [1, 2]

w_default = np.ones((n_j_default,), dtype=np.float64)
# w_default[0] = 1.5
v_default = np.ones((n_j_default,), dtype=np.float64)

# Global and group parameters
params_all_t0 = Params(
    ab_0=ab_0_ones_all,
    delta=0.96,
    v_all=v_default,
    wage=w_default,
    group_dict=group_dict_all,
    field_dict=field_dict_all
)

# number of each agent type in each new period
h_t0_gj = params_all_t0.ab_0[:, :, 0] * params_all_t0.v_all
ability_gj = 0.5 * np.ones_like(h_t0_gj)

num_agent_t_g = np.array([100, 100])

sim = DynamicAgentSimulation(
    params_t0=params_all_t0,
    n_agent_t_g=num_agent_t_g,
    belief_update_rule=find_alpha_beta,
    h0_t0_gj=h_t0_gj,
    theta_gj=ability_gj,
    t_periods=20
)

# %%
fig, ax = plt.subplots()

j_idx = 0
# Dataframe with t on x-axis, group for columns, for j = j_idx
agent_choose_field_j = sim.n_agent_gj[j_idx].unstack('g')
# Convert into a fraction
agent_choose_field_j = agent_choose_field_j / num_agent_t_g

plt_title = f'Fraction choosing {sim.params_t0.field_dict[j_idx]}'
plot_df(
    df=agent_choose_field_j, ax=ax,
    col_labels=sim.params_t0.group_dict,
    title=plt_title,
    x_lim=25,
    x_title='Cohort $\\tau$',
    label_edit = {0: .05, 1: -.08}
)

ax.set_ylim(-.05, 1.05)
# ax.set_xlabel('')
#
# tpl.clean_figure()
# tpl.save(tex_img_path + '/agent_sim_t.tex')
# plt.show()
tplf.save_subplots(
    tex_img_path + '/agent_sim_t.tex',
    # figure=fig,
    # node_code=subplot_titles, caption=group_caption,
    # height=size.h(1.25), width=size.w(1.05),
    clean_figure=True,
    # extra_tikzpicture_parameters={
    #     'every node/.style={font=\\footnotesize}',
    #     'align=left'
    # },
)
