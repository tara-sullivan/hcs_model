import numpy as np
import matplotlib.pyplot as plt

import os

from agent_behavior import Params
from agent_behavior import AgentParams
from agent_behavior import Agent

from sim_agent_history import AgentSimulation

from sim_agent_plots import make_sim_plots
from sim_agent_plots import course_history_df
from sim_agent_plots import median_specialize_idx
from sim_agent_plots import fmt_subtitle

from img_tools import tikzplotlib_functions as tplf

# Set image path
dirname = os.path.dirname(__file__)
tex_img_path = os.path.join(dirname, 'fig_tex_code')

# %%
'''
Beta example plots

    * Beta distribution example - men and women with same mean, different var
    * Beta distribution example - how beta changes (in slides)
    * Beta distribution exampel - B(1, 1) vs B(2, 2)
'''

from model import beta_distribution_example

plt.close('all')
beta_distribution_example.main()
plt.close('all')

print('Saved examples of beta distribution.')

# %%
sim_num_default = 10000
# Helpful for debugging purposes
make_group_plot = True
save_tikz = True

n_j_default, n_g_default, = 2, 2
group_idx = 0

ab_0_ones_all = np.ones((n_g_default, n_j_default, 2), dtype=np.float64)
ab_0_ones = ab_0_ones_all[group_idx]

w_default = np.ones((n_j_default,), dtype=np.float64)
v_default = np.ones((n_j_default,), dtype=np.float64)

h_0_default = ab_0_ones[:, 0] * v_default
ability_default = 0.5 * np.ones((n_j_default,), dtype=np.float64)

# Global and group parameters
params = Params(
    ab_0=ab_0_ones_all,
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

alph_list = ['X', 'Y']
alph_dict = {key: alph_list[key] for key in range(len(alph_list))}

# %% Baseline simulation

# Initialize figure for paper
if make_group_plot:
    fig_all, ax_subplots = plt.subplots(3, 2)
else:
    ax_subplots = None
# Will add a caption for the whole figure and individual titles at the end.
group_caption = 'Simulations of simple version of model.'
subplot_titles = ''
ref_name = 'fig:sim'
# node code for making subplot titles

# %%

#####################################
# Baseline simulation - 10000 times #
#####################################
# Create simulation and dataframe
np.random.seed(125)
agent_sim = AgentSimulation(agent=agent, sim_num=sim_num_default)
agent_sim_df = course_history_df(agent_sim)
spec_idx = median_specialize_idx(agent_sim)
# arguments for plotting
plt_title = 'Baseline Simulation'
label_edit = {0: -.08, 1: .03}
ax_loc = [0, 0]
subtitle_id = 'a'

make_sim_plots(figname='simulation_10000',
               sim_df=agent_sim_df,
               save_tikz_code=save_tikz, group_plot=make_group_plot,
               subject_labels={0: 'Field X', 1: 'Field Y'},
               ax_idx=ax_loc, group_ax=ax_subplots,
               plt_title=plt_title, sim_num=agent_sim.sim_num,
               plot_vline=spec_idx,
               xticks=[0, 5, 10, 15, 20], yticks=[0, 0.25, 0.5, 0.75, 1],
               label_edit=label_edit,
               )

if (save_tikz is False) and (make_group_plot is False):
    plt.show()

if make_group_plot:
    group_caption = group_caption + ' Figure (' + subtitle_id + ')' \
        + ' presents the baseline for ' + r'$N = {:,}$'.format(agent_sim.sim_num) \
        + ' simulations;'
    subplot_titles = subplot_titles + tplf.subplot_title(
        ax_loc=ax_loc, ref_name=ref_name, subtitle_id=subtitle_id,
        plt_title=plt_title.replace('Field selection and ', '').capitalize(),
    )
# %%

##################################
# Baseline simulation - 50 times #
##################################
# Create simulation and dataframe
np.random.seed(125)
agent_sim = AgentSimulation(agent=agent, sim_num=50)
agent_sim_df = course_history_df(agent_sim)
spec_idx = median_specialize_idx(agent_sim)

# arguments for plotting
plt_title = 'Baseline Simulation (zoomed in)'
label_edit = {0: -.08, 1: .03}
ax_loc = [0, 1]
subtitle_id = 'b'

if (save_tikz is False) and (make_group_plot is False):
    plt.close('all')

make_sim_plots(figname='simulation_50',
               sim_df=agent_sim_df,
               save_tikz_code=save_tikz, group_plot=make_group_plot,
               subject_labels={0: 'Field X', 1: 'Field Y'},
               ax_idx=ax_loc, group_ax=ax_subplots,
               plt_title=plt_title, sim_num=agent_sim.sim_num,
               plot_vline=spec_idx,
               xticks=[0, 5, 10, 15, 20], yticks=[0, 0.25, 0.5, 0.75, 1],
               label_edit=label_edit,
               )

if (save_tikz is False) and (make_group_plot is False):
    plt.show()

group_caption = group_caption + ' figure (' + subtitle_id + ')' \
    + ' does the same for the first {:,} simulations.'.format(agent_sim.sim_num)
subplot_titles = subplot_titles + tplf.subplot_title(
    ax_loc=ax_loc, ref_name=ref_name, subtitle_id=subtitle_id,
    plt_title=plt_title.replace('Field selection and ', '').capitalize(),
)

# %%

################
# Change wages #
################
np.random.seed(29)

w_delta = w_default.copy() * np.array([1, 1.5])
params_delta = Params(ab_0=ab_0_ones_all, delta=0.96,
                      v_all=v_default, wage=w_delta)
agent_params_delta = AgentParams(
    params=params_delta, group_idx=group_idx,
    h_0=h_0_default, theta=ability_default)
agent_delta = Agent(agent_params_delta)

agent_sim = AgentSimulation(agent=agent_delta, sim_num=sim_num_default)
agent_sim_df = course_history_df(agent_sim)
spec_idx = median_specialize_idx(agent_sim)

# arguments for plotting
plt_title = 'Field selection and wages '
subtitle_dict = {'var_tex': r'w', 'var': w_delta}
label_edit = {0: .03, 1: .03}
ax_loc = [1, 0]
subtitle_id = 'c'

if (save_tikz is False) and (make_group_plot is False):
    plt.close('all')

make_sim_plots(figname='wage_effect',
               sim_df=agent_sim_df,
               save_tikz_code=save_tikz, group_plot=make_group_plot,
               subject_labels={0: 'Field X', 1: 'Field Y'},
               ax_idx=ax_loc, group_ax=ax_subplots,
               plt_title=plt_title, sim_num=agent_sim.sim_num,
               plot_vline=spec_idx,
               xticks=[0, 5, 10, 15, 20], yticks=[0, 0.25, 0.5, 0.75, 1],
               label_edit=label_edit,
               )

if (save_tikz is False) and (make_group_plot is False):
    plt.show()

group_caption = group_caption \
    + ' The remaining figures have ' + r'$N = {:,}$'.format(agent_sim.sim_num) \
    + ' simulations.' \
    + ' Figure (' + subtitle_id + ') repeats the simulations' \
    + ' for ' + fmt_subtitle(subtitle_dict, 'short', ' and') + '.'
subplot_titles = subplot_titles + tplf.subplot_title(
    ax_loc=ax_loc, ref_name=ref_name, subtitle_id=subtitle_id,
    plt_title=plt_title.replace('Field selection and ', '').capitalize(),
)

# %%

##################
# Ability effect #
##################
np.random.seed(29)
ability_delta = np.array([.4, .6])
agent_params_delta = AgentParams(
    params=params, group_idx=group_idx, h_0=h_0_default, theta=ability_delta)
agent_delta = Agent(agent_params_delta)

agent_sim = AgentSimulation(agent=agent_delta, sim_num=sim_num_default)
agent_sim_df = course_history_df(agent_sim)
spec_idx = median_specialize_idx(agent_sim)

# arguments for plotting
plt_title = 'Field selection and ability to succeed'
subtitle_dict = {'var_tex': r'\theta', 'var': ability_delta}
label_edit = {0: .03, 1: .03}
ax_loc = [1, 1]
subtitle_id = 'd'

if (save_tikz is False) and (make_group_plot is False):
    plt.close('all')

make_sim_plots(figname='ability_effect',
               sim_df=agent_sim_df,
               save_tikz_code=save_tikz, group_plot=make_group_plot,
               subject_labels={0: 'Field X', 1: 'Field Y'},
               ax_idx=ax_loc, group_ax=ax_subplots,
               plt_title=plt_title, sim_num=agent_sim.sim_num,
               plot_vline=spec_idx,
               xticks=[0, 5, 10, 15, 20], yticks=[0, 0.2, 0.4, 0.6, 0.8, 1],
               label_edit=label_edit,
               )

if (save_tikz is False) and (make_group_plot is False):
    plt.show()

group_caption = group_caption \
    + ' Figure (' + subtitle_id + ') repeats the simulations when ' \
    + fmt_subtitle(subtitle_dict, 'short', ' and') + '.'
subplot_titles = subplot_titles + tplf.subplot_title(
    ax_loc=ax_loc, ref_name=ref_name, subtitle_id=subtitle_id,
    plt_title=plt_title.replace('Field selection and ', '').capitalize(),
)

# %%
##########################
# Change initial beliefs #
##########################
np.random.seed(519)

ab_0_all_delta = ab_0_ones_all.copy()
ab_0_all_delta[group_idx, 1, :] = ab_0_all_delta[group_idx, 1, :] * 2
ab_0_delta = ab_0_all_delta[group_idx]

h_0_delta = ab_0_delta [:, 0] * v_default
ability_default = 0.5 * np.ones((n_j_default,), dtype=np.float64)

params_delta = Params(ab_0=ab_0_all_delta, delta=0.96,
                      v_all=v_default, wage=w_default)
agent_params_delta = AgentParams(
    params=params_delta, group_idx=group_idx,
    h_0=h_0_delta, theta=ability_default)
agent_delta = Agent(agent_params_delta)

agent_sim = AgentSimulation(agent=agent_delta, sim_num=sim_num_default)
agent_sim_df = course_history_df(agent_sim)
spec_idx = median_specialize_idx(agent_sim)

# arguments for plotting
plt_title = 'Field selection and initial beliefs'
subtitle_dict = {'var_tex': 'ab_0', 'var': ab_0_delta}
label_edit = {}
ax_loc = [2, 0]
subtitle_id = 'e'

if (save_tikz is False) and (make_group_plot is False):
    plt.close('all')

make_sim_plots(figname='belief_effect',
               sim_df=agent_sim_df,
               save_tikz_code=save_tikz, group_plot=make_group_plot,
               subject_labels={0: 'Field X', 1: 'Field Y'},
               ax_idx=ax_loc, group_ax=ax_subplots,
               plt_title=plt_title, sim_num=agent_sim.sim_num,
               plot_vline=spec_idx,
               xticks=[0, 5, 10, 15, 20], yticks=[0, 0.2, 0.4, 0.6, 0.8, 1],
               label_edit=label_edit,
               )

if (save_tikz is False) and (make_group_plot is False):
    plt.show()

group_caption = group_caption \
    + ' Figure (' + subtitle_id + ') repeats the simulations when ' \
    + fmt_subtitle(subtitle_dict, 'short', ' and') + '.'
subplot_titles = subplot_titles + tplf.subplot_title(
    ax_loc=ax_loc, ref_name=ref_name, subtitle_id=subtitle_id,
    plt_title=plt_title.replace('Field selection and ', '').capitalize(),
)

# %% Save group plots
###################
# Save group plot #
###################

ax_subplots[2, 1].axis('off')

tplf.save_subplots(
    filepath=os.path.join(tex_img_path, 'sim_plots.tex'),
    figure=fig_all, xlabel_loc='right',
    node_code=subplot_titles, caption=group_caption,
)

plt.close('all')
