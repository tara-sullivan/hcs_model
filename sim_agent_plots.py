import numpy as np
import pandas as pd
import string

import os

import matplotlib.pyplot as plt
import tikzplotlib as tpl

from agent_behavior import Params
from agent_behavior import AgentParams
from agent_behavior import Agent

from sim_agent_history import AgentSimulation

from img_tools.figsize import ArticleSize
from img_tools import plot_line_labels

plot_df = plot_line_labels.plot_df
# get figure height and width
size = ArticleSize()

# Set image path
dirname = os.path.dirname(__file__)
tex_img_path = os.path.join(dirname, 'fig_tex_code')

alph_list = ['X', 'Y']
alph_dict = {key: alph_list[key] for key in range(len(alph_list))}

# %%
if __name__ == '__main__':
    sim_num_default = 10000

    n_j_default, n_g_default, = 2, 2
    group_idx = 0

    ab_0_all_default = np.ones((n_g_default, n_j_default, 2), dtype=np.float64)
    ab_0_all_default[0, 1, :] = ab_0_all_default[0, 1, :] * 2
    ab_0_default = ab_0_all_default[0]

    w_default = np.ones((n_j_default,), dtype=np.float64)
    v_default = np.ones((n_j_default,), dtype=np.float64)

    h_0_default = ab_0_default[:, 0] * v_default
    ability_default = 0.5 * np.ones((n_j_default,), dtype=np.float64)

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
    agent_sim = AgentSimulation(agent, sim_num=sim_num_default)

    alph_list = ['X', 'Y']
    alph_dict = {key: alph_list[key] for key in range(len(alph_list))}


# %%
def course_history_df(sim: AgentSimulation):
    """
    Create dataframe from simulation of agents to plot
    """
    course_history = sim.course_history
    chosen_field = sim.chosen_field

    # For some buggy reason, margins=True doesn't seem to work
    # Create a table that shows the fraction of simulated agents studying
    # a subject at each time period
    df = (course_history.pivot_table(index='t', columns='subject',
                                     aggfunc='count', margins=False
                                     ).fillna(0))
    # remove extraneous column
    df = df['outcome']

    # counts at each time index
    # Last time period for each student
    graduation = (
        course_history.reset_index()
        .pivot_table('t', index='student', aggfunc='max')
    )
    # by field, when did each group exit
    min_t = graduation.groupby(chosen_field).min()
    # the maximum of these min values determines the end of df
    max_idx = min_t.max()[0]
    # eliminate observations after maximum idx
    df = df.loc[:max_idx]
    # replace with NaN for other values
    for subject in min_t.index.tolist():
        if min_t.loc[subject].item() < max_idx:
            grad_t = min_t.loc[subject][0] + 1
            df[subject].loc[grad_t:] = np.nan

    # Create the frequency table
    df = df / df.loc[0, :].sum()

    return df


if __name__ == '__main__':
    agg_sim_df = course_history_df(agent_sim)


# %%
def median_specialize_idx(sim):
    idx = sim.course_history.index.unique(level=0)
    spec_df = pd.DataFrame(
        {'subject': sim.chosen_field,
         'specialize_idx': sim.specialize_idx},
        index=idx)
    spec_df = spec_df.groupby('subject').median()
    return spec_df


if __name__ == '__main__':
    spec_idx = median_specialize_idx(agent_sim)


# %%
def make_sim_plots(
        sim_df,
        save_tikz_code=False,
        group_plot=False,
        plt_title: str = '',
        subject_labels: dict = None,
        sim_num: int=None,
        plot_vline=None,
        figname: str = None,
        ax_idx=None,
        group_ax=None,
        plt_subtitle=False,
        subtitle_dict=None,
        # subtitle_id=None,
        *args, **kwargs):
    """
    Run the plotting code twice; once to save a standalone figure that can
    be used in a presentation, and once to save a combined figure for the
    paper.

    Required:

    * sim_df: dataframe of simulations to plot

    Optional:

    * save_tikz_code: Use tikzplotlib to save code
    * group_plot: if False, only create a standalone figure. If true,
    create a standalone figure and plot a subplot
    * subject labels: dictionary to map subject labels (columns in the
    sim_df dataframe) to different labels.
    * plt_title: title of plot
    * plot_vline: vertial line (i.e. median specializtion index)
    * sim_num: Number of simulations; if not none, labeled on figure

    Required if save_tikz_code True:

    * figname: string to name the figure; required if saving code

    Required if group_plot:
    * group_ax: axes (numpy array of all axes)
    * ax_idx: index of axes for the subplot

    Optional for group_plot:


    * plt_subtitle: optional subtitle of plot
    * subtitle_id: ID for each subtitle (i.e. '(a)')
    * (*args, **kwargs) wraps plot_df signature
    """

    # Create the standalone version of the figure
    # Shared characteristics
    kwargs['x_title'] = 't'
    kwargs['y_title'] = 'Fraction students enrolled in field'
    kwargs['x_lim'] = 30
    if subject_labels is not None:
        kwargs['col_labels'] = subject_labels

    if save_tikz_code:
        assert figname is not None, "Please provide figname"

    if group_plot:
        assert(group_ax is not None and ax_idx is not None), \
            'Please pass group_ax and ax_idx'

    # create the text to add
    if sim_num is not None:
        if sim_num < 999:
            loc_num = 18
        else:
            loc_num = 16
        kwargs['add_text'] = {'add_loc': (loc_num, .03),
                              'add_str': 'N simulations = '
                              + '{:,}'.format(sim_num)}

    # Create the title for the standalone version
    if plt_subtitle is False:
        kwargs['title'] = plt_title
    else:
        kwargs['title'] = plt_title + ' \\\\ ' \
            + fmt_subtitle(subtitle_dict=subtitle_dict, length='long')

    # Initialize standalone figure
    fig_standalone, ax_standalone = plt.subplots()
    # Create standalone figure
    plot_df(df=sim_df, ax=ax_standalone, *args, **kwargs)

    # Initialize objects to plot vertical line at median
    if plot_vline is not None:
        vline_dict = {}
        spec_idx = plot_vline

    # Change some properties after plotting
    for line in ax_standalone.lines:
        # Change line width
        line.set_linewidth(2.0)
        # Add marker
        line.set_marker('x')
        if plot_vline is not None:
            # find the subject
            subject = int(line.get_label())
            vline_dict[subject] = {}
            # find the line color
            line_color = line.get_color()
            vline_dict[subject]['color'] = line_color
            # point where you want to plot vline
            vline = spec_idx.loc[subject].squeeze()
            vline_dict[subject]['idx'] = vline

    # Plot the median values using the same line color as above
    if plot_vline is not None:
        for subject in vline_dict.keys():
            ax_standalone.axvline(
                vline_dict[subject]['idx'],
                linestyle='--', alpha=0.5,
                color=vline_dict[subject]['color']
            )

    # Save standalone version of the graph
    if save_tikz_code:
        img_name = os.path.join(tex_img_path, figname + '.tex')
        # tpl.clean_figure(fig_standalone)
        tpl.save(
            figure=fig_standalone, filepath=img_name,
            axis_height=size.h(1.2), axis_width=size.w(1.3),)
        print('saved ' + figname)
        plt.close(fig_standalone)

    if group_plot is False:
        return
    else:
        # get axes for particular subplot
        subplot_ax = group_ax[ax_idx[0], ax_idx[1]]

        # Only label y axis for graphs on the left
        if ax_idx[1] == 0:
            kwargs['y_title'] = 'Fraction enrolled in field'
        else:
            kwargs['y_title'] = ''

        # Label all x-axes
        kwargs['x_title'] = r'$t$'

        # Create the title for the grouped version
        # if subtitle_id is not None:
        if subtitle_dict is None:
            kwargs['title'] = ''
        elif subtitle_dict is not None:
            kwargs['title'] = fmt_subtitle(subtitle_dict=subtitle_dict,
                                           length='short')

        # Remove labels
        # kwargs['col_labels'] = {k: '' for k, v in
        #                         kwargs['col_labels'].items()}
        # Remove the text 'simulation'
        if sim_num is not None:
            kwargs['add_text']['add_str'] = (kwargs['add_text']['add_str']
                                             .replace('simulations ', ''))
            # Change location of the simulation number
            kwargs['add_text']['add_loc'] = \
                (3, kwargs['add_text']['add_loc'][1])
        # Casually undoing the previous three lines of code because i'm hungry
        kwargs['add_text'] = None
        # Move x axis to the inside
        subplot_ax.tick_params(axis='x', direction='in')

        plot_df(df=sim_df, ax=subplot_ax, *args, **kwargs)

        # Initialize objects to plot vertical line at median
        if plot_vline is not None:
            vline_dict = {}
            spec_idx = plot_vline

        # Change some properties after plotting
        for line in subplot_ax.lines:
            # Change line width
            line.set_linewidth(2.0)
            # Add marker
            line.set_marker('x')
            if plot_vline is not None:
                # find the subject
                subject = int(line.get_label())
                vline_dict[subject] = {}
                # find the line color
                line_color = line.get_color()
                vline_dict[subject]['color'] = line_color
                # point where you want to plot vline
                vline = spec_idx.loc[subject].squeeze()
                vline_dict[subject]['idx'] = vline

        # Plot the median values using the same line color as above
        if plot_vline is not None:
            for subject in vline_dict.keys():
                subplot_ax.axvline(
                    vline_dict[subject]['idx'],
                    linestyle='--', alpha=0.5,
                    color=vline_dict[subject]['color']
                )

# %%


def fmt_subtitle(subtitle_dict, length, sep=';'):
    """
    Format subtitle strings for plots. Creates two versions:

    long length: 'Field X: w = 1; Field Y: w = 1.5'
    short length: w_X = 1; w_Y = 1.5

    subtitle dict is a dictionary with the format:
        {
            'var_tex': xx,
            'var': yy
        }
    where xx is the LaTeX name for a variable (i.e. \theta or w) and yy is
    variable name in this program.

    Separately calculated for ab_0.
    """
    # String for the variable name ; i.e. '\theta' or '\theta_x'
    if length == 'long':
        var_str = subtitle_dict['var_tex']
        # Exception for ab_0
        if subtitle_dict['var_tex'] == 'ab_0':
            var_str = r'(\alpha_{0}, \beta_{0})'
        # One version for j=0, one version for j=1
        str0 = r'$' + var_str
        str1 = r'$' + var_str
    if length == 'short':
        # a string for j=0 and j=1
        str0 = r'${0}_{{{1}}}'.format(subtitle_dict['var_tex'], alph_dict[0])
        str1 = r'${0}_{{{1}}}'.format(subtitle_dict['var_tex'], alph_dict[1])
        # Exception for ab_0
        if subtitle_dict['var_tex'] == 'ab_0':
            str0 = r'$(\alpha_{{{x}0}}, \beta_{{{x}0}})'.format(x=alph_dict[0])
            str1 = r'$(\alpha_{{{y}0}}, \beta_{{{y}0}})'.format(y=alph_dict[1])
    # Add 'Field j: ' to long version
    if length == 'long':
        str0 = r'Field {j}: {exp}'.format(j=alph_dict[0], exp=str0)
        str1 = r'Field {j}: {exp}'.format(j=alph_dict[1], exp=str1)
    # Find value for j = 0 and j = 1
    if subtitle_dict['var_tex'] == 'ab_0':
        val0 = '({0[0]:g}, {0[1]:g})'.format(subtitle_dict['var'][0, :])
        val1 = '({0[0]:g}, {0[1]:g})'.format(subtitle_dict['var'][1, :])
    else:
        val0 = '{0:g}'.format(subtitle_dict['var'][0])
        val1 = '{0:g}'.format(subtitle_dict['var'][1])
    str0 = str0 + ' = ' + val0 + r'$'
    str1 = str1 + ' = ' + val1 + r'$'
    str_all = str0 + sep + ' ' + str1
    return str_all

#     def print_v(self, v, length):
#         '''
#         Print human capital accumulation
#         '''
#         if length == 'long':
#             str0 = r'Field ' + self.alph_dict[0] + r': $\nu'
#             str1 = r'Field ' + self.alph_dict[1] + r': $\nu'
#         if length == 'short':
#             str0 = r'$\theta_{' + self.alph_dict[0] + '}'
#             str1 = r'$\theta_{' + self.alph_dict[1] + '}'
#         str0 = str0 + ' = {0:g}'.format(np.round(v[0], 2)) + r'$'
#         str1 = str1 + ' = {0:g}'.format(np.round(v[1], 2)) + r'$'
#         # str0 = r'Field ' + self.alph_dict[0] \
#         #        + r': $\nu=' + str(np.round(v[0], 2)) + r'$'
#         # str1 = r'Field ' + self.alph_dict[1] \
#         #        + r': $\nu=' + str(np.round(v[1], 2)) + r'$'
#         str_all = str0 + '; ' + str1

#         return str_all


# %%
if __name__ == '__main__':

    ##################################
    # Baseline simulation - 50 times #
    ##################################
    # Create simulation and dataframe
    np.random.seed(125)

    alph_list = ['X', 'Y']
    alph_dict = {key: alph_list[key] for key in range(len(alph_list))}

    # arguments for plotting
    plt_title = 'Baseline Simulation (zoomed in)'
    label_edit = {0: -.08, 1: .03}
    ax_loc = [0, 1]
    subtitle_id = 'b'

    plt.close('all')

    # make_sim_plots(figname='simulation_50',
    #                sim_df=agg_sim_df,
    #                plt_title=plt_title, sim_num=agent_sim.sim_num,
    #                subject_labels={0: 'Field X', 1: 'Field Y'},
    #                plot_vline=spec_idx,
    #                subtitle_id=subtitle_id,
    #                xticks=[0, 5, 10, 15, 20], yticks=[0, 0.25, 0.5, 0.75, 1],
    #                label_edit=label_edit,
    #                save_tikz_code=True, group_plot=False
    #                )
    # # plt.show()
    # plt.close('all')

    fig_all, ax = plt.subplots(3, 2)

    make_sim_plots(figname='simulation_50',
                   sim_df=agg_sim_df,
                   ax_idx=ax_loc, group_ax=ax,
                   plt_title=plt_title, sim_num=agent_sim.sim_num,
                   subject_labels={0: 'Field X', 1: 'Field Y'},
                   # subtitle_id=subtitle_id,
                   plot_vline=spec_idx,
                   subtitle_id='b',
                   xticks=[0, 5, 10, 15, 20], yticks=[0, 0.25, 0.5, 0.75, 1],
                   label_edit=label_edit,
                   save_tikz_code=False, group_plot=True
                   )

    plt.show()

    # ability = np.array([.25, .75])
    # subtitle_dict = {'var_tex': r'\theta', 'var': ability}
    # print(fmt_subtitle(subtitle_dict, 'short'))
    # print(fmt_subtitle(subtitle_dict, 'long'))

    # ab_0 = np.array([[1., 1.], [2., 2.]])
    subtitle_dict = {'var_tex': 'ab_0', 'var': ab_0_default}
    print(fmt_subtitle(subtitle_dict, 'short'))
    print(fmt_subtitle(subtitle_dict, 'long'))
