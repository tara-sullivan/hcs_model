import matplotlib.pyplot as plt
import numpy as np
# simulate multiple agents at time period t
from scipy.stats.contingency import expected_freq

import itertools

from agent_behavior import Params
from agent_behavior import AgentParams
from agent_behavior import Agent

from sim_agent_history import AgentSimulation

from typing import Union
import numpy.typing as npt


# %%
def find_alpha_beta(
        n_agent_t_gj,
        replace_ones: Union[list[int], int] = None,
        method: str = 'grid',
):
    """
    Find the alpha and beta, given the current count of students at t.

    Parameters
    ----------
    n_agent_t_gj : array, [n_g x n_j]
        Number of agents of type g in field j at time t; presented as
        an n_g x n_j matrix.
    replace_ones : list or int
        group indexes to replace with uninformative prior
    method : str ['grid', 'relative_count']
        Set the method for calculating alpha and beta. See notes below.

    Returns
    -------
    alpha_beta_0_all : array, [n_g x n_j x 2]
        Values of alpha and beta given the number of agents

    Remarks
    -------
    Types of relive counts:
        relative_counts : find alpha and beta by setting alpha equal to
        the number of students in the field, and beta equal the relative
        count of other student types
    """
    n_g, _ = n_agent_t_gj.shape

    if method == 'grid':

        scale_val = 5

        # Find the total number for your group type [n_g x 1]
        n_agent_t_g = n_agent_t_gj.sum(axis=1)

        # Find alpha first; based on how popular your major is for your group
        # fraction of your group in each field
        alpha = (n_agent_t_gj / n_agent_t_g[:, None])
        # Find beta next
        # Difference between number of agents in a field and the expected
        # frequency in each field, rounded
        scaled_diff = (expected_freq(n_agent_t_gj) - n_agent_t_gj) / n_agent_t_g[:, None]
        # Add this
        beta = alpha + scaled_diff

        # Round up
        alpha = np.round(alpha * scale_val, 0)
        beta = np.round(beta * scale_val, 0)
        # replace zeros with 1
        alpha[alpha == 0] = 1
        beta[beta == 0] = 1

        alpha_beta_0_all = np.stack([alpha, beta], axis=2)

    if method == 'relative_count':
        ab_0_list = []
        for g_idx in np.arange(n_g):
            # First index; will result in two arrays
            if g_idx == 0:
                split_idx = np.array([1])
            # Last index; will result in three arrays
            elif g_idx == n_g - 1:
                split_idx = np.array([n_g - 1])
            # Results in three arrays; before g_idx, g_idx, and after g_idx
            else:
                split_idx = np.array([g_idx, g_idx+1])

            split_n = np.split(n_agent_t_gj, axis=0,
                               indices_or_sections=split_idx)

            if g_idx == 0:
                alpha = split_n.pop(0)
            elif g_idx == n_g - 1:
                alpha = split_n.pop(1)
            else:
                alpha = split_n.pop(1)
            # concatenating along axis=1 won't do anything for first and last
            # idx, but will combine the index for those in the middle.
            beta = np.concatenate(split_n, axis=0).sum(axis=0)[None, :]

            ab_0_list.append(np.concatenate([alpha, beta], axis=0).T[None, :, :])

        alpha_beta_0_all = np.concatenate(ab_0_list, axis=0)

    if replace_ones is not None:
        if isinstance(replace_ones, int):
            alpha_beta_0_all[replace_ones] = np.ones_like(
                alpha_beta_0_all[replace_ones])
        elif isinstance(replace_ones, list):
            for g_idx in replace_ones:
                alpha_beta_0_all[g_idx] = np.ones_like(alpha_beta_0_all[g_idx])

    return alpha_beta_0_all



# %%
def simulate_agents_t(
        params_t,
        n_agent_t_g,
        share_h_0: bool = True,
        share_theta: bool = True,
        h_0_t_gj=None,
        theta_t_gj=None,
):
    """
    Simulate agent decision-making at time t.

    Parameters
    ----------
    params_t : economy parameters
        All non-agent parameters. See class Params (agent_history.py)
        for details.
    n_agent_t_g: array
        Number of agent's of type g at time t [n_g x 1]
    share_h_0 : bool, optional
        All agents of type g share h_0; default is True
    share_theta: bool, optional
        All agents of type g share theta 0.
    h_0_t_gj : array, optional
        Initial human capital by group at time t. Default will be
        h_0 = alpha * v. If share_h_0 True, this has dimension
        [n_g x n_j]. If share_h_0 False, this is a list of length n_g,
        with each element being an [n_i x n_j array] (i.e. where each
        item in the list holds all h_0 for all agents of type g at time
        t across all fields j).
    theta_t_gj : array, optional
        Initial abilities by group at time t. Default will be 0.5.
        Dimensions follow same parameters as for h_0_t_gj, but using the
        share_theta boolean.
    """
    if share_h_0:
        if h_0_t_gj is not None:
            assert (params_t.n_g, params_t.n_j) == h_0_t_gj.shape, \
                "Please pass correct dimensions of h_0_t_gj [n_g x n_j]"
        else:
            h_0_t_gj = params_t.ab_0[:, :, 0] * params_t.v_all
    else:
        if h_0_t_gj is None:
            alpha_0_t_list = np.split(params_t.ab_0[:, :, 0], 3, axis=0)
            h_0_t_gj = [
                np.tile(alpha_0_t_list[idx], (n_agent_t_g[idx], 1))
                for idx in np.arange(params_t.n_g)
            ]
        else:
            print('Code not currently configured for individual human capital.')
            return

    if share_theta:
        if theta_t_gj is None:
            theta_t_gj = 0.5 * np.ones_like(h_0_t_gj, dtype=np.float64)
        else:
            assert (params_t.n_g, params_t.n_j) == theta_t_gj.shape, \
                "Please pass correct dimensions of theta_t_gj [n_g x n_j]"
    else:
        if theta_t_gj is None:
            theta_t_gj = [
                0.5 * np.ones_like(h_0_t_gj[idx], dtype=np.float64)
                for idx in np.arange(params_t.n_g)
            ]
        else:
            print('Code not currently configured for individual abilities.')
            return

    if (share_h_0 is True) and (share_theta is True):
        # Simulate each agent's history
        # Initialize list; this will be the number of agents of group g
        # starting education at t who end up specializing in each field
        num_agents_gt1_list = []
        for idx in np.arange(n_g_all):
            # Create agent's parameters; because h_0 and theta shared among
            # all group members, we only need to do this once for each group.
            agent_params_gt = AgentParams(
                params=params_t,
                group_idx=idx,
                h_0=h_0_t_gj[idx, :],
                theta=theta_t_gj[idx, :]
            )
            # create agent
            agent_gt = Agent(agent_params_gt)
            # Simulate agent's decision for the appropriate number of time
            sim_gt = AgentSimulation(agent_gt, sim_num=n_agent_t_g[idx])

            j_values, j_counts = np.unique(
                sim_gt.chosen_field, return_counts=True)

            # Using the np.unique command means that, in cases where 0
            # people choose a field, you will not have 0 count. Need to
            # insert 0 for appropriate fields.
            # There's almost surely a better way to do this
            j_counts_all = []
            for j_idx in np.arange(params_t.n_j):
                if j_idx not in j_values:
                    j_counts_all.append(0)
                else:
                    j_counts_all.append(
                        j_counts[np.argwhere(j_values == j_idx)].squeeze()[()])
            num_agents_gt1_list.append(j_counts_all)
        num_agents_gt1 = np.array(num_agents_gt1_list)
        return num_agents_gt1

    # if share_h_0 is False | share_theta is False
    else:
        # placeholder in case I generalize this.
        return "Code only configured for shared h_0 and theta_0"


# %%
if __name__ == '__main__':
    np.random.seed(10)

    # Static parameters

    # Dictionary of fields
    field_dict_all = {0: 'Science',
                      1: 'Math',
                      2: 'Education', 3: 'Social Science'}
    group_dict_all = {0: 'Women', 1: 'Men', 2: 'NB'}

    n_g_all = len(group_dict_all)
    n_j_all = len(field_dict_all)

    wage_all = np.array([1.25, 1.5, .5, 1])
    v_all = np.ones_like(wage_all)

    # Number of agents of each group type in each field, n_g x n_j at
    # t = 0
    num_agent_t0_gj = np.array([[2, 1, 4, 3], [2, 2, 2, 2], [1, 0, 1, 0]])
    # Initial belief states based on historical number of agents at t=0
    ab_0_all_t0 = find_alpha_beta(num_agent_t0_gj)

    # Make copies of above items to be updated in loop
    num_agent_t_gj = num_agent_t0_gj.copy()
    ab_0_all_t = ab_0_all_t0.copy()

    # number of each agent type in each new period
    num_agent_t_g = np.array([11, 9, 1])

    for t_idx in np.arange(10):

        # Define economy parameters at t
        params_all_t = Params(
            ab_0_all_t,
            field_dict=field_dict_all,
            wage=wage_all,
            v_all=v_all,
            delta=0.96
        )

        # num_agent_gt = num_agent_t_gj.sum(axis=1)

        # Initial human capital for beta-bernoulli example
        h_0_all_t = ab_0_all_t[:, :, 0] * params_all_t.v_all
        ability_default = 0.5 * np.ones_like(h_0_all_t, dtype=np.float64)

        agent_choice_t_gj = simulate_agents_t(
            params_t=params_all_t,
            n_agent_t_g=num_agent_t_g,
            h_0_t_gj=h_0_all_t,
            theta_t_gj=ability_default
        )

        # print(agent_choice_t_gj)
        # Update the number state variable
        num_agent_t_gj = num_agent_t_gj + agent_choice_t_gj
        ab_0_all_t = find_alpha_beta(num_agent_t_gj)
        print(ab_0_all_t)
        print(ab_0_all_t0)

