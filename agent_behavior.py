import numpy as np
import numba as nb

use_numba = True


class Params:
    """
    Set parameters in the economy. These can broadly be divided into:
        * group parameters: ab_0, n_j, n_g, group_dict, w_r
        * field parameters: field_dict, n_j [compatible with ab_0]
        * aggregate parameters: v_all, w, delta [compatible with ab_0]

    Group parameters in the economy:
        * ab_0:         Initial abilities for each group [n_g x n_j x 2]
        * n_g:          Number of groups
        * wage_r:       Reservation wage (for drop-out)

    Field parameters in the economy:
        * field_dict:   Dictionary for J fields
        * n_j:          Number of fields J

    Aggregate parameters in the economy:
        * v_all:    Human capital update amount (n_j x 1)
        * wage:     Wages upon graduating (n_j x 1)
        * delta:    Discount rate
    """

    def __init__(self,
                 ab_0,
                 delta=0.96,
                 v_all=None,
                 wage=None,
                 wage_r=None,
                 group_dict=None,
                 field_dict=None):
        self.ab_0 = ab_0
        self.n_g, self.n_j, _ = ab_0.shape

        self.delta: float = delta
        if wage_r is None:
            self.wage_r = 0
        else:
            self.wage_r = wage_r
        if v_all is None:
            self.v_all = np.ones(self.n_j, dtype=np.float64)
        else:
            self.v_all = v_all
            assert len(self.v_all) == self.n_j, \
                "Dimension of v_all and ab_0 must match"
        if wage is None:
            self.wage = np.ones(n_j_all, dtype=np.float64)
        else:
            self.wage = wage
            assert len(self.wage) == self.n_j, \
                "Dimension of wage and ab_0 must match"

        # Optional dictionary parameters
        if group_dict is not None:
            self.group_dict = group_dict
            assert len(self.group_dict) == self.n_g, \
                "Dimension of group_dict and ab_0 must match"
        if field_dict is not None:
            self.field_dict = field_dict
            assert len(self.field_dict) == self.n_j, \
                "Dimension of field_dict and ab_0 must match"


if __name__ == '__main__':
    np.random.seed(10)

    # Dictionary of fields
    field_dict_all = {0: 'Science',
                      1: 'Math',
                      2: 'Humanities', 3: 'Social Science'}
    group_dict_all = {0: 'Women', 1: 'Men', 2: 'NB'}

    n_g_all, n_j_all = 3, 4
    ab_0_all = np.ones((n_g_all, n_j_all, 2), dtype=np.int8)
    # Women better at humanities
    ab_0_all[0, 2, 0] = 2
    # Women know about SS
    ab_0_all[0, 3, :] = 2
    # Men more certain about math
    ab_0_all[1, 1, :] = 2
    # Men slightly better at science
    ab_0_all[1, 0, 0] = 2

    wage_all = np.array([1., 1.25, 1, 1])

    # Define economy parameters
    params_all = Params(
        ab_0_all, field_dict=field_dict_all, wage=wage_all)

    # variables for testing purposes
    # wage = params.wage
    nu_all = params_all.v_all
    delta_all = params_all.delta

    # Initial human capital for beta-bernoulli example
    h_0_all = ab_0_all[:, :, 0] * nu_all


class AgentParams:
    """
    Set parameters for an agent. These include:
        * params: economy-wide parameters; see class Params
        * group: group designation
        * h_0: agent's initial human capital
        * theta: agent's true ability theta
    """
    def __init__(self,
                 params,
                 group_idx,
                 h_0,
                 theta):
        self.h_0 = h_0
        self.theta = theta

        # params inherited from group parameters
        self.group_idx = group_idx
        self.ab_0 = params.ab_0[group_idx]

        # params inherited from macro parameters
        self.v_all = params.v_all
        self.wage = params.wage
        self.delta = params.delta


if __name__ == '__main__':
    true_ability_i = np.random.beta(ab_0_all[0, :, 0], ab_0_all[0, :, 1])
    agent_i_params = AgentParams(
        params_all,
        group_idx=0,
        h_0=h_0_all[0],
        theta=true_ability_i
    )


# %% Define stop condition and in grad region functions
def get_stop_condition(
        m_courses, s_success,
        alpha_0, beta_0, h_0,
        v_all, delta):
    """
    Stopping condition under monotonicity assumption.
    """
    # if ab_0.shape=(2,), want alpha = first element
    # if ab_0.shape=(n_j, 2), want alpha = n_j equals first axis
    # alpha_0 = np.take(ab_0, 0, axis=-1)
    # beta_0 = np.take(ab_0, 1, axis=-1)

    stop_condition = (
            delta / (1 - delta) * (
                (v_all * (alpha_0 + s_success))
                / (h_0 + v_all * s_success)
        ) - alpha_0 - beta_0
    )

    return stop_condition


def in_grad_region(
        m_courses, s_success,
        alpha_0, beta_0, h_0,
        v_all, delta):
    """
    Note: evaluated using a single agent's vector of beliefs (ab_0)
    Be careful not to confuse this with the vector of beliefs for all groups
    """
    stop_condition = get_stop_condition(
        m_courses, s_success, alpha_0, beta_0, h_0, v_all, delta
    )
    grad_region = (m_courses >= stop_condition)
    return grad_region


if use_numba:
    get_stop_condition = nb.njit(get_stop_condition)
    in_grad_region = nb.njit(in_grad_region)

# test that in_grad_region function works
if __name__ == '__main__':
    # h_0 = np.ones(n_j, np.float64)

    m_courses_i = np.zeros(n_j_all, dtype=np.int8)
    s_success_i = np.zeros(n_j_all, dtype=np.int8)
    # Test functions
    get_stop_condition(
        m_courses=m_courses_i, s_success=s_success_i,
        alpha_0=agent_i_params.ab_0[:, 0], beta_0=agent_i_params.ab_0[:, 1],
        h_0=agent_i_params.h_0, v_all=agent_i_params.v_all, delta=agent_i_params.delta
    )
    in_grad_region(
        m_courses=m_courses_i, s_success=s_success_i,
        alpha_0=agent_i_params.ab_0[:, 0], beta_0=agent_i_params.ab_0[:, 1],
        h_0=agent_i_params.h_0, v_all=agent_i_params.v_all, delta=agent_i_params.delta
    )
    # Create array of whether courses in grad region
    grad_region_array = in_grad_region(
        m_courses=m_courses_i, s_success=s_success_i,
        alpha_0=agent_i_params.ab_0[:, 0], beta_0=agent_i_params.ab_0[:, 1],
        h_0=agent_i_params.h_0, v_all=agent_i_params.v_all, delta=agent_i_params.delta)

    print('in_grad_region function test outcome:')
    print(grad_region_array)


# %%

def find_index(
        m_courses, s_success,
        h_0,
        ab_0,
        v_all, wage, delta,
        m_star=None,
):
    """
    m_star: total number of courses in each field; indicative of deterministic
    """
    # Initialize index
    index = np.empty_like(h_0, dtype=np.float64)
    # Re-index problem
    a_0_hat = ab_0[:, 0] + s_success
    h_0_hat = h_0 + v_all * s_success
    b_0_hat = np.sum(ab_0, axis=1) + m_courses - a_0_hat
    # vector of in grad region

    grad_region_mask = in_grad_region(
        m_courses=m_courses, s_success=s_success,
        alpha_0=ab_0[:, 0], beta_0=ab_0[:, 1], h_0=h_0,
        v_all=v_all, delta=delta
    )

    if m_star is not None:
        # total number of courses left
        # m_star = (np.ceil(delta / (1 - delta)) - a_0_hat - b_0_hat)

        # current payoff
        current_payoff = (1 / (1 - delta)) * (wage * h_0_hat)
        # expected discounted human capital accumulation
        h_star = np.where(
            grad_region_mask,
            h_0_hat,
            (
                    delta ** m_star
                    * (h_0_hat + v_all * m_star * (a_0_hat / (a_0_hat + b_0_hat))))
        )
        index = wage / (1 - delta) * h_star

    return index


if use_numba:
    find_index = nb.njit(find_index)

if __name__ == '__main__':
    m_star_i = (
            np.ceil(agent_i_params.delta / (1 - agent_i_params.delta))
            - agent_i_params.ab_0[:, 0] + s_success_i
            - (
                    np.sum(agent_i_params.ab_0, axis=1)
                    + m_courses_i
                    - agent_i_params.ab_0[:, 0] - s_success_i)
    )
    index_it = find_index(
        m_courses=m_courses_i, s_success=s_success_i,
        h_0=agent_i_params.h_0, ab_0=agent_i_params.ab_0,
        m_star=m_star_i,
        v_all=agent_i_params.v_all, wage=agent_i_params.wage, delta=agent_i_params.delta)
    print(index_it)

# %%


def is_specialized(
    choose_j, total_courses, index,
    m_courses, s_success,
    ab_0, h_0,
    v_all, wage, delta,
):
    '''
    Determine whether an agent is specialized in a field. Here, specialization
    is defined as the point where an agent choosing to student field j would
    continue to choose j, even if they failed all remaining courses.

    Note that this may only work for the beta-bernoulli case, and may need to
    be adjusted in the more general setting; check this
    '''

    # number of times you've taken j
    m_count = m_courses[choose_j]
    # number of times you've succeeded at j
    s_count = s_success[choose_j]
    # course_counts = len(c_t[np.where(c_t == choose_j)])
    # calculate index for choose_j as if you failed all
    # remaining courses in j
    fail_index = (
        (1 / (1 - delta))
        * delta ** (
            total_courses[choose_j] - m_count)
        * wage[choose_j]
        * (h_0[choose_j] + v_all[choose_j] * s_count)
    )
    # add failure index to index
    index_specialize = index.copy()
    # replace index with index where all remaining courses in j
    # are failed
    index_specialize[choose_j] = fail_index
    # find max in new version of index
    max_specialize = np.argwhere(
        index_specialize == np.max(index_specialize))
    # If you would still choose to study j, knowing you'd fail all
    # remaining courses
    return max_specialize[0, 0] == choose_j


if use_numba:
    is_specialized = nb.njit(is_specialized)


if __name__ == '__main__':
    if is_specialized(
        choose_j=3, total_courses=m_star_i, index=index_it,
        m_courses=m_courses_i, s_success=s_success_i,
        h_0=agent_i_params.h_0, ab_0=agent_i_params.ab_0,
        v_all=agent_i_params.v_all, wage=agent_i_params.wage,
        delta=agent_i_params.delta
    ):
        print('Is specialized.')
    else:
        print('Is not specialized.')


# %% Find an agent's full history

def _find_agent_history(
        h_0, theta, ab_0,
        v_all, wage, delta,
        fail_first=0, choose_first=-1,
):
    '''
    Find agent's complete course history.

    * fail_first: fail first n classes; helpful for checks
    * choose_first: first required course is j; helpful for checks

    '''
    # make a copy of initial human capital levels
    ab_t = np.copy(ab_0)
    # convenient notation
    n_j, _ = ab_0.shape
    dd = np.ceil(delta / (1 - delta))

    # calculate total courses = m_star
    total_courses = dd - np.sum(ab_0, axis=1)

    # Initialize course history
    m_courses = np.zeros(n_j, dtype=np.int16)
    s_success = np.zeros(n_j, dtype=np.int16)

    # timelines
    # course history timeline; records index of each course chosen from t=0 on
    course_tl = np.empty(0, dtype=np.int16)
    # outcome timeline; records outcome of each course chosen from t=0 on
    outcome_tl = np.empty(0, dtype=np.int16)

    # find initial index
    index_t = find_index(
        m_courses=m_courses, s_success=s_success,
        h_0=h_0, ab_0=ab_0,
        v_all=v_all, wage=wage, delta=delta,
        m_star=total_courses
    )

    # Initialize number of switches made
    n_switch = 0
    # Initial time index where specialization decision is made
    specialize_idx = 0

    # Find full course history
    keep_studying = True
    while keep_studying:
        # If no first course is specified, pick a random one
        if choose_first == -1:
            # Find the largest indices (there may be more than one)
            # Numba doesn't play nice with using np.argwhere + np.reshape.
            # This might have to do with the fact that np.argwhere is not
            # suitable for indexing arrays.
            max_idx_array = np.nonzero(index_t == np.max(index_t))[0]
            # Randomly choose the largest index
            choose_j = np.random.choice(max_idx_array)
        # Otherwise, specify which course comes first
        else:
            choose_j = choose_first
            choose_first = -1

        # Determine whether you are specialized
        # may need to flag if non-beta bernoulli; use m_star flag
        if specialize_idx == 0:
            # If you would still choose to study j, knowing you'd fail all
            # remaining courses
            if is_specialized(
                choose_j=choose_j, total_courses=total_courses,
                index=index_t,
                m_courses=m_courses, s_success=s_success,
                ab_0=ab_0, h_0=h_0,
                v_all=v_all, wage=wage, delta=delta
            ):
                # then you are specialized.
                # TO DO: check this is the correct index adjustment
                specialize_idx = len(course_tl) - 1

        # Graduate if in graduation region
        if in_grad_region(
            m_courses=m_courses[choose_j], s_success=s_success[choose_j],
            alpha_0=ab_0[choose_j, 0], beta_0=ab_0[choose_j, 1],
            h_0=h_0[choose_j],
            v_all=v_all[choose_j], delta=delta
        ):
            chosen_field = choose_j
            chosen_field_state = ab_t[choose_j, :]
            keep_studying = False
        # Study and record outcomes if not
        else:
            # study and pass/fail
            if fail_first == 0:
                # outcomes usually random binomial
                outcome_j = np.random.binomial(1, theta[choose_j])
            else:
                # ensure agent fails first n classes
                outcome_j = 0
                fail_first = max(fail_first - 1, 0)

            # update variables
            ab_t[choose_j, :] = ab_t[choose_j, :] + \
                np.array([outcome_j, 1 - outcome_j])

            # record number of switches
            if len(course_tl) > 0:
                if course_tl[-1] != choose_j:
                    n_switch = n_switch + 1

            # update timeline
            course_tl = np.append(course_tl, np.int16(choose_j))
            outcome_tl = np.append(outcome_tl, np.int16(outcome_j))

            # update m_course and s_success
            m_courses[choose_j] += 1
            s_success[choose_j] += outcome_j

            # update index
            index_t = find_index(
                m_courses=m_courses, s_success=s_success,
                h_0=h_0, ab_0=ab_0,
                v_all=v_all, wage=wage, delta=delta,
                m_star=total_courses
            )

    # Numba can't return a dictionary with heterogeneous values (at least
    # according to my current reading). So instead I'm returning a
    # tuple of values and a tuple of names of those values.
    # This is a dictionary if you run: dict(zip(*return_tuple))

    # Values to return
    return_vals_tuple = (
        m_courses,
        s_success,
        course_tl,
        outcome_tl,
        chosen_field,
        chosen_field_state,
        n_switch,
        specialize_idx
    )
    # Names to return
    return_names_tuple = (
        'm_courses',
        's_success',
        'course_tl',
        'outcome_tl',
        'chosen_field',
        'chosen_field_state',
        'n_switch',
        'specialize_idx',
    )
    # Turn this into a dictionary with:  dict(zip(*return_tuple))
    return_tuple = (return_names_tuple, return_vals_tuple)

    return return_tuple


# if use_numba:
#     _find_agent_history = nb.njit(_find_agent_history)

if __name__ == '__main__':
    history = _find_agent_history(
        h_0=agent_i_params.h_0,
        theta=agent_i_params.theta,
        ab_0=agent_i_params.ab_0,
        v_all=agent_i_params.v_all,
        wage=agent_i_params.wage,
        delta=agent_i_params.delta,
        fail_first=1, choose_first=1,
    )
    print(dict(zip(*history)))


# %% Agent class
class Agent:
    """
    Class defining an agent at time t with group type g.

    Arguments:
        * AgentParams
        Parameters for a particular agent. See class documentation.

    Methods:
        * find_agent_history(self)
        Finds the agent's academic history given their state variables
        at the beginning of their education.
    """
    def __init__(self, agent_params: AgentParams):
        self.h_0 = agent_params.h_0
        self.theta = agent_params.theta

        # params inherited from group parameters
        self.group_idx = agent_params.group_idx
        self.ab_0 = agent_params.ab_0

        # params inherited from macro parameters
        self.v_all = agent_params.v_all
        self.wage = agent_params.wage
        self.delta = agent_params.delta

        # # Run history argument
        # _history = self.find_agent_history()
        #
        # self.m_courses = _history['m_courses']
        # self.s_success = _history['s_success']
        # self.course_tl = _history['course_tl']
        # self.outcome_tl = _history['outcome_tl'],
        # self.chosen_field = _history['chosen_field'],
        # self.chosen_field_state = _history['chosen_field_state'],
        # self.n_switch = _history['n_switch'],
        # self.specialize_idx = _history['specialize_idx']

    def find_agent_history(self):
        _history = _find_agent_history(
            h_0=self.h_0,
            theta=self.theta,
            ab_0=self.ab_0,
            v_all=self.v_all,
            wage=self.wage,
            delta=self.delta,
        )

        return dict(zip(*_history))


if __name__ == '__main__':
    agent_i = Agent(agent_i_params)
    history = agent_i.find_agent_history()

    def print_i(agent_i, history):
        # # Create a nice dataframe that summarizes the above output
        # print('True ability: ' + str(agent.theta))
        # print('Course History: ')
        # print(agent.course_tl)
        # print('Final state: ' + str(agent.chosen_field_state))
        # print('Number switches: ' + str(agent.n_switch))
        # print('Specialize index: ' + str(agent.specialize_idx))
        # Create a nice dataframe that summarizes the above output
        print('True ability: ' + str(agent_i.theta))
        print('Course History: ')
        print(history['course_tl'])
        print('Final state: ' + str(history['chosen_field_state']))
        print('Number switches: ' + str(history['n_switch']))
        print('Specialize index: ' + str(history['specialize_idx']))

    print_i(agent_i, history)
