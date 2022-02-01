import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
import tol_colors

import os

from scipy.stats import beta
import string

from img_tools import tikzplotlib_functions as tplf
from img_tools.figsize import ArticleSize

# Set color map
color_list = list(tol_colors.tol_cset('bright'))
# get figure height and width
size = ArticleSize()



# Set image path
dirname = os.path.dirname(__file__)
tex_img_path = os.path.join(dirname, 'fig_tex_code')

# %% Define beta example program
def beta_example_gender():
    '''
    Plot example beta distribution for males and females
    '''

    a_f, b_f = 3 * 1, 2 * 1
    a_m, b_m = a_f * 2, b_f * 2

    # Mean and sample size
    mu_m = beta.moment(1, a_m, b_m)
    mu_f = beta.moment(1, a_f, b_f)
    n_m, n_f = a_m + b_m, a_f + b_f

    x = np.linspace(0, 1, 1000)

    fig, ax = plt.subplots()

    # Function to create label
    def label_str(g, mu, n):
        label_str = g + ' \\\\ ' \
            + r'($\mu = $' + str(round(mu, 2)) + r', $n = $' + str(n) + ')'
        return label_str

    ax.plot(x, beta.pdf(x, a_m, b_m),
            label=label_str('Men', mu_m, n_m),
            color=color_list[0])

    ax.plot(x, beta.pdf(x, a_f, b_f),
            label=label_str('Women', mu_f, n_f),
            color=color_list[1])

    ax.legend(loc='upper left',
              handlelength=0.5,
              frameon=False,
              borderpad=0)

    # get the current limit of y-axis
    _, y_max = ax.get_ylim()

    # Dotted line at mean
    ax.plot([mu_f, mu_f], [0, y_max], ':', color='#BBBBBB')

    # set y-limit
    ax.set_ylim(0, y_max)


def beta_example_change(ab0, history):
    '''
    Create beta distribution examples

      * ab_0: prior parameters for two fields
      * history: history of passing/failing for three periods
    '''

    x_range = np.linspace(0, 1, 50)

    a0, b0 = ab0
    y0 = beta.pdf(x_range, a0, b0)

    if len(history) >= 1:
        a1 = a0 + history[0]
        b1 = b0 + (1 - history[0])
        y1 = beta.pdf(x_range, a1, b1)
    if len(history) >= 2:
        a2 = a1 + history[1]
        b2 = b1 + (1 - history[1])
        y2 = beta.pdf(x_range, a2, b2)
    if len(history) >= 3:
        a3 = a2 + history[2]
        b3 = b2 + (1 - history[2])
        y3 = beta.pdf(x_range, a3, b3)

    # plot initial distribution
    fig, ax = plt.subplots()

    ax.plot(x_range, y0, linewidth=3)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 2.25)

    ax.set_xticks((0, 0.5, 1))
    ax.set_yticks([0, 1, 2])

    title_str = r'Beliefs $p(\theta | \alpha, \beta)$'
    ax.set_title(title_str)

    tikz_save('beta_example0')
    # plt.show()

    # plot t=1 distribution
    fig, ax = plt.subplots()

    ax.plot(x_range, y0, 'k', alpha=0.5)
    ax.plot(x_range, y1, linewidth=3)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 2.25)

    ax.set_xticks((0, 0.5, 1))
    ax.set_yticks([0, 1, 2])

    # title_str = r'$p(\theta | \alpha, \beta)$'
    ax.set_title(title_str)

    tikz_save('beta_example1')
    # plt.show()

    # plot t=2 distribution
    fig, ax = plt.subplots()

    ax.plot(x_range, y0, 'k', alpha=0.3)
    ax.plot(x_range, y1, 'k', alpha=0.5)
    ax.plot(x_range, y2, linewidth=3)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 2.25)

    ax.set_xticks((0, 0.5, 1))
    ax.set_yticks([0, 1, 2])

    # title_str = r'$p(\theta | \alpha, \beta)$'
    ax.set_title(title_str)

    tikz_save('beta_example2')
    # plt.show()

    # plot t=3 distribution
    fig, ax = plt.subplots()

    ax.plot(x_range, y0, 'k', alpha=0.1)
    ax.plot(x_range, y1, 'k', alpha=0.3)
    ax.plot(x_range, y2, 'k', alpha=0.5)
    ax.plot(x_range, y3, linewidth=3)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 2.25)

    ax.set_xticks((0, 0.5, 1))
    ax.set_yticks([0, 1, 2])

    # title_str = r'$p(\theta | \alpha, \beta)$'
    ax.set_title(title_str)

    tikz_save('beta_example3')


def beta_example_change_paper():
    '''
    Beta example plot. Two lines, legend in figure. aimed at paper
    '''
    plt.close('all')
    fig, ax = plt.subplots(3, 3, sharex=True, sharey=True)
    # initial groupplot stuff
    subplot_titles = ''
    ref_name = 'fig:beta_ex'

    x_range = np.linspace(0, 1, 50)

    i, j = 0, 0
    for i in np.arange(3):
        for j in np.arange(3):
            alpha0 = j + 1
            beta0 = i + 1
            subtitle_id = string.ascii_lowercase[i * 3 + j]

            y_range = beta.pdf(x_range, alpha0, beta0)

            ax[i, j].plot(x_range, y_range, color=color_list[0], linewidth=2)
            plt_title = \
                r'$(\alpha_0, \beta_0) = ({a0}, {b0})$'.format(a0=alpha0,
                                                               b0=beta0)

            # direction of ticks
            ax[i, j].tick_params(direction="in")
            # start at 0
            ax[i, j].set_xlim(left=0)
            if j == 2:
                ax[i, j].set_ylim(bottom=0)

            subplot_titles += tplf.subplot_title(
                ax_loc=[i, j], ref_name=ref_name,
                subtitle_id=subtitle_id, plt_title=plt_title,
                text_width=size.w(0.65)
            )
    caption_str = 'Evolution of the Beta distribution ' \
        + r'$\mathcal{B} (\alpha_{0}, \beta_{0})$' \
        + r' for different values of $(\alpha_{0}, \beta_{0})$.'

    tplf.save_subplots(
        os.path.join(tex_img_path, 'beta_example_change.tex'),
        height=size.h(.7), width=size.w(0.7),
        clean_figure=True,
        node_code=subplot_titles, caption=caption_str,
        extra_groupstyle_parameters={
            'horizontal sep=0.75cm',
        })


def ab_str(j, t, a, b):
    ab_str = r'$(\alpha_{{{0}{1:g}}}, \beta_{{{0}{1:g}}})'.format(j, t) \
             + r' = ({0:g},{1:g})$'.format(a, b)
    return ab_str


def tikz_save(figname):
    tikzplotlib.clean_figure()
    # default tikz (width, height) = (240pt, 207pt) (manual 4.10.01)
    tikzplotlib.save(os.path.join(tex_img_path, figname + '.tex'),
                     axis_height='120pt', axis_width='150pt')


def main():
    # Create plot for male - female beta distribution example
    beta_example_gender()
    # Save figure
    tikzplotlib.clean_figure()
    tikzplotlib.save(os.path.join(tex_img_path, 'beta_example_gender.tex'),
                     axis_width=size.w(1.25),
                     axis_height=size.h(1.25))

    # Create beta distribution examples that show how they change
    # This is for the slide deck and wil be smaller
    beta_example_change(ab0=(1, 1), history=[1, 0, 1])

    # Create plot for paper
    beta_example_change_paper()


if __name__ == '__main__':
    plt.close('all')

    # main()
    beta_example_change_paper()

    plt.show()