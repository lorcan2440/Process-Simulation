import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import animation as anim, pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import axes3d
from matplotlib.widgets import Slider


INITIAL_T = 0                         # simulation start time
INITIAL_VALUES = np.array([5, 3, 1])  # initial values of [x, y, z]
MAX_STEP_T = 0.0005                     # maximum integration step size
ANIM_STEP_T = 0.05                     # update frame every interval
SLIDING_WINDOW_T = 5                 # start moving along with the graph after this time
WINDOW = int(ANIM_STEP_T / MAX_STEP_T)              # number of integrations per frame
DEP_VAR_NAMES = ('$x$', '$y$', '$z$')  # labels for dependent variables
history = {
    'last_vals': INITIAL_VALUES,
    'all_t': np.array([INITIAL_T]),
    'all_y_vec': np.array([[y_val] for y_val in INITIAL_VALUES])
    }


def ode_system(t: float, X: np.ndarray) -> np.ndarray:

    x, y, z = X

    A = np.array([50, 50, 50])
    #A = A / np.linalg.norm(A)
    a, b, c = A

    # form system of differential equations: [dx/dt = ..., dy/dt = ..., dz/dt = ...]
    return np.array([
        b * z - c * y,
        c * x - a * z,
        a * y - b * x
    ])


def animate(frame: int, history: dict[str, np.ndarray], lines: tuple) -> None:

    line_C, line_x, line_y, line_z = lines

    # the new values of t to be plotted in this frame
    t_range = np.array([INITIAL_T + MAX_STEP_T * frame * WINDOW, \
                        INITIAL_T + MAX_STEP_T * (frame + 1) * WINDOW])

    # set first values for this range
    last_vals, all_t, all_y_vec = history['last_vals'], history['all_t'], history['all_y_vec']
    last_vals = all_y_vec[:, -1] if frame != 0 else INITIAL_VALUES

    # solve system over these t bounds
    sol = solve_ivp(ode_system, t_range, last_vals, method='LSODA', max_step=MAX_STEP_T)  # solve the system over these t
    
    # record values up to this latest point
    all_t = np.concatenate((all_t, sol.t))
    all_y_vec = np.concatenate((all_y_vec, sol.y), axis=1)

    # chop down lists and set new axis limits
    new_cutoff = all_t[-1] - SLIDING_WINDOW_T
    all_t = all_t[all_t >= new_cutoff]
    all_y_vec = all_y_vec[:, -1 * len(all_t):]

    # set x axis range
    _, t_max = ax2d.get_xlim()
    if all_t[-1] < INITIAL_T + SLIDING_WINDOW_T:
        ax2d.set_xlim(INITIAL_T, INITIAL_T + SLIDING_WINDOW_T)
    elif all_t[-1] >= t_max:
        ax2d.set_xlim(all_t[-1] - SLIDING_WINDOW_T, all_t[-1])
    
    # set y axis range (fix T)
    ax2d.relim()
    ax2d.autoscale_view()

    # update the record of the previous values, passed into the next animation frame
    history.update({'last_vals': last_vals, 'all_t': all_t, 'all_y_vec': all_y_vec})

    # update the lines
    line_C.set_data(all_y_vec[0], all_y_vec[1])
    line_C.set_3d_properties(all_y_vec[2])
    line_x.set_data(all_t, all_y_vec[0])
    line_y.set_data(all_t, all_y_vec[1])
    line_z.set_data(all_t, all_y_vec[2])

    #ax3d.plot(all_y_vec[0][-1], all_y_vec[1][-1], all_y_vec[2][-1], 'o', color='red')


if __name__ == '__main__':

    fig = plt.figure(figsize=plt.figaspect(0.5))

    ax3d = fig.add_subplot(1, 2, 1, projection='3d')
    ax2d = fig.add_subplot(1, 2, 2)

    fig.subplots_adjust(wspace=0.5)

    line_C, = ax3d.plot([], [])
    line_x, = ax2d.plot([], [], label=DEP_VAR_NAMES[0])
    line_y, = ax2d.plot([], [], label=DEP_VAR_NAMES[1])
    line_z, = ax2d.plot([], [], label=DEP_VAR_NAMES[2])
    lines = (line_C, line_x, line_y, line_z)

    ax2d.legend(loc='upper left')

    def init_graph():

        plt.rcParams["figure.autolayout"] = True

        ax2d.set_title(r'(x, y, z) coordinates')
        ax2d.set_xlabel(r'time / s')
        ax2d.set_ylabel(r'position coordinate / $ m $')
        ax2d.tick_params(axis='both', which='both', labelsize=7, labelbottom=True)

        ax3d.set_xlabel('x')
        ax3d.set_ylabel('y')
        ax3d.set_zlabel('z')

        ax3d.set_xlim3d([-5, 5])
        ax3d.set_ylim3d([-5, 5])
        ax3d.set_zlim3d([-5, 5])

        line_C.set_data([], [])
        line_x.set_data([], [])
        line_y.set_data([], [])
        line_z.set_data([], [])
        return line_x, line_y, line_z

    ani = anim.FuncAnimation(fig, animate, fargs=(history, lines), init_func=init_graph, \
                             interval=30, save_count=100)
    
    fig.suptitle(r'Solution to vector equation, $\dot{\mathbf{r}} = \mathbf{a} \times \mathbf{r}$')
    plt.show()
