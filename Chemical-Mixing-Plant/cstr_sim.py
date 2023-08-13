import numpy as np
from scipy.integrate import solve_ivp
from scipy.misc import derivative
from matplotlib import animation as anim, pyplot as plt
import matplotlib as mpl
from matplotlib.widgets import Slider
from CoolProp.CoolProp import PropsSI

###############
#### SETUP ####
###############

## application constants

FFMPEG_PATH = r'C:\LibsAndApps\ffmpeg-build\bin\ffmpeg.exe'
MPL_STYLESHEET = r'C:\LibsAndApps\Python config files\proplot_style.mplstyle'

## solver settings

INITIAL_T = 0                         # simulation start time
MAX_STEP_T = 0.5                     # maximum integration step size
ANIM_STEP_T = 5                     # update frame every interval
SLIDING_WINDOW_T = 60                 # start moving along with the graph after this time

AXIS_TEMP_MIN = 300
AXIS_TEMP_RANGE = 100
AXIS_CONC_MIN = 0
AXIS_CONC_RANGE = 0.3

## simulation settings

# chemistry and geometry (fixed)
A = 3.274e7         # Arrhenius constant in rate equation               [mol^-1 dm^3 s^-1]
E_A = 5.0e4         # activation energy for reaction in rate equation   [J]
DH = -2.632e4       # enthalpy change for reaction                      [J mol^-1]
R = 8.314           # gas constant                                      [J mol^-1 K^-1]
V = 120             # volume of vessel                                  [dm^3 = litres]
Q = 20              # total flow rate                                   [dm^3 s^-1]
UA = 880 * 3.95     # steam heater overall HTC * surface area           [W K^-1]

# functions of solvent and chemistry
RHO = lambda T: 1e-3 * PropsSI('D', 'T|liquid', T, 'P', 1e5, 'Water')   # solvent density
D_RHO_DT = lambda T: derivative(RHO, T, 1e-3)                           # [kg dm^-3]
C_P = lambda T: PropsSI('C', 'T|liquid', T, 'P', 1e5, 'Water')          # solvent SHC
D_CP_DT = lambda T: derivative(C_P, T, 1e-3)                            # [J kg^-1 K^-1]
K = lambda T: A * np.exp(-E_A / (R * T))                                # rate constant
                                                                        # [mol^-1 dm^3 s^-1]


# manipulated vars (initial values)
Q_A = 10            # inflow rate of A                                  [dm^3 s^-1]
C_A1 = 1.5          # concentration of A at inflow                      [mol dm^-3]
C_B1 = 1.8          # concentration of B at inflow                      [mol dm^-3]
T_A1 = 300          # temperature of inflow of A                        [K]
T_B1 = 320          # temperature of inflow of A                        [K]
T_S = 973           # steam heater temperature                          [K]
T_2_INIT = 300      # initial solvent temperature                       [K]

# constrained vars
Q_B = Q - Q_A                                       # inflow rate of B  [dm^3 s^-1]
WINDOW = int(ANIM_STEP_T / MAX_STEP_T)              # number of integrations per frame
INITIAL_VALUES = np.array([0, 0, 0, T_2_INIT])      # initial c_A, c_B, c_C, T in vessel
DEP_VAR_NAMES = ('$c_A$', '$c_B$', '$c_C$', '$T$')  # labels for dependent variables
history = {
    'last_vals': INITIAL_VALUES,
    'all_t': np.array([INITIAL_T]),
    'all_y_vec': np.array([[y_val] for y_val in INITIAL_VALUES])
    }

## apply settings

vid_writer = anim.FFMpegWriter(fps=30, codec='libx264', bitrate=-1)


###############
#### MODEL ####
###############

def ode_system(t: float, y: np.ndarray,
        const_rho: bool = False, const_shc: bool = False, const_rate: bool = False) -> np.ndarray:

    # get dependent vars
    c_A2, c_B2, c_C2, T_2 = y

    # get slider values
    C_A1 = slider_c_A.val
    C_B1 = slider_c_B.val

    # set temperature-dependent quantities
    k = 0.05584 if const_rate else K(T_2)        # rate constant            [mol^-1 dm^3 s^-1]
    rho = 1.000 if const_rho else RHO(T_2)       # density                  [kg dm^-3]
    c_p = 4200 if const_shc else C_P(T_2)        # isobaric SHC             [J kg^-1 K^-1]
    dcp_dt = 0 if const_shc else D_CP_DT(T_2)    # temperature derivative   [J kg^-1 K^-2]
    drho_dt = 0 if const_rho else D_RHO_DT(T_2)  # temperature derivative   [kg dm^-3 K^-1]
    energy_balance_numer = rho * c_p * (Q_A * T_A1 + Q_B * T_B1 - Q * T_2) \
        - k * V * DH * c_A2 * c_B2 + UA * (T_S - T_2)                     # [J s^-1]
    energy_balance_denom = rho * V * (c_p + T_2 * dcp_dt) \
        + c_p * V * drho_dt                                               # [J K^-1]
    
    # form system of differential equations
    return np.array([
        C_A1 * Q_A / V - c_A2 * Q / V - k * c_A2 * c_B2,
        C_B1 * Q_B / V - c_B2 * Q / V - k * c_A2 * c_B2,
        -c_C2 * Q / V + k * c_A2 * c_B2,
        energy_balance_numer / energy_balance_denom
    ])


###################
#### ANIMATION ####
###################

def animate(frame: int, history: dict[str, np.ndarray], lines: tuple) -> None:

    line_cA, line_cB, line_cC, line_T = lines

    # the new values of t to be plotted in this frame
    t_range = np.array([INITIAL_T + MAX_STEP_T * frame * WINDOW, \
                        INITIAL_T + MAX_STEP_T * (frame + 1) * WINDOW])

    # set first values for this range
    last_vals, all_t, all_y_vec = history['last_vals'], history['all_t'], history['all_y_vec']
    last_vals = all_y_vec[:, -1] if frame != 0 else INITIAL_VALUES

    # solve system over these t bounds
    sol = solve_ivp(ode_system, t_range, last_vals, args=(True, True, False),
        method='LSODA', max_step=MAX_STEP_T)  # solve the system over these t
    
    # record values up to this latest point
    all_t = np.concatenate((all_t, sol.t))
    all_y_vec = np.concatenate((all_y_vec, sol.y), axis=1)

    # chop down lists and set new axis limits
    new_cutoff = all_t[-1] - SLIDING_WINDOW_T
    all_t = all_t[all_t >= new_cutoff]
    all_y_vec = all_y_vec[:, -1 * len(all_t):]

    # set x axis range
    _, t_max = ax_T.get_xlim()
    if all_t[-1] < INITIAL_T + SLIDING_WINDOW_T:
        ax_c.set_xlim(INITIAL_T, INITIAL_T + SLIDING_WINDOW_T)
        ax_T.set_xlim(INITIAL_T, INITIAL_T + SLIDING_WINDOW_T)
    elif all_t[-1] >= t_max:
        ax_c.set_xlim(all_t[-1] - SLIDING_WINDOW_T, all_t[-1])
        ax_T.set_xlim(all_t[-1] - SLIDING_WINDOW_T, all_t[-1])
    
    # set y axis range (fix T)
    ax_c.relim()
    ax_c.autoscale_view()
    ax_T.set_ylim(AXIS_TEMP_MIN, AXIS_TEMP_MIN + AXIS_TEMP_RANGE)

    # update the record of the previous values, passed into the next animation frame
    history.update({'last_vals': last_vals, 'all_t': all_t, 'all_y_vec': all_y_vec})

    # update the lines
    line_cA.set_data(all_t, all_y_vec[0])
    line_cB.set_data(all_t, all_y_vec[1])
    line_cC.set_data(all_t, all_y_vec[2])
    line_T.set_data(all_t, all_y_vec[3])


if __name__ == '__main__':

    # set up figure and axes
    fig = plt.gcf()
    ax_c = plt.subplot(2, 2, 1)
    ax_T = plt.subplot(2, 2, 3)
    ax_img = plt.subplot(1, 2, 2)
    plt.subplots_adjust(bottom=0.20)

    line_cA, = ax_c.plot([], [], label=DEP_VAR_NAMES[0])
    line_cB, = ax_c.plot([], [], label=DEP_VAR_NAMES[1])
    line_cC, = ax_c.plot([], [], label=DEP_VAR_NAMES[2])
    line_T, = ax_T.plot([], [], label=DEP_VAR_NAMES[3])
    lines = (line_cA, line_cB, line_cC, line_T)

    def init_graph():

        mpl.rcParams['animation.ffmpeg_path'] = FFMPEG_PATH
        plt.rcParams["figure.autolayout"] = True
        plt.style.use(MPL_STYLESHEET)

        ax_c.set_title(r'Reaction: $ A + B \rightarrow C $')
        ax_T.set_xlabel(r'time / s')
        ax_c.set_ylabel(r'concentration / $ mol \ dm^{-3} $')
        ax_T.set_ylabel(r'temperature / $ K $')

        ax_c.tick_params(axis='both', which='both', labelsize=7, labelbottom=True)
        ax_T.tick_params(axis='both', which='both', labelsize=7)

        line_cA.set_data([], [])
        line_cB.set_data([], [])
        line_cC.set_data([], [])
        line_T.set_data([], [])
        return line_cA, line_cB, line_cC, line_T

    # set up sliders
    ax_slider_c_A = plt.axes([0.125, 0.1, 0.5, 0.02], facecolor="lightgrey")
    ax_slider_c_B = plt.axes([0.125, 0.06, 0.5, 0.02], facecolor="lightgrey")
    slider_c_A = Slider(ax_slider_c_A, '$ c_A^{(in)} $', 0, 5, valinit=C_A1,
        valfmt=r'%0.3f $ mol \ dm^{-3} $')
    slider_c_B = Slider(ax_slider_c_B, '$ c_B^{(in)} $', 0, 5, valinit=C_B1,
        valfmt=r'%0.3f $ mol \ dm^{-3} $')
    ax_slider_c_B.add_artist(ax_slider_c_B.xaxis)
    ax_slider_c_B.set_xticks(np.arange(0, 5.01, 1))

    # add still image on right
    ax_img.imshow(plt.imread(r'Chemical-Mixing-Plant/cstr_pic.png'))
    ax_img.set_axis_off()

    # render animation
    ani = anim.FuncAnimation(fig, animate, fargs=(history, lines), init_func=init_graph, \
                             interval=30, save_count=100)
    plt.show()

    #ani.save('Chemical-Mixing-Plant/Simulation.mp4', writer=vid_writer)
    print('Finished')