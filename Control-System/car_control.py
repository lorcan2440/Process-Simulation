from matplotlib import animation as anim
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import matplotlib as mpl
import numpy as np

from filterpy.kalman import KalmanFilter

import sys
import logging

from utils import ColouredLoggingFormatter


FFMPEG_PATH = r'C:\LibsAndApps\ffmpeg-build\bin\ffmpeg.exe'  # path to ffmpeg executable
MPL_STYLESHEET = r'C:\LibsAndApps\Python config files\proplot_style.mplstyle'  # path to mpl stylesheet


class Car:

    T_0 = 0
    V_SP_0 = 0
    INITIAL_STATE = [0, 0, 0]  # x, v, a
    DATA_PERIOD = 0.1
    WINDOW_PERIOD = 10.0

    MEASURE_NOISE_STD_DEV = 4.0
    PROCESS_NOISE_STD_DEV = 0.5

    _LOG_FILE = 'car_log.txt'

    def __init__(self) -> None:
        '''
        Create a new car object.
        '''
        # get initial state
        self.t = [self.T_0]
        self.x, self.v_true, self.a = [[var] for var in self.INITIAL_STATE]  # these are all 1D lists
        self.v_meas = [self.v_true[0]]
        self.v_est = [self.v_true[0]]
        self.v_sp = [self.V_SP_0]
        self.e = [self.v_sp[0] - self.v_est[0]]  # error in velocity
        self.e_integral = 0  # running total of error in velocity to speed up integration
        self.num_points_per_frame = int(self.WINDOW_PERIOD / self.DATA_PERIOD)
        self.init_logger()

        # create Kalman filter - track velocity using velocity measurements
        # state dynamics model: x_{k+1} = F x_{k} + B u_{k} + w_{k}
        # measurement model: z_{k} = H x_{k} + v_{k}
        self.kf = KalmanFilter(dim_x=1, dim_z=1, dim_u=1)
        self.kf.x = np.array([[self.v_meas[0]]])  # initial state - column vector
        self.kf.F = np.array([[1]])  # state transition matrix
        self.kf.B = np.array([[self.DATA_PERIOD]])  # control transition matrix
        self.kf.H = np.array([[1]])  # measurement transition matrix
        self.kf.P = np.array([[100]])  # initial covariance matrix in state space
        self.kf.Q = np.array([[self.PROCESS_NOISE_STD_DEV ** 2]])  # process noise covariance matrix
        self.kf.R = np.array([[self.MEASURE_NOISE_STD_DEV ** 2]])  # measurement noise covariance matrix
    
    def init_logger(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(self._LOG_FILE)
        file_handler.setFormatter(file_formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(ColouredLoggingFormatter())
        self.logger.addHandler(file_handler)  # send logs to file
        self.logger.addHandler(stream_handler)  # send logs to console

    def pid_controller(self, Kp: float = 0, Ki: float = 0, Kd: float = 0,
                       Ti: float = None, Td: float = None) -> float:
        '''
        Returns the control signal (acceleration) for the next time step, given the
        current state and the PID gains.

        Input in either the form:
        - u = Kp * e + Ki * (integral e dt) + Kd * de/dt, or
        - u = Kp * (e + 1 / Ti * (integral e dt) + Td * de/dt)
        
        ### Arguments
        #### Required
        #### Optional
        - `Kp` (float, default = 0): proportional gain
        - `Ki` (float, default = 0): integral gain
        - `Kd` (float, default = 0): derivative gain
        - `Ti` (float, default = None): integral time constant
        - `Td` (float, default = None): derivative time constant
        
        ### Returns
        - `float`: _description_
        '''

        if Ti is not None and Td is not None:
            Ki = Kp / Ti
            Kd = Kp * Td

        self.e_integral += self.DATA_PERIOD * (self.e[-1] + self.e[-2]) / 2  # cumulative trapezoidal rule

        prop_term = Kp * self.e[-1]  # proportional term
        integral_term = Ki * self.e_integral  # integral of error
        derivative_term = Kd * (self.e[-1] - self.e[-2]) / self.DATA_PERIOD  # derivative of error
        return prop_term + integral_term + derivative_term
    
    def kalman_filter(self, v_meas: float) -> float:
        '''
        Estimates the velocity using a Kalman filter to smooth the noisy measured velocity.
        
        ### Arguments
        #### Required
        - `v_meas` (float): last measured velocity
        #### Optional
        
        ### Returns
        - `float`: estimation of the velocity
        '''        
        self.kf.predict()
        self.kf.update(z=v_meas)
        return self.kf.x[0, 0]

    def gaussian_noise(self, mean: float = 0, std_dev: float = 1) -> float:
        '''
        Generate a random number from a normal distribution with the given mean and standard deviation.
        Default: drawn from standard normal distribution.
        
        ### Arguments
        #### Required
        #### Optional
        - `mean` (float, default = 0): mean of the normal distribution
        - `std_dev` (float, default = 1): standard deviation of the normal distribution
        
        ### Returns
        - `float`: a single random number from the normal distribution
        '''
        return np.random.normal(mean, std_dev)
    
    def iterate_system(self, slider: Slider):
        # new iteration: calculate system dynamics
        self.t.append(self.t[-1] + self.DATA_PERIOD)
        self.v_true.append(self.v_true[-1] + self.a[-1] * self.DATA_PERIOD)
        # add measurement noise (output disturbance) to velocity reading
        self.v_meas.append(self.v_true[-1] + self.gaussian_noise(std_dev=self.MEASURE_NOISE_STD_DEV))
        # estmate the velocity using a Kalman filter
        self.v_est.append(self.kalman_filter(self.v_meas[-1]))
        # read the setpoint velocity from the slider
        self.v_sp.append(slider.val)
        # calculate the error
        self.e.append(self.v_sp[-1] - self.v_est[-1])

        # compute the control signal
        u = self.pid_controller(Kp=1.5, Ki=0.1, Kd=0.1)
        
        # add process noise (input disturbance) to the acceleration
        self.a.append(u + self.gaussian_noise(std_dev=self.PROCESS_NOISE_STD_DEV))

    def animate(self, frame: int, slider: Slider, ax: plt.Axes, lines: tuple, log: bool = True):
        '''
        Display the next frame, making all calculations required.
        
        ### Arguments
        #### Required
        - `frame` (int): the current frame number
        - `slider` (Slider): the slider object for setting the set point velocity
        - `ax` (plt.Axes): the main axes to make the graph on
        #### Optional
        '''

        # make all calculations
        self.iterate_system(slider)

        # log the current state
        if log:
            self.logger.info(f'Frame = {frame}, t = {self.t[-1]}, '
                            f'v_true = {self.v_true[-1]}, v_meas = {self.v_meas[-1]}, '
                            f'v_est = {self.v_est[-1]}, v_sp = {self.v_sp[-1]}')

        # calculate arrays to be shown in the plot
        t_range = self.t[-1 - self.num_points_per_frame:]  #  upto and including the current time
        v_sp_range = self.v_sp[-1 - self.num_points_per_frame:]
        v_true_range = self.v_true[-1 - self.num_points_per_frame:]
        v_meas_range = self.v_meas[-1 - self.num_points_per_frame:]
        v_est_range = self.v_est[-1 - self.num_points_per_frame:]

        # trim axes, plot data
        ax.set_xlim(max(0, self.t[-1] - self.WINDOW_PERIOD), max(self.WINDOW_PERIOD, self.t[-1]))
        ax.set_ylim(-20, 60)

        line_sp, line_true, line_meas, line_est = lines
        line_sp.set_data(t_range, v_sp_range)
        line_true.set_data(t_range, v_true_range)
        line_meas.set_data(t_range, v_meas_range)
        line_est.set_data(t_range, v_est_range)

        ax.legend(loc='upper left')
    
    def drive(self, save_or_view: str = 'view'):

        fig, axs = plt.subplots(2, 1, figsize=(10, 6),
            gridspec_kw={'height_ratios': [8, 1], 'hspace': 0.3})
        ax_main, ax_slider = axs

        ax_main.set_xlabel('Time / $ s $')
        ax_main.set_ylabel('Velocity / $ ms^{-1} $')

        line_sp, = ax_main.plot([], [], color='gray', linestyle='-', alpha=0.3, label='$v_{sp}$')
        line_true, = ax_main.plot([], [], color='red', alpha=0.3, label='$v_{true}$')
        line_meas, = ax_main.plot([], [], color='red', marker='x', linestyle='', label='$v_{meas}$')
        line_est, = ax_main.plot([], [], color='black', label='$v_{est}$')
        lines = (line_sp, line_true, line_meas, line_est)

        def init_graph():
            plt.style.use(MPL_STYLESHEET)
            mpl.rcParams['animation.ffmpeg_path'] = FFMPEG_PATH

            line_sp.set_data([], [])
            line_true.set_data([], [])
            line_meas.set_data([], [])
            line_est.set_data([], [])
            return line_sp, line_true, line_meas, line_est
        
        slider = Slider(ax_slider, '$v_{sp}$', valmin=-10, valmax=50, valinit=self.v_sp[0])
        ani = anim.FuncAnimation(fig, self.animate, fargs=(slider, ax_main, lines, False),
            init_func=init_graph, interval=self.DATA_PERIOD * 1000, save_count=sys.maxsize)
        
        if save_or_view == 'save':
            ani.save('MyVideo.mp4', writer=anim.FFMpegWriter(fps=30, codec='libx264', bitrate=-1))
        
        plt.show()


if __name__ == '__main__':

    car = Car()
    car.drive(save_or_view='view')