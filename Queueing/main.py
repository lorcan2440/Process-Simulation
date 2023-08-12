import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MultipleLocator

import simpy

import random
import math
import itertools


class MMcNK_Queue:
    def __init__(self, lambda_: float = 1, mu: float = 1, c: int = 1,
                 N: int = float('inf'), K: int = float('inf'), env: simpy.Environment = None):     
        '''
        Define an M/M/c/N/K queue, a continuous-time Markov chain (CTMC) and stochastic process
        in which 'customers' from a 'population' arrive and are served acoording to a Poisson point process,
        with options for a finite queue size and finite population size.
        
        #### Arguments
        
        - `lambda_` (float, default = 1): the rate parameter for customers attempting to
        enter the system.
        - `mu` (float, default = 1): the rate parameter for each customer leaving a service node.
        - `c` (int, default = 1): the number of service nodes.
        - `N` (int, default = float('inf')): the maximum possible size of the system, so
        that the maximum queue length is `N - c`. If unset, the size is unlimited.
        - `K` (int, default = float('inf')): the maximum possible size of the population.
        If `K = N` then the queue is closed, so that customers leaving the service node
        immediately may wish to enter the queue again.
        - `env` (simpy.Environment, default = None): a `SimPy` environment instance to use
        in this simulation.
        ''' 
        self.env = simpy.Environment() if env is None else env
        self.server = simpy.Resource(self.env, capacity=c)
        self.queue = simpy.Store(self.env) if N == float('inf') else simpy.Store(self.env, capacity=N - c)
        self.lamda = lambda_
        self.mu = mu
        self.c = c
        self.K = K
        self.N = N
        self.num_people_served = 0  # people who entered the queue, went to a server, and left
        self.num_people_blocked_data = [(0, 0)]  # people who tried to enter but found a full queue, so did not get served never got served
        self.results = {}

    def run_until_time(self, runtime: float):
        '''
        Run the simulation of the queue for a set simulation end time.
        
        #### Arguments
        
        `runtime` (float): amount of time after which to stop the simulation, measuring in the simulated world.

        #### Returns

        `self.results` (dict[int: list[float]]): information about all customers who entered the system.
        The key is the index of the customer.
        The value is `[time_of_entering_queue, time_moved_to_counter, time_left_system]`.
        These entries are `None` if the customer did not make it to the counter
        or leave the system by the time the simulation ended.
        '''
        self.env.process(self.generate_arrivals_forever())
        self.env.run(until=runtime)
        return self.results

    def generate_arrivals_forever(self, end_condition_customers: int = float('inf')):
        id_num = 0
        while True:
            # wait for the next customer
            inter_arrival_time = random.expovariate(self.lamda)
            yield self.env.timeout(inter_arrival_time)  # wait a random entry time

            # check there are some customers in the population - if not, cycle
            is_people_in_population = (self.K - len(self.queue.items) + len(self.server.users) > 0)
            if not is_people_in_population:
                continue

            # a wild customer appears! Can they fit in the queue?
            is_free_queue_space = (len(self.queue.items) < self.queue.capacity)
            if is_free_queue_space:
                # admit to queue and enter process for them
                self.results.update({id_num: [self.env.now, None, None, True]})
                self.queue.put(id_num)
                self.env.process(self.process_customer(id_num))
            else:
                # record their attempted entry but do not enter the queue
                self.num_people_blocked_data.append((self.env.now, self.num_people_blocked_data[-1][1] + 1))
                self.results.update({id_num: [self.env.now, None, None, False]})

            # end simulation if end condition met
            if id_num >= end_condition_customers:
                break
            id_num += 1

    def process_customer(self, id_num):
        # serve a customer at the front of the queue
        with self.server.request() as request:
            yield request
            _id_moved = self.queue.get()  # remove this customer from the queue
            self.results[id_num][1] = self.env.now  # time moved to the counter
            service_time = random.expovariate(self.mu)
            yield self.env.timeout(service_time)
            self.results[id_num][2] = self.env.now  # time left the system
            self.num_people_served += 1

    def get_statistics(self):

        # calculate wait times per customer
        system_wait_times, queue_wait_times, service_times = zip(*[
            [val[2] - val[0], val[1] - val[0], val[2] - val[1]] for _, val in self.results.items() \
                if all([x is not None for x in val])])
        mean_queue_wait_time = sum(queue_wait_times) / len(system_wait_times)

        # calculate queue lengths
        join_times = [(1, val) for val in [data[0] for data in self.results.values() if data[3]]]
        leave_times = [(-1, val) for val in [data[1] for data in self.results.values() if data[3]]]
        leave_times = list(filter(lambda x: x[1] is not None, leave_times))
        change_times = sorted(join_times + leave_times, key=lambda x: x[1])
        queue_changes = [c[0] for c in change_times]
        queue_cumulative_changes = itertools.accumulate(queue_changes)
        queue_lengths = [(0, 0)] + [(total, time) for (total, (_, time)) in \
                                    zip(queue_cumulative_changes, change_times)]

        # get time weighted average queue length
        time_sum = 0
        for x, y in zip(queue_lengths[:-1], queue_lengths[1:]):
            time_sum += x[0] * (y[1] - x[1])
        mean_queue_length = time_sum / queue_lengths[-1][1]

        # get theoretical stats
        expected_queue_time, expected_queue_length = self.get_theoretical_statistics()

        print(f'Queue length: Simulated {mean_queue_length}, Theoretical {expected_queue_length}, '
              f'Diff {100 * (1 - mean_queue_length / expected_queue_length):.4f}%')
        print(f'Queue time: Simulated {mean_queue_wait_time}, Theoretical {expected_queue_time}, '
              f'Diff {100 * (1 - mean_queue_wait_time / expected_queue_time):.4f}%')
    
    def get_theoretical_statistics(self):
        '''
        Only valid for M/M/c/N queue (K is infinite)
        '''
        rho = self.lamda / (self.c * self.mu)  # server utilisation
        pi_0 = (1 + sum([(self.c * rho) ** n / math.factorial(n) for n in range(1, self.c + 1)])
                + (self.c * rho) ** self.c / math.factorial(self.c) * sum([
                    rho ** (n - self.c) for n in range(self.c + 1, self.N + 1)
                ])) ** -1
        pi_n = lambda n: pi_0 * (self.c * rho) ** n / math.factorial(n) if 0 <= n < self.c \
            else pi_0 * (self.c * rho) ** n * self.c ** (self.c - n) / math.factorial(self.c)
        pi_N = pi_0 * (self.c * rho) ** self.N / (math.factorial(self.c) * self.c ** (self.N - self.c))
        E_queue_length = pi_0 * (self.c * rho) ** self.c * rho / (math.factorial(self.c) * (1 - rho) ** 2) * (
            1 - rho ** (self.N - self.c) - (self.N - self.c) * (1 - rho) * rho ** (self.N - self.c)
        )
        lambda_eff = self.lamda * (1 - pi_N)
        E_queue_time = E_queue_length / lambda_eff  # Little's lemma

        return E_queue_time, E_queue_length

    def animate_queue_length(self, window_size: float = 1):
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        self.event_timepoints = [] 
        self.num_people_served_data = []
        self.queue_length_data = []
        self.server_length_data = []

        def init():
            pass

        def update(frame):

            # process up to next event
            self.env.step()

            # prepare axes
            ax1.cla()
            ax1.set_xlabel("Time (minutes)")
            ax1.set_ylabel("Number of People")
            ax1.set_xlim(max(0, self.env.now - window_size), max(window_size, self.env.now))
            ax1.yaxis.set_major_locator(MultipleLocator(base=1.0))

            ax2.cla()
            ax2.set_ylabel("Cumulative Number of People")
            ax2.yaxis.set_label_position("right")
            ax2.yaxis.tick_right()
            ax2.grid(False)
            ax2.minorticks_off()

            # get current state and update history
            queue_length = len(self.queue.items)
            server_length = len(self.server.users)
            self.queue_length_data.append((self.env.now, queue_length))  # Store (time, queue_length)
            self.server_length_data.append((self.env.now, server_length))  # Store (time, server_length)
            self.num_people_served_data.append((self.env.now, self.num_people_served))
            self.num_people_blocked_data.append((self.env.now, self.num_people_blocked_data[-1][1]))
            self.event_timepoints.append(self.env.now)

            # add data points before step changes to make to the plot
            # look like rectangles rather than lines joining points
            if frame != 0 and self.queue_length_data[-1][1] != self.queue_length_data[-2][1]:
                self.queue_length_data.insert(-1, (self.env.now, self.queue_length_data[-2][1]))
            if frame != 0 and self.server_length_data[-1][1] != self.server_length_data[-2][1]:
                self.server_length_data.insert(-1, (self.env.now, self.server_length_data[-2][1]))
            if frame != 0 and self.num_people_served_data[-1][1] != self.num_people_served_data[-2][1]:
                self.num_people_served_data.insert(-1, (self.env.now, self.num_people_served_data[-2][1]))
            if frame != 0 and self.num_people_blocked_data[-1][1] != self.num_people_blocked_data[-2][1]:
                self.num_people_blocked_data.insert(-1, (self.env.now, self.num_people_blocked_data[-2][1]))

            # plot
            ax1.plot(*zip(*self.queue_length_data), label='In the Queue')
            ax1.plot(*zip(*self.server_length_data), color='black', alpha=0.3, label='Being Served')
            ax2.plot(*zip(*self.num_people_served_data), color='green', linestyle=':', alpha=0.3, label='Total Served')
            ax2.plot(*zip(*self.num_people_blocked_data), color='red', linestyle='--', alpha=0.3, label='Total Blocked')
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper center')

        ani = FuncAnimation(fig, update, init_func=init, interval=30, blit=False)
        plt.show()

def main():
    # Simulation
    q = MMcNK_Queue(lambda_=5, mu=3, c=2, N=7)
    q.env.process(q.generate_arrivals_forever())  # Start customer arrival process

    # Animate queue length
    q.animate_queue_length(window_size=3)
    q.get_statistics()

if __name__ == '__main__':
    main()