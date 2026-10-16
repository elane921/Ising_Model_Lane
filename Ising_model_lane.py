#pip install scipy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Set a seed as a starting point for generating random values, so that test runs are reproducible
# Set "seed = None" when finished testing
seed = 12
rng = np.random.default_rng(seed)

N = 100
# rng.choice (random number generator) picks from [-1, 1] for each of the N*N cells,
# with assigned probability of 0.3 and 0.7, necessary to bias initial magnetization

spins = rng.choice([-1, 1], size=(N, N),p=[0.3, 0.7])

print(spins)
#Generating a plot to show the initial state of the spins
plt.imshow(spins)
plt.show()

from scipy.signal import convolve2d

# Generating an array to define the nearest neighbors at any given point within the matrix
nn = np.array([[0, 1, 0],
                [1, 0, 1],
                [0, 1, 0]])

# Convolve function works by aligning the "nn" array with each point in the "spins" matrix
# and multiplying the values aligned between the array and matrix then summing the products
# "mode='same'" generates an output from the convolution that is also 50x50
# "boundary='wrap'" wraps the 2d matrix into a torus, so that none of the edges in the matrix
#are missing nearest neighbors
neighbor_sum = convolve2d(spins, nn, mode='same', boundary='wrap')

# Derived from Hamiltonian = - J * summation across matrix
# (central term in martix * summation of nearest neighbor spins)
J = 1.0  # In context of neural networks, coupling constant set as dimensionless constant
total_energy = -J * np.sum(spins * neighbor_sum)

print(total_energy)


# spins = initial matrix, n_iters = number of iterations to run in algorithm
# BJ = inverse of temperature * Boltzmann's constant, energy = total energy of matrix from start
# snapshot_interval = how often to save an image of the matrix for the animation
def metropolis(spins, n_iters, BJ, energy, snapshot_interval=None):
#give the length of one dimension of the spins matrix, redefining N from earlier in code
    N = spins.shape[0]

    # making a duplicate matrix of spins for iterations to be run over
    spins = spins.copy()

    # Making empty arrays to track net magnetization and net energy at each step which are
    #filled with each successive iteration
    net_spins = np.zeros(n_iters)
    net_energy = np.zeros(n_iters)
    # Creating an empty list called snapshots to hold captured images of matrix
    snapshots = []

    for t in range(n_iters):
        # Selecting a randomly generated integer between 0 and N (length/width) of matrix
        #for both a x and y
        x = rng.integers(0, N)
        y = rng.integers(0, N)

        spin_i = spins[x, y]      # naming the current randomly selected spin
        spin_f = -spin_i          # naming the proposed spin flip and assigning the opposite value

        # Calculating the sum of the nearest neighbors for the randomly selected spin_i
        # "% N" wraps the matrix into a torus so that there are no edges or corners missing
        # any nearest neighbors
        neighbor_sum = (spins[(x + 1) % N, y] + spins[(x - 1) % N, y] +
                         spins[x, (y + 1) % N] + spins[x, (y - 1) % N])

        # Calculating the change in energy for just the one flipped spin rather than
        # recalculating the total energy across the whole matrix for each iteration
        E_i = -J * spin_i * neighbor_sum
        E_f = -J * spin_f * neighbor_sum
        dE = E_f - E_i

        if dE > 0:
            # an increase in energy within the system is not energetically favorable,
            # so only some of the time will the algorithm accept this change
            # rng.random generate a random number between 0 and 1
            # np.exp(-BJ * dE) = e^(-BJ* dE) to meet the detailed balance condition
            # converging to the Boltzmann distribution
            if rng.random() < np.exp(-BJ * dE):
                spins[x, y] = spin_f
                energy += dE
        else:
            # if the net energy of the system will decrease with the proposed flip
            #then the change is always accepted
            spins[x, y] = spin_f
            energy += dE

        # Recording the spins and energy after each run
        net_spins[t] = spins.sum()
        net_energy[t] = energy

        # Save a copy of the matrix every snapshot_interval iterations, only saves snapshots
        # if interval is included within input for algorithm "t % snapshot_interval == 0" =
        # save a snapshot each time the remainder of t is equal to zero
        if snapshot_interval is not None and t % snapshot_interval == 0:
            snapshots.append(spins.copy())
        #takes a copy for the selected matrix and adds to snapshots list
# after algorithm is run the filled arrays with information tracked across each iteration
# is returned along with the final matrix of spins
    return net_spins, net_energy, spins, snapshots
#input for the algorithm
net_spins, net_energy, final_spins, snapshots = metropolis(spins, n_iters=1000000, BJ=0.4, energy=total_energy, snapshot_interval=5000)

#plotting total energy over iterations
plt.plot(net_energy)
plt.xlabel("Iteration")
plt.ylabel("Total energy")
plt.show()

#plotting average spins over iterations
avg_spins = net_spins / (N * N)

plt.plot(avg_spins)
plt.xlabel("Iteration")
plt.ylabel("Average spin")
plt.show()
# selecting figure structure to animate on
fig, ax = plt.subplots()
# creating initial image to animate on, "ax.imshow" = matplot method to display a 2D matrix
# as an image, selecting first saved snapshot matrix, "vmin=-1, vmax=1" = fixing color scale
# for assigned values in matrix
im = ax.imshow(snapshots[0], vmin=-1, vmax=1)
ax.set_title("Spin configuration at iteration 0")

# update function is called once per frame, swapping in the next saved snapshot
def update(frame):
    im.set_data(snapshots[frame])
    ax.set_title(f"Spin configuration at iteration {frame * 5000}")
    return [im]
# animation function from matplot, pulls the figure to animate frames on, "frames=len(snapshot)" =
# plays all captured snapshots in order of saving, "interval=100" = play a frame every 100 ms,
# "blit=False" = turns of blitting optimization
ani = animation.FuncAnimation(fig, update, frames=len(snapshots), interval=100, blit=False)
plt.show()

# Modeling phase transition over range of temperatures

# Generates a range of temperatures to test across, starting 1 and going to 4 with 25 total temperatures
#  BJ = 1/T since J = 1
T_range = np.linspace(1.0, 4.0, 25)
#  BJ = 1/T since J = 1
BJ_range = 1 / T_range

n_iters_per_T = 1000000       # iterations to run at each temperature
remove_in_frac = 0.5           # fraction of early iterations discarded as equilibration

# Creating an array filled with zeros the same length as T_range to be filled
# as for loop iterates
avg_mag_vs_T = np.zeros(len(T_range))

# enumerate takes the values from the BJ_range array and assigns them an index which corresponds
# based upon position in array, matching with "num:" in "T_range"
for i, BJ_val in enumerate(BJ_range):
    # Set starting conditions for each matrix, calculating nearest neighbors sums as before
    start_spins = rng.choice([-1, 1], size=(N, N), p=[0.3, 0.7])
    start_neighbor_sum = convolve2d(start_spins, nn, mode='same', boundary='wrap')
    start_energy = -J * np.sum(start_spins * start_neighbor_sum)

    # Rerunning the metropolis algorithm at this temperature's BJ value, with "_" to indicate that
    # energy and snapshots will not be captured across each iteration of this run
    net_spins_T, net_energy_T, _, _ = metropolis(start_spins, n_iters=n_iters_per_T, BJ=BJ_val, energy=start_energy)

    # Removing designated fraction of runs from "remove_in_frac" for period before the system has equilibrated,
    remove_in = int(n_iters_per_T * remove_in_frac)
    # Calculating average magnetization, "net_spins_T[remove_in:]" keeping only elements from
    # array from remove_in and onwards, taking absolute value, then calculating mean, dividing by
    # total elements in matrix to get mean spin
    avg_mag_vs_T[i] = np.mean(np.abs(net_spins_T[remove_in:])) / (N * N)

# Plotting average magnetization per spin against temperature
plt.plot(T_range, avg_mag_vs_T, marker='o')
plt.xlabel("Temperature")
plt.ylabel("Average |magnetization| per spin")
plt.title("Phase transition in the 2D Ising model")
plt.show()