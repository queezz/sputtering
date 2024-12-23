import numpy as np
import matplotlib.pyplot as plt

# Parameters
lattice_size = 100  # Number of lattice sites
num_atoms = 10  # Number of diffusing atoms
steps = 1000  # Total number of Monte Carlo steps

# Initialize the lattice: 1 for atoms, 0 for empty sites
lattice = np.zeros(lattice_size, dtype=int)
atom_positions = np.random.choice(lattice_size, num_atoms, replace=False)
lattice[atom_positions] = 1


# Function to visualize the lattice
def plot_lattice(lattice, step):
    plt.figure(figsize=(8, 1))
    plt.scatter(range(len(lattice)), lattice, c="blue", marker="|", s=100)
    plt.title(f"Step {step}")
    plt.yticks([])
    plt.show()


# Monte Carlo simulation
for step in range(steps):
    # Select a random atom
    atom_index = np.random.choice(np.where(lattice == 1)[0])

    # Choose a direction: -1 (left) or +1 (right)
    direction = np.random.choice([-1, 1])
    new_position = (
        atom_index + direction
    ) % lattice_size  # Periodic boundary conditions

    # Attempt a move if the new site is empty
    if lattice[new_position] == 0:
        lattice[atom_index] = 0
        lattice[new_position] = 1

    # Visualize the lattice every 100 steps
    if step % 100 == 0:
        plot_lattice(lattice, step)
