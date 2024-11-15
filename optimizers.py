import numpy as np

class AdamOptimizer:
    def __init__(self, gamma, similarity_matrix, update_sim_matr, objective_function, gradient_function, num_iterations=100, lambda_v=.99, lambda_s=.9999, epsilon=1e10-8, alpha=10):
        self.gamma = gamma
        self.similarity_matrix = similarity_matrix
        self.generate_edge_weights = update_sim_matr
        self.objective_function = objective_function
        self.gradient_function = gradient_function
        self.num_iterations = num_iterations
        self.lambda_v = lambda_v
        self.lambda_s = lambda_s
        self.epsilon = epsilon
        self.alpha = alpha

    def adam_computation(self):
        print("Beggining Optimizations")
        v_curr = np.zeros_like(self.gamma)
        s_curr = np.zeros_like(self.gamma)
        curr_sim_matr = self.similarity_matrix
        curr_gamma = self.gamma
        step = 0

        min_error = float("inf")

        for i in range(num_iterations):
            print("Current Iteration: ", str(i+1))
            print("Computing Gradient")
            gradient = self.gradient_function(curr_sim_matr, curr_gamma)
            print("Current Gradient: ", gradient)
            print("Computing Error")
            curr_error = self.objective_function(adj_matr, gamma)
            print("Current Error: ", curr_error)

            v_next = (self.lambda_v * v_curr) + (1 - self.lambda_v)*gradient

            s_next = (self.lambda_s * s_curr) + (1 - self.lambda_s)*(gradient**2)

            step += 1

            corrected_v = v_next / (1 - self.lambda_v**step)

            corrected_s = s_next / (1 - self.lambda_s**step)

            #print(v_next, " ", s_next, " ", corrected_v, " ", corrected_s)

            print("Current Gamma: ", curr_gamma)

            curr_gamma = curr_gamma - (alpha*(corrected_v))/(epsilon + np.sqrt(corrected_s))

            v_curr = v_next

            s_curr = s_next

            #print((self.alpha*(corrected_v))/(self.epsilon + np.sqrt(corrected_s)))

            print("Next Gamma: ", curr_gamma)

            if curr_error <= min_error:
                min_error = curr_error
                min_gamma = curr_gamma
                #min_sim_matrix = deepcopy(curr_sim_matr)
            curr_sim_matr = self.generate_edge_weights()

        #self.gamma = min_gamma
        #self.similarity_matrix = min_sim_matrix
        return curr_gamma



class SimulatedAnnealingOptimizer:
    def __init__(self, gamma, similarity_matrix, update_sim_matr, objective_function, temperature, min_temp, cooling_rate):
        self.gamma = gamma
        self.similarity_matrix = similarity_matrix
        self.objective_function = objective_function
        self.generate_edge_weights =  update_sim_matr

        self.temperature = temperature
        self.min_temp = min_temp
        self.cooling_rate = cooling_rate
        self.max_iterations = 1000


    def simulated_annealing(self, num_iterations):
        curr_gamma = self.gamma
        curr_energy = self.objective_function(self.similarity_matrix, self.gamma)
        curr_sim_matr = self.similarity_matrix

        for idx in range(num_iterations):
            new_position = self.solution_transition(curr_gamma, self.temperature)

            curr_adj_matr = self.generate_edge_weights(new_position)

            new_energy = self.objective_function(curr_adj_matr, new_position)

            print("Potential New Position: ", new_position)
            print("Potential New Postion Error: ", new_energy)

            alpha = self.acceptance_probability_computation(curr_energy, new_energy, self.temperature)
            print("Potential New Position Acceptance Probability: ", alpha)

            if new_energy < curr_energy and np.random.rand() < alpha:
                curr_gamma = new_position
                curr_energy = new_energy

            self.temperature *= self.cooling_rate

            print("Current Gamma: ", curr_gamma)
            print("Current Error: ", curr_energy)
            print("Current Temperature: ", self.temperature)
            if self.temperature < self.min_temp:
                break

        #self.gamma = curr_gamma
        print("Final Error: ", curr_energy)
        print("Final Gamma: ", self.gamma)
        return curr_gamma

    def solution_transition(self, curr_gamma):
        new_position = curr_gamma + np.random.normal(0, self.temperature, size = len(curr_gamma))
        return new_position

    def acceptance_probability_computation(self, curr_energy, new_energy):
        if new_energy < curr_energy:
            return 1.0
        else:
            return np.exp((curr_energy - new_energy) / self.temperature )

class ParticleSwarmOptimizer:
    def __init__(self, similarity_matrix, gamma, objective_function, update_sim_matr, num_particles, dimensions, max_iter, w=0.5, c1=1.5, c2=1.5):
        self.similarity_matrix = similarity_matrix
        self.gamma = gamma
        self.objective_function = objective_function
        self.generate_edge_weights = update_sim_matr

        self.num_particles = num_particles         # Number of particles in the swarm
        self.dimensions = dimensions               # Number of dimensions in the search space
        self.max_iter = max_iter                   # Maximum number of iterations
        self.w = w                                 # Inertia weight (controls velocity)
        self.c1 = c1                               # Cognitive coefficient (individual learning factor)
        self.c2 = c2                               # Social coefficient (swarm learning factor)

        # Initialize the swarm (random positions and velocities)
        self.positions = np.random.uniform(-100, 100, (num_particles, dimensions))  # Random initial positions
        self.velocities = np.random.uniform(-1, 1, (num_particles, dimensions))  # Random initial velocities
        
        # Best known positions and their corresponding fitness values
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_fitness = np.array([self.objective_function(self.similarity_matrix, p) for p in self.positions])
        
        # Global best position (the best solution found by the swarm)
        self.global_best_position = self.personal_best_positions[np.argmin(self.personal_best_fitness)]
        self.global_best_fitness = np.min(self.personal_best_fitness)

    def update_velocity(self, particle_idx):
        # Update the velocity of each particle
        r1 = np.random.random(self.dimensions)  # Random number for cognitive component
        r2 = np.random.random(self.dimensions)  # Random number for social component
        
        cognitive_velocity = self.c1 * r1 * (self.personal_best_positions[particle_idx] - self.positions[particle_idx])
        social_velocity = self.c2 * r2 * (self.global_best_position - self.positions[particle_idx])
        
        new_velocity = self.w * self.velocities[particle_idx] + cognitive_velocity + social_velocity
        return new_velocity

    def update_position(self, particle_idx):
        # Update the position of each particle based on its velocity
        new_position = self.positions[particle_idx] + self.velocities[particle_idx]
        #return np.clip(new_position, -5, 5)  # Keep positions within bounds (-5, 5)
        return new_position

    def optimize(self):
        curr_adj_matr = self.similarity_matrix
        for iteration in range(self.max_iter):
            for i in range(self.num_particles):
                # Update velocity
                new_velocity = self.update_velocity(i)
                self.velocities[i] = new_velocity

                # Update position
                new_position = self.update_position(i)
                self.positions[i] = new_position
                
                print("Current Position for Agent ", i, ":", new_position)

                curr_adj_matr = self.generate_edge_weights(new_position)

                # Evaluate new fitness
                fitness = self.objective_function(curr_adj_matr, self.positions[i])
                    
                print("Current Fitness for Agent ", i, ":", fitness)

                # Update personal best if necessary
                if fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness
                    self.personal_best_positions[i] = self.positions[i]

            # Update global best
            min_fitness_idx = np.argmin(self.personal_best_fitness)
            if self.personal_best_fitness[min_fitness_idx] < self.global_best_fitness:
                self.global_best_fitness = self.personal_best_fitness[min_fitness_idx]
                self.global_best_position = self.personal_best_positions[min_fitness_idx]

            # Print progress
            if iteration % 10 == 0:
                print(f"Iteration {iteration}/{self.max_iter}, Best Fitness: {self.global_best_fitness}")
        
        return self.global_best_position #, self.global_best_fitness

class SwarmBasedAnnealingOptimizer:
    def __init__(self, similarity_matrix, gamma, objective_function, gradient_function, update_sim_matr, num_particles, dimensions, max_iter, h=0.95, c1=1.5, c2=1.5):
        self.similarity_matrix = similarity_matrix
        self.gamma = gamma
        self.objective_function = objective_function
        self.generate_edge_weights = update_sim_matr
        self.gradient_function = gradient_function

        self.num_particles = num_particles         # Number of particles in the swarm
        self.dimensions = dimensions               # Number of dimensions in the search space
        self.max_iter = max_iter                   # Maximum number of iterations
        self.h = h                                 # step size
        self.c1 = c1                               # Cognitive coefficient (individual learning factor)
        self.c2 = c2                               # Social coefficient (swarm learning factor)

        self.provisional_minimum = float('inf')

        # Initialize the swarm (random positions and velocities)
        self.positions = np.random.uniform(-100, 100, (num_particles, len(self.gamma)))  # Random initial positions
        self.masses = np.ones((1, num_particles))[0] * (1/num_particles)

        # Best known positions and their corresponding fitness values
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_fitness = np.array([self.objective_function(self.similarity_matrix, p) for p in self.positions])

        # Global best position (the best solution found by the swarm)
        self.global_best_position = self.personal_best_positions[np.argmin(self.personal_best_fitness)]
        self.global_best_fitness = np.min(self.personal_best_fitness)

    def update_mass(self, particle_idx):
        # Update the velocity of each particle

        new_mass = self.masses[particle_idx] - ((self.masses[particle_idx] * self.h)*(self.personal_best_fitness[particle_idx] - self.provisional_minimum))
        
        return new_mass

    def update_position(self, particle_idx, eta, curr_adj_matrix):
        
        gradient = self.gradient_function(curr_adj_matrix, self.positions[particle_idx])
        
        '''
        print(self.h)

        print(self.h*gradient)

        print(self.masses[particle_idx])

        print(eta)

        print(np.sqrt(2*self.h*self.masses[particle_idx]))

        print(np.sqrt(2*self.h*self.masses[particle_idx])* eta)
        '''

        new_position = self.positions[particle_idx] - (self.h*gradient*self.personal_best_fitness[particle_idx]) + (np.sqrt(2*self.h*self.masses[particle_idx])* eta)
        return new_position

    def provisional_min_computation(self):
        return np.sum([self.masses[y] * self.personal_best_fitness[y] for y in range(self.num_particles)]) / np.sum(self.masses)

    def optimize(self):
        print("Beggining Optimization")
        curr_adj_matr = self.similarity_matrix
        self.provisional_minimum = self.provisional_min_computation()
        print("Provisional Minimum: ", self.provisional_minimum)
        
        for iteration in range(self.max_iter):
            for i in range(self.num_particles):
                # Update velocity
                new_mass = self.update_mass(i)
                self.masses[i] = new_mass

            ### finished in second for here
            # Update position

            eta = np.random.normal(0, 1, size = len(self.gamma))

            #eta = np.random.normal(0, , size = len(self.gamma))

            for i in range(self.num_particles):

                curr_adj_matr = self.generate_edge_weights(self.positions[i])

                print("Initial Position for Agent ", i, ":", self.positions[i])

                new_position = self.update_position(i, eta, curr_adj_matr)
                    
                self.positions[i] = new_position

                print("Current Position for Agent ", i, ":", new_position)

                # Evaluate new fitness
                fitness = self.objective_function(curr_adj_matr, self.positions[i])

                print("Current Fitness for Agent ", i, ":", fitness)

                # Update personal best if necessary
                if fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness
                    self.personal_best_positions[i] = self.positions[i]

                # Update global best
                min_fitness_idx = np.argmin(self.personal_best_fitness)
                if self.personal_best_fitness[min_fitness_idx] < self.global_best_fitness:
                    self.global_best_fitness = self.personal_best_fitness[min_fitness_idx]
                    self.global_best_position = self.personal_best_positions[min_fitness_idx]

            self.provisional_minimum = self.provisional_min_computation()
            print("Provisional Minimum: ", self.provisional_minimum)


            # Print progress
            print(f"Iteration {iteration}/{self.max_iter}, Best Fitness: {self.global_best_fitness}")
        print("Completed Optimization")
        return self.global_best_position #, self.global_best_fitness

