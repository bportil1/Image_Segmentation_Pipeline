import numpy as np
import scipy.sparse as sp

class AdamOptimizer:
    def __init__(self, similarity_matrix, gamma, update_sim_matr, objective_function, gradient_function, num_iterations=100, lambda_v=.99, lambda_s=.9999, epsilon=1e-8, alpha=10):
        '''
        Adaptive momentum optimizer
        '''
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

    def optimize(self):
        '''
        Optimization function
        '''
        print("Beginning Optimizations")
        v_curr = np.zeros_like(self.gamma)
        s_curr = np.zeros_like(self.gamma)
        curr_sim_matr = self.similarity_matrix
        curr_gamma = self.gamma
        step = 0
        min_error = float("inf")
        
        for i in range(self.num_iterations):
            print(f"Current Iteration: {i+1}")
            print("Computing Gradient")
            gradient = self.gradient_function(curr_sim_matr, curr_gamma)
            print("Current Gradient: ", gradient)
            
            print("Computing Error")
            curr_error = self.objective_function(curr_sim_matr, curr_gamma)
            print("Current Error: ", curr_error)

            v_next = (self.lambda_v * v_curr) + (1 - self.lambda_v) * gradient
            s_next = (self.lambda_s * s_curr) + (1 - self.lambda_s) * (gradient ** 2)
            step += 1
            corrected_v = v_next / (1 - self.lambda_v ** step)
            corrected_s = s_next / (1 - self.lambda_s ** step)

            print("Current Gamma: ", curr_gamma)
            curr_gamma = curr_gamma - (self.alpha * (corrected_v)) / (self.epsilon + np.sqrt(corrected_s))
            print("Updated Gamma: ", curr_gamma)

            v_curr = v_next
            s_curr = s_next


            if curr_error <= min_error:
                min_error = curr_error
                min_gamma = curr_gamma

            curr_sim_matr = self.generate_edge_weights(curr_gamma)
        return curr_gamma

class SimulatedAnnealingOptimizer:
    def __init__(self, similarity_matrix, gamma, update_sim_matr, objective_function, num_iterations=1000, temperature=10, min_temp=.001, cooling_rate=.9):
        '''
        Simulated Annealing Optimizer
        '''
        self.gamma = gamma
        self.similarity_matrix = similarity_matrix
        self.objective_function = objective_function
        self.generate_edge_weights = update_sim_matr
        self.num_iterations = num_iterations
        self.temperature = temperature
        self.min_temp = min_temp
        self.cooling_rate = cooling_rate

    def optimize(self):
        '''
        Optimization function
        '''
        print("Beggining Simulated Annealing Optimization")
        update_ctr = 0
        curr_gamma = self.gamma
        curr_energy = self.objective_function(self.similarity_matrix, self.gamma)
        curr_sim_matr = self.similarity_matrix
        
        for idx in range(self.num_iterations):
            new_position = self.solution_transition(curr_gamma)
            curr_adj_matr = self.generate_edge_weights(new_position)
            new_energy = self.objective_function(curr_adj_matr, new_position)
            
            print("Potential New Position: ", new_position)
            print("Potential New Position Error: ", new_energy)

            alpha = self.acceptance_probability_computation(curr_energy, new_energy)
            print("Potential New Position Acceptance Probability: ", alpha)

            if new_energy < curr_energy :
                curr_gamma = new_position
                curr_energy = new_energy
                update_ctr = 0
            elif np.random.rand() > (1-alpha):
                curr_gamma = new_position
                curr_energy = new_energy
                update_ctr = 0
            else: 
                update_ctr += 1

            self.temperature *= self.cooling_rate

            print("Current Gamma: ", curr_gamma)
            print("Current Error: ", curr_energy)
            print("Current Temperature: ", self.temperature)

            if self.temperature < self.min_temp or update_ctr > 20:
                print("Early Convergence, Breaking")
                break

        print("Final Error: ", curr_energy)
        print("Final Gamma: ", curr_gamma)
        return curr_gamma

    def solution_transition(self, curr_gamma):
        '''
        Compute new possible position
        '''
        new_position = curr_gamma + np.random.normal(0, self.temperature, size=len(curr_gamma))
        return new_position

    def acceptance_probability_computation(self, curr_energy, new_energy):
        '''
        Compute probability that new position will be accepted
        '''
        if new_energy < curr_energy:
            return 1.0
        else:
            return np.exp(-((new_energy-curr_energy) / self.temperature))

class ParticleSwarmOptimizer:
    def __init__(self, similarity_matrix, gamma, objective_function, update_sim_matr, num_particles, dimensions, max_iter, w=0.5, c1=1.5, c2=1.5):
        '''
        Particle Swarm Optimization function
        '''
        self.similarity_matrix = similarity_matrix
        self.gamma = gamma
        self.objective_function = objective_function
        self.generate_edge_weights = update_sim_matr
        self.num_particles = num_particles
        self.dimensions = dimensions
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.positions = np.random.uniform(-100, 100, (num_particles, dimensions))
        self.velocities = np.random.uniform(-1, 1, (num_particles, dimensions))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_fitness = np.array([self.objective_function(self.similarity_matrix, p) for p in self.positions])
        self.global_best_position = self.personal_best_positions[np.argmin(self.personal_best_fitness)]
        self.global_best_fitness = np.min(self.personal_best_fitness)

    def update_velocity(self, particle_idx):
        '''
        Update particles movement rates
        '''
        r1 = np.random.random(self.dimensions)
        r2 = np.random.random(self.dimensions)
        cognitive_velocity = self.c1 * r1 * (self.personal_best_positions[particle_idx] - self.positions[particle_idx])
        social_velocity = self.c2 * r2 * (self.global_best_position - self.positions[particle_idx])
        new_velocity = self.w * self.velocities[particle_idx] + cognitive_velocity + social_velocity
        return new_velocity

    def update_position(self, particle_idx):
        '''
        Update particle's position
        '''
        new_position = self.positions[particle_idx] + self.velocities[particle_idx]
        return new_position

    def optimize(self):
        '''
        Optimization function
        '''
        curr_adj_matr = self.similarity_matrix
        for iteration in range(self.max_iter):
            for i in range(self.num_particles):
                new_velocity = self.update_velocity(i)
                self.velocities[i] = new_velocity
                new_position = self.update_position(i)
                self.positions[i] = new_position
                
                print("Current Position for Agent ", i, ":", new_position)
                curr_adj_matr = self.generate_edge_weights(new_position)
                fitness = self.objective_function(curr_adj_matr, self.positions[i])
                
                print("Current Fitness for Agent ", i, ":", fitness)
                if fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness
                    self.personal_best_positions[i] = self.positions[i]

            min_fitness_idx = np.argmin(self.personal_best_fitness)
            if self.personal_best_fitness[min_fitness_idx] < self.global_best_fitness:
                self.global_best_fitness = self.personal_best_fitness[min_fitness_idx]
                self.global_best_position = self.personal_best_positions[min_fitness_idx]
                print(f"Iteration {iteration}/{self.max_iter}, Best Fitness: {self.global_best_fitness}")

        return self.global_best_position


