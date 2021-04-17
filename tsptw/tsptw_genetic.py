import pandas as pd
import random
import numpy as np
import copy
import os


class TSPTWGenetic:
    def __init__(self):
        pass

    # Function: Build Coordinates
    def build_coordinates(self, distance_matrix):
        a           = distance_matrix[0,:].reshape(distance_matrix.shape[0], 1)
        b           = distance_matrix[:,0].reshape(1, distance_matrix.shape[0])
        m           = (1/2)*(a**2 + b**2 - distance_matrix**2)
        w, u        = np.linalg.eig(np.matmul(m.T, m))
        s           = (np.diag(np.sort(w)[::-1]))**(1/2)
        coordinates = np.matmul(u, s**(1/2))
        coordinates = coordinates.real[:,0:2]
        return coordinates

    # Function: Build Distance Matrix
    def build_distance_matrix(self, coordinates):
       a = coordinates
       b = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
       return np.sqrt(np.einsum('ijk,ijk->ij',  b - a,  b - a)).squeeze()

    # Function: Subroute Distance
    def evaluate_distance(self, distance_matrix, depot, subroute):
        subroute_i    = depot + subroute
        subroute_j    = subroute + depot
        subroute_ij   = [(subroute_i[i], subroute_j[i]) for i in range(0, len(subroute_i))]
        distance      = list(np.cumsum(distance_matrix[tuple(np.array(subroute_ij).T)]))
        distance[0:0] = [0.0]
        return distance

    # Function: Subroute Time
    def evaluate_time(self, distance_matrix, parameters, depot, subroute):
        tw_early   = parameters[:, 0]
        tw_late = parameters[:, 1]
        tw_st      = parameters[:, 2]
        subroute_i = depot + subroute
        subroute_j = subroute + depot
        wait       = [0]*len(subroute_j)
        time       = [0]*len(subroute_j)
        late = [0] * len(subroute_j)
        for i in range(0, len(time)):
            time[i] = time[i] + distance_matrix[(subroute_i[i], subroute_j[i])]
            if (time[i] > tw_late[subroute_j][i]):
                late[i] = time[i] - tw_late[subroute_j][i]

            if (time[i] < tw_early[subroute_j][i]):
                wait[i] = tw_early[subroute_j][i] - time[i]
                time[i] = tw_early[subroute_j][i]

            time[i] = time[i] + tw_st[subroute_j][i]
            if (i + 1 <= len(time) - 1):
                time[i + 1] = time[i]
        time[0:0] = [0]
        wait[0:0] = [0]
        late[0:0] = [0]
        return wait, time, late

    # Function: Subroute Cost
    def evaluate_cost(self, dist, wait, late, parameters, depot, subroute):
        tw_wc     = np.array([10.0 for _ in range(len(parameters))])
        tw_lc     = np.array([100.0 for _ in range(len(parameters))])
        subroute_ = depot + subroute + depot
        cost      = [0]*len(subroute_)
        cost = [1.0 + y*wait_val + z*late_val if x == 0 else 1.0 + x*1.0 + y*wait_val + z*late_val for x, y, z, wait_val, late_val in zip(dist, wait, late, tw_wc[subroute_], tw_lc[subroute_])]
        return cost

    # Function: Subroute Cost
    def evaluate_cost_penalty(self, dist, time, wait, late, parameters, depot, subroute, penalty_value, route):
        tw_late = parameters[:, 1]
        tw_st   = parameters[:, 2]
        tw_wc   = np.array([1.0 for _ in range(len(parameters))])
        tw_lc   = np.array([100.0 for _ in range(len(parameters))])
        if (route == 'open'):
            subroute_ = depot + subroute
        else:
            subroute_ = depot + subroute + depot
        pnlt = 0
        cost = [0]*len(subroute_)
        pnlt = pnlt + sum(x > y + z for x, y, z in zip(time, tw_late[subroute_] , tw_st[subroute_]))
        cost = [1.0 + y*val + z*val if x == 0 else cost[0] + x*1.0 + y*val + z*val for x, y, z, val in zip(dist, wait, late, tw_wc[subroute_])]

        cost[-1] = cost[-1] + pnlt*penalty_value
        return cost[-1]

    # Function: Solution Report
    def show_report(self, solution, distance_matrix, parameters, route):
        column_names = ['Route', 'Activity', 'Job', 'Wait_Time', 'Arrive_Time',
                        'Leave_Time', 'Distance', 'Late_Time']
        tt         = 0
        td         = 0
        lt         = 0
        wt         = 0
        tw_st      = parameters[:, 2]
        report_lst = []
        for i in range(0, len(solution[1])):
            dist       = self.evaluate_distance(distance_matrix, solution[0][i], solution[1][i])
            wait, time, late = self.evaluate_time(distance_matrix, parameters, solution[0][i], solution[1][i])
            if (route == 'closed'):
                subroute = [solution[0][i] + solution[1][i] + solution[0][i] ]
            elif (route == 'open'):
                subroute = [solution[0][i] + solution[1][i] ]
            else:
                subroute = None

            for j in range(0, len(subroute[0])):
                if (j == 0):
                    activity    = 'start'
                    arrive_time = round(time[j],2)
                else:
                    activity = None
                    arrive_time = round(time[j] - tw_st[subroute[0][j]] - wait[j],2)
                if (j > 0 and j < len(subroute[0]) - 1):
                    activity = 'service'
                if (j == len(subroute[0]) - 1):
                    activity = 'finish'
                    if (time[j] > tt):
                        tt = time[j]
                    td = td + dist[j]

                lt = lt + late[j]
                wt = wt + wait[j]

                report_lst.append(['#' + str(i+1), activity, subroute[0][j], round(wait[j], 2), arrive_time, round(time[j], 2), round(dist[j], 2), round(late[j], 2)])
        report_lst.append(['-//-', '-//-', '-//-', '-//-', '-//-', '-//-', '-//-', '-//--'])
        report_lst.append(['TOTAL', '', '', round(wt, 2), '', round(tt, 2), round(td, 2), round(lt, 2)])
        td = td - dist[1]
        wt = wt - wait[1]
        tt = tt - time[1]
        report_lst.append(['DIST W\O FIRST', '', '', round(wt, 2), '', round(tt, 2), round(td, 2), round(lt, 2)])
        report_df = pd.DataFrame(report_lst, columns=column_names)
        return report_df

    # Function: Route Evalution & Correction
    def target_function(self, population, distance_matrix, parameters, penalty_value, route):
        cost     = [[0] for _ in range(len(population))]
        tw_late  = parameters[:, 1]
        tw_st    = parameters[:, 2]

        if (route == 'open'):
            end =  2
        else:
            end =  1
        for k in range(0, len(population)): # k individuals
            individual = copy.deepcopy(population[k])
            size       = len(individual[1])
            i          = 0
            pnlt       = 0
            while (size > i): # i subroutes
                dist = self.evaluate_distance(distance_matrix, individual[0][i], individual[1][i])
                wait, time, late = self.evaluate_time(distance_matrix, parameters, depot = individual[0][i], subroute = individual[1][i])

                cost_s = self.evaluate_cost(dist, wait, late, parameters, depot = individual[0][i], subroute = individual[1][i])
                if (route == 'open'):
                    subroute_ = individual[0][i] + individual[1][i]
                else:
                    subroute_ = individual[0][i] + individual[1][i] + individual[0][i]
                # pnlt = pnlt + sum(x > y + z for x, y, z in zip(time, tw_late[subroute_] , tw_st[subroute_]))

                cost[k][0] = cost[k][0] + cost_s[-end]
                # + pnlt*penalty_value
                size = len(individual[1])
                i = i + 1
        cost_total = copy.deepcopy(cost)
        return cost_total, population

    # Function: Initial Population
    def initial_population(self, coordinates = 'none', distance_matrix = 'none', population_size = 5):
        try:
            distance_matrix.shape[0]
        except:
            distance_matrix = self.build_distance_matrix(coordinates)

        depots     = [[0]]
        vehicles   = [[0]]
        clients    = list(range(1, distance_matrix.shape[0]))
        population = []
        for i in range(0, population_size):
            clients_temp    = copy.deepcopy(clients)
            routes          = []
            routes_depot    = []
            routes_vehicles = []
            while (len(clients_temp) > 0):
                e = random.sample(vehicles, 1)[0]
                d = random.sample(depots, 1)[0]
                c = random.sample(clients_temp, len(clients_temp))

                routes_vehicles.append(e)
                routes_depot.append(d)
                routes.append(c)
                clients_temp = [item for item in clients_temp if item not in c]
            population.append([routes_depot, routes, routes_vehicles])
        return population

    # Function: Fitness
    def fitness_function(self, cost, population_size):
        fitness = np.zeros((population_size, 2))
        for i in range(0, fitness.shape[0]):
            fitness[i,0] = 1/(1 + cost[i][0] + abs(np.min(cost)))
        fit_sum      = fitness[:,0].sum()
        fitness[0,1] = fitness[0,0]
        for i in range(1, fitness.shape[0]):
            fitness[i,1] = (fitness[i,0] + fitness[i-1,1])
        for i in range(0, fitness.shape[0]):
            fitness[i,1] = fitness[i,1]/fit_sum
        return fitness

    # Function: Selection
    def roulette_wheel(self, fitness):
        ix     = 0
        random = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
        for i in range(0, fitness.shape[0]):
            if (random <= fitness[i, 1]):
              ix = i
              break
        return ix

    # Function: TSP Crossover - BRBAX (Best Route Better Adjustment Recombination)
    def crossover_tsp_brbax(self, parent_1, parent_2):
        offspring = copy.deepcopy(parent_2)
        cut       = random.sample(list(range(0,len(parent_1[1][0]))), 2)
        cut.sort()
        rand      = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
        A         = parent_1[1][0][cut[0]:cut[1]]
        B         = [item for item in parent_2[1][0] if item not in A ]
        if (rand > 0.5):
            A.reverse()
        offspring[1][0] = A + B
        return offspring

    # Function: TSP Crossover - BCR (Best Cost Route Crossover)
    def crossover_tsp_bcr(self, parent_1, parent_2, distance_matrix, penalty_value, parameters, route):
        offspring = copy.deepcopy(parent_2)
        cut       = random.sample(list(range(0,len(parent_1[1][0]))), 2)
        for i in range(0, 2):
            d_1            = float('+inf')
            A              = parent_1[1][0][cut[i]]
            best           = []
            parent_2[1][0] = [item for item in parent_2[1][0] if item not in [A] ]
            insertion      = copy.deepcopy([ parent_2[0][0], parent_2[1][0], parent_2[2][0] ])
            dist_list      = [self.evaluate_distance(distance_matrix, insertion[0], insertion[1][:n] + [A] + insertion[1][n:]) for n in range(0, len(parent_2[1][0]) + 1)]
            wait_time_list = [self.evaluate_time(distance_matrix, parameters, insertion[0], insertion[1][:n] + [A] + insertion[1][n:]) for n in range(0, len(parent_2[1][0]) + 1)]
            insertion_list = [insertion[1][:n] + [A] + insertion[1][n:] for n in range(0, len(parent_2[1][0]) + 1)]
            d_2_list       = [self.evaluate_cost_penalty(dist_list[n], wait_time_list[n][1], wait_time_list[n][0], wait_time_list[n][2], parameters, insertion[0], insertion_list[n], penalty_value, route) for n in range(0, len(dist_list))]
            d_2 = min(d_2_list)
            if (d_2 <= d_1):
                d_1   = d_2
                best  = insertion_list[d_2_list.index(min(d_2_list))]
            parent_2[1][0] = best
            if (d_1 != float('+inf')):
                offspring = copy.deepcopy(parent_2)
        return offspring

    # Function: Breeding
    def breeding(self, cost, population, fitness, distance_matrix, elite, penalty_value, parameters, route):
        offspring = copy.deepcopy(population)
        if (elite > 0):
            cost, population = (list(t) for t in zip(*sorted(zip(cost, population))))
            for i in range(0, elite):
                offspring[i] = copy.deepcopy(population[i])
        for i in range (elite, len(offspring)):
            parent_1, parent_2 = self.roulette_wheel(fitness), self.roulette_wheel(fitness)
            while parent_1 == parent_2:
                parent_2 = random.sample(range(0, len(population) - 1), 1)[0]
            parent_1 = copy.deepcopy(population[parent_1])
            parent_2 = copy.deepcopy(population[parent_2])
            rand = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
            # TSP - Crossover
            if (len(parent_1[1]) == 1 and len(parent_2[1]) == 1):
                if (rand > 0.5):
                    offspring[i] = self.crossover_tsp_brbax(parent_1, parent_2)
                    offspring[i] = self.crossover_tsp_bcr(offspring[i], parent_2, distance_matrix, penalty_value, parameters = parameters, route = route)
                elif (rand <= 0.5):
                    offspring[i] = self.crossover_tsp_brbax(parent_2, parent_1)
                    offspring[i] = self.crossover_tsp_bcr(offspring[i], parent_1, distance_matrix, penalty_value, parameters = parameters, route = route)
        return offspring

    # Function: Mutation - Swap
    def mutation_tsp_vrp_swap(self, individual):
        if (len(individual[1]) == 1):
            k1 = random.sample(list(range(0, len(individual[1]))), 1)[0]
            k2 = k1
        else:
            k  = random.sample(list(range(0, len(individual[1]))), 2)
            k1 = k[0]
            k2 = k[1]
        cut1                    = random.sample(list(range(0, len(individual[1][k1]))), 1)[0]
        cut2                    = random.sample(list(range(0, len(individual[1][k2]))), 1)[0]
        A                       = individual[1][k1][cut1]
        B                       = individual[1][k2][cut2]
        individual[1][k1][cut1] = B
        individual[1][k2][cut2] = A
        return individual

    # Function: Mutation - Insertion
    def mutation_tsp_vrp_insertion(self, individual):
        if (len(individual[1]) == 1):
            k1 = random.sample(list(range(0, len(individual[1]))), 1)[0]
            k2 = k1
        else:
            k  = random.sample(list(range(0, len(individual[1]))), 2)
            k1 = k[0]
            k2 = k[1]
        cut1 = random.sample(list(range(0, len(individual[1][k1])))  , 1)[0]
        cut2 = random.sample(list(range(0, len(individual[1][k2])+1)), 1)[0]
        A    = individual[1][k1][cut1]
        del individual[1][k1][cut1]
        individual[1][k2][cut2:cut2] = [A]
        if (len(individual[1][k1]) == 0):
            del individual[0][k1]
            del individual[1][k1]
            del individual[2][k1]
        return individual

    # Function: Mutation
    def mutation(self, offspring, mutation_rate, elite):
        for i in range(elite, len(offspring)):
            probability = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
            if (probability <= mutation_rate):
                rand = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
                if (rand <= 0.5):
                    offspring[i] = self.mutation_tsp_vrp_insertion(offspring[i])
                elif(rand > 0.5):
                    offspring[i] = self.mutation_tsp_vrp_swap(offspring[i])
            for k in range(0, len(offspring[i][1])):
                if (len(offspring[i][1][k]) >= 2):
                    probability = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
                    if (probability <= mutation_rate):
                        rand = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
                        cut  = random.sample(list(range(0,len(offspring[i][1][k]))), 2)
                        cut.sort()
                        C    = offspring[i][1][k][cut[0]:cut[1]+1]
                        if (rand <= 0.5):
                            random.shuffle(C)
                        elif(rand > 0.5):
                            C.reverse()
                        offspring[i][1][k][cut[0]:cut[1]+1] = C
        return offspring

    # Function: Elite Distance
    def elite_distance(self, individual, distance_matrix, route):
        if (route == 'open'):
            end = 2
        else:
            end = 1
        td = 0
        for n in range(0, len(individual[1])):
            td = td + self.evaluate_distance(distance_matrix, depot = individual[0][n], subroute = individual[1][n])[-end]
        return round(td,2)

    # GA-VRP Function
    def genetic_algorithm_tsp(self, coordinates, distance_matrix, parameters, population_size = 5, route = 'closed', mutation_rate = 0.1, elite = 0, generations = 50, penalty_value = 100000, graph = True):
        count           = 0
        solution_report = ['None']

        parameters[0, 0] = 0

        population       = self.initial_population(coordinates, distance_matrix, population_size = population_size)
        cost, population = self.target_function(population, distance_matrix, parameters, penalty_value, route = route)
        fitness          = self.fitness_function(cost, population_size)
        cost, population = (list(t) for t in zip(*sorted(zip(cost, population))))
        elite_ind        = self.elite_distance(population[0], distance_matrix, route = route)
        cost             = copy.deepcopy(cost)
        solution         = copy.deepcopy(population[0])
        print('Generation = ', count, ' Distance = ', elite_ind, ' f(x) = ', round(cost[0][0],2))
        while (count <= generations-1):
            offspring        = self.breeding(cost, population, fitness, distance_matrix, elite, penalty_value, parameters, route)
            offspring        = self.mutation(offspring, mutation_rate = mutation_rate, elite = elite)
            cost, population = self.target_function(offspring, distance_matrix, parameters, penalty_value, route = route)
            fitness          = self.fitness_function(cost, population_size)
            cost, population = (list(t) for t in zip(*sorted(zip(cost, population))))
            elite_child      = self.elite_distance(population[0], distance_matrix, route = route)
            if(elite_ind > elite_child):
                elite_ind = elite_child
                solution  = copy.deepcopy(population[0])
            count = count + 1
            print('Generation = ', count, ' Distance = ', elite_ind, ' f(x) = ', round(cost[0][0],2))

        solution_report = self.show_report(solution, distance_matrix, parameters, route = route)

        return solution_report, solution
