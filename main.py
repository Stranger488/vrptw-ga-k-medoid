import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import rgb2hex
from mpl_toolkits.mplot3d.axis3d import Axis

from libs.pyVRP import plot_tour_coordinates
from pyvrp_solver import PyVRPSolver
from spatiotemporal import Spatiotemporal
from solver import Solver

from config_reduced import *
# from config_standard import *


# fix wrong z-offsets in 3d plot
def _get_coord_info_new(self, renderer):
    mins, maxs, cs, deltas, tc, highs = self._get_coord_info_old(renderer)
    correction = deltas * [0, 0, 1.0 / 4]
    mins += correction
    maxs -= correction
    return mins, maxs, cs, deltas, tc, highs


if not hasattr(Axis, "_get_coord_info_old"):
    Axis._get_coord_info_old = Axis._get_coord_info
Axis._get_coord_info = _get_coord_info_new

numpy_rand = np.random.RandomState(42)


def make_solution(init_dataset, tws_all, service_time_all, k=None, distance='spatiotemp', plot=False, text=False,
                  output_dir='cluster_result/', eval_method='default'):
    # Init and calculate all spatiotemporal distances
    spatiotemporal = Spatiotemporal(init_dataset, tws_all, service_time_all, k1, k2, k3, alpha1, alpha2)
    spatiotemporal.calculate_all_distances()

    # Reduce depot
    dataset_reduced = init_dataset[1:][:]
    tws_reduced = tws_all[1:]

    spatio_points_dist = np.delete(spatiotemporal.euclidian_dist_all, 0, 0)
    spatio_points_dist = np.delete(spatio_points_dist, 0, 1)

    spatiotemporal_points_dist = np.delete(spatiotemporal.spatiotemporal_dist_all, 0, 0)
    spatiotemporal_points_dist = np.delete(spatiotemporal_points_dist, 0, 1)

    if distance == 'spatiotemp':
        solver = Solver(Z, spatiotemporal_points_dist, P, ng, Pc, Pm, Pmb, k=k, numpy_rand=numpy_rand)
    else:
        solver = Solver(Z, spatio_points_dist, P, ng, Pc, Pm, Pmb, k=k, numpy_rand=numpy_rand)

    # Result will be an array of clusters, where row is a cluster, value in column - point index
    result = solver.solve()

    # Collect result, making datasets of space data and time windows
    res_dataset = np.array([[dataset_reduced[point] for point in cluster] for cluster in result])
    res_tws = np.array([[tws_reduced[point] for point in cluster] for cluster in result])

    for i, cluster in enumerate(res_dataset):
        # Create coords file
        coord_df = pd.DataFrame(res_dataset[i], columns=['X', 'Y'])

        coord_df.loc[-1] = init_dataset[0]
        coord_df.index = coord_df.index + 1  # shifting index
        coord_df.sort_index(inplace=True)

        coord_df.to_csv('cluster_result/' + output_dir + 'coords{}.txt'.format(i), sep=' ', index=False)

        # Create time parameters file
        tw_df = pd.DataFrame(res_tws[i], columns=['TW_early', 'TW_late'])

        tw_df.loc[-1] = tws_all[0]
        tw_df.index = tw_df.index + 1  # shifting index
        tw_df.sort_index(inplace=True)

        tw_df.insert(2, 'TW_service_time', [service_time_all[i][0] for i in range(len(tw_df))])

        tw_df.to_csv('cluster_result/' + output_dir + 'params{}.txt'.format(i), index=False, sep=' ')

    # Output distance matrix
    distance_df = pd.DataFrame(spatiotemporal.euclidian_dist_all)
    distance_df.to_csv('cluster_result/' + output_dir + 'distance_matrix.txt', sep=' ', index=False, header=False)

    tsptw_solver = PyVRPSolver(method='tsp')
    tsptw_results, plots_data = tsptw_solver.solve_tsp(res_dataset.shape[0], data_dir=output_dir)

    if plot:
        plot_clusters(dataset_reduced, res_dataset, res_tws, spatiotemporal.MAX_TW,
                      np.array(init_dataset[0]), np.array(tws_all[0]), plots_data, axes_text=distance, text=text)

    # Evaluate solution
    evaluation = evaluate_solution(tsptw_results, eval_method=eval_method)

    return evaluation


def make_solution_pyvrp(points_dataset, tws_all, service_time_all, k=None, plot=False, text=False,
                        output_dir='pyvrp_result/'):
    pyvrp_solver = PyVRPSolver(method='vrp')
    pyvrp_results, plots_data = pyvrp_solver.solve_vrp(points_dataset, tws_all, service_time_all, output_dir=output_dir)

    # Evaluate solution
    evaluation = evaluate_solution(pyvrp_results)

    # Reduce depot
    dataset_reduced = points_dataset[1:][:]
    tws_reduced = tws_all[1:]

    if plot:
        max_TW = max(np.subtract(tws_all[:, 1], tws_all[:, 0]))
        plot_clusters(dataset_reduced, points_dataset, tws_reduced, max_TW,
                      np.array(points_dataset[0]), np.array(tws_all[0]), plots_data, axes_text='pyvrp', text=text)

    return evaluation


def read_standard_dataset(dataset, points_dataset, tws_all, service_time_all):
    for i in range(dataset.shape[0]):
        tws_all = np.concatenate((tws_all, [[dataset['READY_TIME'][i],
                                             dataset['DUE_DATE'][i]]]), axis=0)

        service_time_all = np.concatenate((service_time_all, [[dataset['SERVICE_TIME'][i]]]), axis=0)

        points_dataset = np.concatenate((points_dataset, [[dataset['XCOORD'][i],
                                                           dataset['YCOORD'][i]]]), axis=0)

    return points_dataset, tws_all, service_time_all


def solve(filename, distance='spatiotemp', plot=False, k=None, output_dir='cluster_result/', text=False,
          method='cluster', eval_method='default'):
    dataset = pd.read_fwf('data/' + filename)

    points_dataset = np.empty((0, 2))
    tws_all = np.empty((0, 2))
    service_time_all = np.empty((0, 1))

    points_dataset, tws_all, service_time_all = read_standard_dataset(dataset, points_dataset, tws_all,
                                                                      service_time_all)
    if method == 'cluster':
        val = make_solution(points_dataset, tws_all, service_time_all, k=int(dataset['VEHICLE_NUMBER'][0]),
                            distance=distance, plot=plot, output_dir=output_dir, text=text, eval_method=eval_method)
    elif method == 'pyvrp':
        val = make_solution_pyvrp(points_dataset, tws_all, service_time_all, output_dir=output_dir)
    else:
        val = None

    return val


def plot_clusters(init_dataset, dataset, tws, max_tw, depo_spatio, depo_tws, plots_data, axes_text=None, text=False):
    plt.rc('font', size=5)  # controls default text sizes
    plt.rc('xtick', labelsize=8)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=8)

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=figsize_standart,
                             dpi=dpi_standart, subplot_kw={'projection': '3d'})

    axes.set_xlabel('x')
    axes.set_ylabel('y')
    axes.set_zlabel('z')

    axes.set_title(axes_text)

    colors = [rgb2hex([np.random.random_sample(), np.random.random_sample(), np.random.random_sample()])
              for _ in dataset]

    for i in range(dataset.shape[0]):
        plot_with_tws(dataset[i], tws[i], max_tw, colors[i], axes)
        plot_tour_coordinates(plots_data[i]['coordinates'], plots_data[i]['ga_vrp'], axes, colors[i],
                              route=plots_data[i]['route'])

    axes.scatter(depo_spatio[0], depo_spatio[1], 0.0, c='black', s=1)

    axes.scatter(depo_spatio[0], depo_spatio[1], depo_tws[0], c='black', s=1)
    axes.scatter(depo_spatio[0], depo_spatio[1], depo_tws[1], c='black', s=1)

    axes.bar3d(depo_spatio[0] - depth / 8., depo_spatio[1] - depth / 8., 0.0, width / 4., depth / 4., max_tw,
               color='black')

    if text:
        for i, data in enumerate(init_dataset):
            axes.text(data[0], data[1], 0.0, str(i + 1))

    axes.set_zlim(0, None)


def plot_with_tws(spatial_data, tws, max_tw, colors, axes):
    cluster_size = spatial_data[0].size

    x_data = np.array([i[0] for i in spatial_data])
    y_data = np.array([i[1] for i in spatial_data])

    z_data1 = np.array([i[0] for i in tws])
    z_data2 = np.array([i[1] for i in tws])
    dz_data = np.abs(np.subtract(z_data1, z_data2))

    axes.bar3d(x_data - depth / 8., y_data - depth / 8., 0.0, width / 4., depth / 4., max_tw)
    axes.bar3d(x_data - depth / 2., y_data - depth / 2., z_data1, width, depth, dz_data)

    axes.scatter(x_data, y_data, 0.0, c=colors, s=cluster_size)
    axes.scatter(x_data, y_data, z_data1, c=colors, s=cluster_size)
    axes.scatter(x_data, y_data, z_data2, c=colors, s=cluster_size)


def evaluate_solution(tsptw_results, eval_method='default'):
    total_dist = 0.0
    wait_time = 0.0
    late_time = 0.0

    for result in tsptw_results:
        total_dist += result['Distance'][len(result) - 1]
        wait_time += result['Wait_Time'][len(result) - 1]
        late_time += result['Late_Time'][len(result) - 1]

    if eval_method == 'by_distance':
        return total_dist

    return c_D * total_dist + c_T * wait_time + c_L * late_time


def solve_and_plot(datasets):
    st = []
    s = []

    pyvrp = []
    for dataset in datasets:
        print(dataset['name'])
        if dataset['method'] == 'pyvrp':
            pyvrp.append(solve(dataset['data_file'], distance='spatial', plot=dataset['plot'],
                               output_dir=dataset['output_dir'], text=dataset['text'], method=dataset['method'],
                               eval_method=dataset['eval_method']))
        else:
            st.append(solve(dataset['data_file'], distance='spatiotemp', plot=dataset['plot'],
                            output_dir=dataset['output_dir'], text=dataset['text'], method=dataset['method'],
                            eval_method=dataset['eval_method']))
            s.append(solve(dataset['data_file'], distance='spatial', plot=dataset['plot'],
                           output_dir=dataset['output_dir'], text=dataset['text'], method=dataset['method'],
                           eval_method=dataset['eval_method']))

    for i, dataset in enumerate(datasets):
        if dataset['method'] == 'pyvrp':
            print("Pyvrp res on {}: {}".format(dataset['name'], pyvrp[i]))
        else:
            print("Spatiotemporal res on {}: {}".format(dataset['name'], st[i]))
            print("Spatial res on {}: {}\n".format(dataset['name'], s[i]))

    if True in [d['plot'] for d in datasets]:
        plt.show()


if __name__ == '__main__':
    test_dataset = {
        'data_file': 'test.txt',
        'output_dir': 'test_output/',
        'plot': True,
        'name': 'test',
        'text': True,
        'method': 'cluster',
        'eval_method': 'by_distance'
    }
    # solve_and_plot([test_dataset, ])

    r109_reduced_dataset = {
        'data_file': 'r109_reduced.txt',
        'output_dir': 'r109_reduced_output/',
        'plot': True,
        'name': 'r109_reduced',
        'text': False,
        'method': 'cluster',
        'eval_method': 'by_distance'
    }
    # solve_and_plot([r109_reduced_dataset, ])

    # --- c104, r110, rc103 --- #
    # --- c201, r201, rc201 --- #

    c104_dataset = {
        'data_file': 'c104_mod.txt',
        'output_dir': 'c104_output/',
        'plot': False,
        'name': 'c104',
        'text': False,
        'method': 'cluster',
        'eval_method': 'by_distance'
    }
    r110_dataset = {
        'data_file': 'r110_mod.txt',
        'output_dir': 'r110_output/',
        'plot': False,
        'name': 'r110',
        'text': False,
        'method': 'cluster',
        'eval_method': 'by_distance'
    }
    rc103_dataset = {
        'data_file': 'rc103_mod.txt',
        'output_dir': 'rc103_output/',
        'plot': False,
        'name': 'rc103',
        'text': False,
        'method': 'cluster',
        'eval_method': 'by_distance'
    }

    # solve_and_plot([c104_dataset])
    solve_and_plot([r110_dataset])
    # solve_and_plot([rc103_dataset])

    c201_dataset = {
        'data_file': 'c201_mod.txt',
        'output_dir': 'c201_output/',
        'plot': False,
        'name': 'c201',
        'text': False,
        'method': 'cluster',
        'eval_method': 'by_distance'
    }
    r201_dataset = {
        'data_file': 'r201_mod.txt',
        'output_dir': 'r201_output/',
        'plot': False,
        'name': 'r201',
        'text': False,
        'method': 'cluster',
        'eval_method': 'by_distance'
    }
    rc201_dataset = {
        'data_file': 'rc201_mod.txt',
        'output_dir': 'rc201_output/',
        'plot': False,
        'name': 'rc201',
        'text': False,
        'method': 'cluster',
        'eval_method': 'by_distance'
    }
    # solve_and_plot([c201_dataset])
    # solve_and_plot([r201_dataset])
    # solve_and_plot([rc201_dataset])
