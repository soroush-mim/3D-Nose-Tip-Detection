import argparse
from utils import *
import os
import time

def sec_sol(cloud):
    most_protruding_point = get_most_protruding_point(cloud)
    clusters = DBSCAN_clustering(cloud)
    nose_cluster = find_nose_cluster(cloud, clusters)
    nose_tip_x = find_middle_point(nose_cluster)
    nose_tip = np.array([nose_tip_x, most_protruding_point[1], most_protruding_point[2]])
    return nose_tip


parser = argparse.ArgumentParser(description='nose tip detection')
parser.add_argument('--path', type=str, help='path to ply file or directory')
args = parser.parse_args()

# determineing which the path is a ply file or a directory
if args.path.endswith('.ply'):

    file_name = os.path.basename(args.path)
    cloud = load_ply_file(args.path)

    # first solution
    # calculate the time needed to run the function
    start = time.time()
    most_protruding_point = get_most_protruding_point(cloud)
    end = time.time()
    print('first solution: most_protruding_point', most_protruding_point)
    print('time(S):', end - start)
    plot_mat(np.asarray(cloud.points), most_protruding_point, save_path=f'Results/{file_name}_first_solution.png')
    

    # sec solution
    start = time.time()
    nose_tip = sec_sol(cloud)
    end = time.time()

    print('sec solution: ', nose_tip)
    print('time(S):', end - start)
    plot_mat(np.asarray(cloud.points), nose_tip, save_path=f'Results/{file_name}_sec_solution.png')


# if the path is a directory, read all the ply files in the directory
# and plot the 3D point clouds
elif os.path.isdir(args.path):
    cloud_files = [os.path.join(args.path, f) for f in os.listdir(args.path) if f.endswith('.ply')]
    clouds = [load_ply_file(f) for f in cloud_files]
    first_sols = []
    sec_sols = []
    time_first = 0
    time_sec = 0
    for i, cloud in enumerate(clouds):
        print(f'Processing cloud {i + 1} of {len(clouds)}')

        start = time.time()
        most_protruding_point = get_most_protruding_point(cloud)
        end = time.time()
        time_first += (end - start)
        first_sols.append(most_protruding_point)

        start = time.time()
        nose_tip = sec_sol(cloud)
        end = time.time()
        time_sec += (end - start)
        sec_sols.append(nose_tip)

    #calculating MAE for both solutions
    ground_truth = [np.loadtxt(f.replace('.ply', '.txt')) for f in cloud_files]
    first_sols = np.array(first_sols)
    sec_sols = np.array(sec_sols)

    mae_first = np.mean(np.abs(first_sols - ground_truth))
    mae_sec = np.mean(np.abs(sec_sols - ground_truth))

    print('MAE first solution:', mae_first)
    print('first solution avg time(S):', time_first/len(clouds))
    print('sec solution avg time(S):', time_sec/len(clouds))
    print('MAE sec solution:', mae_sec)
else:
    print("Invalid path")

