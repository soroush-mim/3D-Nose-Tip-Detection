import argparse
from utils import *
import os
import time
from open3d import visualization


def sec_sol(cloud):
    #perform second solution
    clusters, labels = DBSCAN_clustering(cloud, eps=0.009, min_samples=24)
    nose_cluster = find_nose_cluster(cloud, labels)
    nose_tip_x = find_middle_point(nose_cluster)
    most_protruding_point = nose_cluster[np.argmax(nose_cluster[:, 2])]
    nose_tip = np.array([nose_tip_x, most_protruding_point[1], most_protruding_point[2]])
    return nose_tip


parser = argparse.ArgumentParser(description='nose tip detection')
parser.add_argument('--path', type=str, help='path to ply file or directory')
args = parser.parse_args()

# determineing which the path is a ply file or a directory
if args.path.endswith('.ply'):

    file_name = os.path.basename(args.path)
    cloud = load_ply_file(args.path, normalize=False)
    # visualization.draw_geometries([cloud]) 
    real_nose_tip = np.loadtxt(args.path.replace('.ply', '.txt'))
    plot_mat(np.asarray(cloud.points), real_nose_tip, save_path=f'Results/{file_name}_real_solution.png')

    # first solution
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
    clouds = [load_ply_file(f, normalize=False) for f in cloud_files]
    first_sols = []
    sec_sols = []
    time_first = 0
    time_sec = 0
    for i, cloud in enumerate(clouds):
        print(f'Processing cloud {i + 1} of {len(clouds)}')
        print('File:', cloud_files[i])
        real_nose_tip = np.loadtxt(cloud_files[i].replace('.ply', '.txt'))
        plot_mat(np.asarray(cloud.points), real_nose_tip)

        start = time.time()
        most_protruding_point = get_most_protruding_point(cloud)
        end = time.time()
        time_first += (end - start)
        plot_mat(np.asarray(cloud.points), most_protruding_point)
        first_sols.append(most_protruding_point)

        start = time.time()
        nose_tip = sec_sol(cloud)
        end = time.time()
        time_sec += (end - start)
        plot_mat(np.asarray(cloud.points), nose_tip)
        sec_sols.append(nose_tip)

    #calculating MAE for both solutions
    ground_truth = np.asarray([np.loadtxt(f.replace('.ply', '.txt')) for f in cloud_files])
    first_sols = np.array(first_sols)
    sec_sols = np.array(sec_sols)

    mae_first = np.mean(np.abs(first_sols - ground_truth))
    mae_sec = np.mean(np.abs(sec_sols - ground_truth))

    print('MAE first solution:', mae_first)
    print('first solution avg time(S):', time_first/len(clouds))
    print('MAE sec solution:', mae_sec)
    print('sec solution avg time(S):', time_sec/len(clouds))
    
else:
    print("Invalid path")

