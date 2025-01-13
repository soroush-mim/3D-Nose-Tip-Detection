import argparse
from utils import *
import os
import time
from open3d import visualization


def process_file(path):
    
    file_name = os.path.basename(path)
    cloud = load_ply_file(path, normalize=False)
    # visualization.draw_geometries([cloud]) 

    # plot real nose tip
    real_nose_tip = np.loadtxt(path.replace('.ply', '.txt'))
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


def process_dir(dir):
    # calculates MAE for both solutions
    cloud_files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.ply')]
    clouds = [load_ply_file(f, normalize=False) for f in cloud_files]

    first_sols = []
    sec_sols = []
    time_first = 0
    time_sec = 0

    for i, cloud in enumerate(clouds):
        print(f'Processing cloud {i + 1} of {len(clouds)}')
        print('File:', cloud_files[i])

        #plot reak nose tip
        # real_nose_tip = np.loadtxt(cloud_files[i].replace('.ply', '.txt'))
        # plot_mat(np.asarray(cloud.points), real_nose_tip)

        # first solution
        start = time.time()
        most_protruding_point = get_most_protruding_point(cloud)
        end = time.time()
        time_first += (end - start)
        # plot_mat(np.asarray(cloud.points), most_protruding_point)
        first_sols.append(most_protruding_point)

        # sec solution
        start = time.time()
        nose_tip = sec_sol(cloud)
        end = time.time()
        time_sec += (end - start)
        # plot_mat(np.asarray(cloud.points), nose_tip)
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


if __name__=='__main__':

    #definig parser
    parser = argparse.ArgumentParser(description='nose tip detection')
    parser.add_argument('--path', type=str, help='path to ply file or directory')
    args = parser.parse_args()

    # determineing which the path is a ply file or a directory
    if args.path.endswith('.ply'):
        process_file(args.path)

    # if the path is a directory, read all the ply files in the directory
    # and calculate mae for both solutions
    elif os.path.isdir(args.path):
        process_dir(args.path)
        
    else:
        print("Invalid path")

