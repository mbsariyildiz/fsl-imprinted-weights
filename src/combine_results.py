import os
import argparse
import glob
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--exp_root_dir', type=str,
                    help='directory which includes all experiment folders')
parser.add_argument('--keys', type=str, default='',
                    help='comma separated names of the attributes saved in log file')
args = parser.parse_args()

def main():

    experiment_folders = glob.glob(os.path.join(args.exp_root_dir, '*')) # experiment folder for each different seed
    experiment_folders = [f for f in experiment_folders if os.path.isdir(f)]
    print ('Experiment folders under {}:'.format(args.exp_root_dir))
    for folder in experiment_folders: print ('\t{}'.format(folder))

    keys = args.keys.split(',') # names of the log attributes in log files
    print ('Attributes whose values will be combined:', keys)
    combined_logs = {}
    for k in keys:
        combined_logs[k] = []

    for folder in experiment_folders:
        
        log_file = os.path.join(folder, 'logs.npz') # log file in the experiment folder
        L = np.load(log_file)
        print ('Keys in {}:'.format(folder))
        for k in L.keys(): print ('\t', k)

        for k in keys:
            v = L[k]
            if len(v.shape) < 2:
                v = v.reshape([-1, 1])
            combined_logs[k].append(v)

    for k in keys:
        combined_logs[k] = np.concatenate(combined_logs[k], axis=1).mean(axis=1)

    np.savez(
        os.path.join(args.exp_root_dir, 'combined_logs.npz'),
        **combined_logs)


if __name__ == '__main__':
    main()
