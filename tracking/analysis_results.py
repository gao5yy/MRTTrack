import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist
import argparse

parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
parser.add_argument('--tracker_name', default='select_track')
parser.add_argument('--tracker_param', type=str, default='vitb_256_select_32x1_1e4_lasher_15ep_sot',help='Name of config file.')
parser.add_argument('--dataset_name', type=str, default='lasher_test',help='Name of config file.')
# parser.add_argument('--dataset_name', type=str, default='vtuav_test',help='Name of config file.')
parser.add_argument('--runid', type=int, default=None, help='The run id.')

args = parser.parse_args()

trackers = []
dataset_name = args.dataset_name

trackers.extend(trackerlist(name=args.tracker_name, parameter_name=args.tracker_param, dataset_name=dataset_name,
                            run_ids=args.runid, display_name=args.tracker_param))


dataset = get_dataset(dataset_name)

print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'))



