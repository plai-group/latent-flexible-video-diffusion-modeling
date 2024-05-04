import argparse
import json
import numpy as np
from pathlib import Path
import torch
from typing import List

from improved_diffusion.video_datasets import get_test_dataset
from improved_diffusion.script_util import str2bool
from video_fvd import SampleDataset



def ar(x, y, z):
    return z/2+np.arange(x, y, z, dtype='float')


def get_kernel_maxacts(frame, kernels, n_max=1):
    kernel_size = kernels.size(-1)
    _, C, width, height = frame.shape
    frame_unfolded = frame.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1).permute(0, 2, 3, 1, 4, 5)
    frame_unfolded = frame_unfolded.reshape(-1, C, kernel_size, kernel_size)

    best_stats = [(None, float('inf'), None)] * n_max  # location, distance, kernel index
    for k, kernel in enumerate(kernels):
        distances = (kernel - frame_unfolded).norm(2, dim=(1, 2, 3))
        topk_result = torch.topk(distances, k=n_max * 4, largest=False)
        best_match_indices, min_dists = topk_result.indices, topk_result.values
        min_locs = torch.unravel_index(best_match_indices, (width - kernel_size + 1, height - kernel_size + 1))
        min_locs = torch.stack(min_locs, dim=0).transpose(0, 1)
        curr_stats = [(loc.tolist(), dist.item(), k) for loc, dist in zip(min_locs, min_dists)]

        # HACK: Ensure the centers of balls are at least 4 pixels away from the best match location for this ball.
        curr_stats = [curr_stats[0]] + [stat for stat in curr_stats if (torch.tensor(stat[0])-min_locs[0]).float().norm()>4]

        best_stats = sorted(best_stats+curr_stats, key=lambda e: e[1])[:n_max]
    locs, kernel_idxs = [e[0] for e in best_stats], [e[2] for e in best_stats]
    return locs, kernel_idxs


def align_balls(traj, color):
    first_frame_info = sorted(zip(traj[0], color[0]), key=lambda e: e[0])
    result = [([e[0] for e in first_frame_info], [e[1] for e in first_frame_info])]

    for t, (locs_next, c_next) in enumerate(zip(traj[1:], color[1:])):
        distances = torch.cdist(torch.FloatTensor(result[t][0]), torch.FloatTensor(locs_next))
        min_locs = torch.topk(distances, k=1, largest=False).indices
        assert len(min_locs) == len(min_locs.unique()), "Each location must uniquely be assigned to a ball."
        to_add_traj, to_add_color = [], []
        for min_loc in min_locs:
            to_add_traj.append(locs_next[min_loc])
            to_add_color.append(c_next[min_loc])
        result.append((to_add_traj, to_add_color))
    return [e[0] for e in result], [e[1] for e in result]


def compute_minADE_exemplar(test_frames: torch.Tensor, all_sample_frames: List[torch.Tensor],
                            kernels: torch.Tensor, T: int, n_balls: int):
    num_samples = len(all_sample_frames)
    test_traj = []
    test_color = []
    sample_trajs = [[] for _ in range(num_samples)]
    sample_colors = [[] for _ in range(num_samples)]
    for t in range(T):
        test_frame = test_frames[0, t]
        test_loc, test_kernel = get_kernel_maxacts(test_frame.unsqueeze(0), kernels, n_balls)
        test_traj.append(test_loc)
        test_color.append(test_kernel)

        for sample_idx in range(num_samples):
            sample_frame = all_sample_frames[sample_idx][0, t]
            sample_loc, sample_kernel = get_kernel_maxacts(sample_frame.unsqueeze(0), kernels, n_balls)
            sample_trajs[sample_idx].append(sample_loc)
            sample_colors[sample_idx].append(sample_kernel)

    test_traj, test_color = align_balls(test_traj, test_color)
    sample_results = [align_balls(traj, color) for traj, color in zip(sample_trajs, sample_colors)]
    sample_trajs, sample_colors = [e[0] for e in sample_results], [e[1] for e in sample_results]

    test_traj = torch.tensor(test_traj).unsqueeze(0).float()
    sample_trajs = torch.tensor(sample_trajs).float()
    ADEs = (test_traj-sample_trajs).norm(2, dim=-1).mean(dim=(-1,-2))
    minADE = torch.min(ADEs).item()

    test_color = torch.tensor(test_color).unsqueeze(0)
    sample_colors = torch.tensor(sample_colors)
    color_accs = (test_color == sample_colors).float().mean(dim=(-1,-2))
    max_color_acc = torch.min(color_accs).item()

    return minADE, max_color_acc


def compute_minADE(test_dataset, sample_datasets, n_obs, T, n_balls):
    minADE = 0.
    test_loader = iter(torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                                   shuffle=False, drop_last=False))
    sample_loaders = [iter(torch.utils.data.DataLoader(sample_dataset, batch_size=1,
                                                       shuffle=False, drop_last=False))
                      for sample_dataset in sample_datasets]

    kernel_width = 8
    [I, J] = np.meshgrid(ar(-1.25, 1.25, 2.5/kernel_width), ar(-1.25, 1.25, 2.5/kernel_width))  # Ball radius
    # Make a 3x8x8 kernel based on the bouncing ball dataset's ball shape.
    gaussian_bump = np.exp(-((I**2+J**2)/(1.2**2))**4)
    gaussian_bump = torch.tensor(gaussian_bump).to(torch.float32)
    kernels = (
        torch.stack([1. * gaussian_bump, 0. * gaussian_bump, 0. * gaussian_bump], dim=0),  # red ball
        torch.stack([1. * gaussian_bump, 1. * gaussian_bump, 0. * gaussian_bump], dim=0),  # yellow ball
        torch.stack([0. * gaussian_bump, 1. * gaussian_bump, 0. * gaussian_bump], dim=0),  # green ball
    )
    kernels = torch.stack(kernels, dim=0) * 2 - 1

    # TODO: Have 3 kernels for red, white, yellow

    all_ADEs, all_accs = [], []
    for _ in range(len(test_dataset)):
        test_frames = next(test_loader)[0][:, n_obs:]
        sample_frames = [next(sample_loader)[0][:, n_obs:] for sample_loader in sample_loaders]
        ADE, acc = compute_minADE_exemplar(test_frames, sample_frames, kernels, T, n_balls)
        all_ADEs.append(ADE)
        all_accs.append(acc)

    minADE = sum(all_ADEs)/len(all_ADEs)
    max_color_acc = sum(all_accs)/len(all_accs)
    return minADE, max_color_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", type=str, required=True)
    parser.add_argument("--n_obs", type=int, required=True, help="Number of observed frames at the beginning of the video. These are not counted.")
    parser.add_argument("--num_videos", type=int, default=None, help="Number of generated samples per test video.")
    parser.add_argument("--n_balls", type=int, default=2, help="Number of bouncing balls in a video frame.")
    parser.add_argument("--sample_indices", type=int, nargs='+', default=[0])
    parser.add_argument("--T", type=int, default=None, help="Length of the videos. If not specified, it will be inferred from the dataset.")
    parser.add_argument("--eval_on_train", type=str2bool, default=False)
    args = parser.parse_args()

    if args.eval_on_train:
        save_path = Path(args.eval_dir) / f"minADE-{args.num_videos}-train.txt"
        save_path_color = Path(args.eval_dir) / f"color-acc-{args.num_videos}-train.txt"
    else:
        save_path = Path(args.eval_dir) / f"minADE-{args.num_videos}.txt"
        save_path_color = Path(args.eval_dir) / f"color-acc-{args.num_videos}.txt"

    # if save_path.exists():
    #     content = np.loadtxt(save_path).squeeze()
    #     print(f"minADE/max_color_acc already computed: {content}")
    #     print(f"Results at\n{save_path}\n{save_path_color}")
    #     exit(0)

    # Load model args
    model_args_path = Path(args.eval_dir) / "model_config.json"
    with open(model_args_path, "r") as f:
        model_args = argparse.Namespace(**json.load(f))

    # Set batch size given dataset if not specified
    args.dataset = model_args.dataset
    if args.T is None:
        args.T = model_args.T

    samples_prefix = "samples_train" if args.eval_on_train else "samples"        

    # Prepare datasets
    sample_datasets = [SampleDataset(samples_path=(Path(args.eval_dir) / samples_prefix),
                                     sample_idx=i, length=args.num_videos) for i in args.sample_indices]
    test_dataset_full = get_test_dataset(dataset_name=args.dataset, T=args.T)
    if args.eval_on_train:
        test_dataset_full.is_test = False
    test_dataset = torch.utils.data.Subset(
        dataset=test_dataset_full,
        indices=list(range(args.num_videos)),
    )

    T_sample = args.T - args.n_obs
    minADE, max_color_acc = compute_minADE(test_dataset, sample_datasets, n_obs=args.n_obs, T=T_sample, n_balls=args.n_balls)
    np.savetxt(save_path, np.array([minADE]))
    np.savetxt(save_path_color, np.array([max_color_acc]))
    print(f"minADE: {minADE:.5f}, color-acc: {max_color_acc:.5f}")
    print(f"Results saved to\n{save_path}\n{save_path_color}")
