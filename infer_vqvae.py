import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.motion_dataset import MotionDataset
from models.vqvae import CausalMotionVQVAE
from motion_features import resolve_database_path
from preprocess import quat


def parse_args():
    parser = argparse.ArgumentParser(description='Run VQ-VAE inference on a dataset split and export a Genoview-compatible database.')
    parser.add_argument('--checkpoint', type=Path, required=True, help='Path to best.pt or last.pt.')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--database-path', type=Path, default=None)
    parser.add_argument('--use-full-skeleton', action='store_true')
    parser.add_argument('--window-size', type=int, default=64)
    parser.add_argument('--window-stride', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--pin-memory', dest='pin_memory', action='store_true')
    parser.add_argument('--no-pin-memory', dest='pin_memory', action='store_false')
    parser.set_defaults(pin_memory=torch.cuda.is_available())
    parser.add_argument('--outdir', type=Path, default=Path('outputs/vqvae/infer'))
    parser.add_argument('--tag', type=str, default='recon')
    parser.add_argument('--export-trajectory', action='store_true', help='Also export a trajectory.npz for Genoview from the dataset database.')
    return parser.parse_args()


def build_dataset(args, database_path):
    return MotionDataset(
        split=args.split,
        window_size=args.window_size,
        window_stride=args.window_stride,
        database_path=database_path,
        use_full_skeleton=args.use_full_skeleton,
    )


def build_loader(args, dataset):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory and torch.cuda.is_available()),
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )


def build_model_from_checkpoint(ckpt):
    args = ckpt['args']
    model = CausalMotionVQVAE(
        motion_dim=ckpt['motion_dim'],
        root_cond_dim=args['root_cond_dim'],
        use_root_cond=args['use_root_cond'],
        code_dim=args['code_dim'],
        codebook_size=args['codebook_size'],
        num_heads=args['num_heads'],
        down_t=args['down_t'],
        stride_t=args['stride_t'],
        width=args['width'],
        depth=args['depth'],
        dilation_growth_rate=args['dilation_growth_rate'],
    )
    model.load_state_dict(ckpt['model'])
    return model, args


def export_trajectory(database, out_path):
    from preprocess import quat as q

    positions = database['positions'].astype(np.float32)
    rotations = database['rotations'].astype(np.float32)
    range_starts = database['range_starts'].astype(np.int32)
    range_stops = database['range_stops'].astype(np.int32)
    range_names = database['range_names']
    range_mirror = database['range_mirror'].astype(bool)

    xroot_pos = positions[:, 0]
    xroot_rot = rotations[:, 0]
    xroot_dir = q.mul_vec(xroot_rot, np.array([0.0, 0.0, 1.0], dtype=np.float32))

    future_frames = np.array([20, 40, 60], dtype=np.int32)
    max_future = int(future_frames.max())

    indices = []
    future_positions = []
    future_directions = []
    sample_range_names = []
    sample_mirror = []

    for range_idx, (rs, re) in enumerate(zip(range_starts, range_stops)):
        pose_indices = np.arange(int(rs) + 1, int(re) - max_future, dtype=np.int32)
        if len(pose_indices) == 0:
            continue
        cpos = q.inv_mul_vec(
            xroot_rot[pose_indices][:, None],
            xroot_pos[pose_indices[:, None] + future_frames] - xroot_pos[pose_indices][:, None],
        ).astype(np.float32)
        cdir = q.inv_mul_vec(
            xroot_rot[pose_indices][:, None],
            xroot_dir[pose_indices[:, None] + future_frames],
        ).astype(np.float32)
        indices.append(pose_indices)
        future_positions.append(cpos)
        future_directions.append(cdir)
        sample_range_names.append(np.full(len(pose_indices), str(range_names[range_idx]), dtype=object))
        sample_mirror.append(np.full(len(pose_indices), bool(range_mirror[range_idx]), dtype=bool))

    if not indices:
        raise ValueError('No trajectory samples can be formed from the database.')

    indices = np.concatenate(indices, axis=0)
    future_positions = np.concatenate(future_positions, axis=0)
    future_directions = np.concatenate(future_directions, axis=0)
    sample_range_names = np.concatenate(sample_range_names, axis=0)
    sample_mirror = np.concatenate(sample_mirror, axis=0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        indices=indices.astype(np.int32),
        T=np.concatenate([future_positions, future_directions], axis=-1).reshape(len(indices), -1).astype(np.float32),
        Tpos=future_positions.astype(np.float32),
        Tdir=future_directions.astype(np.float32),
        future_frames=future_frames.astype(np.int32),
        selected_tags=np.array([args.split], dtype=object),
        sample_range_names=sample_range_names,
        sample_mirror=sample_mirror,
        database_path=np.array(str(out_path.parent / 'database.npz'), dtype=object),
    )


def main():
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model, ckpt_args = build_model_from_checkpoint(ckpt)
    database_path = resolve_database_path(use_full_skeleton=args.use_full_skeleton, database_path=args.database_path or ckpt_args.get('database_path'))
    dataset = build_dataset(args, database_path)
    loader = build_loader(args, dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    outdir = args.outdir / args.tag
    outdir.mkdir(parents=True, exist_ok=True)

    motion_dim = dataset.motion_dim
    num_ranges = len(dataset.range_names)
    range_to_recon = {}
    range_to_window = defaultdict(list)
    range_to_sample_meta = defaultdict(list)
    range_to_range_name = {}
    range_to_mirror = {}

    with torch.no_grad():
        for batch in loader:
            motion = batch['motion'].to(device, non_blocking=bool(args.pin_memory and device.type == 'cuda'))
            output = model(motion)
            recon = output['recon_state'].cpu().numpy().astype(np.float32)
            batch_motion = batch['motion'].numpy().astype(np.float32)
            for i in range(motion.shape[0]):
                range_idx = int(batch['range_idx'][i])
                range_to_recon.setdefault(range_idx, np.zeros((0, motion.shape[1], motion_dim), dtype=np.float32))
                range_to_window[range_idx].append((int(batch['start_idx'][i]), int(batch['end_idx'][i]), batch_motion[i], recon[i]))
                range_to_range_name[range_idx] = str(batch['range_name'][i])
                range_to_mirror[range_idx] = bool(batch['mirror'][i])

    positions = dataset.database['positions'].astype(np.float32).copy()
    rotations = dataset.database['rotations'].astype(np.float32).copy()
    contacts = dataset.database['contacts'].astype(np.uint8).copy()
    velocities = dataset.database['velocities'].astype(np.float32).copy()
    angular_velocities = dataset.database['angular_velocities'].astype(np.float32).copy()

    for range_idx in range(num_ranges):
        windows = sorted(range_to_window.get(range_idx, []), key=lambda x: x[0])
        if not windows:
            continue
        first_start, first_end, first_in, first_out = windows[0]
        positions[first_start:first_end] = first_out[:, : positions.shape[1], :]
        for start, end, in_motion, out_motion in windows[1:]:
            positions[start:end] = out_motion[:, : positions.shape[1], :]
        # Leave velocities/rotations/contact as the original database values for now.

    out_db = outdir / 'database.npz'
    np.savez(
        out_db,
        positions=positions,
        velocities=velocities,
        rotations=rotations,
        angular_velocities=angular_velocities,
        parents=dataset.database['parents'].astype(np.int32),
        names=dataset.database['names'],
        range_starts=dataset.range_starts.astype(np.int32),
        range_stops=dataset.range_stops.astype(np.int32),
        range_mirror=dataset.range_mirror.astype(bool),
        range_names=dataset.database['range_names'],
        contacts=contacts,
        tag_range_starts=dataset.database['tag_range_starts'].astype(np.int32),
        tag_range_stops=dataset.database['tag_range_stops'].astype(np.int32),
        tag_range_names=dataset.database['tag_range_names'],
        tag_tags=dataset.database['tag_tags'],
        tag_mirror=dataset.database['tag_mirror'].astype(bool),
        joint_subset=np.array(dataset.joint_subset, dtype=object),
    )

    np.savez(
        outdir / 'recon_windows.npz',
        split=np.array(args.split, dtype=object),
        database_path=np.array(str(database_path), dtype=object),
        checkpoint=np.array(str(args.checkpoint), dtype=object),
    )

    if args.export_trajectory:
        db = np.load(database_path, allow_pickle=True)
        export_trajectory(db, outdir / 'trajectory.npz')

    print(f'Exported database to {out_db}')
    print(f'Exported recon metadata to {outdir / "recon_windows.npz"}')
    if args.export_trajectory:
        print(f'Exported trajectory to {outdir / "trajectory.npz"}')


if __name__ == '__main__':
    main()
