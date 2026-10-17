"""Convert separate physical episodes to the local LeWM HDF5 schema."""
from pathlib import Path
import argparse, hashlib, json
import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path, nargs='+', required=True)
    parser.add_argument('--validation', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    items = [(split, str(i), path) for split, paths in [('train', args.train), ('validation', args.validation)] for i, path in enumerate(paths)]
    hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for _, _, path in items]
    if len(set(hashes)) != len(hashes):
        raise ValueError('Duplicate episodes are not allowed across or within splits')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest, flat_actions = [], []
    with h5py.File(args.output, 'w') as f:
        f.attrs.update(schema='lewm_tube_episode_adapter_v1', frame_dt=.1, history_size=3,
                       camera_names=json.dumps(['overhead', 'wrist']), target_xy=json.dumps([.466, -.040]))
        for (split, name, path), digest in zip(items, hashes):
            with np.load(path) as a:
                frames = a['frames']
                actions = a['actions'].astype('float32').reshape(-1, 300)
                if len(frames) != len(actions) + 1 or not np.isfinite(actions).all():
                    raise ValueError(f'Invalid episode: {path}')
                g = f.create_group(f'{split}/{name}')
                for key, values in [('pixels', frames), ('actions', actions), ('qpos', a['qpos'])]:
                    g.create_dataset(key, data=values, compression='lzf')
                g.attrs['source_sha256'] = digest
                if split == 'train': flat_actions.append(actions)
                manifest.append(dict(split=split, episode=name, frames=len(frames), source=str(path), sha256=digest))
        train = np.concatenate(flat_actions)
        f.create_dataset('action_mean', data=train.mean(0))
        f.create_dataset('action_std', data=np.maximum(train.std(0, ddof=1), 1e-6))
    args.output.with_suffix('.manifest.json').write_text(json.dumps(dict(episodes=manifest, sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(), note='Whole-episode split; small initial-position variations only.'), indent=2))
    print('Prepared', args.output)

if __name__ == '__main__': main()
