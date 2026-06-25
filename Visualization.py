import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from preprocess import quat


def _set_axes_equal(ax, points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(np.maximum(maxs - mins, 1e-3))

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[2] - radius, center[2] + radius)
    ax.set_zlim(center[1] - radius, center[1] + radius)


class Visualization:
    def __init__(self, database_path: Path, trajectory_path: Path | None, sample_index: int = 0, fps: int = 60):
        self.database_path = database_path
        self.trajectory_path = trajectory_path
        self.fps = fps

        self.database = np.load(database_path, allow_pickle=True)
        self.positions = self.database["positions"].astype(np.float32)
        self.rotations = self.database["rotations"].astype(np.float32)
        self.parents = self.database["parents"].astype(np.int32)
        self.range_names = self.database["range_names"]
        self.range_starts = self.database["range_starts"].astype(np.int32)
        self.range_stops = self.database["range_stops"].astype(np.int32)

        self.global_rotations, self.global_positions = quat.fk(self.rotations, self.positions, self.parents)

        self.trajectory = None
        if trajectory_path is not None:
            self.trajectory = np.load(trajectory_path, allow_pickle=True)
            self.indices = self.trajectory["indices"].astype(np.int32)
            self.tpos = self.trajectory["Tpos"].astype(np.float32)
            self.tdir = self.trajectory["Tdir"].astype(np.float32)
            self.sample_range_names = self.trajectory["sample_range_names"]
            self.sample_mirror = self.trajectory["sample_mirror"].astype(bool)
            self.future_frames = self.trajectory["future_frames"].astype(np.int32)
            self.sample_index = int(np.clip(sample_index, 0, len(self.indices) - 1))
            self.frame_index = int(self.indices[self.sample_index])
        else:
            self.indices = None
            self.tpos = None
            self.tdir = None
            self.sample_range_names = None
            self.sample_mirror = None
            self.future_frames = None
            self.sample_index = 0
            self.frame_index = int(np.clip(sample_index, 0, len(self.positions) - 1))

        self.is_paused = False
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.timer = self.fig.canvas.new_timer(interval=max(1, int(round(1000.0 / self.fps))))
        self.timer.add_callback(self.on_timer)

    def _frame_range_name(self, frame_index: int):
        for name, start, stop in zip(self.range_names, self.range_starts, self.range_stops):
            if start <= frame_index < stop:
                return str(name)
        return "unknown"

    def _world_trajectory(self):
        if self.trajectory is None:
            return None, None

        root_pos = self.positions[self.frame_index, 0]
        root_rot = self.rotations[self.frame_index, 0]
        world_pos = quat.mul_vec(root_rot[None], self.tpos[self.sample_index]) + root_pos[None]
        world_dir = quat.mul_vec(root_rot[None], self.tdir[self.sample_index])
        return world_pos, world_dir

    def draw(self):
        self.ax.clear()

        frame_pos = self.global_positions[self.frame_index]
        plot_points = [frame_pos]

        for bone_index in range(1, len(self.parents)):
            parent_index = self.parents[bone_index]
            a = frame_pos[parent_index]
            b = frame_pos[bone_index]
            self.ax.plot(
                [a[0], b[0]],
                [a[2], b[2]],
                [a[1], b[1]],
                color="black",
                linewidth=1.2,
            )

        self.ax.scatter(
            frame_pos[:, 0],
            frame_pos[:, 2],
            frame_pos[:, 1],
            c="royalblue",
            s=12,
        )

        world_pos, world_dir = self._world_trajectory()
        if world_pos is not None:
            plot_points.append(world_pos)
            self.ax.scatter(
                world_pos[:, 0],
                world_pos[:, 2],
                world_pos[:, 1],
                c="tomato",
                s=40,
            )
            self.ax.plot(
                world_pos[:, 0],
                world_pos[:, 2],
                world_pos[:, 1],
                color="tomato",
                linewidth=1.5,
                linestyle="--",
            )
            self.ax.quiver(
                world_pos[:, 0],
                world_pos[:, 2],
                world_pos[:, 1],
                world_dir[:, 0],
                world_dir[:, 2],
                world_dir[:, 1],
                color="darkorange",
                length=0.25,
                normalize=True,
            )
            for i, future_frame in enumerate(self.future_frames):
                self.ax.text(
                    world_pos[i, 0],
                    world_pos[i, 2],
                    world_pos[i, 1],
                    str(int(future_frame)),
                    color="darkred",
                    fontsize=9,
                )

        all_points = np.concatenate(plot_points, axis=0)
        _set_axes_equal(self.ax, all_points)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Z")
        self.ax.set_zlabel("Y")

        if self.trajectory is not None:
            title = (
                f"Sample {self.sample_index} | Frame {self.frame_index} | "
                f"Range {self.sample_range_names[self.sample_index]} | "
                f"Mirror {bool(self.sample_mirror[self.sample_index])} | "
                f"{'Paused' if self.is_paused else f'Playing {self.fps} FPS'}"
            )
        else:
            title = (
                f"Frame {self.frame_index} | Range {self._frame_range_name(self.frame_index)} | "
                f"{'Paused' if self.is_paused else f'Playing {self.fps} FPS'}"
            )

        self.ax.set_title(title)
        self.ax.text2D(
            0.02,
            0.02,
            "Space: pause/resume | Left/Right: switch sample",
            transform=self.ax.transAxes,
            fontsize=10,
        )
        self.fig.canvas.draw_idle()

    def on_key_press(self, event):
        if event.key == " ":
            self.is_paused = not self.is_paused
            self.draw()
        elif event.key in ["right", "d", "n"]:
            self.switch_sample(1)
        elif event.key in ["left", "a", "p"]:
            self.switch_sample(-1)

    def switch_sample(self, delta: int):
        if self.trajectory is not None:
            self.sample_index = (self.sample_index + delta) % len(self.indices)
            self.frame_index = int(self.indices[self.sample_index])
        else:
            self.frame_index = (self.frame_index + delta) % len(self.positions)
        self.draw()

    def advance_playback(self):
        if self.trajectory is not None:
            self.sample_index = (self.sample_index + 1) % len(self.indices)
            self.frame_index = int(self.indices[self.sample_index])
        else:
            self.frame_index = (self.frame_index + 1) % len(self.positions)

    def on_timer(self):
        if not self.is_paused:
            self.advance_playback()
            self.draw()

    def show(self):
        self.draw()
        self.timer.start()
        print("Controls: space to pause/resume, left/right arrow to switch sample.")
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Minimal database + trajectory visualizer for motion data.")
    parser.add_argument("--database", type=Path, required=True, help="Path to database.npz")
    parser.add_argument("--trajectory", type=Path, default=None, help="Optional path to trajectory.npz")
    parser.add_argument("--sample", type=int, default=0, help="Initial sample index or frame index")
    parser.add_argument("--fps", type=int, default=60, help="Playback FPS")
    args = parser.parse_args()

    viewer = Visualization(
        database_path=args.database,
        trajectory_path=args.trajectory,
        sample_index=args.sample,
        fps=args.fps,
    )
    viewer.show()


if __name__ == "__main__":
    main()
