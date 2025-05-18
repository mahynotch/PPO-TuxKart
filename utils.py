import numpy as np
import pystk

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import dense_transforms
import torch

RESCUE_TIMEOUT = 30
TRACK_OFFSET = 15
DATASET_PATH = 'drive_data'


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path=DATASET_PATH, transform=dense_transforms.ToTensor()):
        from PIL import Image
        from glob import glob
        from os import path
        self.data = []
        for f in glob(path.join(dataset_path, '*.csv')):
            i = Image.open(f.replace('.csv', '.png'))
            i.load()
            self.data.append((i, np.loadtxt(f, dtype=np.float32, delimiter=',')))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data = self.transform(*data)
        return data


def load_data(dataset_path=DATASET_PATH, transform=dense_transforms.ToTensor(), num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path, transform=transform)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


class PyTux:
    _singleton = None

    def __init__(self, screen_width=128, screen_height=96):
        assert PyTux._singleton is None, "Cannot create more than one pytux object"
        PyTux._singleton = self
        self.config = pystk.GraphicsConfig.hd()
        self.config.screen_width = screen_width
        self.config.screen_height = screen_height
        pystk.init(self.config)
        self.k = None

    @staticmethod
    def _point_on_track(distance, track, offset=0.0):
        """
        Get a point at `distance` down the `track`. Optionally applies an offset after the track segment if found.
        Returns a 3d coordinate
        """
        node_idx = np.searchsorted(track.path_distance[..., 1],
                                   distance % track.path_distance[-1, 1]) % len(track.path_nodes)
        d = track.path_distance[node_idx]
        x = track.path_nodes[node_idx]
        t = (distance + offset - d[0]) / (d[1] - d[0])
        return x[1] * t + x[0] * (1 - t)

    @staticmethod
    def _to_image(x, proj, view):
        p = proj @ view @ np.array(list(x) + [1])
        return np.clip(np.array([p[0] / p[-1], -p[1] / p[-1]]), -1, 1)

    def rollout(self, track, controller, planner=None, max_frames=1000, verbose=False, data_callback=None):
        """
        Play a level (track) for a single round.
        :param track: Name of the track
        :param controller: low-level controller, see controller.py
        :param planner: high-level planner, see planner.py
        :param max_frames: Maximum number of frames to play for
        :param verbose: Should we use matplotlib to show the agent drive?
        :param data_callback: Rollout calls data_callback(time_step, image, 2d_aim_point) every step, used to store the
                              data
        :return: Number of steps played
        """
        if self.k is not None and self.k.config.track == track:
            self.k.restart()
            self.k.step()
        else:
            if self.k is not None:
                self.k.stop()
                del self.k
            config = pystk.RaceConfig(num_kart=1, laps=1,  track=track)
            config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL

            self.k = pystk.Race(config)
            self.k.start()
            self.k.step()

        state = pystk.WorldState()
        track = pystk.Track()

        last_rescue = 0

        if verbose:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1)

        for t in range(max_frames):

            state.update()
            track.update()

            kart = state.players[0].kart

            if np.isclose(kart.overall_distance / track.length, 1.0, atol=2e-3):
                if verbose:
                    print("Finished at t=%d" % t)
                break

            proj = np.array(state.players[0].camera.projection).T
            view = np.array(state.players[0].camera.view).T
            aim_point_world = self._point_on_track(kart.distance_down_track+TRACK_OFFSET, track)
            aim_point_image = self._to_image(aim_point_world, proj, view)
            if data_callback is not None:
                data_callback(t, np.array(self.k.render_data[0].image), aim_point_image)

            if planner:
                image = np.array(self.k.render_data[0].image)
                aim_point_image = planner(TF.to_tensor(image)[None]).squeeze(0).cpu().detach().numpy()

            current_vel = np.linalg.norm(kart.velocity)
            action = controller(aim_point_image, current_vel)

            if current_vel < 1.0 and t - last_rescue > RESCUE_TIMEOUT:
                last_rescue = t
                action.rescue = True

            if verbose:
                ax.clear()
                ax.imshow(self.k.render_data[0].image)
                WH2 = np.array([self.config.screen_width, self.config.screen_height]) / 2
                ax.add_artist(plt.Circle(WH2*(1+self._to_image(kart.location, proj, view)), 2, ec='b', fill=False, lw=1.5))
                ax.add_artist(plt.Circle(WH2*(1+self._to_image(aim_point_world, proj, view)), 2, ec='r', fill=False, lw=1.5))
                if planner:
                    ap = self._point_on_track(kart.distance_down_track + TRACK_OFFSET, track)
                    ax.add_artist(plt.Circle(WH2*(1+aim_point_image), 2, ec='g', fill=False, lw=1.5))
                plt.pause(1e-3)

            self.k.step(action)
            t += 1
        return t, kart.overall_distance / track.length
    
    def ppo_rollout(self, track, policy, reward_fn, control_fn, max_frames=1000, verbose=False, device='cuda'):
        """
        PPO-specific rollout function that collects all data needed for PPO training.

        Args:
            track: Name of the track
            policy: PPO policy model (e.g., PPOPlanner instance)
            reward_fn: Function to calculate rewards
            control_fn: Function to convert aim points to control actions
            max_frames: Maximum number of frames to play for
            verbose: Should we use matplotlib to show the agent drive?
            device: Device to run policy inference on ('cpu' or 'cuda')

        Returns:
            Dictionary with all data needed for PPO training
        """
        # Setup race same as original rollout
        # if self.k is not None and self.k.config.track == track:
        #     self.k.restart()
        #     self.k.step()
        # else:
        if self.k is not None:
            self.k.stop()
            del self.k
        config = pystk.RaceConfig(num_kart=1, laps=1,  track=track)
        config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL

        self.k = pystk.Race(config)
        self.k.start()
        self.k.step()

        state = pystk.WorldState()
        state.update()
        kart = state.players[0].kart
        track_obj = pystk.Track()
        track_obj.update()
        track_length = track_obj.length

        observations = []
        actions_to_store = [] # Stores integer pixel indices
        log_probs_to_store = [] # Stores log_probs of the pixel indices
        values_from_rollout = [] # Stores V(s_t) from policy during rollout
        rewards = []
        dones = []

        last_rescue = 0
        time_steps = 0
        episode_reward = 0.0

        if verbose:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1)

        previous_overall_distance = 0.0
        current_overall_distance = 0.0

        last_best = 0.0

        for t in range(max_frames):
            image = np.array(self.k.render_data[0].image)
            image_tensor = TF.to_tensor(image)[None].to(device) # Shape (1, C, H, W)

            with torch.no_grad():
                action_normalized_coords_tensor, action_pixel_indices_tensor, \
                    log_prob_of_pixel_tensor, value_tensor = policy.get_action(image_tensor)

                # For controlling the kart (normalized coordinates)
                action_for_env_numpy = action_normalized_coords_tensor.squeeze(0).cpu().numpy() # Shape (2,)

                # For PPO buffer (pixel index and its log_prob)
                action_pixel_idx_to_store = action_pixel_indices_tensor.squeeze(0).cpu().numpy().item() # Scalar integer
                log_prob_pixel_to_store = log_prob_of_pixel_tensor.squeeze(0).squeeze(-1).cpu().numpy().item() # Scalar float
                value_st_to_store = value_tensor.squeeze(0).squeeze(-1).cpu().numpy().item() # Scalar float

            observations.append(image_tensor.cpu().numpy()[0])
            actions_to_store.append(action_pixel_idx_to_store)
            log_probs_to_store.append(log_prob_pixel_to_store)
            values_from_rollout.append(value_st_to_store)

            current_vel = np.linalg.norm(np.array(kart.velocity))
            
            aim_point_image_coords = action_for_env_numpy
            control_action = control_fn(aim_point_image_coords, current_vel)

            if current_vel < 1.0 and t - last_rescue > RESCUE_TIMEOUT:
                last_rescue = t
                control_action.rescue = True
                if verbose:
                    print(f"INFO: Agent initiated rescue at step {t}.")
            
            proj = np.array(state.players[0].camera.projection).T
            view = np.array(state.players[0].camera.view).T
            
            aim_point_on_track_world = self._point_on_track(kart.distance_down_track + TRACK_OFFSET, track_obj)
            aim_point_target_image_coords = self._to_image(aim_point_on_track_world, proj, view)

            self.k.step(control_action)
            state.update()
            track_obj.update()
            time_steps += 1

            kart = state.players[0].kart
            current_overall_distance = kart.distance_down_track
            current_progress = current_overall_distance / track_length

            done = False
            if track_length > 0 and (np.isclose(current_progress, 1.0, atol=2e-3) or current_progress >= 1.0):
                print(f"Finished at t={t}, progress={current_progress:.4f}, distance_down={kart.distance_down_track}")
                done = True
            dones.append(done)

            current_vel = np.linalg.norm(np.array(kart.velocity))

            aim_point_diff = np.linalg.norm(aim_point_image_coords - aim_point_target_image_coords)

            kart_location_world = np.array(kart.location)
            kart_on_track_world = self._point_on_track(kart.distance_down_track, track_obj)
            location_diff = np.linalg.norm(kart_location_world - kart_on_track_world)

            progress_this_step = 0.0
            if track_length > 0:
                #  progress_this_step = (current_overall_distance - previous_overall_distance) / track_length
                progress_this_step = current_progress - last_best
                if progress_this_step > 0.0:
                    last_best = current_progress
                else:
                    progress_this_step = 0.0

            if done or t >= max_frames - 1:
                finished = True
            else:
                finished = False
            
            reward = reward_fn(
                observation=image,
                action=control_action,
                done=done,
                progress_diff=progress_this_step,
                velocity=current_vel,
                aim_point_diff=aim_point_diff,
                location_diff=location_diff,
                finished=finished,
            )
            # reward = reward / 6
            rewards.append(reward)
            episode_reward += reward

            previous_overall_distance = current_overall_distance

            if verbose:
                ax.clear()
                ax.imshow(image)
                WH2 = np.array([self.config.screen_width, self.config.screen_height]) / 2
                viz_aim_point = WH2 * (1 + aim_point_image_coords)
                ax.add_artist(plt.Circle(viz_aim_point, 5, ec='g', fill=False, lw=1.5)) # Increased circle size
                plt.title(f"t={t}, r={reward:.2f}, prog={current_progress:.2f}, vel={current_vel:.1f}")
                plt.pause(1e-3)

            if done:
                state.update()
                track_obj.update()
                break


        final_bootstrap_value = 0.0
        if not done and len(observations) > 0:
            with torch.no_grad():
                final_image_after_last_action = np.array(self.k.render_data[0].image)
                final_tensor_for_bootstrap = TF.to_tensor(final_image_after_last_action)[None].to(device)
                _, V_s_prime_tensor = policy(final_tensor_for_bootstrap)
                final_bootstrap_value = V_s_prime_tensor.squeeze(0).squeeze(-1).cpu().numpy().item() # Scalar float

        array_lengths = {
            'observations': len(observations),
            'actions': len(actions_to_store),
            'log_probs': len(log_probs_to_store),
            'rewards': len(rewards),
            'values': len(values_from_rollout),
            'dones': len(dones)
        }

        # Check if all arrays have the same length
        lengths_match = len(set(array_lengths.values())) == 1

        if not lengths_match:
            print("WARNING: Data arrays have inconsistent lengths:")
            for name, length in array_lengths.items():
                print(f"  {name}: {length}")

        return {
            'obs': np.array(observations),
            'actions': np.array(actions_to_store),
            'log_probs': np.array(log_probs_to_store),
            'rewards': np.array(rewards),
            'values': np.array(values_from_rollout),
            'dones': np.array(dones),
            'last_value': final_bootstrap_value,
            'episode_reward': episode_reward,
            'episode_length': time_steps,
            'final_progress': current_progress
        }

    def close(self):
        """
        Call this function, once you're done with PyTux
        """
        if self.k is not None:
            self.k.stop()
            del self.k
        pystk.clean()



if __name__ == '__main__':
    from controller import control
    from argparse import ArgumentParser
    from os import makedirs


    def noisy_control(aim_pt, vel):
        return control(aim_pt + np.random.randn(*aim_pt.shape) * aim_noise,
                       vel + np.random.randn() * vel_noise)


    parser = ArgumentParser("Collects a dataset for the high-level planner")
    parser.add_argument('track', nargs='+')
    parser.add_argument('-o', '--output', default=DATASET_PATH)
    parser.add_argument('-n', '--n_images', default=10000, type=int)
    parser.add_argument('-m', '--steps_per_track', default=20000, type=int)
    parser.add_argument('--aim_noise', default=0.1, type=float)
    parser.add_argument('--vel_noise', default=5, type=float)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    try:
        makedirs(args.output)
    except OSError:
        pass
    pytux = PyTux()
    for track in args.track:
        n, images_per_track = 0, args.n_images // len(args.track)
        aim_noise, vel_noise = 0, 0


        def collect(_, im, pt):
            from PIL import Image
            from os import path
            global n
            id = n if n < images_per_track else np.random.randint(0, n + 1)
            if id < images_per_track:
                fn = path.join(args.output, track + '_%05d' % id)
                Image.fromarray(im).save(fn + '.png')
                with open(fn + '.csv', 'w') as f:
                    f.write('%0.1f,%0.1f' % tuple(pt))
            n += 1


        while n < args.steps_per_track:
            steps, how_far = pytux.rollout(track, noisy_control, max_frames=1000, verbose=args.verbose, data_callback=collect)
            print(steps, how_far)
            # Add noise after the first round
            aim_noise, vel_noise = args.aim_noise, args.vel_noise
    pytux.close()
