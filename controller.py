import pystk

def control(aim_point, current_vel, steer_gain=6, skid_thresh=0.2, target_vel=27):
    # Originally 6, 0.2, 25
    import numpy as np
    action = pystk.Action()
    
    # Steering based on aim_point horizontal position
    # aim_point[0] ranges from -1 (left) to 1 (right)
    action.steer = np.clip(aim_point[0] * steer_gain, -1, 1)
    
    # Adjust target velocity based on turn sharpness
    # Slow down for sharp turns
    turn_factor = abs(aim_point[0])
    adjusted_target_vel = target_vel * (1 - 0.5 * turn_factor)
    vel_error = adjusted_target_vel - current_vel
    if vel_error > 0:
        action.acceleration = min(1.0, 0.1 + vel_error / 10)
        action.brake = False
    else:
        action.acceleration = 0.0
        action.brake = vel_error < -5
    if turn_factor > 0.3 and current_vel > adjusted_target_vel:
        action.brake = True
        action.acceleration = 0.0
    action.drift = turn_factor > skid_thresh and current_vel > 15
    action.nitro = turn_factor < 0.1 and vel_error > 5 and current_vel > 10
    
    return action

if __name__ == '__main__':
    from utils import PyTux
    from argparse import ArgumentParser

    def test_controller(args):
        import numpy as np
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()


    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_controller(args)