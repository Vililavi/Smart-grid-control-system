"""Test that our GridV0Env runs with gym."""

import gym
import custom_envs.grid_v0


def test_grid_v0_with_gym():
    env = gym.make("Grid-v0", max_total_steps=24*100)

    max_episodes = 10

    for episode in range(max_episodes):
        step_count = 0
        ep_reward = 0
        terminated = False
        _state = env.reset()

        while not terminated:
            next_state, reward, terminated, _, _info = env.step(env.action_space.sample())
            env.render()
            step_count += 1
            ep_reward += reward
            _state = next_state

        print(f"Episode: {episode}, Step count: {step_count}, Episode reward: {ep_reward}")


if __name__ == "__main__":
    test_grid_v0_with_gym()
