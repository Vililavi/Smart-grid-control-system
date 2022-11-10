"""
Super-yksinkertainen testi siihen, että custom-environmentit toimii.
"""
import gym
import custom_cartpole

env = gym.make('CustomCartPole-v1', render_mode='human')

max_ep = 10

for ep_cnt in range(max_ep):
    step_cnt = 0
    ep_reward = 0
    done = False
    state = env.reset()

    while not done:
        next_state, reward, done, _, _ = env.step(env.action_space.sample())
        env.render()
        step_cnt += 1
        ep_reward += reward
        state = next_state

    print('Episode: {}, Step count: {}, Episode reward: {}'.format(ep_cnt, step_cnt, ep_reward))
env.close()