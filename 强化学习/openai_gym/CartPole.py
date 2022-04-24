import gym
import time


def random_cartpole(render=False, sleep=0.1):
    print('-------------random cartpole----------------------')
    e = gym.make('CartPole-v0')
    a_space = e.action_space
    o_space = e.observation_space
    print("Action space: {}, \nObservation space: {}".format(a_space, o_space))
    total_reward = 0.
    total_step = 0
    # reset the environment and return the initial observation
    obs = e.reset()
    print("Initial observation: {}".format(obs))

    while True:
        # random sample an action
        action = a_space.sample()
        obs, reward, done, _ = e.step(action)
        total_reward += reward
        total_step += 1
        if render:
            e.render()
            time.sleep(sleep)
        if done:
            e.close()
            break

    print("Episode done in {} steps\nTotal reward: {}".format(total_step, total_reward))
    print('-----------------------------------------')


