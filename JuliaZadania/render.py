import argparse
import gym

from julia.api import Julia
jl = Julia(compiled_modules=False)

from julia import Main


parser = argparse.ArgumentParser(description='Runs TD(0) on Walking5-v0.')

parser.add_argument('-e', '--env',
                    type=str,
                    default='CliffWalking-v0',
                    choices = ['CliffWalking-v0', 'Taxi-v3'],
                    help='Environment. One of CliffWalking-v0, Taxi-v3. Default: CliffWalking-v0'
                   )


parser.add_argument('-g', '--greedy',
                    type=lambda s: s.lower() in ['true', 't', 'yes', '1'],
                    default=True,
                    help='Choose greedy action only. Default: True'
                   )



parser.add_argument('-a', '--agent',
                    type=str,
                    default='SARSA',
                    choices = ['QLearning', 'SARSA'],
                    help='Agent. One of QLearning, SARSA. Default: QLearning'
                   )


parser.add_argument('-f', '--filename',
                    type=str,
                    default=None,
                    help='Filename to load Q_est from. Default: None (it is choosen by the Agent then)'
                   )





def render_env(agent, env, greedy=True):
    path = []
    state, info = env.reset()
    states = [state]
    total_reward = 0
    total_steps = 0
    done = False
    env.render()
    while not done:
        if greedy:
            action = Main.get_Q_est(agent)[state].argmax()
        else:
            #action = Main.get_action(state)
            1+1
            
        next_state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        state = next_state
        total_steps += 1
        states.append(state)
        env.render()
    return states, total_reward, total_steps



if __name__ == "__main__":
    args = parser.parse_args()
    env = gym.make(args.env, render_mode = 'human')
    alpha, gamma, epsilon = 0.5, 0.9, 0.1

    if args.agent == 'QLearning':
        Main.include("./Julia-code/q_learning_agent.jl")
    else:
        Main.include("./Julia-code/sarsa_agent.jl")
    
    agent = Main.load_agent(env, alpha, gamma, epsilon)

    states, total_reward, total_steps = render_env(agent, env, greedy=True)
    print (f'Reward {total_reward} achieved in {total_steps} steps.')
    print ('States: {}'.format(states))
    
    