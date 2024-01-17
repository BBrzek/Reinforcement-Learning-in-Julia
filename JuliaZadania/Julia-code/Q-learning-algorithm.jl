using Random
using CSV
using DataFrames

include("Q_learning_agent.jl")




function runQ_learning(env)
    Q_Rewards = Float64[]
    
    alpha, gamma, epsilon = 0.1, 0.95, 0.1
    agent = QLearningAgent(env, alpha, gamma, epsilon)
        
    for episode in 1:20000
        episode_reward = 0

        state = env.reset()
        
        done = false
        while !done
            action = get_action(agent, state[1])
            
            next_state, reward, done, truncated, info = env.step(action)

            update(agent, state[1], action, reward, next_state[1])
            
            state = next_state[1]
            
            episode_reward += reward
        end
        push!(Q_Rewards, episode_reward)
    end
    save(agent)
    return Q_Rewards
end