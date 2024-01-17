using Random
using CSV
using DataFrames

include("SARSA_agent.jl")




function runSARSA(env)
    sarsa_Rewards = Float64[]
    
    alpha, gamma, epsilon = 0.1, 0.95, 0.1
    agent = SarsaAgent(env, alpha, gamma, epsilon)
   
    
    for episode in 1:20000
        episode_reward = 0
        
        state = env.reset()
        action = get_action(agent, state[1])
        
        done = false
        while !done
            
            next_state, reward, done, truncated, info = env.step(action)
            
            next_action = get_action(agent, next_state)
            
            update(agent, state[1], action, reward, next_state[1], next_action)
            
            state = next_state[1]
            action = next_action
            
            episode_reward += reward
        end
        push!(sarsa_Rewards, episode_reward)
    end
    save(agent)
    return sarsa_Rewards
end