using Random
using CSV
using DataFrames


"""
    QLearningAgent

    A reinforcement learning agent using the Q-learning algorithm.

    Fields:
    - `env`: The environment in which the agent operates.
    - `alpha::Float64`: The learning rate (alpha) for Q-learning.
    - `gamma::Float64`: The discount factor (gamma) for future rewards.
    - `epsilon::Float64`: The exploration-exploitation trade-off parameter (epsilon).
    - `Q_est`: The estimated Q-values table.
"""
struct QLearningAgent
    env
    alpha::Float64
    gamma::Float64
    epsilon::Float64
    Q_est
end


"""
    QLearningAgent(env, alpha, gamma, epsilon)

    Initializes a Q-learning agent with the specified parameters.

    Parameters:
    - `env`: The environment in which the agent operates.
    - `alpha::Float64`: The learning rate (alpha) for Q-learning.
    - `gamma::Float64`: The discount factor (gamma) for future rewards.
    - `epsilon::Float64`: The exploration-exploitation trade-off parameter (epsilon).

    Returns:
    - A Q-learning agent with initialized Q-values.
"""
function QLearningAgent(env, alpha, gamma, epsilon)
    Q_est = zeros(Float64, env.observation_space.n, env.action_space.n)
    return QLearningAgent(env, alpha, gamma, epsilon, Q_est)
end


"""
    get_alpha(agent::QLearningAgent)

    Retrieves the learning rate (alpha) of a Q-learning agent.

    Parameters:
    - `agent::QLearningAgent`: The Q-learning agent from which to retrieve alpha.

    Returns:
    - The learning rate (alpha) of the agent.
"""
function get_alpha(agent::QLearningAgent)
    return agent.alpha
end


"""
    get_gamma(agent::QLearningAgent)

    Retrieves the discount factor (gamma) of a Q-learning agent.

    Parameters:
    - `agent::QLearningAgent`: The Q-learning agent from which to retrieve gamma.

    Returns:
    - The discount factor (gamma) of the agent.
"""
function get_gamma(agent::QLearningAgent)
    return agent.gamma
end


"""
    get_epsilon(agent::QLearningAgent)

    Retrieves the exploration-exploitation trade-off parameter (epsilon) of a Q-learning agent.

    Parameters:
    - `agent::QLearningAgent`: The Q-learning agent from which to retrieve epsilon.

    Returns:
    - The exploration-exploitation trade-off parameter (epsilon) of the agent.
"""
function get_epsilon(agent::QLearningAgent)
    return agent.epsilon
end


"""
    get_Q_est(agent::QLearningAgent)

    Retrieves the estimated Q-values table of a Q-learning agent.

    Parameters:
    - `agent::QLearningAgent`: The Q-learning agent from which to retrieve Q-values.

    Returns:
    - The estimated Q-values table of the agent.
"""
function get_Q_est(agent::QLearningAgent)
    return agent.Q_est
end


"""
    get_action(agent::QLearningAgent, state)

    Determines an action for a Q-learning agent based on the epsilon-greedy policy.

    Parameters:
    - `agent::QLearningAgent`: The Q-learning agent for which to select an action.
    - `state`: The current state of the agent.

    Returns:
    - The selected action based on the epsilon-greedy policy.
"""
function get_action(agent::QLearningAgent, state)
    if rand() > agent.epsilon
        action_values = agent.Q_est[state+1, :]
        max_value = maximum(action_values)
        action = rand(findall(x -> x == max_value, action_values)) - 1
    else
        action = rand(0:(agent.env.action_space.n - 1))
    end
    return action
end
    
    
"""
    update(agent::QLearningAgent, state, action, reward, next_state)

    Updates the Q-values of a Q-learning agent based on the observed transition.

    Parameters:
    - `agent::QLearningAgent`: The Q-learning agent to update.
    - `state`: The current state.
    - `action`: The action taken in the current state.
    - `reward`: The received reward for the action.
    - `next_state`: The next state after taking the action.

    Note:
    - The action index is adjusted by 1 for Julia indexing compatibility.

"""
function update(agent::QLearningAgent, state, action, reward, next_state)
    # Zwiększ indeksy action i next_action o 1 dla zgodności z indeksowaniem w Julii
    julia_action = action + 1

    td_target = reward + agent.gamma * maximum(agent.Q_est[next_state+1])
    td_error = td_target - agent.Q_est[state+1, julia_action]
    agent.Q_est[state+1, julia_action] += agent.alpha * td_error
end

"""
    save(agent::QLearningAgent, path="./QLearning_Q_est.csv")

    Saves the estimated Q-values of a Q-learning agent to a CSV file.

    Parameters:
    - `agent::QLearningAgent`: The Q-learning agent to save.
    - `path::String`: The file path where the Q-values will be saved (default: "./QLearning_Q_est.csv").

"""
function save(agent::QLearningAgent, path="./QLearning_Q_est.csv")
    CSV.write(path, DataFrame(agent.Q_est, :auto))
end

        
        
"""
    load(agent::QLearningAgent, path="./QLearning_Q_est.csv")

    Loads Q-values from a CSV file into a Q-learning agent.

    Parameters:
    - `agent::QLearningAgent`: The Q-learning agent to load the Q-values into.
    - `path::String`: The file path from which to load the Q-values (default: "./QLearning_Q_est.csv").

"""
function load(agent::QLearningAgent, path="./QLearning_Q_est.csv")
    Q_est_dataframe = CSV.File(path) |> DataFrame
    agent.Q_est = Matrix(Q_est_dataframe)
end

        
"""
    load(agent::QLearningAgent, path="./QLearning_Q_est.csv")

    Loads Q-values from a CSV file into a Q-learning agent.

    Parameters:
    - `agent::QLearningAgent`: The Q-learning agent to load the Q-values into.
    - `path::String`: The file path from which to load the Q-values (default: "./QLearning_Q_est.csv").

"""
function load_agent(env, alpha, gamma, epsilon)
    Q_est_dataframe = CSV.File("./QLearning_Q_est.csv") |> DataFrame
    Q_est = Matrix(Q_est_dataframe)
    return QLearningAgent(env, alpha, gamma, epsilon, Q_est)
end