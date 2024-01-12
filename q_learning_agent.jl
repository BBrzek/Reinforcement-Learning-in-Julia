using Random
using CSV
using DataFrames

struct QLearningAgent
    env
    alpha::Float64
    gamma::Float64
    epsilon::Float64
    Q_est
end


function QLearningAgent(env, alpha, gamma, epsilon)
    Q_est = zeros(Float64, env.observation_space.n, env.action_space.n)
    return QLearningAgent(env, alpha, gamma, epsilon, Q_est)
end

function get_alpha(agent::QLearningAgent)
    return agent.alpha
end

function get_gamma(agent::QLearningAgent)
    return agent.gamma
end

function get_epsilon(agent::QLearningAgent)
    return agent.epsilon
end

function get_Q_est(agent::QLearningAgent)
    return agent.Q_est
end

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

function update(agent::QLearningAgent, state, action, reward, next_state)
    # Zwiększ indeksy action i next_action o 1 dla zgodności z indeksowaniem w Julii
    julia_action = action + 1

    td_target = reward + agent.gamma * maximum(agent.Q_est[next_state+1])
    td_error = td_target - agent.Q_est[state+1, julia_action]
    agent.Q_est[state+1, julia_action] += agent.alpha * td_error
end

function save(agent::QLearningAgent, path="./QLearning_Q_est.csv")
    CSV.write(path, DataFrame(agent.Q_est,:auto))
end

function load(agent::QLearningAgent, path="./QLearning_Q_est.csv")
    Q_est_dataframe = CSV.File(path) |> DataFrame
    agent.Q_est = Matrix(Q_est_dataframe)
end

function load_agent(env, alpha, gamma, epsilon)
    Q_est_dataframe = CSV.File("./QLearning_Q_est.csv") |> DataFrame
    Q_est = Matrix(Q_est_dataframe)
    return QLearningAgent(env, alpha, gamma, epsilon, Q_est)
end