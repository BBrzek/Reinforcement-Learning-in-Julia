using Random
using CSV
using DataFrames

struct SarsaAgent
    env
    alpha::Float64
    gamma::Float64
    epsilon::Float64
    Q_est
end


function SarsaAgent(env, alpha, gamma, epsilon)
    Q_est = zeros(Float64, env.observation_space.n, env.action_space.n)
    return SarsaAgent(env, alpha, gamma, epsilon, Q_est)
end

function get_alpha(agent::SarsaAgent)
    return agent.alpha
end

function get_gamma(agent::SarsaAgent)
    return agent.gamma
end

function get_epsilon(agent::SarsaAgent)
    return agent.epsilon
end

function get_Q_est(agent::SarsaAgent)
    return agent.Q_est
end

function get_action(agent::SarsaAgent, state)
    if rand() > agent.epsilon
        action_values = agent.Q_est[state+1, :]
        max_value = maximum(action_values)
        # Losowo wybierz jedną z akcji o maksymalnej wartości
        action = rand(findall(x -> x == max_value, action_values)) - 1
    else
        # Losowo wybierz akcję z zakresu od 0 do (n-1), gdzie n to liczba akcji
        action = rand(0:(agent.env.action_space.n - 1))
    end
    return action
end

function update(agent::SarsaAgent, state, action, reward, next_state, next_action)
    # Zwiększ indeksy action i next_action o 1 dla zgodności z indeksowaniem w Julii
    julia_action = action + 1
    julia_next_action = next_action + 1

    td_target = reward + agent.gamma * agent.Q_est[next_state+1, julia_next_action]
    td_error = td_target - agent.Q_est[state+1, julia_action]
    agent.Q_est[state+1, julia_action] += agent.alpha * td_error
end

function save(agent::SarsaAgent, path="./SARSA_Q_est.csv")
    CSV.write(path, DataFrame(agent.Q_est,:auto))
end

function load(agent::SarsaAgent, path="./SARSA_Q_est.csv")
    Q_est_dataframe = CSV.File(path) |> DataFrame
    agent.Q_est = Matrix(Q_est_dataframe)
end

function load_agent(env, alpha, gamma, epsilon)
    Q_est_dataframe = CSV.File("./SARSA_Q_est.csv") |> DataFrame
    Q_est = Matrix(Q_est_dataframe)
    return SarsaAgent(env, alpha, gamma, epsilon, Q_est)
end