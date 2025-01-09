import numpy as np
from tqdm import tqdm

def init(numFeatures, numArms):
    A = np.array([np.eye(numFeatures) for _ in range(numArms)])
    b = np.zeros((numArms, numFeatures, 1))
    theta = np.zeros((numArms, numFeatures, 1))
    return A, b, theta

# Omniscient
def omniscient(data_array, alpha=None):
    total_payoff = np.sum(data_array[:, 1])
    return total_payoff / len(data_array)

# UCB
def ucb(data_array, alpha):
    numArms = 10
    trials = data_array.shape[0]
    arm_counts = np.zeros(numArms)
    arm_rewards = np.zeros(numArms)
    total_payoff = 0
    for t in range(trials):
        if t < numArms:
            selected_arm = t
        else:
            upper_bounds = (arm_rewards / np.maximum(arm_counts, 1)) + alpha * np.sqrt(2 * np.log(t + 1) / np.maximum(arm_counts, 1))
            selected_arm = np.argmax(upper_bounds)
        arm = int(data_array[t, 0]) - 1
        payoff = data_array[t, 1]
        arm_counts[selected_arm] += 1
        arm_rewards[selected_arm] += payoff
        total_payoff += payoff if selected_arm == arm else 0
    return total_payoff / trials

# Warm-Started UCB
def warm_started_ucb(data_array, alpha):
    numArms = 10
    trials = data_array.shape[0]
    arm_rewards = np.zeros(numArms)
    arm_counts = np.ones(numArms)
    total_payoff = 0
    for t in range(trials):
        arm = int(data_array[t, 0]) - 1
        payoff = data_array[t, 1]
        upper_bounds = arm_rewards / arm_counts + alpha * np.sqrt(2 * np.log(t + 1) / arm_counts)
        selected_arm = np.argmax(upper_bounds)
        arm_counts[selected_arm] += 1
        if selected_arm == arm:
            arm_rewards[selected_arm] += payoff
        total_payoff += payoff if selected_arm == arm else 0
    return total_payoff / trials

# Segmented UCB
def segmented_ucb(data_array, alpha):
    segment_size = 100
    num_segments = len(data_array) // segment_size
    if len(data_array) % segment_size != 0:
        num_segments += 1
    total_payoff = 0
    for i in range(num_segments):
        segment = data_array[i * segment_size : (i + 1) * segment_size]
        total_payoff += ucb(segment, alpha)
    return total_payoff / num_segments

# Disjoint LinUCB
def disjoint_linucb(data_array, alpha):
    numArms = 10
    trials = data_array.shape[0]
    numFeatures = data_array.shape[1] - 2
    A, b, theta = init(numFeatures, numArms)
    total_payoff = 0
    for t in range(trials):
        arm = int(data_array[t, 0]) - 1
        payoff = data_array[t, 1]
        x_t = np.expand_dims(data_array[t, 2:], axis=1)
        p = [theta[a].T @ x_t + alpha * np.sqrt(x_t.T @ np.linalg.inv(A[a]) @ x_t) for a in range(numArms)]
        selected_arm = np.argmax(p)
        if selected_arm == arm:
            A[selected_arm] += x_t @ x_t.T
            b[selected_arm] += payoff * x_t
            theta[selected_arm] = np.linalg.inv(A[selected_arm]) @ b[selected_arm]
        total_payoff += payoff if selected_arm == arm else 0
    return total_payoff / trials

# Hybrid LinUCB
def hybrid_linucb(data_array, alpha):
    numArms = 10
    trials = data_array.shape[0]
    numFeatures = data_array.shape[1] - 2
    A_0 = np.eye(numFeatures)
    b_0 = np.zeros((numFeatures, 1))
    A = [np.eye(numFeatures) for _ in range(numArms)]
    B = [np.zeros((numFeatures, numFeatures)) for _ in range(numArms)]
    b = [np.zeros((numFeatures, 1)) for _ in range(numArms)]
    total_reward = 0
    for t in range(trials):
        arm = int(data_array[t, 0]) - 1
        reward = data_array[t, 1]
        z_t = np.expand_dims(data_array[t, 2:], axis=1)
        x_t = z_t.copy()
        beta_hat = np.linalg.solve(A_0, b_0)
        p_values = []
        for a in range(numArms):
            if t == 0 or (np.all(A[a] == np.eye(numFeatures)) and np.all(B[a] == 0) and np.all(b[a] == 0)):
                A[a] = np.eye(numFeatures)
                B[a] = np.zeros((numFeatures, numFeatures))
                b[a] = np.zeros((numFeatures, 1))
            theta_hat_a = np.linalg.solve(A[a], b[a] - B[a] @ beta_hat)
            s_t_a = (z_t.T @ np.linalg.solve(A_0, z_t)
                - 2 * z_t.T @ np.linalg.solve(A_0, B[a].T @ np.linalg.solve(A[a], x_t))
                + x_t.T @ np.linalg.solve(A[a], x_t)
                + x_t.T @ np.linalg.solve(A[a], B[a] @ np.linalg.solve(A_0, B[a].T @ np.linalg.solve(A[a], x_t))))[0, 0]
            p_t_a = (z_t.T @ beta_hat + x_t.T @ theta_hat_a + alpha * np.sqrt(s_t_a))[0, 0]
            p_values.append(p_t_a)
        selected_arm = np.argmax(p_values)
        if selected_arm == arm:
            A_0 += B[arm].T @ np.linalg.solve(A[arm], B[arm])
            b_0 += B[arm].T @ np.linalg.solve(A[arm], b[arm])
            A[arm] += x_t @ x_t.T
            B[arm] += x_t @ z_t.T
            b[arm] += reward * x_t
            A_0 += z_t @ z_t.T - B[arm].T @ np.linalg.solve(A[arm], B[arm])
            b_0 += reward * z_t - B[arm].T @ np.linalg.solve(A[arm], b[arm])
        total_reward += reward if selected_arm == arm else 0
    return total_reward / trials
