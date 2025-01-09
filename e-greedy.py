import numpy as np
from tqdm import tqdm

def init(numFeatures, numArms):
    A = np.array([np.eye(numFeatures) for _ in range(numArms)])
    b = np.zeros((numArms, numFeatures, 1))
    theta = np.zeros((numArms, numFeatures, 1))
    return A, b, theta

# Omniscient
def omniscient(data_array, epsilon=None):
    total_payoff = np.sum(data_array[:, 1])
    return total_payoff / len(data_array)

# e-greedy
def e_greedy(data_array, epsilon):
    numArms = 10
    trials = data_array.shape[0]
    arm_counts = np.zeros(numArms)
    arm_rewards = np.zeros(numArms)
    total_payoff = 0
    for t in range(trials):
        if np.random.rand() < epsilon:
            # Exploration: choose a random arm
            selected_arm = np.random.randint(numArms)
        else:
            # Exploitation: choose arm with the highest average reward
            selected_arm = np.argmax(arm_rewards / np.maximum(arm_counts, 1))
        arm = int(data_array[t, 0]) - 1
        payoff = data_array[t, 1]
        arm_counts[selected_arm] += 1
        arm_rewards[selected_arm] += payoff
        total_payoff += payoff if selected_arm == arm else 0
    return total_payoff / trials

# Warm-Started e-greedy
def warm_started_e_greedy(data_array, epsilon):
    numArms = 10
    trials = data_array.shape[0]
    arm_rewards = np.zeros(numArms)
    arm_counts = np.ones(numArms)
    total_payoff = 0
    for t in range(trials):
        arm = int(data_array[t, 0]) - 1
        payoff = data_array[t, 1]
        if np.random.rand() < epsilon:
            selected_arm = np.random.randint(numArms)
        else:
            selected_arm = np.argmax(arm_rewards / np.maximum(arm_counts, 1))
        arm_counts[selected_arm] += 1
        if selected_arm == arm:
            arm_rewards[selected_arm] += payoff
        total_payoff += payoff if selected_arm == arm else 0
    return total_payoff / trials

# Segmented e-greedy
def segmented_e_greedy(data_array, epsilon):
    segment_size = 100
    num_segments = len(data_array) // segment_size
    if len(data_array) % segment_size != 0:
        num_segments += 1
    total_payoff = 0
    for i in range(num_segments):
        segment = data_array[i * segment_size : (i + 1) * segment_size]
        total_payoff += e_greedy(segment, epsilon)
    return total_payoff / num_segments

# Disjoint e-greedy
def disjoint_e_greedy(data_array, epsilon):
    numArms = 10
    trials = data_array.shape[0]
    numFeatures = data_array.shape[1] - 2
    A, b, theta = init(numFeatures, numArms)
    total_payoff = 0
    for t in range(trials):
        arm = int(data_array[t, 0]) - 1
        payoff = data_array[t, 1]
        x_t = np.expand_dims(data_array[t, 2:], axis=1)
        if np.random.rand() < epsilon:
            selected_arm = np.random.randint(numArms)
        else:
            p = [theta[a].T @ x_t for a in range(numArms)]
            selected_arm = np.argmax(p)
        if selected_arm == arm:
            A[selected_arm] += x_t @ x_t.T
            b[selected_arm] += payoff * x_t
            theta[selected_arm] = np.linalg.inv(A[selected_arm]) @ b[selected_arm]
        total_payoff += payoff if selected_arm == arm else 0
    return total_payoff / trials

# Hybrid e-greedy
def hybrid_e_greedy(data_array, epsilon):
    numArms = 10
    trials = data_array.shape[0]
    numFeatures = data_array.shape[1] - 2
    d = numFeatures + numArms
    B = np.eye(d)
    b = np.zeros((d, 1))
    A = np.array([np.eye(numFeatures) for _ in range(numArms)])
    a = np.zeros((numArms, numFeatures, 1))
    total_payoff = 0
    for t in range(trials):
        arm = int(data_array[t, 0]) - 1
        payoff = data_array[t, 1]
        x_t = np.expand_dims(data_array[t, 2:], axis=1)
        z_t = np.zeros((d, 1))
        z_t[numFeatures + arm] = 1
        theta = np.linalg.inv(B) @ b
        beta = np.linalg.inv(A[arm]) @ a[arm]
        if np.random.rand() < epsilon:
            selected_arm = np.random.randint(numArms)
        else:
            p_values = []
            for i in range(numArms):
                if i == arm:
                    p_t_a = (z_t.T @ theta + x_t.T @ beta)[0, 0]
                else:
                    p_t_a = (np.zeros((d, 1)).T @ theta + np.zeros((numFeatures, 1)).T @ beta)[0, 0]
                p_values.append(p_t_a)
            selected_arm = np.argmax(p_values)
        if selected_arm == arm:
            B += z_t @ z_t.T
            b += payoff * z_t
            A[arm] += x_t @ x_t.T
            a[arm] += payoff * x_t
        total_payoff += payoff if selected_arm == arm else 0
    return total_payoff / trials
