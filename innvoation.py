# segmented_disjoint_linucb
def segmented_disjoint_linucb(data_array, alpha):
    segment_size = 100
    num_segments = len(data_array) // segment_size
    if len(data_array) % segment_size != 0:
        num_segments += 1
    total_payoff = 0
    for i in range(num_segments):
        segment = data_array[i * segment_size : (i + 1) * segment_size]
        numArms = 10
        trials = segment.shape[0]
        numFeatures = segment.shape[1] - 2
        A, b, theta = init(numFeatures, numArms)
        segment_payoff = 0
        for t in range(trials):
            arm = int(segment[t, 0]) - 1
            payoff = segment[t, 1]
            x_t = np.expand_dims(segment[t, 2:], axis=1)
            p = [theta[a].T @ x_t + alpha * np.sqrt(x_t.T @ np.linalg.inv(A[a]) @ x_t) for a in range(numArms)]
            selected_arm = np.argmax(p)
            if selected_arm == arm:
                A[selected_arm] += x_t @ x_t.T
                b[selected_arm] += payoff * x_t
                theta[selected_arm] = np.linalg.inv(A[selected_arm]) @ b[selected_arm]
            segment_payoff += payoff if selected_arm == arm else 0
        total_payoff += segment_payoff / trials
    return total_payoff / num_segments

# disjoint_e_greedy_linucb
def disjoint_e_greedy_linucb(data_array, alpha):
    epsilon=0.2
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
            p = [theta[a].T @ x_t + alpha * np.sqrt(x_t.T @ np.linalg.inv(A[a]) @ x_t) for a in range(numArms)]
            selected_arm = np.argmax(p)
        if selected_arm == arm:
            A[selected_arm] += x_t @ x_t.T
            b[selected_arm] += payoff * x_t
            theta[selected_arm] = np.linalg.inv(A[selected_arm]) @ b[selected_arm]
        total_payoff += payoff if selected_arm == arm else 0
    return total_payoff / trials
