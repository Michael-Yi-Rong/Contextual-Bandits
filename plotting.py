import matplotlib.pyplot as plt

data_array = np.loadtxt('dataset.txt', dtype=int)
epsilon_values = np.linspace(0, 1, 21)
algorithms = {
    "Omniscient": omniscient,
    "e-greedy": e_greedy,
    "Warm-Started e-greedy": warm_started_e_greedy,
    "Segmented e-greedy": segmented_e_greedy,
    "Disjoint e-greedy": disjoint_e_greedy,
    "Hybrid e-greedy": hybrid_e_greedy,
}
plt.figure(figsize=(12, 8), dpi=160)
for name, func in algorithms.items():
    ctr_values = []
    for epsilon in tqdm(epsilon_values, desc=f"Running {name}"):
        ctr = func(data_array, epsilon)
        ctr_values.append(ctr)
    if name == "Omniscient":
        marker_style = '_'
    else:
        marker_style = 'o'
    plt.plot(epsilon_values, ctr_values, marker=marker_style, label=name)
plt.xlabel("Epsilon")
plt.ylabel("CTR")
plt.xlim(0, 1)
plt.ylim(0,)
plt.title("CTR vs Epsilon")
plt.legend()
plt.grid()
plt.show()


base_ctr = np.zeros(10)
for arm in range(10):
    base_ctr[arm] = np.mean(data_array[data_array[:, 0] == arm + 1, 1])
epsilon = 0.2
alpha = 0.05
lifted_e_greedy = e_greedy(data_array, epsilon)
lifted_ucb = ucb(data_array, alpha)
plt.figure(figsize=(8, 8), dpi=160)
plt.scatter(base_ctr, lifted_e_greedy, color='red', label='e-greedy', marker='x', s=100)
plt.scatter(base_ctr, lifted_ucb, color='blue', label='UCB', marker='o', s=100)
plt.plot([0, 1], [0, 1], color='black', linestyle='--', label='y = x')
plt.xlabel('Base CTR')
plt.ylabel('Lifted CTR')
plt.title('Base CTR vs Lifted CTR for e-greedy & UCB')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.show()


def get_subsets(data_array, percentages):
    subsets = [data_array[:int(len(data_array) * p / 100)] for p in percentages]
    return subsets
percentages = [1, 5, 10, 20, 30, 100]
subsets = get_subsets(data_array, percentages)
all_results = {}
for i, subset in enumerate(subsets):
    print(f"Evaluating for {percentages[i]}% of data...")
    all_results[percentages[i]] = evaluate_algorithms(subset, epsilon) # alpha
labels = list(all_results[100].keys())
x = np.arange(len(percentages))
width = 0.15
plt.figure(dpi=160)
fig, ax = plt.subplots(figsize=(10, 6))
for i, label in enumerate(labels):
    rewards = [all_results[percentage][label] for percentage in percentages]
    ax.bar(x + i * width, rewards, width, label=label)
ax.set_xlabel('Data Subset Size (%)')
ax.set_ylabel('Average Reward')
ax.set_title(f'Algorithm Performance by Data Subset Size (epsilon={epsilon})')
ax.set_xticks(x + width * (len(labels) - 1) / 2)
ax.set_xticklabels(percentages)
ax.legend(title="Algorithms")
plt.tight_layout()
plt.show()
