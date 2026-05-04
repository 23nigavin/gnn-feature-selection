# Functions for plotting results
import matplotlib.pyplot as plt

def plot_method_accuracy_vs_noise(noise_percent, results_by_method, filename="accuracy_vs_noise_all_methods.png", title="Accuracy vs. noise level"):
    plt.figure(figsize=(10, 6))
    for method_name, accuracies in results_by_method.items():
        plt.plot(noise_percent, accuracies, marker="o", label=method_name)
    plt.xlabel("Noise level (%)")
    plt.ylabel("Test accuracy")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def plot_accuracy_vs_k(k_values, clean_baseline_acc, corrupted_baseline_acc, k_selection_acc):
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, [clean_baseline_acc] * len(k_values), marker='o', label='No noise, no feature selection')
    plt.plot(k_values, [corrupted_baseline_acc] * len(k_values), marker='o', label='With noise, no feature selection')
    plt.plot(k_values, k_selection_acc, marker='o', label='With noise, with feature selection')
    plt.xlabel("Number of selected features (k)")
    plt.ylabel("Test accuracy")
    plt.title("Accuracy vs. number of selected features")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("accuracy_vs_k.png")
    plt.show()