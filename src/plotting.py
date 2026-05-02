# Functions for plotting results
import matplotlib.pyplot as plt

def plot_accuracy_vs_noise(noise_percent, clean_baseline_acc, noise_no_selection_acc, noise_with_selection_acc):
    plt.figure(figsize=(8, 5))
    plt.plot(noise_percent, [clean_baseline_acc] * len(noise_percent), marker='o', label='No noise, no feature selection')
    plt.plot(noise_percent, noise_no_selection_acc, marker='o', label='With noise, no feature selection')
    plt.plot(noise_percent, noise_with_selection_acc, marker='o', label='With noise, with feature selection')
    plt.xlabel("Extra junk features added (% of original feature count)")
    plt.ylabel("Test accuracy")
    plt.title("Accuracy vs. noise level")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("accuracy_vs_noise.png")
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