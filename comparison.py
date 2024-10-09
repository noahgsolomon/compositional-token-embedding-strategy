import matplotlib.pyplot as plt

# Read the losses from the files
with open("training_loss_mapped.txt", "r") as f:
    losses_mapped = [float(line.strip()) for line in f]

with open("training_loss_anew.txt", "r") as f:
    losses_anew = [float(line.strip()) for line in f]

# Plot the losses
plt.figure(figsize=(10, 6))
plt.plot(losses_mapped, label='Tokens Mapped (Custom Embeddings)')
plt.plot(losses_anew, label='Tokens Anew (Random Embeddings)')
plt.xlabel('Training Iterations')
plt.ylabel('Loss')
plt.title('Training Loss Comparison')
plt.legend()
plt.grid(True)

# Save the plot to a file instead of showing it
plt.savefig('loss_comparison.png', dpi=300, bbox_inches='tight')
plt.close()  # Close the figure to free up memory