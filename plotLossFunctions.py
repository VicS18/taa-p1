import os
import pickle
import argparse
import matplotlib.pyplot as plt

def plot_loss_functions(optimizer, data_aug, lr, bsz, project_dir):
    plots_dir = os.path.join(project_dir, optimizer, data_aug, "plots")
    loss_dir = os.path.join("loss", f"lr_{lr}", f"bsz_{bsz}_fig.obj")

    train_loss_file = os.path.join(plots_dir, "train", loss_dir)
    val_loss_file = os.path.join(plots_dir, "validation", loss_dir)

    print(train_loss_file)

    # Check if the files exist and plot the loss curves
    if os.path.exists(train_loss_file) and os.path.exists(val_loss_file):
        with open(train_loss_file, 'rb') as f:
            train_loss_fig = pickle.load(f)
            train_loss = train_loss_fig.gca().lines[0].get_ydata()
        with open(val_loss_file, 'rb') as f:
            val_loss_fig = pickle.load(f)
            val_loss = val_loss_fig.gca().lines[0].get_ydata()

        epochs = range(1, len(train_loss) + 1)
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_loss, label='Training Loss')
        plt.plot(epochs, val_loss, label='Validation Loss')
        plt.title(f'Training and Validation Loss for {optimizer} (LR: {lr}, Batch Size: {bsz})')
        plt.xlabel('Epochs')
        plt.ylabel('$J(\\Theta)$')
        plt.legend()
        plt.show()
    else:
        print("Training or validation loss figure not found.")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Plot training and validation loss functions.')
    parser.add_argument('optimizer', help='Optimizer used (e.g., Adam)')
    parser.add_argument('data_aug', help='Whether data augmentation was used (e.g., False_DATA_AUG)')
    parser.add_argument('lr', type=float, help='Learning rate')
    parser.add_argument('bsz', type=int, help='Batch size')
    parser.add_argument('--project_dir', default='.', help='Project directory path')
    args = parser.parse_args()
    print(args)
    # Plot the loss functions
    plot_loss_functions(args.optimizer, args.data_aug, args.lr, args.bsz, args.project_dir)
