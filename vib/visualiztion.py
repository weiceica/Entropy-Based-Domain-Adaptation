import matplotlib.pyplot as plt
import re

def parse_file(filepath):
    epochs = []
    accuracies = []
    
    # Regular expression to match the required data
    pattern = re.compile(r"Epoch: \[(\d+)/\d+\], .* acc: ([\d.]+)")

    with open(filepath, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                epoch = int(match.group(1))
                accuracy = float(match.group(2))
                epochs.append(epoch)
                accuracies.append(accuracy)
                
    return epochs, accuracies

def parse_and_plot(filepaths, labels, output_filepath):
    plt.figure(figsize=(14, 7))
    
    colors = ['b', 'g', 'r']
    offsets = [0.05, 0.10, 0.15]  # Different offsets for each annotation

    for i, (filepath, label) in enumerate(zip(filepaths, labels)):
        epochs, accuracies = parse_file(filepath)
        plt.plot(epochs, accuracies, marker='o', linestyle='-', color=colors[i], label=label, zorder=i)
        max_acc = max(accuracies)
        max_epoch = epochs[accuracies.index(max_acc)]
        xytext_pos = (max_epoch, max_acc - offsets[i])
        
        plt.annotate(f'Max Acc: {max_acc:.4f}', 
                     xy=(max_epoch, max_acc), 
                     xytext=xytext_pos,
                     arrowprops=dict(facecolor=colors[i], shrink=0.05, zorder=i),
                     horizontalalignment='center', 
                     verticalalignment='top', 
                     fontsize=9, 
                     color=colors[i],
                     bbox=dict(boxstyle='round,pad=0.3', edgecolor=colors[i], facecolor='white', alpha=0.6),
                     zorder=i+1)

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Epoch vs Accuracy')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.savefig(output_filepath, format='png')
    plt.close()
input_filepaths = ['res/test1.txt', 'res/ddc_kl_bottleneck.txt', 'res/ddc_original.txt']
labels = ['viib', 'vib + ddc', 'original']
output_filepath = 'webcam_to_amazonMay21.png'
parse_and_plot(input_filepaths, labels, output_filepath)
