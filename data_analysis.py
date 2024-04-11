import matplotlib.pyplot as plt

def generate_precision_plot(study, dataAugDict, pltName, xName, yName):
    num_classes = len(study)
    bar_width = 0.15

    plt.figure(figsize=(10, 6))
    count = 0
    # Display each bar
    for key,value in dataAugDict.items():
        if count == 0:
            plt.bar(range(num_classes), value, width=bar_width, label=key)
        else:
            plt.bar([i + count * bar_width for i in range(num_classes)], value, width=bar_width, label=key)
        count += 1

        # Display values above the bars
        for i, j in enumerate(value):
            if count >= (num_classes/2):
                shift = i + abs(count - (num_classes/2)) * 0.145 + 0.1
            else:
                shift = i - abs(count - (num_classes/2)) * 0.145 + 0.1
            plt.text(shift, j + 1, str(j), fontsize=8, color='black', fontweight='bold')

    plt.xlabel(xName)
    plt.ylabel(yName)
    plt.title(pltName)
    plt.xticks([i + (count/2) * bar_width for i in range(num_classes)], study)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    study = ['Ablation 1', 'Ablation 2', 'Ablation 3', 'Ablation 4']
    dataAug = ['Original', 'Gaussian Noise', 'Sharpen', 'Anisotropic Diffusion', 'Rotation', 'Translation']
    dataAugDict = {dataAug[0] : [5, 5, 5, 5],
                dataAug[1] : [50, 50, 50, 50],
                dataAug[2] : [70, 70, 70, 70],
                dataAug[3] : [40, 40, 40, 40],
                dataAug[4] : [55, 55, 55, 55],
                dataAug[5] : [35, 35, 35, 35]}
    pltName = 'map_all for all augmentations?'
    xAxis = 'Type of ablation study'
    yAxis = 'map_all?'
    generate_precision_plot(study, dataAugDict, pltName, xAxis, yAxis)