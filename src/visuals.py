import os
import yaml
import datetime
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from . import utils


def plot_encode_assay_sample_distribution(files_samples_df, 
        assay_col='Assay term name', sample_col='Biosample name', 
        name=None, assay_label='Assay Term Name', sample_label='Biosample Name', plot_path=None):

    assay_counts = Counter(files_samples_df[assay_col])
    assay_names = list(assay_counts.keys())
    assay_values = [assay_counts[name] for name in assay_names]
    # sort by values
    assay_names, assay_values = zip(*sorted(zip(assay_names, assay_values), key=lambda x: x[1], reverse=True))
    plt.figure(figsize=(10,6))
    plt.bar(assay_names, assay_values)
    # show the percentages on top of bars
    total = sum(assay_values)
    for i, v in enumerate(assay_values):
        plt.text(i, v + 1, f"{v}\n({v/total:.1%})", ha='center', va='bottom', fontsize=10)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel(assay_label)
    plt.ylabel('Number of Files')
    plt.title(f'Distribution of Assay Types (N={len(assay_names)}) in {name} Dataset')
    plt.tight_layout()
    # plt.savefig("data/entex/plots/entex_final_files_samples_assay_term_name_distribution.png")
    plt.show()

    # show the distrubtion of "Biosample name"
    sample_counts = Counter(files_samples_df[sample_col])
    sample_names = list(sample_counts.keys())
    sample_values = [sample_counts[name] for name in sample_names]
    # sort by values
    sample_names, sample_values = zip(*sorted(zip(sample_names, sample_values), key=lambda x: x[1], reverse=True))
    plt.figure(figsize=(10,6))
    plt.bar(sample_names, sample_values)
    # # show the percentages on top of bars
    # total = sum(sample_values)
    # for i, v in enumerate(sample_values):
    #     plt.text(i, v + 1, f"{v}\n({v/total:.1%})", ha='center', va='bottom', fontsize=10)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel(sample_label)
    plt.ylabel('Number of Files')
    plt.title(f'Distribution of Biosample Names (N={len(sample_names)}) in {name} Dataset')
    plt.tight_layout()
    # plt.savefig("data/entex/plots/entex_final_files_samples_biosample_name_distribution.png")
    plt.show()

    
def plot_train_feats(feats, tissue_idx=0, meta_dict={}):

    fig, axes = plt.subplots(nrows=3, sharex=True, figsize=(16, 6))
    for ax in axes:
        ax.margins(y=0)
    # panel 1: onehot encoding of sequence
    seq_onehot = feats['seq_onehot'][:, 5000:10000]
    seq_onehot = np.concatenate([seq_onehot, 1 - seq_onehot.sum(axis=0, keepdims=True)], axis=0)
    axes[0].imshow(seq_onehot, aspect='auto', cmap='Greys', interpolation='nearest')
    axes[0].set_title(f'Sequence and Splice Feats {meta_dict}')
    axes[0].set_yticks([0, 1, 2, 3, 4])
    axes[0].set_yticklabels(['A', 'C', 'G', 'T', 'N'])
    axes[0].set_ylabel('One-hot Encoding')

    # panel 2: cls
    cls_label = feats['cls'][tissue_idx] if feats['cls'].ndim == 2 else feats['cls']
    axlines = axes[1].plot(cls_label, label='true')

    cls_max = max([1, cls_label.max()])

    if len(feats['seq']) == 15000:
        seq = feats['seq'][5000:10000]
    elif len(feats['seq']) == 5000:
        seq = feats['seq']
    else:
        raise ValueError(f'Unexpected seq length: {len(feats["seq"])}')
    
    if isinstance(seq, bytes):
        seq = seq.decode('utf-8')

    for p in np.where(cls_label > 0)[0]:
        axes[1].text(p, cls_max * 1.05, f'{seq[p]}', fontsize=12, color=axlines[0].get_color(), ha='center', va='center')
        axes[1].text(p, cls_max * 1.16, f'{p}', fontsize=10, color=axlines[0].get_color(), ha='center', va='bottom', rotation=90)        

    axes[1].set_ylim(0, cls_max * 1.35)
    axes[1].set_ylabel('Splice Site\nClassification')

    # panel 3: psi
    psi_label = feats['psi'][tissue_idx] if feats['psi'].ndim == 2 else feats['psi']
    axlines = axes[2].plot(psi_label, label='true')
    axes[2].set_ylim(-2.8, 1.2)

    psi_pos = np.where(psi_label > 0.05)[0]
    for p in psi_pos:
        axes[2].text(p, 1.05, seq[p], fontsize=12, color=axlines[0].get_color(), ha='center', va='center')

    axes[2].set_ylim(0, 1.15)
    
    axes[2].set_xlabel('Sequence Position')
    axes[2].set_ylabel('Psi Value\nRegression')

    axes[0].grid(axis='x')
    axes[1].grid(axis='x')
    axes[2].grid(axis='x')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.show()


def load_model_dataset_metrics(model_dataset_paths):
    model_dataset_metrics = {}
    for model, dataset_paths in model_dataset_paths.items():
        dataset_metrics = {}
        for dataset, yaml_path in dataset_paths.items():
            with open(yaml_path, 'r') as f:
                dataset_metrics[dataset] = yaml.safe_load(f)
        model_dataset_metrics[model] = dataset_metrics
    return model_dataset_metrics


def plot_model_dataset_metrics(model_dataset_metrics, ys=['cls_loss', 'psi_loss', 'cls_f1', 'psi_mse', 'auprc', 'auroc'], save_prefix=None, display=True):
    import matplotlib.pyplot as plt
    
    fig = plt.figure(figsize=(12, 8))

    model_names = list(model_dataset_metrics.keys())
    dataset_names = list(model_dataset_metrics[model_names[0]].keys())
    n_models = len(model_dataset_metrics)
    n_datasets = len(model_dataset_metrics[model_names[0]])

    for i, y in enumerate(ys):
        plt.subplot((len(ys) + 1) // 2, 2, i + 1)
        plt.title(f'{y}')
        xpos = np.arange(n_datasets)
        for j, model in enumerate(model_names):
            dataset_metrics = model_dataset_metrics[model]

            yval = [dataset_metrics[dataset][y] for dataset in dataset_names]
            plt.bar(xpos + ((j+0.5) / n_models - 0.5)*0.85, yval, width=1/(n_models + 1)*0.9, align='center', alpha=0.7, label=model)
        plt.xticks(xpos, dataset_names)
        plt.ylabel(y)
        if y.endswith('loss'):
            plt.yscale('log')
        plt.grid()
        plt.tight_layout()

    # add model_names as legends at the top of the entire figure
    fig.legend(model_names, loc='upper center', bbox_to_anchor=(0.5, 0.01), ncol=min([4, n_models]))
    fig.tight_layout()
    if save_prefix:
        out_png = f'{save_prefix}_model_dataset_metrics.png'
        plt.savefig(out_png)
        print(f'Saved figure to {out_png}')
    if display:
        plt.show()


def plot_agg_metrics(agg_metrics, save_prefix=None, display=True):
    plt.figure(figsize=(13, 10))

    # display all scalar number metrics in sum_metrics as a table
    plt.subplot(2, 2, 1)
    scalar_metrics = {k: v for k, v in agg_metrics.items() if utils.is_scalar(v)}
    table_data = [[k, f"{int(v):,}" if int(v) == v else f"{v:.6f}"] for k, v in scalar_metrics.items()]
    table = plt.table(cellText=table_data, colLabels=['Metric', 'Value'], loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    plt.axis('off')
    # plt.title('Overall Metrics')

    plt.subplot(2, 2, 2)
    plt.title('Top-K Metrics')
    metrics_to_plot = ['topk_accuracy', 'topk_precision', 'topk_recall', 'topk_f1']
    width = 1.0 / len(metrics_to_plot) * 0.8  # the width of the bars
    x_pos = np.arange(len(agg_metrics['topk_accuracy'])) - (width * (len(metrics_to_plot) - 1) / 2)

    for i, metric in enumerate(metrics_to_plot):
        plt.bar(x_pos + i * width, agg_metrics[metric], width=width, label=metric.split('_')[-1])
    plt.xticks(np.arange(len(agg_metrics['topk_accuracy'])),
            [f"K={int(k) if k >= 1 else k}\n({agg_metrics['k'][i]})" for i, k in enumerate(agg_metrics['k0'])])
    plt.grid()
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.title('Precision-Recall Curve')
    plt.plot(agg_metrics['recall'], agg_metrics['precision'], linestyle='-', marker=None, label='PRC curve')
    # mark top-k points
    markers = ['o', 's', '^', 'D', 'v', '*', 'P', 'X', 'h', '+', 'x']
    for i, (k0, precision, recall) in enumerate(zip(agg_metrics['k0'], agg_metrics['topk_precision'], agg_metrics['topk_recall'])):
        marker = markers[i % len(markers)]
        plt.scatter(recall, precision, marker=marker, s=100, edgecolors='k',
            label=f"Top K={int(k0) if k0 >= 1 else k0}")
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid()
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.title('Receiver Operating Characteristic Curve')
    plt.plot(agg_metrics['fpr'], agg_metrics['tpr'], linestyle='-', marker=None, label='ROC curve')
    # try to calculate fpr at top-k points from k0 (the ratio between predicted positives and total positives), precision, and recall
    # for i, (k0, precision, recall) in enumerate(zip(agg_metrics['k0'], agg_metrics['topk_precision'], agg_metrics['topk_recall'])):
    #     fpr = (recall - precision * recall) / (precision + 1e-8)
    #     plt.scatter(fpr, precision, marker=markers[i % len(markers)], s=100, edgecolors='k',
    #                 label=f"Top K={int(k0) if k0 >= 1 else k0}")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid()
    plt.legend()            

    plt.figtext(0.5, 0.03, f"{save_prefix}.png\nCreation time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", horizontalalignment='center', fontsize=9)

    if save_prefix:
        plt.savefig(f"{save_prefix}.png", dpi=300, bbox_inches='tight')

    if display:
        plt.show()


def plot_one_hot(array, title="One-Hot Encoded Sequence"):
    seq_len = array.shape[0]
    plt.figure(figsize=(12, 3))
    plt.imshow(array.T, aspect='auto', cmap='Greys', interpolation='nearest')
    
    # Y-axis: bases
    plt.yticks(ticks=[0, 1, 2, 3], labels=['A', 'C', 'G', 'T'])

    # X-axis: sequence positions (e.g., 1 to 1500)
    step = max(seq_len // 20, 1)  # show ~20 ticks
    xtick_positions = list(range(0, seq_len, step))
    xtick_labels = [str(i + 1) for i in xtick_positions]
    plt.xticks(ticks=xtick_positions, labels=xtick_labels, rotation=90)

    plt.xlabel("Position in sequence")
    plt.title(title)
    plt.colorbar(label="One-hot value")
    plt.tight_layout()
    plt.show()


def Plot_2D_Array(array):
    ## 2D torch tensor, shape =  torch.Size([3, 15000])
    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 6), sharex=True)
    colors = ['gray', 'blue', 'red']
    labels = ['Un', 'S', 'Usage']
    # X-axis (sequence positions)
    x = np.arange(len(array[0]))
    
    for i in range(3):
        axes[i].plot(x, array[i], color = colors[i])
        axes[i].set_ylabel(labels[i])
        axes[i].set_ylim(0, 1)
        axes[i].grid(True)
    
    axes[-1].set_xlabel("Sequence Position")
    fig.suptitle("Individual Class Predictions Over Sequence", y=1.02)
    plt.tight_layout()
    plt.show()
    return None


def plot_one_hot_and_labels_zoom(one_hot_array, label_array, zoom_start=0, zoom_end=None, title="Zoomed Sequence and Labels"):
    """
    Plot one-hot encoded sequence, expression data, label classes (U, S), and usage.

    Args:
        one_hot_array: (>=4, sequence_length) numpy array
        label_array: (>=3, label_length) numpy array
        zoom_start: start position of zoom window in input coordinates
        zoom_end: end position of zoom window in input coordinates
        title: plot title
    """
    seq_len = one_hot_array.shape[1]
    label_len = label_array.shape[1]
    input_start_for_labels = (seq_len - label_len) // 2

    if zoom_end is None:
        zoom_end = seq_len
    zoom_len = zoom_end - zoom_start

    # Slice one-hot array
    one_hot_zoom = one_hot_array[:, zoom_start:zoom_end]  # shape (channels, zoom_len)

    # Expression starts from index 4
    sequence = one_hot_zoom[0:4, :]
    expression = one_hot_zoom[4:, :] if one_hot_zoom.shape[0] > 4 else np.zeros((1, zoom_len))

    # Calculate label slice range
    label_start_idx = max(0, zoom_start - input_start_for_labels)
    label_end_idx = max(0, zoom_end - input_start_for_labels)
    label_zoom_len = label_end_idx - label_start_idx

    if label_start_idx >= label_len:
        label_zoom = np.zeros((label_array.shape[0], zoom_len))
        label_positions = np.arange(zoom_start, zoom_end)
    else:
        label_zoom = label_array[:, label_start_idx:label_end_idx]
        label_positions = np.arange(input_start_for_labels + label_start_idx,
                                    input_start_for_labels + label_end_idx)

    # Extract U, S, Usage
    unspliced = label_zoom[0, :]
    spliced = label_zoom[1, :]
    usage = label_zoom[2, :] if label_zoom.shape[0] > 2 else np.zeros(label_zoom.shape[1])

    # Plot setup
    fig, axes = plt.subplots(4, 1, figsize=(14, 8), sharex=True, 
                             gridspec_kw={"height_ratios": [1, 1, 1, 1]})

    # --- Plot 1: Sequence TGCA ---
    axes[0].imshow(sequence, aspect='auto', cmap='Greys', interpolation='nearest',
                   extent=[zoom_start, zoom_end, 0, 4])
    axes[0].set_yticks([0.5, 1.5, 2.5, 3.5])
    axes[0].set_yticklabels(['T', 'G', 'C', 'A'])
    axes[0].set_title("One-Hot Encoded Sequence (TGCA)")

    # --- Plot 2: Expression ---
    axes[1].imshow(expression, aspect='auto', cmap='viridis', interpolation='nearest',
                   extent=[zoom_start, zoom_end, 0, expression.shape[0]])
    axes[1].set_yticks(np.arange(0.5, expression.shape[0]))
    axes[1].set_yticklabels([f'Exp{i}' for i in range(expression.shape[0])])
    axes[1].set_title("Expression (Additional One-Hot Channels)")

    # --- Plot 3: Labels U/S (as line plots like Usage) ---
    if label_zoom.shape[1] > 0:
        axes[2].plot(label_positions, unspliced[:label_zoom_len], label='U', color='blue', linewidth=1)
        axes[2].plot(label_positions, spliced[:label_zoom_len], label='S', color='green', linewidth=1)
    else:
        axes[2].plot(np.arange(zoom_start, zoom_end), np.zeros(zoom_len), label='U', color='blue')
        axes[2].plot(np.arange(zoom_start, zoom_end), np.zeros(zoom_len), label='S', color='green')

    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Label")
    axes[2].set_title("Label Classes (U: blue, S: green)")
    axes[2].legend(loc='upper right')


    # --- Plot 4: Usage ---
    if label_zoom.shape[1] > 0:
        axes[3].plot(label_positions, usage[:label_zoom_len], color='Red')
    else:
        axes[3].plot(np.arange(zoom_start, zoom_end), np.zeros(zoom_len), color='gray')
    axes[3].set_ylim(0, 1)
    axes[3].set_ylabel("Usage")
    axes[3].set_title("Estimated Usage Level")

    # X-axis ticks
    step = max(zoom_len // 20, 1)
    xtick_positions = list(range(zoom_start, zoom_end, step))
    xtick_labels = [str(i) for i in xtick_positions]
    axes[3].set_xticks(xtick_positions)
    axes[3].set_xticklabels(xtick_labels, rotation=90)
    axes[3].set_xlabel("Position in Sequence")

    plt.tight_layout()
    plt.suptitle(title, y=1.02)
    plt.show()
    return None


def plot_one_hot_and_predictions_zoom(one_hot_array, prediction_array, zoom_start=8600, zoom_end=9001, title="Zoomed Prediction View"):
    """
    Plot one-hot encoded input and model predictions for a zoomed region.

    Args:
        one_hot_array: (4, 15000) array, one-hot encoded input sequence (A, C, G, T)
        prediction_array: (12, 5000) array, model output aligned to positions 5000:10000
        zoom_start: start position in input coordinates (0–14999)
        zoom_end: end position in input coordinates
    """
    full_seq_len = one_hot_array.shape[1]
    pred_len = prediction_array.shape[1]
    pred_start = (full_seq_len - pred_len) // 2  # e.g., 5000
    pred_end = pred_start + pred_len             # e.g., 10000

    if zoom_start < 0 or zoom_end > full_seq_len:
        raise ValueError("Zoom region is outside of input sequence bounds.")

    zoom_len = zoom_end - zoom_start

    # --- One-hot ---
    one_hot_zoom = one_hot_array[:, zoom_start:zoom_end]  # shape (4, zoom_len)

    # --- Predictions ---
    pred_start_idx = max(0, zoom_start - pred_start)
    pred_end_idx = max(0, zoom_end - pred_start)

    if pred_start_idx >= pred_len or pred_end_idx > pred_len:
        pred_zoom = np.zeros((12, zoom_len))
        pred_positions = np.arange(zoom_start, zoom_end)
    else:
        pred_zoom = prediction_array[:, pred_start_idx:pred_end_idx]  # (12, L)
        pred_positions = np.arange(pred_start + pred_start_idx, pred_start + pred_end_idx)

    # --- Plot ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 6), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1]})



    # Predicted splice class
     # Convert class labels: 0 = unknown, 1 = unspliced, 2 = spliced
    if pred_zoom.shape[1] > 0:
        unspliced = pred_zoom[0, :]
        spliced = pred_zoom[1, :]
        class_labels = []
        for u, s in zip(unspliced, spliced):
            if (u - s) > 0.9:
                class_labels.append(1)
            elif (s - u) > 0.2:
                class_labels.append(2)
            else:
                class_labels.append(0)

        # Define custom colormap
        cmap = mcolors.ListedColormap(['gray', 'white', 'red'])
        bounds = [-0.5, 0.5, 1.5, 2.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        
        axes[0].imshow([class_labels], aspect='auto', cmap=cmap, norm=norm, interpolation='nearest',
                       extent=[pred_positions[0], pred_positions[-1] + 1, 0, 1])
    else:
        axes[0].imshow([[0] * zoom_len], aspect='auto', cmap=mcolors.ListedColormap(['gray']),
                       interpolation='nearest', extent=[zoom_start, zoom_end, 0, 1])
    axes[0].set_yticks([0.5])
    axes[0].set_yticklabels(["Splice Class"])
    axes[0].set_title("Splice Class (Grey=unknown, White=unspliced, Red=spliced)")

    # Predicted usage
    if pred_zoom.shape[1] > 0:
        usage = pred_zoom[2, :]
        axes[1].plot(pred_positions, usage, color='red')
    else:
        axes[1].plot(np.arange(zoom_start, zoom_end), np.zeros(zoom_len), color='gray')
        
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Usage")
    axes[1].set_title("Predicted Usage Level")

    # X ticks
    step = max(zoom_len // 10, 1)
    xtick_positions = list(range(zoom_start, zoom_end, step))
    xtick_labels = [str(i) for i in xtick_positions]
    axes[1].set_xticks(xtick_positions)
    axes[1].set_xticklabels(xtick_labels, rotation=90)
    axes[1].set_xlabel("Position in Sequence")

        # One-hot sequence
    axes[2].imshow(one_hot_zoom, aspect='auto', cmap='Greys', interpolation='nearest',
                   extent=[zoom_start, zoom_end, 0, 4])
    axes[2].set_yticks([0.5, 1.5, 2.5, 3.5])
    axes[2].set_yticklabels(['A', 'C', 'G', 'T'])
    axes[2].set_title("One-Hot Encoded Sequence")
    
    plt.tight_layout()
    plt.suptitle(title, y=1.03, fontsize=14)
    plt.show()
    return None
    #class_labels
