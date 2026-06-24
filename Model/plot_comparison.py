import os
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Performance metrics
    models = ['BPR-MF', 'LightGCN']
    recall_scores = [0.1238 , 0.1384]
    ndcg_scores = [0.0461, 0.0540]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    # Plotting bars
    rects1 = ax.bar(x - width/2, recall_scores, width, label='Recall@20', color='skyblue', edgecolor='black')
    rects2 = ax.bar(x + width/2, ndcg_scores, width, label='NDCG@20', color='lightgreen', edgecolor='black')

    # Add text for labels, title and custom x-axis tick labels
    ax.set_ylabel('Scores')
    ax.set_title('Top-20 Ranking Comparison: MF vs LightGCN')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 0.16)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend()

    # Function to attach a text label above each bar
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    # Save the plot
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'comparison_bar_chart.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison chart successfully saved to {save_path}")

if __name__ == '__main__':
    main()
