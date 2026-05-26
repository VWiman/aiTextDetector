
import matplotlib.pyplot as plt
# ============================================================
# EDA (Exploratory Data Analysis)
# ============================================================
def eda(df_sample):
    # Class distribution
    print("\nLabel distribution:")
    print(df_sample['label'].value_counts())

    # Text length statistics
    df_sample['text_length'] = df_sample['text'].str.len()
    df_sample['word_count'] = df_sample['text'].str.split().str.len()

    print("\nText length statistics by label:")
    print(df_sample.groupby('label')[['text_length', 'word_count']].describe().round(1))

    # --- Plot: EDA distributions ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    df_sample['label'].value_counts().plot(kind='bar', ax=axes[0], color=["#008CFF", '#FF5722'])
    axes[0].set_title('Class Distribution', fontsize=14)
    axes[0].set_xlabel('Label')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=0)

    for label, color in zip(['Human', 'AI'], ['#2196F3', '#FF5722']):
        subset = df_sample[df_sample['label'] == label]
        axes[1].hist(subset['text_length'], bins=50, alpha=0.6, label=label, color=color)
    axes[1].set_title('Text Length Distribution', fontsize=14)
    axes[1].set_xlabel('Character Count')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()

    for label, color in zip(['Human', 'AI'], ["#008CFF", '#FF5722']):
        subset = df_sample[df_sample['label'] == label]
        axes[2].hist(subset['word_count'], bins=50, alpha=0.6, label=label, color=color)
    axes[2].set_title('Word Count Distribution', fontsize=14)
    axes[2].set_xlabel('Word Count')
    axes[2].set_ylabel('Frequency')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('eda_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved: eda_distributions.png")