import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import textwrap

# def plot_method_performance(df, metric_type, figsize=(12, 6), ylim=(60, 100)):
#     """
#     Create a bar plot of method performance using seaborn.
    
#     Parameters:
#     -----------
#     df : pandas DataFrame
#         DataFrame containing the data in the flat structure with columns:
#         'method', 'type', 'metric_type', 'metric', 'performance'
#     metric_type : str
#         Type of metric to plot ('blooms_level' or 'question_type')
#     figsize : tuple, optional (default=(12, 8))
#         Figure size in inches
#     ylim : tuple, optional (default=(60, 100))
#         Y-axis limits
    
#     """
#     # Filter data for the specified metric type
#     plot_df = df[df['metric_type'] == metric_type].copy()
    
#     # Calculate y-axis lower limit based on data
#     ymin = plot_df['performance'].min()
#     y_margin = (100 - ymin) * 0.1  # 10% of the range to max
#     ylim = (max(0, ymin - y_margin), 100)  # Keep upper limit at 100
    
#     # Create color palette
#     colors = {
#         # Open Source - Blues
#         'OS-1': '#2c7bb6',
#         'OS-2': '#539cc6',
#         'OS-3': '#7cbdd6',
#         'OS-4': '#a5dee6',
#         # Closed Source - Oranges
#         'CS-1': '#e66101',
#         'CS-2': '#ec8843',
#         'CS-3': '#f1af85',
#         # Specialized - Purples
#         'S-1': '#5e3c99',
#         'S-2': '#806bb3',
#         'S-3': '#a39acc'
#     }
    
#     # Create figure
#     fig, ax = plt.subplots(figsize=figsize)
    
#     # Create the plot using seaborn
#     sns.barplot(
#         data=plot_df,
#         x='metric',
#         y='performance',
#         hue='method',
#         palette=colors,
#         ax=ax
#     )
    
#     # Set title based on metric type
#     title = f"Performance by {metric_type}"
#     ax.set_title(title, fontsize=20, pad=20)
    
#     # Customize axes with bigger font sizes
#     ax.set_ylabel('Performance', fontsize=18)
#     ax.set_xlabel('')
#     ax.set_ylim(ylim)
    
#     # Format x-axis labels - split into two lines if needed
#     if metric_type == 'question_type':
#         labels = ['Expert Visual\nUnderstanding', 'Hypothesis\nGeneration', 'Experimental\nProposal']
#         ax.set_xticklabels(labels, fontsize=16)
#     else:
#         ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)
    
#     # Increase tick label size
#     ax.tick_params(axis='both', which='major', labelsize=16)
    
#     # Adjust legend with bigger font
#     ax.legend(
#         bbox_to_anchor=(1.05, 1),
#         loc='upper left',
#         borderaxespad=0,
#         title='Method',
#         fontsize=16,
#         title_fontsize=18
#     )
    
#     # Add grid lines
#     ax.yaxis.grid(True, linestyle='--', alpha=0.3)
#     ax.set_axisbelow(True)  # Put grid behind bars
    
#     # Adjust layout
#     plt.tight_layout()

    
#     plt.savefig(f'{metric_type}_performance.png', dpi=300)

def wrap_labels(text, width=20):
    """
    Wrap text at specified width using textwrap.
    
    Parameters:
    text : str
        Text to wrap
    width : int, optional (default=20)
        Maximum line length
    """
    return '\n'.join(textwrap.wrap(text, width=width))

def plot_model_performance(performance_df, exp_name, save_path, method_order, figsize=(12, 6)):
    """
    Create a bar plot of model performance using seaborn.
    Uses colors from the performance_df['color'] column.
    
    Parameters:
    performance_df : pandas DataFrame
        DataFrame containing 'model_name', 'performance', and 'color' columns
    exp_name : str
        Name of the experiment for the plot title
    figsize : tuple, optional (default=(12, 6))
        Figure size in inches
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # # Calculate y-axis limits with margin
    # ymin = performance_df['performance'].min()
    # y_margin = (1 - ymin) * 0.1  # 10% of the range to max
    # ylim = (max(0, ymin - y_margin), 1)
    
    # Create color mapping dictionary from the DataFrame
    color_dict = dict(zip(performance_df['method'].unique(),
                         performance_df.groupby('method')['color'].first()))
    
    # Create the grouped bar plot using seaborn
    bars = sns.barplot(
        data=performance_df,
        x=exp_name,
        y='performance',
        hue='method',
        hue_order=method_order,
        palette=color_dict,
        ax=ax
    )
    
    # Get current tick positions and labels
    ticks = ax.get_xticks()
    labels = [item.get_text() for item in ax.get_xticklabels()]
    
    # Wrap the labels
    wrapped_labels = [wrap_labels(label) for label in labels]
    
    # Set the ticks and labels explicitly
    ax.set_xticks(ticks)
    ax.set_xticklabels(wrapped_labels)
    
    # # Add value labels on the bars
    # for container in bars.containers:
    #     bars.bar_label(container, fmt='%.1f%%', padding=3, fontsize=10)
    
    # Customize appearance
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)  # Put grid behind bars
    
    # Move legend below the plot with multiple rows if needed
    n_models = len(performance_df['method'].unique())
    ncols = min(3, n_models)  # Maximum 3 columns in legend
    
    ax.legend(
        title='Model',
        bbox_to_anchor=(0.5, -0.15),  # Adjust vertical position
        loc='upper center',
        ncol=ncols,
        fontsize=11,
        title_fontsize=12
    )
    
    # Adjust layout to make room for legend at bottom
    plt.subplots_adjust(bottom=0.25)  # Increased bottom margin
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_path, f'performance_by_{exp_name}.png'), dpi=300)

def create_fake_data():
    """Create sample data for all metrics including sublevels"""
    methods = {
        'Open Source': ['OS-1', 'OS-2', 'OS-3', 'OS-4'],
        'Closed Source': ['CS-1', 'CS-2', 'CS-3'],
        'Specialized': ['S-1', 'S-2', 'S-3']
    }
    
    blooms_levels = {
        'Analyze (4)': {'base': 80, 'decay': 3},
        'Evaluate (5)': {'base': 75, 'decay': 3},
        'Create (6)': {'base': 70, 'decay': 3}
    }
    
    question_types = {
        'Expert Visual Understanding': {'base': 85, 'decay': 3},
        'Hypothesis Generation': {'base': 83, 'decay': 3},
        'Experimental Proposal': {'base': 87, 'decay': 3}
    }
    
    sublevels = {
        '1.1': {'base': 85, 'decay': 3},
        '1.2': {'base': 83, 'decay': 3},
        '1.3': {'base': 84, 'decay': 3},
        '2.1': {'base': 82, 'decay': 3},
        '2.2': {'base': 80, 'decay': 3},
        '3.1': {'base': 78, 'decay': 3},
        '3.2': {'base': 76, 'decay': 3}
    }
    
    data = []
    
    def get_performance(base, decay, method_index):
        return base - (decay * method_index) + (2 if 'CS' in method else 5 if 'S' in method else 0)
    
    for type_name, method_list in methods.items():
        for method_index, method in enumerate(method_list):
            # Add Bloom's levels data
            for level_name, level_info in blooms_levels.items():
                data.append({
                    'method': method,
                    'type': type_name,
                    'metric_type': 'blooms_level',
                    'metric': level_name,
                    'performance': get_performance(level_info['base'], level_info['decay'], method_index)
                })
            
            # Add question types data
            for question_name, question_info in question_types.items():
                data.append({
                    'method': method,
                    'type': type_name,
                    'metric_type': 'question_type',
                    'metric': question_name,
                    'performance': get_performance(question_info['base'], question_info['decay'], method_index)
                })
            
            # Add sublevel data
            for sublevel_name, sublevel_info in sublevels.items():
                data.append({
                    'method': method,
                    'type': type_name,
                    'metric_type': 'sublevel',
                    'metric': sublevel_name,
                    'performance': get_performance(sublevel_info['base'], sublevel_info['decay'], method_index)
                })
    
    return pd.DataFrame(data)

def calculate_model_performance(df, tag):
    """
    Calculate fractional performance by model_name using the 'correct' column.
    Returns one row per model.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing 'model_name' and 'correct' columns
    
    Returns:
    pandas.DataFrame: Summary DataFrame with model_name and performance columns
    """
    # Calculate performance as fraction of correct responses per model
    performance_df = df.groupby(['method', tag]).agg({
        'correct': 'mean',
        'color': 'first'  # Keep the first color for each model
    }).reset_index()
    
    
    # Rename the column for clarity
    performance_df = performance_df.rename(columns={'correct': 'performance'})
    
    return performance_df

def compile_pred_data():
    def clean_pred_df(path, method_name="", type_name=""):
        df = pd.read_csv(path)
        df = df[['key_question', 'correct']]
        df['method'] = method_name
        df['model_type'] = type_name
        return df
    
    base_path = '/pasteur/data/microchat/dataset_versions/nov11/model_results'
    tag_path = '/pasteur/data/microchat/dataset_versions/nov11/tagging_results_2/tagged_nov11_2_bio_blooms.csv'

    model_info = {
        # 'gpt-4o': {'path': os.path.join(base_path, 'eval_gpt-4o-2024-08-06_stage2.csv'), 'type': 'Closed source'},
        'gpt-4-turbo': {'path': os.path.join(base_path, 'eval_gpt-4-turbo-2024-04-09_stage2_prompt0.csv'), 'type': 'Closed source'},
        # 'claude3.5-sonnet': {'path': os.path.join(base_path, 'eval_anthropicclaude-35-sonnet_stage2.csv'), 'type': 'Closed source'},
        'claude3.5-opus': {'path': os.path.join(base_path, 'eval_anthropicclaude-3-opus_stage2_prompt0.csv'), 'type': 'Closed source'},
        'gemini 1.5 pro': {'path': os.path.join(base_path, 'eval_googlegemini-pro-15_stage2.csv'), 'type': 'Closed source'},
        'Qwen2-VL-72B': {'path': os.path.join(base_path, 'eval_QwenQwen2-VL-72B-Instruct_stage2.csv'), 'type': 'Open source'},
        # 'Qwen2-VL-7B': {'path': os.path.join(base_path, 'eval_QwenQwen2-VL-7B-Instruct_stage2_prompt0.csv'), 'type': 'Open source'},
        'Llama 3.2 90B': {'path': os.path.join(base_path, 'eval_meta-llamallama-32-90b-vision-instruct_stage2_prompt0.csv'), 'type': 'Open source'},
        # 'Llama 3.2 11B': {'path': os.path.join(base_path, 'eval_meta-llamallama-32-11b-vision-instruct_stage2_prompt0.csv'), 'type': 'Open source'},
        'Pixtral 12B': {'path': os.path.join(base_path, 'eval_mistralaipixtral-12b_stage2_prompt0.csv'), 'type': 'Open source'},
        'LlaVA-Med-7B': {'path': os.path.join(base_path, 'microsoft-llava-med-v1.5-mistral-7b (1).csv'), 'type': 'Specialized'},
    }
    # colors = sns.color_palette("muted", n_colors=len(model_info))
    # colors = [mcolors.rgb2hex(c) for c in colors]
    # colors = ['#2c7bb6', '#539cc6', '#7cbdd6', '#a5dee6', '#e66101', '#ec8843', '#f1af85', '#5e3c99', '#806bb3', '#a39acc']
    colors = ['#2c7bb6', '#539cc6', '#a5dee6', '#e66101', '#ec8843', '#f1af85', '#5e3c99', '#806bb3', '#a39acc']
    method_order = list(model_info.keys())
  
    all_df = pd.DataFrame()
    for idx, (method_name, value_dict) in enumerate(model_info.items()):
        method_df = clean_pred_df(value_dict['path'], method_name=method_name, type_name=value_dict['type'])
        method_df['color'] = colors[idx]
        all_df = pd.concat([all_df, method_df])
    tag_df = pd.read_csv(tag_path)
    #update blooms to make question type 6 --> 5
    tag_df['blooms_level'] = tag_df['blooms_level'].apply(lambda x: 5 if x == 6 else x)
    tag_df['image_counts'] = tag_df['image_counts'].apply(lambda x: '3+' if x >= 3 else str(x))

    return all_df, tag_df, method_order

# Example usage with fake data
if __name__ == "__main__":
    # # Create the fake data
    # df = create_fake_data()
    save_path = '/pasteur/data/microchat/dataset_versions/nov11/model_results/figs'
    os.makedirs(save_path, exist_ok=True)
    all_df, tag_df, method_order = compile_pred_data()
    tag_names = ['blooms_level', 'use_case', '_sub_use_case2', 'organism', 
                 'specimen', 'research_subject', 'modality', 'scale', 'image_counts']
    for tag in tag_names:
        exp_tag_df = tag_df[[tag, 'key_question']]
        exp_df = pd.merge(all_df, exp_tag_df, on='key_question')
        perf_df = calculate_model_performance(exp_df, tag)
        plot_model_performance(perf_df, tag, save_path, method_order)
    # # Create plots
    # plot_method_performance(df, metric_type='blooms_level')
    
    # plot_method_performance(df, metric_type='question_type')

    # plot_method_performance(df, metric_type='sublevel', figsize=(16, 6))