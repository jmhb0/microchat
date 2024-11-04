import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_method_performance(df, metric_type, figsize=(12, 8), ylim=(60, 100)):
    """
    Create a bar plot of method performance using seaborn.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the data in the flat structure with columns:
        'method', 'type', 'metric_type', 'metric', 'performance'
    metric_type : str
        Type of metric to plot ('blooms_level' or 'question_type')
    figsize : tuple, optional (default=(12, 8))
        Figure size in inches
    ylim : tuple, optional (default=(60, 100))
        Y-axis limits
    
    """
    # Filter data for the specified metric type
    plot_df = df[df['metric_type'] == metric_type].copy()
    
    # Create color palette
    colors = {
        # Open Source - Blues
        'OS-1': '#2c7bb6',
        'OS-2': '#539cc6',
        'OS-3': '#7cbdd6',
        'OS-4': '#a5dee6',
        # Closed Source - Oranges
        'CS-1': '#e66101',
        'CS-2': '#ec8843',
        'CS-3': '#f1af85',
        # Specialized - Purples
        'S-1': '#5e3c99',
        'S-2': '#806bb3',
        'S-3': '#a39acc'
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create the plot using seaborn
    sns.barplot(
        data=plot_df,
        x='metric',
        y='performance',
        hue='method',
        palette=colors,
        ax=ax
    )
    
    # Set title based on metric type
    title = "Performance by Bloom's Level" if metric_type == 'blooms_level' else 'Performance by Question Type'
    ax.set_title(title, fontsize=14, pad=20)
    
    # Customize axes
    ax.set_ylabel('Performance', fontsize=12)
    ax.set_xlabel('')
    ax.set_ylim(ylim)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Adjust legend
    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0,
        title='Method'
    )
    
    # Adjust layout
    plt.tight_layout()
    
    plt.savefig(f'{metric_type}_performance.png', dpi=300)

# Example usage with fake data
if __name__ == "__main__":
    # Create fake data
    def create_fake_data():
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
        
        return pd.DataFrame(data)

    # Create the fake data
    df = create_fake_data()
    
    # Create plots
    plot_method_performance(df, metric_type='blooms_level')
    
    plot_method_performance(df, metric_type='question_type')
