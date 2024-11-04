import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_method_performance(df, column_name, types, method_col='method', perf_col='performance', 
                          title=None, figsize=(12, 8), ylim=(60, 100)):
    """
    Create a bar plot of method performance across different types of a given column.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the data
    column_name : str
        Name of the column containing the types to plot (e.g., 'question', 'blooms_level')
    types : list
        List of values to plot from the column (e.g., ['Expert Visual Understanding', ...])
    method_col : str, optional (default='method')
        Name of the column containing method names
    perf_col : str, optional (default='performance')
        Name of the column containing performance values
    title : str, optional
        Plot title
    figsize : tuple, optional (default=(12, 8))
        Figure size in inches
    ylim : tuple, optional (default=(60, 100))
        Y-axis limits
    
    Returns:
    --------
    fig : matplotlib Figure
        The generated figure
    """
    # Set the style
    plt.style.use('seaborn')
    
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
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get unique methods
    methods = df[method_col].unique()
    n_methods = len(methods)
    
    # Calculate bar positions and width
    bar_width = 0.8 / n_methods
    
    # Plot bars for each method
    for i, method in enumerate(methods):
        # Get data for this method
        method_data = []
        for type_val in types:
            val = df[(df[method_col] == method) & 
                    (df[column_name] == type_val)][perf_col].values
            method_data.append(val[0] if len(val) > 0 else 0)
        
        # Calculate bar positions
        positions = range(len(types))
        bar_positions = [p - 0.4 + bar_width * (i + 0.5) for p in positions]
        
        # Plot bars
        ax.bar(bar_positions, 
               method_data, 
               width=bar_width,
               label=method,
               color=colors.get(method, '#333333'))  # default color if method not in colors
    
    # Customize the plot
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels(types, wrap=True)
    
    # Set axis labels and title
    ax.set_ylabel('Performance')
    if title:
        ax.set_title(title)
    
    # Set y-axis limits
    ax.set_ylim(ylim)
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    plt.savefig(f'method_performance_by_{column_name}.png')

# Example usage for question types:
if __name__ == "__main__":
    # Create sample data
    data = []
    
    # Question types
    question_types = [
        'Expert Visual\nUnderstanding',
        'Hypothesis\nGeneration',
        'Experimental\nProposal'
    ]
    
    # Performance values
    performances = {
        'Expert Visual\nUnderstanding': {
            'OS-1': 85, 'OS-2': 82, 'OS-3': 78, 'OS-4': 75,
            'CS-1': 88, 'CS-2': 85, 'CS-3': 82,
            'S-1': 90, 'S-2': 87, 'S-3': 84
        },
        'Hypothesis\nGeneration': {
            'OS-1': 83, 'OS-2': 80, 'OS-3': 76, 'OS-4': 73,
            'CS-1': 86, 'CS-2': 83, 'CS-3': 80,
            'S-1': 88, 'S-2': 85, 'S-3': 82
        },
        'Experimental\nProposal': {
            'OS-1': 87, 'OS-2': 84, 'OS-3': 80, 'OS-4': 77,
            'CS-1': 90, 'CS-2': 87, 'CS-3': 84,
            'S-1': 92, 'S-2': 89, 'S-3': 86
        }
    }
    
    # Create DataFrame
    for question in question_types:
        for method, score in performances[question].items():
            method_type = 'Open Source' if method.startswith('OS') else \
                         'Closed Source' if method.startswith('CS') else 'Specialized'
            data.append({
                'method': method,
                'type': method_type,
                'question': question,
                'performance': score
            })
    
    df = pd.DataFrame(data)
    
    # Create and show the plot for question types
    fig1 = plot_method_performance(
        df,
        column_name='question',
        types=question_types,
        title='Performance by Question Type'
    )
    
    # Example for Bloom's levels
    blooms_types = ['Analyze (4)', 'Evaluate (5)', 'Create (6)']
    # You would normally read this from your data
    # Here just showing how to use the same function for different types
    plot_method_performance(
        df,  # You would use your Bloom's data here
        column_name='blooms_level',
        types=blooms_types,
        title="Performance by Bloom's Level"
    )


# import React from 'react';
# import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

# const QuestionViz = () => {
#   const data = [
#     {
#       type: "Expert Visual\nUnderstanding",
#       // Open Source
#       "OS-1": 85,
#       "OS-2": 82,
#       "OS-3": 78,
#       "OS-4": 75,
#       // Closed Source
#       "CS-1": 88,
#       "CS-2": 85,
#       "CS-3": 82,
#       // Specialized
#       "S-1": 90,
#       "S-2": 87,
#       "S-3": 84
#     },
#     {
#       type: "Hypothesis\nGeneration",
#       "OS-1": 83,
#       "OS-2": 80,
#       "OS-3": 76,
#       "OS-4": 73,
#       "CS-1": 86,
#       "CS-2": 83,
#       "CS-3": 80,
#       "S-1": 88,
#       "S-2": 85,
#       "S-3": 82
#     },
#     {
#       type: "Experimental\nProposal",
#       "OS-1": 87,
#       "OS-2": 84,
#       "OS-3": 80,
#       "OS-4": 77,
#       "CS-1": 90,
#       "CS-2": 87,
#       "CS-3": 84,
#       "S-1": 92,
#       "S-2": 89,
#       "S-3": 86
#     }
#   ];

#   const colors = {
#     // Open Source - Blues
#     'OS-1': '#2c7bb6',
#     'OS-2': '#539cc6',
#     'OS-3': '#7cbdd6',
#     'OS-4': '#a5dee6',
#     // Closed Source - Oranges
#     'CS-1': '#e66101',
#     'CS-2': '#ec8843',
#     'CS-3': '#f1af85',
#     // Specialized - Purples
#     'S-1': '#5e3c99',
#     'S-2': '#806bb3',
#     'S-3': '#a39acc'
#   };

#   const CustomTooltip = ({ active, payload, label }) => {
#     if (active && payload && payload.length) {
#       return (
#         <div className="bg-white p-2 border border-gray-200 rounded shadow-sm">
#           <p className="font-medium">{label.replace('\n', ' ')}</p>
#           {payload.map((entry, index) => (
#             <p key={index} style={{color: entry.color}}>
#               {entry.name}: {entry.value}
#             </p>
#           ))}
#         </div>
#       );
#     }
#     return null;
#   };

#   return (
#     <div style={{ width: '100%', height: 500 }}>
#       <ResponsiveContainer>
#         <BarChart
#           data={data}
#           margin={{ top: 20, right: 30, left: 60, bottom: 20 }}
#         >
#           <CartesianGrid strokeDasharray="3 3" />
#           <XAxis 
#             dataKey="type" 
#             height={60}
#             interval={0}
#             tick={{ width: 100, wordBreak: 'break-word' }}
#           />
#           <YAxis 
#             domain={[60, 100]}
#             label={{ value: 'Performance', angle: -90, position: 'insideLeft', offset: 0 }}
#           />
#           <Tooltip content={<CustomTooltip />} />
#           <Legend />
          
#           {/* Open Source Methods */}
#           <Bar dataKey="OS-1" fill={colors['OS-1']} name="Open Source 1" />
#           <Bar dataKey="OS-2" fill={colors['OS-2']} name="Open Source 2" />
#           <Bar dataKey="OS-3" fill={colors['OS-3']} name="Open Source 3" />
#           <Bar dataKey="OS-4" fill={colors['OS-4']} name="Open Source 4" />
          
#           {/* Closed Source Methods */}
#           <Bar dataKey="CS-1" fill={colors['CS-1']} name="Closed Source 1" />
#           <Bar dataKey="CS-2" fill={colors['CS-2']} name="Closed Source 2" />
#           <Bar dataKey="CS-3" fill={colors['CS-3']} name="Closed Source 3" />
          
#           {/* Specialized Methods */}
#           <Bar dataKey="S-1" fill={colors['S-1']} name="Specialized 1" />
#           <Bar dataKey="S-2" fill={colors['S-2']} name="Specialized 2" />
#           <Bar dataKey="S-3" fill={colors['S-3']} name="Specialized 3" />
#         </BarChart>
#       </ResponsiveContainer>
#     </div>
#   );
# };

# export default QuestionViz;