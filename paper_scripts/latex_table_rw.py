import pandas as pd
import numpy as np

def format_column_title(title):
    """Format column title for LaTeX, splitting on spaces and special characters"""
    # Replace underscores with spaces
    title = title.replace("_", " ")
    # Split title on spaces and special characters
    words = title.split()
    
    # If title is short, return as is
    if len(words) <= 2 and len(title) <= 12:
        return title.title()
    
    # For longer titles, split into roughly equal parts
    mid = len(words) // 2
    first_half = " ".join(words[:mid])
    second_half = " ".join(words[mid:])
    
    return f"\\shortstack{{{first_half} \\\\ {second_half}}}"

def csv_to_latex_table(csv_path, text_width="2.5cm"):
    """
    Convert CSV file to a LaTeX table format with text wrapping and checkmarks.
    
    Parameters:
    csv_path (str): Path to the CSV file
    text_width (str): Width for text columns in LaTeX units
    
    Returns:
    str: Formatted LaTeX table
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Define the LaTeX preamble with required packages and commands
    latex_preamble = """% Required packages:
% \\usepackage{array}
% \\usepackage{pifont}
% \\usepackage{amssymb}
% \\usepackage{booktabs}
% Command for crossmark:
% \\newcommand{\\crossmark}{\\ding{53}}
% Command for checkmark (optional, but for consistency):
% \\newcommand{\\checkmark}{\\ding{51}}

"""
    
    # Start the LaTeX table
    latex_code = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Benchmark Comparison}",
        "\\label{tab:benchmark-comparison}",
        "\\resizebox{\\textwidth}{!}{%",  # Make table fit page width
    ]
    
    # Determine which columns are primarily numeric
    numeric_cols = {}
    for col in df.columns:
        # Convert to numeric, counting how many succeed
        numeric_vals = pd.to_numeric(df[col], errors='coerce')
        # A column is numeric if >50% of its non-null values are numbers
        # and it contains numbers >1 (to exclude binary 0/1 columns)
        valid_numeric = numeric_vals.notna()
        if valid_numeric.mean() > 0.5 and (numeric_vals > 1).any():
            numeric_cols[col] = True
        else:
            numeric_cols[col] = False
    
    # Create column specifications with centering
    col_specs = []
    for col in df.columns:
        if numeric_cols[col]:
            col_specs.append('c')  # Center numeric columns
        else:
            # >{\centering\arraybackslash} centers the text within p columns
            col_specs.append(f'>{{\\centering\\arraybackslash}}p{{{text_width}}}')
    
    latex_code.append("\\begin{tabular}{" + " ".join(col_specs) + "}")
    latex_code.append("\\toprule")  # Thick top line
    
    # Add headers with wrapping for long titles
    headers = " & ".join([
        format_column_title(col)
        for col in df.columns
    ])
    latex_code.append(headers + " \\\\")
    latex_code.append("\\midrule")  # Line after headers
    
    # Add data rows
    for i, (_, row) in enumerate(df.iterrows()):
        # Convert row values to strings and replace any problematic characters
        row_values = []
        for col, val in zip(df.columns, row):
            if pd.isna(val):
                val = "-"
            elif not numeric_cols[col]:  # Only convert to checkmarks in non-numeric columns
                # Convert to string and strip whitespace
                val_str = str(val).strip()
                # Replace 1/0 with checkmarks/crossmarks
                if val_str == "1":
                    val = "\\checkmark"
                elif val_str == "0":
                    val = "\\crossmark"
            elif isinstance(val, (int, float)):
                val = f"{val:,}"  # Add thousands separator
            
            # Clean and format the text
            val = str(val).replace("%", "\\%")  # Escape percentage signs
            val = val.replace("&", "\\&")  # Escape ampersands
            val = val.replace("+", "\\+")  # Escape plus signs
            val = val.strip()
            
            row_values.append(val)
        
        latex_code.append(" & ".join(row_values) + " \\\\")
        
    # Add thick bottom line
    latex_code.append("\\bottomrule")
    
    # Close the table
    latex_code.extend([
        "\\end{tabular}",
        "}",  # Close resizebox
        "\\end{table}"
    ])
    
    return latex_preamble + "\n".join(latex_code)

# # Example usage with the provided CSV data
# csv_data = """Benchmark,difficulty level,reasoning classes,domain,source,curation,taxonomy,open-source,metadata,modalities,multi-image comp.,"size (close, multimodal, microscopy)","Size (closed, multimodal)",total
# MicroVQA,research,1,microscopy,original,M + A,1,1,1,,1,1050,1050,1050
# PathVQA,graduate,0,pathology,texbooks,template qs,0,1,0,2,0,16334,16334,32799
# OmnimedVQA,graduate,0,medical,existing datasets,template qs,0,1,"types, modality",12,0,4196,127995,127995
# Microbench,graduate,0,microscopy,existing datasets,existing + M + template qs,1,1,"modality, type",8,0,17235,17235,17235
# LabBench,graduate,0,biology,webscraped,template qa + M,0,1,0,1,1,0,181,2400
# SciEval,college,1,Science,"web QA, existing datasets",existing + A,1,1,ability,3,0,0,is it multi-modal?,16522
# MMMU,college,0,general,"textbooks, web QA, original",existing + M + A,1,1,type ,6,1,0,11264,11550
# MMMU Pro,college,0,general,existing dataset,existing + M + A,1,1,type,?,1,0,1730,3460
# Science QA,school,0,science,existing datasets,existing,1,1,"type, domain","?, 21 domains",0,0,16864,21208"""

csv_path = "table_rw_clean_table.csv"

latex_table = csv_to_latex_table(csv_path)
print(latex_table)