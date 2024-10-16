import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipdb
import math
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch

import argparse

def preprocess_dfs(res_path, tag_path, baseline_path):
    def read_response(res_path):
        df = pd.read_csv(res_path)
        # parse answer for a correct/incorrect column
        gt_idx = df['choices'].apply(lambda x: ast.literal_eval(x)['correct_index']).to_numpy()
        choices = np.vstack(df['choices'].apply(lambda x: ast.literal_eval(x)['choices']))
        df['is_correct'] = (gt_idx == df['gpt_prediction'])
        # calculate length of distractor and correct choices
        df['correct_length'] = np.char.str_len(choices[np.arange(len(choices)), gt_idx])
        mask = np.ones_like(choices, dtype=bool)
        mask[np.arange(len(choices)), gt_idx] = False
        df['mean_distractor_length'] = np.mean(np.char.str_len(choices[mask].reshape(-1, 5)), axis=1)
        # keep only basic columns
        cols = ['key_question', 'key_image', 'question_number', 'fname_images',
                'gpt_response', 'gpt_prediction', 'is_correct', 
                'correct_length', 'mean_distractor_length']
        df = df[cols]
        return df
    
    def read_tagged(tag_path):
        df = pd.read_csv(tag_path)
        return df
    
    def read_baseline(base_path):
        base_df = read_response(base_path)
        base_df.rename(columns={'gpt_response': 'base_gpt_response',
                                'gpt_prediction': 'base_gpt_prediction',
                                'is_correct': 'base_is_correct',
                                'correct_length': 'base_correct_length',
                                'mean_distractor_length': 'base_mean_distractor_length',
                                'fname_images': 'base_fname_images'},
                                inplace=True)
        return base_df
    
    print('Loading response file from ', res_path)
    print('Loading tagged file from ', tag_path)
    print('Loading baseline file from ', baseline_path)

    # read tagged and response files
    res_df = read_response(res_path)
    tag_df = read_tagged(tag_path)
    base_df = read_baseline(baseline_path)
    # merge files
    tag_df = pd.merge(res_df, tag_df, on=['key_question', 'key_image', 'question_number'])
    tag_df = pd.merge(tag_df, base_df, on=['key_question', 'key_image', 'question_number'])
    # # calculate delta with baseline
    # tag_df['delta_correct_length'] = tag_df['correct_length'] - tag_df['base_correct_length']
    # tag_df['delta_mean_distractor_length'] = tag_df['mean_distractor_length'] - tag_df['base_mean_distractor_length']
    # tag_df['flipped_is_correct'] = tag_df['base_is_correct'] - tag_df['is_correct'] # pos - neg = 1, neg - pos = -1, 0 = no change

    
    # tag_df['flipped_is_correct'] * tag_df['correct_is_longer'] # 1 became neg, -1 became pos, 0 no change or isn't longer
    return tag_df

def png_to_pdf(input_folder, output_pdf):
    # Get all PNG files in the input folder
    png_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.png')]
    png_files.sort()  # Sort the files to ensure consistent order

    # Create a new PDF file
    c = canvas.Canvas(output_pdf)

    for png_file in png_files:
        img_path = os.path.join(input_folder, png_file)
        img = Image.open(img_path)
        
        # Get image size
        width, height = img.size
        
        # Calculate scaling factor to fit the image on a PDF page
        pdf_width, pdf_height = 8.5 * inch, 11 * inch  # US Letter size
        scale = min(pdf_width / width, pdf_height / height)
        
        # Add image to the PDF
        c.setPageSize((width * scale, height * scale))
        c.drawImage(img_path, 0, 0, width=width*scale, height=height*scale)
        c.showPage()

    c.save()


def compare_length_results(args, df):
    # Identify questions that became shorter and those that remained longer
    became_shorter = df[(df['base_correct_length'] > df['base_mean_distractor_length']) & 
                        (df['correct_length'] <= df['mean_distractor_length'])]
    remained_longer = df[(df['base_correct_length'] > df['base_mean_distractor_length']) & 
                         (df['correct_length'] > df['mean_distractor_length'])]
    
    # Function to calculate counts for a given dataset
    def get_counts(data):
        baseline_correct = data['base_is_correct'].sum()
        baseline_incorrect = len(data) - baseline_correct
        current_correct = data['is_correct'].sum()
        current_incorrect = len(data) - current_correct
        return baseline_correct, baseline_incorrect, current_correct, current_incorrect

    # Get counts for both datasets
    shorter_counts = get_counts(became_shorter)
    longer_counts = get_counts(remained_longer)
    
    # Set up the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    conditions = ['Baseline', 'Current']
    x = range(len(conditions))
    width = 0.35
    
    # Function to create a single subplot
    def create_subplot(ax, counts, title):
        baseline_correct, baseline_incorrect, current_correct, current_incorrect = counts
        ax.bar([i - width/2 for i in x], [baseline_correct, current_correct], width, label='Correct', color='green', alpha=0.7)
        ax.bar([i + width/2 for i in x], [baseline_incorrect, current_incorrect], width, label='Incorrect', color='red', alpha=0.7)
        
        ax.set_ylabel('Number of Questions')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(conditions)
        ax.legend()
        
        # Add value labels on the bars
        for i, v in enumerate([baseline_correct, current_correct]):
            ax.text(i - width/2, v, str(v), ha='center', va='bottom')
        for i, v in enumerate([baseline_incorrect, current_incorrect]):
            ax.text(i + width/2, v, str(v), ha='center', va='bottom')
        
        # Add a text box with additional information
        total = sum(counts) // 2
        baseline_accuracy = baseline_correct / total
        current_accuracy = current_correct / total
        info_text = f'Total questions: {total}\n'
        info_text += f'Baseline accuracy: {baseline_accuracy:.2%}\n'
        info_text += f'Current accuracy: {current_accuracy:.2%}'
        
        ax.text(0.95, 0.95, info_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Create the two subplots
    create_subplot(ax1, shorter_counts, 'Questions that Became Shorter')
    create_subplot(ax2, longer_counts, 'Questions that Remained Longer')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'comparison_plot.png'))

def create_pdf_from_dataframes(output_filename, dataframes_and_conditions):
    def split_long_text(text, max_length=100):
        """Split long text into smaller chunks."""
        words = text.split()
        chunks = []
        current_chunk = []

        for word in words:
            if len(' '.join(current_chunk + [word])) <= max_length:
                current_chunk.append(word)
            else:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return '\n'.join(chunks)
    doc = SimpleDocTemplate(output_filename, pagesize=landscape(letter), topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()
    elements = []

    # Define colors for each condition
    condition_colors = {
        'stayed_correct': colors.green,
        'stayed_incorrect': colors.red,
        'correct_to_incorrect': colors.orange,
        'incorrect_to_correct': colors.blue,
    }

    for df, condition in dataframes_and_conditions:
        # Add a page break before each condition (except the first one)
        if elements:
            elements.append(PageBreak())

        # Determine the color for this condition
        condition_color = condition_colors[condition]

        # Create custom styles with color
        title_style = ParagraphStyle(
            'ConditionTitle',
            parent=styles['Heading1'],
            alignment=TA_CENTER,
            spaceAfter=12,
            textColor=condition_color
        )
        subtitle_style = ParagraphStyle(
            'ExampleTitle',
            parent=styles['Heading2'],
            alignment=TA_CENTER,
            spaceAfter=12,
            textColor=condition_color
        )

        # Add the condition title with the specific color
        title = Paragraph(condition, title_style)
        title.textColor = condition_color
        elements.append(title)

        # Select the columns to display
        columns_to_display = list(df.columns)

        # Create a page for each example
        for index, row in df.iterrows():
            if index > 0:
                elements.append(PageBreak())
            
            # Add example number as subtitle with the same color as the condition
            subtitle = Paragraph(f"Example {index + 1}", subtitle_style)
            subtitle.textColor = condition_color
            elements.append(subtitle)
            elements.append(Spacer(1, 12))

            # Create a list to hold the formatted data for the table
            table_data = []

            # Add header and data as separate rows
            for col in columns_to_display:
                header = Paragraph(split_long_text(col, 20), styles['Normal'])
                cell_content = split_long_text(str(row[col]), 100)
                cell = Paragraph(cell_content, ParagraphStyle('Normal', fontSize=8, leading=10))
                table_data.append([header, cell])

            # Create the table
            table = Table(table_data, colWidths=[2*inch, 6*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ]))

            elements.append(table)

    # Build the PDF
    doc.build(elements)



def plot_experiment_change_results(args, df, sample_qs=True, num_samples=3):
    def sample_qual_qs(df, cond, num_samples=3):
        this_df = df[cond]
        sample_df = this_df.sample(num_samples)
        sample_df = sample_df[['key_question', 'key_image', 'question_number',
                               'question_and_answer_and_context', 'choices', 'gpt_response',
                               '_image_modality', '_image_scale',
                               '_sub_use_case', '_question_has_answer']]
        return sample_df

    # Count total questions
    total_questions = len(df)
    
    # Count correct and incorrect for baseline and experiment
    baseline_correct = df['base_is_correct'].sum()
    baseline_incorrect = total_questions - baseline_correct
    experiment_correct = df['is_correct'].sum()
    experiment_incorrect = total_questions - experiment_correct
    
    # Find transitions
    stayed_correct = ((df['base_is_correct'] == True) & (df['is_correct'] == True))
    stayed_incorrect = ((df['base_is_correct'] == False) & (df['is_correct'] == False))
    correct_to_incorrect = ((df['base_is_correct'] == True) & (df['is_correct'] == False))
    incorrect_to_correct = ((df['base_is_correct'] == False) & (df['is_correct'] == True))

    if sample_qs:
        df_1 = sample_qual_qs(df, stayed_correct, num_samples=num_samples)
        df_2 = sample_qual_qs(df, stayed_incorrect, num_samples=num_samples)
        df_3 = sample_qual_qs(df, correct_to_incorrect, num_samples=num_samples)
        df_4 = sample_qual_qs(df, incorrect_to_correct, num_samples=num_samples)
        dataframes_and_conditions = [
            (df_1, 'stayed_correct'),
            (df_2, 'stayed_incorrect'),
            (df_3, 'correct_to_incorrect'),
            (df_4, 'incorrect_to_correct')]
        print('Creating PDF with sample questions')
        create_pdf_from_dataframes(os.path.join(args.save_dir, 'sample_questions.pdf'),
                                   dataframes_and_conditions)     
    
    # Count transitions
    stayed_correct = stayed_correct.sum()
    stayed_incorrect = stayed_incorrect.sum()
    correct_to_incorrect = correct_to_incorrect.sum()
    incorrect_to_correct = incorrect_to_correct.sum()
    
    # Set up the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Baseline vs Experiment
    conditions = ['Baseline', 'Experiment']
    correct_counts = [baseline_correct, experiment_correct]
    incorrect_counts = [baseline_incorrect, experiment_incorrect]
    
    x = np.arange(len(conditions))
    width = 0.35
    
    ax1.bar(x - width/2, correct_counts, width, label='Correct', color='green', alpha=0.7)
    ax1.bar(x + width/2, incorrect_counts, width, label='Incorrect', color='red', alpha=0.7)
    
    ax1.set_ylabel('Number of Questions')
    ax1.set_title('Baseline vs Experiment Results')
    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions)
    ax1.legend()
    
    # Add value labels
    for i, v in enumerate(correct_counts):
        ax1.text(i - width/2, v, str(v), ha='center', va='bottom')
    for i, v in enumerate(incorrect_counts):
        ax1.text(i + width/2, v, str(v), ha='center', va='bottom')
    
    # Plot 2: Transitions
    transitions = ['Stayed\nCorrect', 'Stayed\nIncorrect', 'Correct to\nIncorrect', 'Incorrect to\nCorrect']
    counts = [stayed_correct, stayed_incorrect, correct_to_incorrect, incorrect_to_correct]
    colors = ['green', 'red', 'orange', 'blue']
    
    ax2.bar(transitions, counts, color=colors, alpha=0.7)
    ax2.set_ylabel('Number of Questions')
    ax2.set_title('Question Transitions')
    
    # Add value labels
    for i, v in enumerate(counts):
        ax2.text(i, v, str(v), ha='center', va='bottom')
    
    # Add text boxes with additional information
    info_text1 = f'Total questions: {total_questions}\n'
    info_text1 += f'Baseline accuracy: {baseline_correct/total_questions:.2%}\n'
    info_text1 += f'Experiment accuracy: {experiment_correct/total_questions:.2%}'
    
    ax1.text(0.95, 0.95, info_text1, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    info_text2 = f'Improved: {incorrect_to_correct}\n'
    info_text2 += f'Worsened: {correct_to_incorrect}\n'
    info_text2 += f'Net change: {incorrect_to_correct - correct_to_incorrect}'
    
    ax2.text(0.95, 0.95, info_text2, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'change_correct_plot.png'))

        

def analyze_tags(args, tag_df):
    os.makedirs(args.save_dir, exist_ok=True)
    # create histograms and calculate statistics
    def bar_col_by_col(tag_df, col_name, second_col_name='is_correct'):
        # Group by the specified column and 'is_correct', then count occurrences
        hist_df = tag_df[[second_col_name, col_name]].groupby([col_name, second_col_name]).size().unstack(fill_value=0)
        
        # Plot a bar chart instead of a histogram
        hist_df.plot(kind='bar', stacked=True, figsize=(10, 6))

        # Add title and labels
        plt.title(f'{second_col_name} Distribution by {col_name}')
        plt.xlabel(col_name)
        plt.ylabel('Count')

        # Save the plot to the specified directory
        plt.savefig(os.path.join(args.save_dir, f'{second_col_name}_by_{col_name}_hist.png'))
    
    def pie_col_by_col(tag_df, col_name, second_col_name='is_correct'):
        
        hist_df = tag_df[[second_col_name, col_name]].groupby([col_name, second_col_name]).size().unstack(fill_value=0)
        num_categories = len(hist_df.index)
    
        # Calculate the number of rows and columns for the subplots
        num_cols = math.ceil(math.sqrt(num_categories))
        num_rows = math.ceil(num_categories / num_cols)
        
        fig = plt.figure(figsize=(4*num_cols, 4*num_rows))
        
        for i, category in enumerate(hist_df.index, 1):
            ax = fig.add_subplot(num_rows, num_cols, i)
            wedges, texts, autotexts = ax.pie(hist_df.loc[category], autopct='%1.1f%%', textprops=dict(color="w"))
            ax.set_title(f'{col_name}: {category}')
    
        # Add a legend
        fig.legend(wedges, hist_df.columns, title=second_col_name, loc="center left", bbox_to_anchor=(1, 0.5))
        
        plt.suptitle(f'{second_col_name} Distribution by {col_name}', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(args.save_dir, f'{second_col_name}_by_{col_name}_pie.png'), bbox_inches='tight')
        plt.close()

    def plot_hist(tag_df, col_name, col_name2=None, bins=30):
        save_name = os.path.join(args.save_dir, f'{col_name}_hist.png')
        plt.figure(figsize=(10, 6))
        # Plot the first histogram
        plt.hist(tag_df[col_name], bins=bins, alpha=0.5, label=col_name)
        if col_name2 is not None:
            # Plot the second histogram
            plt.hist(tag_df[col_name2], bins=bins, alpha=0.5, label=col_name2)
            save_name = os.path.join(args.save_dir, f'{col_name}_{col_name2}_hist.png')
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.title(f'Histogram')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_name)


    tag_df['correct_is_longer'] = tag_df['correct_length'] > tag_df['mean_distractor_length']
    # for each type of question make a histogram of correct/incorrect answers
    tag_cols = ['_use_case', '_sub_use_case', '_image_scale','_image_modality']
    second_cols = ['is_correct', 'correct_is_longer']
    for col in tag_cols:
        for second_col in second_cols:
            # print(f'Creating histogram for {second_col} by {col}')
            # hist_col_by_col(tag_df, col, second_col)
            pie_col_by_col(tag_df, col, second_col)
    plot_hist(tag_df, 'correct_length', 'mean_distractor_length')



def main(args):
    tag_df = preprocess_dfs(args.res_path, args.tag_path, args.base_path)
    analyze_tags(args, tag_df)
    compare_length_results(args, tag_df)
    plot_experiment_change_results(args, tag_df)
    # create pdf with images
    print('Creating PDF with all images')
    png_to_pdf(args.save_dir, os.path.join(args.save_dir, 'analysis_report.pdf'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--base_path', type=str, default='benchmark/data/formdata_0/question_strategy_0/df_questions_key_choices_2_evalclosed_gpt-4o-2024-08-06.csv')
    parser.add_argument('--base_path', type=str, default='benchmark/data/formdata_0/question_strategy_0/df_questions_key_choices_3_evalclosed_gpt-4o-2024-08-06.csv')
    parser.add_argument('--res_path', type=str, default='benchmark/data/formdata_0/question_strategy_0/df_questions_key_choices_4_evalclosed_gpt-4o-2024-08-06.csv') # for error distractors, base should be length
    # parser.add_argument('--res_path', type=str, default='benchmark/data/formdata_0/question_strategy_0/df_questions_key_choices_3_evalclosed_gpt-4o-2024-08-06.csv') # for length
    # parser.add_argument('--res_path', type=str, default='benchmark/data/formdata_0/question_strategy_0/df_questions_key_choices_2_evalclosed_blind_gpt-4o-2024-08-06.csv') # for blind
    # parser.add_argument('--res_path', type=str, default='benchmark/data/formdata_0/question_strategy_3/df_questions_key_choices_3_evalclosed_gpt-4o-2024-08-06.csv') # for no context
    parser.add_argument('--tag_path', type=str, default='analysis_scripts/results/20240925_llm_tagging/df_choices_with_llm_preds.csv')
    parser.add_argument('--save_dir', type=str, default='analysis_scripts/results/20241015_errormode_version')
    args = parser.parse_args()
    main(args)