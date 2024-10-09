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

import argparse

def preprocess_dfs(res_path, tag_path):
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
        cols = ['key_question', 'key_image', 'question_number', 'gpt_response', 'gpt_prediction',
                'is_correct', 'correct_length', 'mean_distractor_length']
        df = df[cols]
        return df
    
    def read_tagged(tag_path):
        df = pd.read_csv(tag_path)
        return df
    
    print('Loading response file from ', res_path)
    print('Loading tagged file from ', tag_path)
    # read tagged and response files
    res_df = read_response(res_path)
    tag_df = read_tagged(tag_path)
    # merge files
    tag_df = pd.merge(res_df, tag_df, on=['key_question', 'key_image', 'question_number'])
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

def analyze_tags(args, tag_df):
    os.makedirs(args.save_dir, exist_ok=True)
    # create histograms and calculate statistics
    def hist_col_by_col(tag_df, col_name, second_col_name='is_correct'):
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

    tag_df['correct_is_longer'] = tag_df['correct_length'] > tag_df['mean_distractor_length']
    # for each type of question make a histogram of correct/incorrect answers
    tag_cols = ['_use_case', '_sub_use_case', '_image_scale','_image_modality']
    second_cols = ['is_correct', 'correct_is_longer']
    for col in tag_cols:
        for second_col in second_cols:
            # print(f'Creating histogram for {second_col} by {col}')
            # hist_col_by_col(tag_df, col, second_col)
            print(f'Creating pie chart for {second_col} by {col}')
            pie_col_by_col(tag_df, col, second_col)
    # create pdf with images
    png_to_pdf(args.save_dir, os.path.join(args.save_dir, 'tag_analysis_report.pdf'))

def main(args):
    tag_df = preprocess_dfs(args.res_path, args.tag_path)
    analyze_tags(args, tag_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--res_path', type=str, default='../benchmark/data/formdata_0/question_strategy_0/df_questions_key_choices_2_evalclosed_gpt-4o-2024-08-06.csv')
    parser.add_argument('--tag_path', type=str, default='results/20240925_llm_tagging/df_choices_with_llm_preds.csv')
    parser.add_argument('--save_dir', type=str, default='results/20241008_analyze_tags')
    args = parser.parse_args()
    main(args)