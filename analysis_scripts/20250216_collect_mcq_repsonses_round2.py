"""
python -m ipdb analysis_scripts/20250216_collect_mcq_repsonses_round2.py

# similar to the first round of form reviews with analysis_scripts/20241212_get_form_mcq_responses.py 
"""
import ipdb
from pathlib import Path
import pandas as pd
import numpy as np

# format: the csv for the data, then the csv for the form response

# november 5
# [4, 12, 13, 17, 18]  # has the data
links_feb7 = {
    # Not done
    # p0 sarina hasan (set 4) missing
    # 0: {
    #     "data": "https://docs.google.com/spreadsheets/d/e/2PACX-1vTWF-ZJ-xvTobsfX_noz-8LRaqbQTxbOOOEphBo22UsesMB1bFu4YPQ1H_pYebvA4UsaMGOVUmChERz/pub?gid=1934510244&single=true&output=csv",
    #     # missing 
    #     "response": "https://docs.google.com/spreadsheets/d/e/2PACX-1vQEJFccCva_MCywBYSDwYSBN5Yj9G2e7FMkXxZZR66xqOHYUCfoB5Xtf-wfR3wTS934Azf3fm18TscG/pub?gid=1033625288&single=true&output=csv",
    #     "response": "",
    # },
    # Not done
    # p6 zach coman
    # WARNING - GET BOTH FORMS 
    # 6 : {
    #     "data": "https://docs.google.com/spreadsheets/d/e/2PACX-1vSrte09KCov4ZstNL8t0xn9CFGUEZP0bVdtrQCXwWQ5CRZ9xxn4_Z_VJoHp0JBSOxhQfq70o87qPst-/pub?gid=693340444&single=true&output=csv",
    #     "response": "",
    # },
    # p12 ridhi
    12: {
        "data": "https://docs.google.com/spreadsheets/d/e/2PACX-1vT_F_8H0dMzF6st1Fym19a1zpQJp7qjzF6Mg1gfroVmziaJdq6mBLzwpapKXIfVkTm4tli64wtpuVad/pub?gid=742450928&single=true&output=csv",
        "response": "https://docs.google.com/spreadsheets/d/e/2PACX-1vRhyRb6PA1tiQVG35mr3R4bRVLlAIemuazWZ_LoSiOUnJ3qWGAAIOJKqiyhbmtc8jdptpu-rUp3P034/pub?gid=914843595&single=true&output=csv",
    },
    # p14 Alexandra
    14: {
        "data": "https://docs.google.com/spreadsheets/d/e/2PACX-1vTQoKxKkhOdBGzmFEU1-crBY7F3UROkjQxO60ZESWOsknULdAb-sv3j9NGezb-PfB42PBJen4ellw2C/pub?gid=1924252430&single=true&output=csv",
        "response": "https://docs.google.com/spreadsheets/d/e/2PACX-1vR_NZI2kchDZxakTiRVzJAf3FRA4Iy7fwoQRAT2uJSck8loBdhQW3Z05wS1cTh3_b6eqFI1wlaeiR7j/pub?gid=543836520&single=true&output=csv",
    },
    # p15 Disha (set 1)
    15: {
        "data": "https://docs.google.com/spreadsheets/d/e/2PACX-1vQiyapacsmRFm4lA6_gURAKuk1jC-1x-E3EYpaDaEfP0KjH22NwPvVWHzVxMIgr8GCsMZL7m2qJda1h/pub?gid=404258516&single=true&output=csv",
        "response": "https://docs.google.com/spreadsheets/d/e/2PACX-1vRwRfu2_qMfObDGs-z90bfvMdIispuIbImA7yJ7E17coMqR0HbbhYHjOg67czER6TZcz4q8IUaGYTvk/pub?gid=1842606541&single=true&output=csv",
    },
    # p17 Alexandra (set 1)
    17: {
        "data": "https://docs.google.com/spreadsheets/d/e/2PACX-1vTWqzickJHECkZGhuAy_hBWXhsZlzGICNmax27S7crGUdGGsiLfDRVDNvXiY7mX698NLvXLWuBj8xn4/pub?gid=1921776394&single=true&output=csv",
        "response": "https://docs.google.com/spreadsheets/d/e/2PACX-1vRzysxMcdZi55WdZW2CYtSHF35uxVArpZnxFf_V7aTCW4tpnt1a8SwQvzLUc6qsj4l_7Zw1zX-AI9-i/pub?gid=1647501777&single=true&output=csv",
    },
    # p18 connor
    18: {
        "data": "https://docs.google.com/spreadsheets/d/e/2PACX-1vTgmWyR4tMmYGwTVi1is3L8Eqr7E6rSbPQOpGNhf5auARcMZwnCFZ45oGfiJ3w8Zv2pkAiHCLYWLELj/pub?gid=673538865&single=true&output=csv",
        "response": "https://docs.google.com/spreadsheets/d/e/2PACX-1vQrU2y-0yRX_FF0-VQXENJ4xBgE57vko4FC0C3JOmAn_AvmHw2VDSFMygz6MX6k6BWodUZ7vRWuRN84/pub?gid=885031496&single=true&output=csv",
    },
        
}

# november 6
links_feb7_next = {
    # p4 chad liu 
    4: {
        "data": "https://docs.google.com/spreadsheets/d/e/2PACX-1vR8GLzfSJNhBL2jmCsfagxJ2XCTllCHfhvsuyxrdWzpmgkkmi3RgbsOUo-BwCRqA9rUoaS6mqcImDjj/pub?gid=474136647&single=true&output=csv",
        "response": "https://docs.google.com/spreadsheets/d/e/2PACX-1vQYA_oaK8735ej-yGEvFHOwpQyZkMZYpg2uj_suS3CeTrQfinE3NWddKcn0yXlA9kDEAbP_NJJt157F/pub?gid=931649053&single=true&output=csv",
    },
    # p7 Jan 
    7: {
        "data": "https://docs.google.com/spreadsheets/d/e/2PACX-1vTZnhqMyUqWY4O8q8NEmCLX1xHUAHUg3C79cMjpTaIM7nzl2RLlfV4wGuQLZ-IBimg1eYdZbxlztZuQ/pub?gid=450751290&single=true&output=csv",
        "response": "https://docs.google.com/spreadsheets/d/e/2PACX-1vSt_urOCTeXemoFiR2wCAVEvOVmUGXOD0GMo4mICH5eewqrmk21P4xhCYzQpXtIachz9wPnA-CsphEK/pub?gid=422220296&single=true&output=csv",
    },
    # p8   Disha
    8: {
        "data": "https://docs.google.com/spreadsheets/d/e/2PACX-1vSWFJWdk2TR5SvFVcO1ZZG5ID0Y4pozHqyG_wYgiUHIGZUIQAHIEy6rfF7-D9iW_t6XAXCqQFbQZXi2/pub?gid=1539789166&single=true&output=csv",
        "response": "https://docs.google.com/spreadsheets/d/e/2PACX-1vQCsqSlb_C-GqgeqfZn_4z4Q-7oBVf5IWpAlME2ttLRVHEvRehzSrXofG03svx_0qaV8VwAM_QMTrDK/pub?gid=671968700&single=true&output=csv",
    },
    # p9 Jesus
    9: {
        "data": "https://docs.google.com/spreadsheets/d/e/2PACX-1vSML0oYQLnjO29EGoOnsE7GSsvTzSIRpti5pGhgVOr5dOrpNbyaIT9viym4i0zOKN6l7CsbjD7VLxBK/pub?gid=711560343&single=true&output=csv",
        "response": "https://docs.google.com/spreadsheets/d/e/2PACX-1vSXlTcWHVaCu6h47DmtxZKcwq01UrdrjNgLmVpcyad9hOrFIXUPTCdsk_8xG4wI2cqURw9yTlR-bMgY/pub?gid=1478386529&single=true&output=csv",
    },
    # p10 will  
    10: {
        "data": "https://docs.google.com/spreadsheets/d/e/2PACX-1vQ2MHp-XMu4dZlQ8wkHheoqD7BQmQXNgrh26Zd_mQIJh_RfgbLGkNNLTE3OKWNOzrBiprcG5CkBNWTu/pub?gid=921703189&single=true&output=csv",
        "response": "https://docs.google.com/spreadsheets/d/e/2PACX-1vTnJJWfgJKO8mctOE8wiVFXE3MFnWEhYXVifbwmMfSDI3eZs-srv0WlJl2zKe_l1LNuhsHCVYn-Psl7/pub?gid=300965382&single=true&output=csv",
    },
    # p13 malvika 
    13: {
        "data": "https://docs.google.com/spreadsheets/d/e/2PACX-1vQW-qf1hlELTRimtSidKfmkctLxVLoNhVFCd9uG2xJAn4WVnAit4qDMAlo9MCRpPH491p17c6SjbWNH/pub?gid=656254209&single=true&output=csv",
        "response": "https://docs.google.com/spreadsheets/d/e/2PACX-1vSXGVRMPXpAOp-59YH0oZBYFevC-zFQjspIAld-dGYnLdUkiSbuQHUnwGG_qOMkgKaYl0SdR3DnHKYS/pub?gid=2032756947&single=true&output=csv",
    },
}



from benchmark.build_raw_dataset.download_data import download_csv

DOWNLOAD = 0
# call: download_csv(url, output_path)
dir_results = Path(__file__).parent / "results" / Path(__file__).stem
dir_results.mkdir(exist_ok=True, parents=True)

skip_idxs_nov10 = [6, 9, 20]
skip_idxs_nov6 = [4, 12, 13, 17, 18]  # has the data

if DOWNLOAD:
    for idx, v in links_feb7.items():
        print("downloading idx ", idx)

        fname = dir_results / f"responses_{idx}_feb7.csv"
        if len(v['response']) > 0:
            download_csv(v['response'], fname)

        fname = dir_results / f"data_{idx}_feb7.csv"
        if len(v['data']) > 0:
            download_csv(v['data'], fname)

    for idx, v in links_feb7_next.items():
        print("downloading idx ", idx)
        fname = dir_results / f"responses_{idx}_feb7next.csv"
        if len(v['response']) > 0:
            download_csv(v['response'], fname)

        fname = dir_results / f"data_{idx}_feb7next.csv"
        if len(v['data']) > 0:
            download_csv(v['data'], fname)


def get_data_and_responses(fname_data, fname_responses, set_name):
    df_responses_ = pd.read_csv(fname_responses)
    df_data = pd.read_csv(fname_data)

    # put responses array in the right format
    email = df_responses_.iloc[0].iloc[1]
    timestamp = df_responses_.iloc[0].iloc[1]
    n_samples = (df_responses_.shape[1] - 2) // 3
    df_responses = pd.DataFrame(
        np.array(df_responses_)[0, 2:].reshape(n_samples, 3),
        columns=['is_same_topic', 'is_best_answer', 'description'])
    df_responses.insert(0, 'email', email)
    df_responses.insert(0, 'idx', idx)
    df_responses.insert(0, 'set', set_name)

    if len(df_data) != len(df_responses):
        raise ValueError("data and feedback not aligned")

    return df_data, df_responses


dfs_responses = []
dfs_data = []
# november 10 + 6 stuff
for idx in links_feb7.keys():

    # get the form responses
    fname_responses = dir_results / f"responses_{idx}_feb7.csv"
    fname_data = dir_results / f"data_{idx}_feb7.csv"
    df_data, df_responses = get_data_and_responses(fname_data,
                                                   fname_responses,
                                                   set_name='feb7')

    dfs_responses.append(df_responses)
    dfs_data.append(df_data)

for idx in links_feb7_next.keys():
    if idx in skip_idxs_nov6:
        continue
    # get the form responses
    fname_responses = dir_results / f"responses_{idx}_feb7next.csv"
    fname_data = dir_results / f"data_{idx}_feb7next.csv"
    df_data, df_responses = get_data_and_responses(fname_data,
                                                   fname_responses,
                                                   set_name='feb7next')

    dfs_responses.append(df_responses)
    dfs_data.append(df_data)

# merge things
df_responses = pd.concat(dfs_responses)
df_data = pd.concat(dfs_data)
assert len(df_responses) == len(df_data)

# todo: maybe uncomment the later stuff to actually save it 
ipdb.set_trace()
pass 


# identify nov 5 data that is the same as nov 6 data


# def get_nov5_data_thats_same():
#     """
# 	nov 5 and nov 6 have the same underlying questions, but *some* of the mcq variations are different
# 	nov 6 is the actual data we want, but some people only reviewed nov 5
# 	however many of those mcq variations are actually identical and therefore usable. 
# 	this function gets those reviewed questions from nov 5 that are still valid. 
# 		They are returned as (df_responses_nov5_same, df_data_nov5_same)
# 	It also outputs the questions that changed, and therefore need reviewing still
# 		They are returned as (df_data_nov5_diff,)
# 	"""
#     dfs_responses_nov5 = []
#     dfs_data_nov5 = []
#     dfs_data_nov6 = []
#     for idx in links_nov5.keys():
#         fname_responses = dir_results / f"responses_{idx}_nov5.csv"
#         fname_data = dir_results / f"data_{idx}_nov5.csv"
#         df_data_nov5, df_responses_nov5 = get_data_and_responses(
#             fname_data, fname_responses, set_name='nov5')

#         fname_data = dir_results / f"data_{idx}_nov6.csv"
#         df_data_nov6 = pd.read_csv(fname_data)

#         dfs_responses_nov5.append(df_responses_nov5)
#         dfs_data_nov5.append(df_data_nov5)
#         dfs_data_nov6.append(df_data_nov6)

#     df_responses_nov5 = pd.concat(dfs_responses_nov5)
#     df_data_nov5 = pd.concat(dfs_data_nov5)
#     df_data_nov6 = pd.concat(dfs_data_nov6)

#     assert len(df_responses_nov5) == len(df_data_nov6)

#     mask_same = (df_data_nov5['mcq_str'] == df_data_nov6['mcq_str']).values
#     mask_diff = ~mask_same

#     df_responses_nov5_same = df_responses_nov5[mask_same]
#     df_data_nov5_same = df_data_nov5[mask_same]

#     df_data_nov5_diff = df_data_nov5[mask_diff]

#     return df_responses_nov5_same, df_data_nov5_same, df_data_nov5_diff


# get the valid nov 5 data and append it
# df_responses_nov5_same, df_data_nov5_same, df_data_nov5_diff = get_nov5_data_thats_same(
# )
# df_responses = pd.concat([df_responses, df_responses_nov5_same])
# df_data = pd.concat([df_data, df_data_nov5_same])

# # stick them together
# assert len(df_responses) == len(df_data)
# df_data = pd.concat([df_data, df_responses], axis=1)
# acc = df_data['is_best_answer'].sum() / len(df_data['is_best_answer'])
# n_samples = len(df_data)
# print(f"Num samples {n_samples} with correctness {acc}")


# def compare_dataset_to_eval():
#     """ check if the questions we have here are the same as what was in the OG dataset """
#     from benchmark.graduate_samples.combine_dataset import get_full_dataset_before_review, get_naive_choices_data
#     df_questions, mcqs = get_full_dataset_before_review()

#     df_merge = pd.merge(df_data,
#                         df_questions.reset_index(drop=True),
#                         left_on='question_key',
#                         right_on='key_question',
#                         right_index=False)
#     sames = []
#     for i in range(len(df_merge)):
#         is_same = df_merge.iloc[i]['question_2'] in df_merge.iloc[i]['mcq_str']
#         sames.append(is_same)
#     df_merge['same'] = sames

#     f_eval = "benchmark/graduate_samples/results/run_eval/eval_o1-mini-2024-09-12_stage2_prompt1.csv"
#     df_eval = pd.read_csv(f_eval)
#     df_eval['correct'] = df_eval['pred'] == df_eval['gt']

#     df_merge_eval = pd.merge(df_merge,
#                              df_eval[['correct', 'key_question']],
#                              left_on='question_key',
#                              right_on='key_question')
#     # df_merge_eval.groupby(['same', 'is_best_answer'])['correct'].mean()
#     df_merge_eval.groupby(['same', 'email']).count()

#     return df_merge_eval


# def filter_for_qs_matching_df():
#     df_merge_eval = compare_dataset_to_eval()
#     df_same = df_merge_eval[df_merge_eval['same']]
#     # Add both filters for is_best_answer and is_same_topic being True
#     df_same_and_reviewed = df_same[
#         (df_same['is_best_answer']) & 
#         (df_same['is_same_topic'] == True)  # Explicitly check for True
#     ]

#     cols = [
#         "key_question_x",
#         "key_image_x",
#         "question",
#         "choices",
#         "description_question_answer_x",
#         "question_0",
#         "answer_0",
#         "question_1",
#         "choices_1",
#         "correct_index_1",
#         "question_2",
#         "choices_2",
#         "correct_index_2",
#     ]
#     df_final = df_same_and_reviewed[cols]
#     # next line 
#     df_final = df_final.rename(columns={
#         'key_question_x': 'key_question',
#         'key_image_x': 'key_image',
#         'description_question_answer_x': 'description_question_answer'
#     })
#     df_final = df_final.sort_values('key_question')
#     df_final.to_csv(dir_results/"final_qs_jan26.csv", index=False)
#     pass


# def get_filtered_items_for_review():
#     df_merge_eval = compare_dataset_to_eval()
#     df_same = df_merge_eval[df_merge_eval['same']]
    
#     # Get items that failed either condition
#     df_needs_review = df_same[
#         ~((df_same['is_best_answer']) & 
#           (df_same['is_same_topic'] == True))
#     ]
    
#     # Load both exclusion datasets
#     url1 = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQGw_NPBcbu_tGelzMaBxPbWI0I5R_7y1VWrEsR2Z-rNhKSFV1FR1UiylwMJ80LwhY9YW-B8bELC42e/pub?gid=0&single=true&output=csv"
#     url2 = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQGw_NPBcbu_tGelzMaBxPbWI0I5R_7y1VWrEsR2Z-rNhKSFV1FR1UiylwMJ80LwhY9YW-B8bELC42e/pub?gid=0&single=true&output=csv"
    
#     df_exclude1 = pd.read_csv(url1)
#     df_exclude2 = pd.read_csv(url2)
    
#     # Combine question keys from both exclusion datasets
#     exclude_keys = set(df_exclude1['key_question'].unique()) | set(df_exclude2['key_question'].unique())
    
#     # Further filter to remove questions with excluded keys
#     df_needs_review = df_needs_review[~df_needs_review['key_question_x'].isin(exclude_keys)]
    
#     # Keep all original columns from df_data plus key review columns
#     review_cols = [
#         'key_question_x',
#         'question',
#         'mcq_str',
#         'is_best_answer',
#         'is_same_topic',
#         'description',
#         'email',
#         'set',
#         'idx',
#         'question_0',
#         'answer_0',
#         'question_1',
#         'choices_1',
#         'correct_index_1',
#         'question_2',
#         'choices_2',
#         'correct_index_2'
#     ]
    
#     df_review = df_needs_review[review_cols]
    
#     # Rename columns for clarity
#     df_review = df_review.rename(columns={
#         'key_question_x': 'key_question',
#         'key_image_x': 'key_image',
#         'description_question_answer_x': 'description_question_answer'
#     })
    
#     # Sort by key_question for easier review
#     df_review = df_review.sort_values('key_question')
    
#     # Save to CSV
#     df_review.to_csv(dir_results/"review_round1_jan26.csv", index=False)
    
#     print(f"Saved {len(df_review)} items that need review")
#     return df_review


# #
# # filter_for_qs_matching_df()
# get_filtered_items_for_review()
# pass
