"""
python -m ipdb analysis_scripts/20241212_get_form_mcq_responses.py
Next steps:
	- merge with the exidsting datasets and make sure it's the same questions


TODO - follow up on responses to:
- Nov 10, Jesus p9
- Nov 10, Zach p6 and p20 

The columns were
	- Question-answer pair is good and generally tests the same topic as the original
	- The revised MC question has one 'best answer', and I believe the starred ***answer*** is the best for the specific question
	- Give more details IF any prior answer was False (e.g., Q-A pair topic change or revised MC answer is not best or possible multiple equally correct (no one best answer))

For Nov 10 and Nov 6 - stick them together. 
Then get the data from Nov 6 and Nov 5 and identify where hte questions are the same and then save those
Then see how many require re-review.

"""
import ipdb
from pathlib import Path
import pandas as pd
import numpy as np

# format: the csv for the data, then the csv for the form response

# november 5
[4, 12, 13, 17, 18]  # has the data
links_nov5 = {
    4: {
        "data":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vRXHpEajcDADEROF5uZUSPnl0T5xOw1VsOnGsLZYAfvL_cdxjH0i6ZHTtyB6Mtvuq89Uy-7S7YAu0UR/pub?gid=1619400255&single=true&output=csv",
        "response":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vSqpErU2Ykk-EQDN6iZFbZDPhLEIGyEVM7z8POcg5urVv9gfcYUudtpMzZKLq3P3NxhFw1K3BT14goU/pub?gid=1592091952&single=true&output=csv",
    },
    12: {
        "data":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vSikyBBFM37QPIlVNBpAtYlmLDp8r6Bo-KpvOsyynAgJk9g_NeLK3y3eqAoJx-jaV3-jwwJFZ3EQnlD/pub?gid=1418645088&single=true&output=csv",
        "response":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vQV77V5sosEfnsh3cqydhvQQGxBTbwDikqpjRfuEftbsvUlbjbl3UNlPlIr_heUhxhBn4Bq1jpti-Ys/pub?gid=1858565892&single=true&output=csv",
    },
    13: {
        "data":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vQSqDXHGGMn5jnHip_P1gi33Zaa6rj__nfZWyKSKjyn0jrte1F26GTxamBCnFF6Knx4OnH2QodTYkhc/pub?gid=836234278&single=true&output=csv",
        "response":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vQLbnp2Opw67RmjS-fkhvM6a_eOk5ARiKIp4OgBkr4MVk4YmsEbv7DrrTsWg-r-5nB6nSS-GCcZ9c5G/pub?gid=347885963&single=true&output=csv",
    },
    18: {
        "data":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vT7-PftgED9U2R5tS5Mnz2mNdY557KLH3oVh-MqQk0YhwruoTIMnDtcz7q0-d2ArTkiQpq4jkcx5uYY/pub?gid=1633376530&single=true&output=csv",
        "response":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vRKB5bIhS_jdYCpVAgHkX_6nInOL0rixm6s5Yg6UHKVlhFazKgAQPQ6-8Etgd1k73cS_qM3qcU6fGaP/pub?gid=544807212&single=true&output=csv",
    },
}

# november 6
links_nov6 = {
    # p0 sarina hasan (set 4)
    0: {
        "data":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vRhCRTRIe_VHKWGzJm3-V7LZKRYwKJdERYS90ifqTi5lkJ9T32Qcs3m_gjdwbVmkYXnaWac1nnli9Mc/pub?gid=2102901315&single=true&output=csv",
        "response":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vTMYTDxU8sRW6_F0De3JXkXtuDq48crueWX0CiJ3jJO_HRtxp24Sp4NAAugvhxkSK3DlAgFlLjZ-n7h/pub?gid=75229662&single=true&output=csv",
    },
    # p4 Chad Liu  (set 4) (it's on Nov 5)
    4: {
        "data":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vRyD5x8owccET088chHncS3xvE9IBQ1x8Oj9Su3nmI1gpANtQafDOJVM8Yv5v3tk97NErsjJH5TauV3/pub?gid=2028060419&single=true&output=csv",
        "response": "",
    },
    # p6 Zach
    6: {
        "data":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vSkLJIPGB8d6tcAkBcGe11BmZmJqQ4EhPR8Fy283nJXbdIGSoSZMitcNi7fjF2V0dyE-4uUB66twpgO/pub?gid=1643890775&single=true&output=csv",
        "response":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vTkxKeKhPKbJ6z2THR-vpZWimJbCDx6FiABUxE9FfYmObRlt3MNbkREeyUFP89v3RSXwVhtY_2uI9wo/pub?gid=1392552787&single=true&output=csv",
    },
    # p7 Jan (set 3)
    7: {
        "data":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vQEhgSURGoOZ1vIJ9L_z-lvZSu4ff9kkOe_LBW3fh_X4k1ZVRKu_l4O0c5b0ozc0tbNlDyguC_qIRGc/pub?gid=1386321722&single=true&output=csv",
        "response":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vSEfRRiBnluB3p1jcy5dcgG4tNRZVnXmgAhV0G27vp6xy1VdAkgKcGuORKW2x9wTXLzDS7_YdT5hlHx/pub?gid=1291569183&single=true&output=csv",
    },
    # p8 Disha (set 2)
    8: {
        "data":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vTO8a4zqr5aylN1ywibe6ikQYhJckItBDL3x2nEcksGo9FFmJMiHyEweEGLgwvb-KridKcIQ1O2fH3p/pub?gid=1104355226&single=true&output=csv",
        "response":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vQ7hxQwbpnp28Q4pw_dgkgdVlHNpgCB94xZNQxAuSWMeDkJDwgcQ8JgB7FXS6J9sro1qIYVFhyM58L8/pub?gid=962927990&single=true&output=csv",
    },
    # p9 Jesus (set 1)
    9: {
        "data":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vStq-0srszagZ4p5q84Kh3hC5oK0gbNZlh20DSuzS2Et9I1VA7b86BZ9tybvEEtuvdfP3G5asCR6Fns/pub?gid=1693010859&single=true&output=csv",
        "response":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vRZLWqwCZCxkayueKoaJd-U4N1UXbnw6iPUHYu0ph_ei79cEuqVBsDIw1K5jL4mE5v57A3jt9sy6w-g/pub?gid=1844130967&single=true&output=csv",
    },
    # p10 Will
    10: {
        "data":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vRHcTg2z84Meg_CSaDbfLnUX8xD7-3hZXyWvGQs00BMu1ojEPGoCvj5lK84wWBSGKEWzB50D8bpGJuq/pub?gid=1174288491&single=true&output=csv",
        "response":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vRtFSYYXpbLCs-AMrb2Ix8C4NmeouSGDe17sTeAno-IYLZrdc7awiTUJTVle-d9PhiCpl8QA-VPDGG3/pub?gid=1045094154&single=true&output=csv",
    },
    # p12 ridhi
    12: {
        "data":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vTUO3_sgKC9VHKyXNNP1xt0enAfp0f2YGFk6pcRpPlKHJryxhoSiUqqOGOvt7iVOcerTeS_ZuSZ5rTU/pub?gid=2122213848&single=true&output=csv",
        "response": "",
    },
    # p13 Malvika (set 2)
    13: {
        "data":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vQLBjRqeaprTssgDqa17ph_OlJsVackHKCJOpjjLgng12A2cQnft_pFOZ6engToy7Y4NQhASDHBgNmC/pub?gid=1272697071&single=true&output=csv",
        "response": "",
    },
    # p14 Alexandra
    14: {
        "data":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vS8egL7-9icRYSqXhrS7e82Cc7Lu3XAYbGZxb7ICtFywAfeaDNNVAKu7DEnYtpykgSU7j62EDHqfOCV/pub?gid=672872591&single=true&output=csv",
        "response":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vQyUd8LPPvPiV0v6o5Wcowj6UjPosCZsdc86cdn22tA0GwcirZW9RkO1-UuX8XWOWiGbu34kOIf0EwU/pub?gid=1583763105&single=true&output=csv",
    },
    # p15 Disha (set 1)
    15: {
        "data":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vTx4fDoHNGTCimegR68XENj94vjJMACdGqUr-sUHM6SodjeW8HuDZD15_7zLnK633R1URcwo4E6eDWb/pub?gid=1236443876&single=true&output=csv",
        "response":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vSoUJIJqKRlil_oS8xm8uWW_khAQ5UhgC80zPyOBtkhuKealhujxz1u2FtfAMSMi92U7Q1sdLst4mPJ/pub?gid=1970585122&single=true&output=csv",
    },
    # p17 Alexandra (set 1)
    17: {
        "data": "",
        "response": "",
    },
    # p18 connor
    18: {
        "data":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vQSGK1it2crfF6KzpLZBHjL5l_0Ov9RiPPsj0e6ettKRQy6c3wj7kejffruphH2ypBxK-5Dx2x0NQGD/pub?gid=2074640506&single=true&output=csv",
        "response": "",
    },
    # p20 zach
    20: {
        "data":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vRfGY9fB07dSkhhArwKjHq6ozjEXCUanwbUhtr6UjeoZ0RFVaPRKT_ku-X4NGJvbjWUKfsTiBhBzaOB/pub?gid=954421452&single=true&output=csv",
        "response":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vT6_87O1rwCMvp5jAk6jm8SMKI4ebVInSXA8kqzQ2aJ2VT6pcQmCx9GTlfApSSvppGaIUxKvb5nPU2z/pub?gid=517010770&single=true&output=csv",
    },
}

# november 10
links_nov10 = {
    # p0 sarina hasan (set 4)
    0: {
        "data":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vTBlc63rDp0SlpIYj0A1bOHEaAh71Knqi6_6-LNg3MXpyMh3Hn_j9I6iB2RdZot57opBFVvtxJtgFQN/pub?gid=1358924341&single=true&output=csv",
        "response":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vTKjIcpbCZMaK-Y2n99t7HBZZ7ZJItCFNzDreAaXQRD4JuxZ-RscJVJTDkXny_jCE3HHH6UG6TfiDWn/pub?gid=845873864&single=true&output=csv",
    },
    # p4 Chad Liu  (set 4)
    4: {
        "data":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vRgA_C3zJRWfEwnXsn3J57z92oIoygD_FSzr3MCH8uSl979c_BRLNcUqp0dbzseDasOAtq_HX0lPM9H/pub?gid=24086450&single=true&output=csv",
        "response":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vThLt-Ex6YEZywhFCbb63TAyOSJME9QCn_agC2iu0j1IJrrhO_BHRk7-BB_0H9QV7g1-tjnD7wdmJZ0/pub?gid=1371851226&single=true&output=csv",
    },
    # p6 Zach
    6: {
        "data":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vRF6cfx6_BNXrlgfZePE_FgQz-2bpk76A56qHkcayFxz7ZinZDNfq4gvjqv2y5gH7tEMEPRZv4QJEtK/pub?gid=2020954052&single=true&output=csv",
        "response": "",
    },
    # p7 Jan (set 3)
    7: {
        "data":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vQaOU-d_X95IkWRR3WwK_ay_zse9RR64uLGp9oYETpePHkhOQW7b5Er8jzFaCX48dsm6eH2GwJh-zTQ/pub?gid=1404959722&single=true&output=csv",
        "response":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vRL5lUldbQY_GxxVodgK4YGzo1X9UFw-FAdlKleDhApb8wQuYrfQgBLNh5HhOtqoMLajcrIUM14jYgk/pub?gid=43543579&single=true&output=csv",
    },
    # p8 Disha (set 2)
    8: {
        "data":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vTPP7pMzB7gcDYol-b6mV9ieklpr6kfFu5PjwK7IODo6VggDGvfu1F9iKyZpeFrVOSoLEne3DpqvNFL/pub?gid=1298106111&single=true&output=csv",
        "response":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vRwMpB-ZfKi3aP0GvheWJ92tfrZUXIbixi-JzRTG_WHi-foV7Xyainw0J2M4rjArlfrA29weESZwKYq/pub?gid=1411774192&single=true&output=csv"
    },
    # # p9 Jesus (set 1)
    9: {
        "data":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vT2sxJkSYjzHKP8cGVg7Orm2EhESre0u5njZ5IFxsSF4kgr6Dn5XRxa_rA9eHtuIlsBAo7h-U50roqJ/pub?gid=1800495280&single=true&output=csv",
        "response": "",
    },
    # p10 Will
    10: {
        "data":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vSzFH3RfAR6PxHtHYoxYk3p2Wy1MiYKH6PtcYASOAhrtOBxXoWNIbxJ4TJeAApclS6fcWzV27BekiQ1/pub?gid=173021364&single=true&output=csv",
        "response":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vQFIKcUo8kgHXBJro0hh68JsXilyyPGmmYhL0OFecIG-Jemt1qG6GfJLgh9D97BesD_Y4Db1iVui6WV/pub?gid=997283126&single=true&output=csv"
    },
    # p12 ridhi
    12: {
        "data":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vThKLsLlA384P55MebIMub-lCGR5lRYaYM87_uPZJuarLRyW5i-h9j3eOlHQ1ZweBzIhIy9Yly32wkR/pub?gid=1663351635&single=true&output=csv",
        "response":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vRrf9BtgpnYQgMGWD2elTvFtAJXx3seE4K1H20J4yeN0-D6anM8qqi6sm4P5J7-Q-HsRWGdBuHwoD2k/pub?gid=1676770126&single=true&output=csv"
    },
    # p13 Malvika (set 2)
    13: {
        "data":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vRqYTf8-2TMI1PKgd_G3P8ZS5s_DS4j-YD9_08pTuFrataO2Fq9vKGBVt_Y_1mnJbL2_5fOWLesXySO/pub?gid=923793682&single=true&output=csv",
        "response":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vTvoy_32DkAWbClPVLUGw3RuMBChqDoOH3aPDWGkVwveczbzscm3H-F1xk0tFGpe0yTCNBHhhk5whMe/pub?gid=268996078&single=true&output=csv"
    },
    # p14 Alexandra
    14: {
        "data":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vSj3V9QQvqGAXlO9BXlkuvJro3QpyUjl_xTWMJrP03unBp1290rT26g2UOxwnReAVS1UaA22IaMdFAT/pub?gid=1967010411&single=true&output=csv",
        "response":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vSbO_ejlhrq3YS53yZvC7uOURWtACEKOqJCKIPy02CSqgyJt6iD6LXgSSm49gM3Ixx54VP1cDtSmqqx/pub?output=csv"
    },
    # p15 Disha (set 1)
    15: {
        "data":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vToeVVLd6aIubuKxEvq6sLf7J6c1jme4Y6yrgLbidbvl0PbwAFdC8mC7TFeQMSC-08jlFezfMZJLsE7/pub?gid=500712645&single=true&output=csv",
        "response":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vSIWFp46fHQzmFbp6lT7v2KzI0ZLR_NTzwYANFd-b5-WDkmucd2csx7V3_dKXsoQ6RCB2G576zRd5Yw/pub?gid=492014467&single=true&output=csv",
    },
    # p17 Alexandra (set 1)
    17: {
        "data":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vQDdj436ozKCfcPWx1x53M_r5wY0j0zCd_zGxqHrBP82J01Brt4ndp3iYw9Yp5ws0ajB23iKXVQWji-/pub?gid=1012021787&single=true&output=csv",
        "response":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vRa1On2ajPH2QobU7iLBlJbOzOZn2ysukgVph-crJ-RWkS_LxYELyyEsFOAXbCU1hNkkz7R6b2_bzPH/pub?gid=1621094640&single=true&output=csv"
    },
    # p18 connor
    18: {
        "data":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vTOD9tcU76jG9JT27OkS6mebzIY_zWw8jw-dufyKseJmOWBcnWxV24d5x3QAk_KTO0iW_o2WdREzHWx/pub?gid=1639500691&single=true&output=csv",
        "response":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vRmfH_abJnwT5IveygwJY2V83c7SCBzzdhZM3mPZweHFCnzf__ToRU5PTvnHKsSLtFbkpYJP_45Oefd/pub?gid=1654916034&single=true&output=csv"
    },
    # p20 zach
    20: {
        "data":
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vSFdk3xERf65ST5eLXz9-yeqc41Uv8oJmvxEdzUfnQme0vAkvOnCFv0-MPYqXhUQV6xWM0T8CvTBgwg/pub?gid=2011371542&single=true&output=csv",
        "response": ""
    }
}

from benchmark.build_raw_dataset.download_data import download_csv

DOWNLOAD = 0
# call: download_csv(url, output_path)
dir_results = Path(__file__).parent / "results" / Path(__file__).stem
dir_results.mkdir(exist_ok=True, parents=True)

skip_idxs_nov10 = [6, 9, 20]
skip_idxs_nov6 = [4, 12, 13, 17, 18]  # has the data

if DOWNLOAD:
    for idx, v in links_nov10.items():
        print("downloading idx ", idx)

        fname = dir_results / f"responses_{idx}_nov10.csv"
        if len(v['response']) > 0:
            download_csv(v['response'], fname)

        fname = dir_results / f"data_{idx}_nov10.csv"
        if len(v['data']) > 0:
            download_csv(v['data'], fname)

    for idx, v in links_nov6.items():
        print("downloading idx ", idx)
        fname = dir_results / f"responses_{idx}_nov6.csv"
        if len(v['response']) > 0:
            download_csv(v['response'], fname)

        fname = dir_results / f"data_{idx}_nov6.csv"
        if len(v['data']) > 0:
            download_csv(v['data'], fname)

    for idx, v in links_nov5.items():
        print("downloading idx ", idx)

        fname = dir_results / f"responses_{idx}_nov5.csv"
        if len(v['response']) > 0:
            download_csv(v['response'], fname)

        fname = dir_results / f"data_{idx}_nov5.csv"
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
for idx in links_nov10.keys():
    if idx in skip_idxs_nov10:
        continue
    # get the form responses
    fname_responses = dir_results / f"responses_{idx}_nov10.csv"
    fname_data = dir_results / f"data_{idx}_nov10.csv"
    df_data, df_responses = get_data_and_responses(fname_data,
                                                   fname_responses,
                                                   set_name='nov10')

    dfs_responses.append(df_responses)
    dfs_data.append(df_data)
for idx in links_nov6.keys():
    if idx in skip_idxs_nov6:
        continue
    # get the form responses
    fname_responses = dir_results / f"responses_{idx}_nov6.csv"
    fname_data = dir_results / f"data_{idx}_nov6.csv"
    df_data, df_responses = get_data_and_responses(fname_data,
                                                   fname_responses,
                                                   set_name='nov6')

    dfs_responses.append(df_responses)
    dfs_data.append(df_data)

# merge things
df_responses = pd.concat(dfs_responses)
df_data = pd.concat(dfs_data)
assert len(df_responses) == len(df_data)

# identify nov 5 data that is the same as nov 6 data


def get_nov5_data_thats_same():
    """
	nov 5 and nov 6 have the same underlying questions, but *some* of the mcq variations are different
	nov 6 is the actual data we want, but some people only reviewed nov 5
	however many of those mcq variations are actually identical and therefore usable. 
	this function gets those reviewed questions from nov 5 that are still valid. 
		They are returned as (df_responses_nov5_same, df_data_nov5_same)
	It also outputs the questions that changed, and therefore need reviewing still
		They are returned as (df_data_nov5_diff,)
	"""
    dfs_responses_nov5 = []
    dfs_data_nov5 = []
    dfs_data_nov6 = []
    for idx in links_nov5.keys():
        fname_responses = dir_results / f"responses_{idx}_nov5.csv"
        fname_data = dir_results / f"data_{idx}_nov5.csv"
        df_data_nov5, df_responses_nov5 = get_data_and_responses(
            fname_data, fname_responses, set_name='nov5')

        fname_data = dir_results / f"data_{idx}_nov6.csv"
        df_data_nov6 = pd.read_csv(fname_data)

        dfs_responses_nov5.append(df_responses_nov5)
        dfs_data_nov5.append(df_data_nov5)
        dfs_data_nov6.append(df_data_nov6)

    df_responses_nov5 = pd.concat(dfs_responses_nov5)
    df_data_nov5 = pd.concat(dfs_data_nov5)
    df_data_nov6 = pd.concat(dfs_data_nov6)

    assert len(df_responses_nov5) == len(df_data_nov6)

    mask_same = (df_data_nov5['mcq_str'] == df_data_nov6['mcq_str']).values
    mask_diff = ~mask_same

    df_responses_nov5_same = df_responses_nov5[mask_same]
    df_data_nov5_same = df_data_nov5[mask_same]

    df_data_nov5_diff = df_data_nov5[mask_diff]

    return df_responses_nov5_same, df_data_nov5_same, df_data_nov5_diff


# get the valid nov 5 data and append it
df_responses_nov5_same, df_data_nov5_same, df_data_nov5_diff = get_nov5_data_thats_same(
)
df_responses = pd.concat([df_responses, df_responses_nov5_same])
df_data = pd.concat([df_data, df_data_nov5_same])

# stick them together
assert len(df_responses) == len(df_data)
df_data = pd.concat([df_data, df_responses], axis=1)
acc = df_data['is_best_answer'].sum() / len(df_data['is_best_answer'])
n_samples = len(df_data)
print(f"Num samples {n_samples} with correctness {acc}")


def compare_dataset_to_eval():
    """ check if the questions we have here are the same as what was in the OG dataset """
    from benchmark.graduate_samples.combine_dataset import get_full_dataset_before_review, get_naive_choices_data
    df_questions, mcqs = get_full_dataset_before_review()

    df_merge = pd.merge(df_data,
                        df_questions.reset_index(drop=True),
                        left_on='question_key',
                        right_on='key_question',
                        right_index=False)
    sames = []
    for i in range(len(df_merge)):
        is_same = df_merge.iloc[i]['question_2'] in df_merge.iloc[i]['mcq_str']
        sames.append(is_same)
    df_merge['same'] = sames

    f_eval = "benchmark/graduate_samples/results/run_eval/eval_o1-mini-2024-09-12_stage2_prompt1.csv"
    df_eval = pd.read_csv(f_eval)
    df_eval['correct'] = df_eval['pred'] == df_eval['gt']

    df_merge_eval = pd.merge(df_merge,
                             df_eval[['correct', 'key_question']],
                             left_on='question_key',
                             right_on='key_question')
    # df_merge_eval.groupby(['same', 'is_best_answer'])['correct'].mean()
    df_merge_eval.groupby(['same', 'email']).count()

    return df_merge_eval


def filter_for_qs_matching_df():
    df_merge_eval = compare_dataset_to_eval()
    df_same = df_merge_eval[df_merge_eval['same']]
    # Add both filters for is_best_answer and is_same_topic being True
    df_same_and_reviewed = df_same[
        (df_same['is_best_answer']) & 
        (df_same['is_same_topic'] == True)  # Explicitly check for True
    ]

    cols = [
        "key_question_x",
        "key_image_x",
        "question",
        "choices",
        "description_question_answer_x",
        "question_0",
        "answer_0",
        "question_1",
        "choices_1",
        "correct_index_1",
        "question_2",
        "choices_2",
        "correct_index_2",
    ]
    df_final = df_same_and_reviewed[cols]
    # next line 
    df_final = df_final.rename(columns={
        'key_question_x': 'key_question',
        'key_image_x': 'key_image',
        'description_question_answer_x': 'description_question_answer'
    })
    df_final = df_final.sort_values('key_question')
    df_final.to_csv(dir_results/"final_qs_jan26.csv", index=False)
    pass


def get_filtered_items_for_review():
    df_merge_eval = compare_dataset_to_eval()
    df_same = df_merge_eval[df_merge_eval['same']]
    
    # Get items that failed either condition
    df_needs_review = df_same[
        ~((df_same['is_best_answer']) & 
          (df_same['is_same_topic'] == True))
    ]
    
    # Load both exclusion datasets
    url1 = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQGw_NPBcbu_tGelzMaBxPbWI0I5R_7y1VWrEsR2Z-rNhKSFV1FR1UiylwMJ80LwhY9YW-B8bELC42e/pub?gid=0&single=true&output=csv"
    url2 = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQGw_NPBcbu_tGelzMaBxPbWI0I5R_7y1VWrEsR2Z-rNhKSFV1FR1UiylwMJ80LwhY9YW-B8bELC42e/pub?gid=0&single=true&output=csv"
    
    df_exclude1 = pd.read_csv(url1)
    df_exclude2 = pd.read_csv(url2)
    
    # Combine question keys from both exclusion datasets
    exclude_keys = set(df_exclude1['key_question'].unique()) | set(df_exclude2['key_question'].unique())
    
    # Further filter to remove questions with excluded keys
    df_needs_review = df_needs_review[~df_needs_review['key_question_x'].isin(exclude_keys)]
    
    # Keep all original columns from df_data plus key review columns
    review_cols = [
        'key_question_x',
        'question',
        'mcq_str',
        'is_best_answer',
        'is_same_topic',
        'description',
        'email',
        'set',
        'idx',
        'question_0',
        'answer_0',
        'question_1',
        'choices_1',
        'correct_index_1',
        'question_2',
        'choices_2',
        'correct_index_2'
    ]
    
    df_review = df_needs_review[review_cols]
    
    # Rename columns for clarity
    df_review = df_review.rename(columns={
        'key_question_x': 'key_question',
        'key_image_x': 'key_image',
        'description_question_answer_x': 'description_question_answer'
    })
    
    # Sort by key_question for easier review
    df_review = df_review.sort_values('key_question')
    
    # Save to CSV
    df_review.to_csv(dir_results/"review_round1_jan26.csv", index=False)
    
    print(f"Saved {len(df_review)} items that need review")
    return df_review


#
# filter_for_qs_matching_df()
get_filtered_items_for_review()
pass
