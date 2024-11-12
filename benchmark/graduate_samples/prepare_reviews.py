"""
python -m ipdb benchmark/graduate_samples/prepare_reviews.py

"""

from benchmark.refine_bot.run_experiments import _download_csv
# yapf: disable

## which ones where the old reviews? p4. p13. p12. p18
lookup_0 = {
	# sarina 
	"p0" : "https://docs.google.com/spreadsheets/d/e/2PACX-1vSk_MB66ZHDRHnoBiirEN1ex7l7R5iz2TJVk6O1IUpJazik2kR0ZfKrGKgdWq5SDkYxUivwTNxZHYlB/pub?gid=836075084&single=true&output=csv",
	# chad 
	"p4" : "https://docs.google.com/spreadsheets/d/e/2PACX-1vSb7dWY6z56q_WzdaoB2Ys7TS9TDldzMA6DVydQ7gFhBdBd31tfU1FV5Cq0buJfuMnb-EkesB8s5bfL/pub?gid=869104732&single=true&output=csv", # none
	# Zach
	#***** web issue https://docs.google.com/forms/u/1/d/1XfPuLYSMqPly_mkk3wBraGw7hdfsqTIS8z3kxC-Jn8o/edit#responses
	"p6" : "",  
	# Jan 
	"p7" : "https://docs.google.com/spreadsheets/d/e/2PACX-1vQ16osbeRDh04JtyLou5CPh7_SLY9r9bUbFy1i8GdwMuM-iCNN6niZaAImZW2PpdenqCWER899mUws4/pub?gid=1119612156&single=true&output=csv",
	# Disha
	"p8" : "https://docs.google.com/spreadsheets/d/e/2PACX-1vTG3uGArM0Wbfnu3M29au_HjBcuXBi14UGeeMjA1RaFPF7UfQwL81QT8Vf-IfwvkoeMa94_iMf8WZ-y/pub?gid=1516194284&single=true&output=csv",
	# Jesus 
	"p9" : "https://docs.google.com/spreadsheets/d/e/2PACX-1vTio1gHmPCCTxSQQ3feGRSPHJK4MuM3mHw5ALcFj388cFg-REnZjVOU766TOrWeEQ62MX5SQ3WTfV5h/pub?gid=21371865&single=true&output=csv",
	# will 
	"p10" : "https://docs.google.com/spreadsheets/d/e/2PACX-1vQVQEOfclpTlR2sAcLu0vm84iB9woXCJ50lfMMiHKKzk0vSlBW_dZoeFjT83l_YoPwLlsAiVaeCMcbp/pub?gid=1596210345&single=true&output=csv",
	# ridhi
	"p12" : "https://docs.google.com/spreadsheets/d/e/2PACX-1vSMoJc6_e37qvwpHaGplq5qoooBRPPOlKjEFConQR_gp94_qMrL-L4lVCjrHKeZvZfvvLp_FJfxUESX/pub?gid=1608002273&single=true&output=csv", # none
	# malvika 
	"p13" :"https://docs.google.com/spreadsheets/d/e/2PACX-1vQ0vvkvfrY8JeS5a26sHhdHz97K0ZW4iL83A2aTi0GQtDyhw4nJwgxOu1cEc7Ps50YRTx_4LnDMjPWc/pub?gid=1366148692&single=true&output=csv", # none
	# alexandra
	"p14" : "https://docs.google.com/spreadsheets/d/e/2PACX-1vSIm3R7jH71fAN1rbsNwYEvZPK2WAID4LU0hYDk98tCMnY2kx0Q_ZJisb-M561BPfpYpZTch2CIJDlD/pub?gid=1764905130&single=true&output=csv",
	# disha - loading issue
	# https://docs.google.com/forms/d/117e6lu-yeycGcxLs8T6VglS-kJBF946v5H6nLVgSxYg/edit#responses
	"p15" : "",	
	# connor 
	"p18" : "https://docs.google.com/spreadsheets/d/e/2PACX-1vTWurvF1JXemW5WBgZ2kmK16uRNZQRjD6wC1BmtTbe4v8qMjTUWkzZpUZTZm3ttM7_nIddKYp4gLp3t/pub?gid=1258993078&single=true&output=csv", # none
	# zach 
	"p20" : "https://docs.google.com/spreadsheets/d/e/2PACX-1vRrlRuafHh3PkQNu1-r3ZpHc2-w6J_FXv03PexwoE9qgVoVwzQ6a4iLqbfHcwBXPDXYQF92GsWLN8gB/pub?gid=1885341724&single=true&output=csv",
	
}
# yapf: enable