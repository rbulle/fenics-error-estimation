import pandas as pd

df = pd.read_pickle('output/results.pkl')
tex = df.to_latex('results.tex')
