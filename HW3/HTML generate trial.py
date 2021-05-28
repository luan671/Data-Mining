import pandas as pd
import pandas_profiling

data = pd.read_csv('D:\Data Mining\HW3\cleveland.csv')
profile = data.profile_report(title='Cleveland')
profile.to_file(output_file="D:\Data Mining\HW3\cleveland.html")