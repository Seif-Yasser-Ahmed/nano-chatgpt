import pandas as pd
import os

def run(samples_path, output_path):
    # os.walk(samples_path)
    df =pd.DataFrame(columns=['file_name','step_number','sample_numper', 'sample_output'])
    for dirpath, dirnames, filenames in os.walk(samples_path):
        for filename in filenames:
            if filename.endswith('.txt'):
                file_path = os.path.join(dirpath, filename)
                with open(file_path, 'r',encoding='utf-8') as f:
                    content = f.read()
                step_number = int(filename.split('_')[0].replace('step',''))
                sample_number = int(filename.split('_')[1].split('.')[0].replace('sample',''))
                df.loc[len(df)] = [file_path, step_number, sample_number, content]
    df.to_csv(os.path.join(output_path, 'samples.csv'), index=False)

run('out_gpt2/sampled', 'out_gpt2/outputs_csv')