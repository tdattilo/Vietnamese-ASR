from typing import Callable, Tuple

import amfm_decompy.basic_tools as basic
import amfm_decompy.pYAAPT as pYAAPT
import numpy as np
import pandas as pd
import pitch
import tensorflow as tf

def get_tone_digit(tone_list: dict) -> int:
    for key, value in tone_list.items():
        if value == 1:
            return int(key[-1])

f0_df = pd.DataFrame()
for val in range(0, 3108):
    signal = basic.SignalObj('/home/dattilo/Documents/Project/Data Sources/Audio2-0/Audio2-' + str(val+1).zfill(2) + '.wav')
    pitchY = pYAAPT.yaapt(signal, frame_length=40, tda_frame_length=40, f0_min=75, f0_max=600)# YIN pitches
    f0_df = f0_df.append(pd.DataFrame([[val] + pitchY.samp_values.tolist()]))
text_df = pd.read_csv('/home/dattilo/Documents/Project/Data Sources/truyenkieuwordnumber.txt', sep=' ', names=['index', 'word'])
text_df['tone_2'] = (text_df['word'].str.contains('á|é|í|ó|ú|ý|ắ|ấ|ế|ố|ớ|ứ')).astype(int)
text_df['tone_3'] = (text_df['word'].str.contains('à|è|ì|ò|ù|ỳ|ằ|ầ|ề|ồ|ờ|ừ')).astype(int)
text_df['tone_4'] = (text_df['word'].str.contains('ả|ẻ|ỉ|ỏ|ủ|ỷ|ẳ|ẩ|ể|ổ|ở|ử')).astype(int)
text_df['tone_5'] = (text_df['word'].str.contains('ã|ẽ|ĩ|õ|ũ|ỹ|ẵ|ẫ|ễ|ỗ|ỡ|ữ')).astype(int)
text_df['tone_6'] = (text_df['word'].str.contains('ạ|ặ|ậ|ẹ|ệ|ị|ọ|ộ|ợ|ụ|ự|ỵ')).astype(int)
text_df['tone_1'] = (text_df[['tone_2', 'tone_3', 'tone_4', 'tone_5', 'tone_6']].any(axis=1)==False).astype(int)
f0_df.rename(columns={0:'index'}, inplace=True)
result_df = f0_df.merge(text_df[['index','tone_1', 'tone_2', 'tone_3', 'tone_4', 'tone_5', 'tone_6']], how='left', left_on='index', right_on='index')
result_df['tone_digit'] = result_df[['tone_1', 'tone_2', 'tone_3', 'tone_4', 'tone_5', 'tone_6']].apply(get_tone_digit, axis=1)

def get_train_and_test_dataframes(source_df: pd.DataFrame, proportion_test: float) ->Tuple[pd.DataFrame, pd.DataFrame]:
    test_df = source_df.groupby('tone_digit', group_keys=False).apply(lambda x: 
                                                                        x.sample(
                                                                            int(
                                                                                np.rint(
                                                                                    len(source_df)*proportion_test*len(x)/len(source_df)
                                                                                )
                                                                            )
                                                                        )
                                                                       ).sample(frac=1).reset_index(drop=True)
    train_df = source_df[~source_df.index.isin(test_df.index)]
    return train_df, test_df

num_samples = 20
train_df, test_df = get_train_and_test_dataframes(result_df, 0.1)

audio_clip_sample_length = len(train_df.columns) - 7 - 1


model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(580,1), batch_size=num_samples),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(num_samples)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(6, activation='sigmoid')
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

history = model.fit(train_df.iloc[:num_samples * (len(train_df)//num_samples), 1:-7].fillna(result_df.iloc[:num_samples * (len(train_df)//num_samples), 1:-7].mean()).to_numpy()[..., np.newaxis], 
                    train_df.iloc[:num_samples * (len(train_df)//num_samples), -7:-1].to_numpy(), epochs=10,
                   batch_size=num_samples)
_, test_acc = model.evaluate(test_df.iloc[:num_samples * (len(test_df)//num_samples), 1:-7].fillna(test_df.iloc[:num_samples * (len(test_df)//num_samples), 1:-7].mean()).to_numpy()[..., np.newaxis],
                                     test_df.iloc[:num_samples * (len(test_df)//num_samples), -7:-1].to_numpy(), batch_size=num_samples)

print('Test Accuracy: {}'.format(test_acc))
