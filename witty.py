import os
import pandas as pd
from pydub import AudioSegment
import librosa as lb
import numpy as np
path='C:\\Users\\HP\\Downloads\\train'
os.chdir(path)
final_data=pd.DataFrame()
pre_data=[]
sample_rate=16000
for folder in os.listdir():
	if folder[0]=='d':
		path1=path+'\\'+folder
		os.chdir(path1)
		for inner_folder in os.listdir():
			path2=path1+'\\'+inner_folder
			os.chdir(path2)
			for phn_file in os.listdir():
				if phn_file[-1]=='n':
					data=pd.read_csv(phn_file,sep=' ', names=['1','2','3'])
					lis =["v", "f" , "th" , "z" , "s" , "zh" , "sh" , "h"]
					for i in range(len(data['3'])):
						if data.iloc[i]['3'] in lis:
							file_name=phn_file[:len(phn_file)-4]
							try:
								Audio = lb.load(file_name+'.wav',sr=sample_rate)
								Audio1=Audio[0]
								Audio2=Audio1[data.iloc[i]['1']:data.iloc[i]['2']]
								fft = lb.stft(y = np.array(Audio2),n_fft = 512,hop_length=240)
								abs = np.abs(fft)
# 								print(type(abs))
# 								mfcc = lb.feature.mfcc(y=np.array([abs]), sr=16000,n_mfcc=13)
# 								print(mfcc.shape)
# 								# mfcc = lb.feature.mfcc(y=np.array(Audio2), sr=sample_rate,n_mfcc=13,n_fft=512,hop_length=512)
# 								# print(mfcc.shape)
# 								pd_data=mfcc.reshape((1,13)).tolist()[0]
# 								pd_data.append(1)
# 								pre_data.append(pd_data)
							except Exception as e:
								pass
final_data=pd.DataFrame(pre_data,columns=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13','y'])
print(final_data)
final_data.to_csv('witty.csv')
