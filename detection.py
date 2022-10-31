import os
import librosa
import numpy as np
from scipy import signal
from scipy.fft import fftshift
from scipy.fftpack import fft
from matplotlib import mlab
import matplotlib.pyplot as plt
path='C:\\Users\\HP\\Downloads\\train'
os.chdir(path)
for folder in os.listdir():
    if folder[0]=='d':
        path1=path+'\\'+folder
        os.chdir(path1)
        for inner_folder in os.listdir():
            path2=path1+'\\'+inner_folder
            os.chdir(path2)
            for wav_file in os.listdir():
                if wav_file[-1]=='v':
                    file_name=wav_file[:len(wav_file)-4]
                    print(file_name)
                    try:
                        Audio = librosa.load(file_name+'.wav',sr=16000)

                        x = x/np.max(np.abs(x))
                        # print('x:', x, '\n')
                        # print('x_shape:', np.shape(x), '\n')
                        # print('Sample rate (KHz):', sample_rate, '\n')
                        # print(f'Length of audio: {np.shape(x)[0]/sample_rate}')

                        # Plotting the sound wave.

                        # plt.figure(figsize=(15, 5))
                        # librosa.display.waveplot(x, sr=sample_rate);
                        # plt.title("sa1.wav", fontsize=20)
                        # plt.show()
                        # # calculation of x2
                        x1 = np.zeros([x.shape[0], 1])
                        x1[0]=x[0]
                        for itr in range(1,len(x)):
                          x1[itr] = x[itr]-x[itr-1]

                        x2 = np.zeros([x.shape[0], 1])
                        x2[0]=x1[0]
                        for itr in range(1,len(x)):
                          x2[itr] = x1[itr]-x1[itr-1]



                        # plt.figure(figsize=(15, 15))
                        # plt.subplot(4, 1, 2)
                        # xspec = librosa.amplitude_to_db(librosa.stft(x, n_fft=512, hop_length=80), ref=np.max)
                        # librosa.display.specshow(xspec, cmap='gray_r', y_axis='linear')
                        # plt.colorbar(format='%+2.0f dB')
                        # plt.title('Linear power spectrogram (grayscale)')


                        # Energy of the audio signal
                        Ex = np.zeros([x.shape[0], 1])
                        for itr in range(0,len(x)):
                          if( (itr-7)>=0 and (itr+8)<len(x)):
                            Ex[itr] = np.sum( np.multiply(x[itr-7:itr+8], x[itr-7:itr+8] ))
                          if( (itr-7)<0):
                            Ex[itr] = np.sum( np.multiply(x[0:itr+8], x[0:itr+8] ))
                          if( (itr+8)>=len(x)):
                            Ex[itr] = np.sum( np.multiply(x[itr-7:], x[itr-7:] ))



                        # res.append( np.sum(np.mulitply(y[itr-7:itr+7], y[itr-7:itr+7]) ))
                        # res.append(sum(list(map(lambda x: x*x, y[itr-7:itr+7]))))
                        time_var = np.arange(0, len(x), 1)*(1/sample_rate)
                        plt.figure(figsize=(15, 5))

                        plt.plot(time_var, Ex)
                        plt.title("En of Audio signal.wav", fontsize=20)
                        # plt.show()


                        # Energy of the x2 signal
                        Ex2 = np.zeros([x.shape[0], 1])
                        for itr in range(0,len(x)):
                          if( (itr-7)>=0 and (itr+8)<len(x)):
                            Ex2[itr] = np.sum( np.multiply(x2[itr-7:itr+8], x2[itr-7:itr+8] ))
                          if( (itr-7)<0):
                            Ex2[itr] = np.sum( np.multiply(x2[0:itr+8], x2[0:itr+8] ))
                          if( (itr+8)>=len(x)):
                            Ex2[itr] = np.sum( np.multiply(x2[itr-7:], x2[itr-7:] ))



                        # res.append( np.sum(np.mulitply(y[itr-7:itr+7], y[itr-7:itr+7]) ))
                        # res.append(sum(list(map(lambda x: x*x, y[itr-7:itr+7]))))
                        time_var = np.arange(0, len(x), 1)*(1/sample_rate)
                        plt.figure(figsize=(15, 5))


                        plt.plot(time_var, Ex2)
                        plt.title("En of x2 signal.wav", fontsize=20)
                        # plt.show()


                        # Calculate H(n)
                        # Energy of the x2 signal
                        H = np.zeros([x.shape[0], 1])
                        L = np.zeros([x.shape[0], 1])
                        for itr in range(0,len(x)):
                          H[itr] = Ex[itr]/Ex2[itr]
                          if(H[itr]<1):
                            L[itr] = 1



                        plt.figure(figsize=(15, 5))

                        plt.subplot(4, 1, 1)
                        plt.plot(time_var, H)
                        plt.title("H(n)", fontsize=20)
                        # plt.show()


                        plt.figure(figsize=(15, 5))

                        plt.subplot(4, 1, 1)
                        plt.plot(time_var, L)
                        plt.title("L(n)", fontsize=20)
                        # plt.show()

                        # mean of the L signal
                        G = np.zeros([x.shape[0], 1])
                        DFRC = np.zeros([x.shape[0], 1])
                        for itr in range(0,len(x)):
                          if( (itr-80)>=0 and (itr+80)<len(x)):
                            G[itr] = np.sum( np.multiply(L[itr-80:itr+80], L[itr-80:itr+80] ))
                            G[itr] = G[itr]/161
                          if( (itr-80)<0):
                            G[itr] = np.sum( np.multiply(L[0:itr+80], L[0:itr+80] ))
                            G[itr] = G[itr]/(itr+80)
                          if( (itr+80)>=len(x)):
                            G[itr] = np.sum( np.multiply(L[itr-80:], L[itr-80:] ))
                            G[itr] = G[itr]/(len(x)-itr+80)

                          if(G[itr]>0.45):
                            DFRC[itr] = 1

                        plt.figure(figsize=(15, 5))

                        plt.subplot(4, 1, 1)
                        plt.plot(time_var, G)
                        plt.title("G(n)", fontsize=20)
                        # plt.show()


                        plt.figure(figsize=(15, 5))

                        plt.subplot(4, 1, 1)
                        plt.plot(time_var, DFRC)
                        plt.title("DFRC(n)", fontsize=20)
                        # plt.show()
                        flag=0



                        start=[]
                        finish=[]
                        count=0;
                        for i,j in zip(time_var,DFRC):
                          if(flag==0 and j==1):
                            # print("hii--->",count,i)
                            start.append(i)
                            flag=1
                          elif(flag==1 and j==0):
                            # print("noo--->",count,i)
                            finish.append(i)
                            flag=0

                          count+=1
                        # print(start)
                        # print(finish)
                        # print(len(start))

                        for i in range(len(finish)):
                          if(finish[i]*16000 - start[i]*16000 > 1300):
                            print(start[i]*16000,finish[i]*16000)
