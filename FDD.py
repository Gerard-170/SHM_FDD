import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.linalg as linalg
import PeakSelect as PS
from multiprocessing import Process, Pipe
from timeit import default_timer as timer

def D_Filter(l_F, h_F, fs, o_filt):
    # Ambos filtros son digitales
    b, a = signal.iirfilter(o_filt, Wn=[l_F/fs, h_F/fs], btype='bandpass', analog=False, ftype='butter', output='ba')   ### Diseño utilizando las frecuencias normalizadas, no se le especifica Frecuencias de sampling
    sos = signal.iirfilter(o_filt, Wn=[l_F, h_F], btype='bandpass', ftype='butter', analog=False, output='sos', fs=fs)
    ## Se manda a graficar utilizando sosfreqz
    w, h = signal.sosfreqz(sos, 2000, fs=fs)
    plt.figure(1)
    Fig_Filtro = plt.subplot(2, 1, 1)
    Fig_Filtro.set_title('Digital filter frequency response')
    Fig_Filtro.semilogx(w, 20 * np.log10(np.maximum(abs(h), 1e-5)))
    Fig_Filtro.set_ylabel('Amplitude Response [dB]')
    Fig_Filtro.set_xlabel('Frequency [Hz]')
    Fig_Filtro.grid()
    Fig_Filtro = plt.subplot(2, 1, 2)
    Fig_Filtro.semilogx(w, np.angle(h))
    Fig_Filtro.set_ylabel('Phase [Degrees]')
    Fig_Filtro.set_xlabel('Frequency [Hz]')

    return sos


def CPSD(D_filtered,fs,nperseg,noverlap,Time):
    Fr = []
    C_PSD = []
    for i in range(D_filtered.shape[1]):
         Fr.append([])
         C_PSD.append([])
         for j in range(D_filtered.shape[1]):
            f, PSD = signal.csd(D_filtered[:,i], D_filtered[:,j], fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, nfft=len(Time))
            Fr[i].append([])
            C_PSD[i].append([])
            Fr[i][j] = f
            C_PSD[i][j] = PSD

    F = np.array(Fr,dtype=float, ndmin=3)
    CPSD_CH = np.array(C_PSD,dtype=complex,ndmin=3)

    return F, CPSD_CH





def SVD_F(CPSD_CH):
    S1 = np.empty([CPSD_CH.shape[0],CPSD_CH.shape[2]], dtype=float)
    U1 = np.empty([CPSD_CH.shape[2],CPSD_CH.shape[0]],dtype=complex)
    for j in range(CPSD_CH.shape[2]):
        U, s, Vh = linalg.svd(CPSD_CH[:,:,j])
        U1[j, :] = U[:,0]     ### No estoy tomando valores distintos del U (por eso el 0) solo estoy tomando el primer vector y por lo tanto podemos decir que tomo de referencia al primer canal
        for i in range(CPSD_CH.shape[0]):
            S1[i,j] = s[i]
    return S1, U1                     #S1 Es un vector fila




### La siguiente funcion normaliza las formas modales dividiendo el vector de formas modales por la forma modal con maxima amplitud del vector
def MS_Norm(U):
    #print(U.shape)
    M = np.empty([U.shape[0],U.shape[1]], dtype=complex)
    for x in range(U.shape[0]):
        i = np.argmax(np.absolute(U[x,:]))
        M[x,:] = U[x,:]/U[x,i]
    return M

### La siguiente funcion calcula la MAC (Modal Assurance Criterion)
def MAC(T, E):    ### Aqui T (Teorico) y E (experimental) deben ser numpy arrays y vectores columnas
    Teorico = np.mat(T)
    Experimental = np.mat(E)
    mac_value = np.square(np.absolute(np.conjugate(Teorico.T)*Experimental)) / ((np.conjugate(Teorico.T) * Teorico) * (np.conjugate(Experimental.T) * Experimental))
    return mac_value

#np.array(np.mat(x)*np.mat(z))


def main():
    start = timer()
    ##Variables de muestreo
    low_fc = 0.05                                 #Frecuencia de corte (Bajo)
    high_fc = 4                                   #Frecuencia de corte (Alto)
    fs = 10                                       #Frecuencia de muestreo
    Data = pd.read_csv('Sample record 12 channels - sampling frequency 10 Hz.csv', header=None)
    Time = np.array(Data.iloc[:, 0])
    Channels = np.array(Data.iloc[:, 1:Data.shape[1]])    #Carga los datos de la siguiente forma: la primera columna es de Tiempo el resto son los acelerometros



    #Se manda a diseñar el filtro pero esta ves usando la funcion creada
    SOS = D_Filter(low_fc,high_fc,fs,o_filt=10)
    #Aplicacion de filtro con Zero Phase
    #D_filtered = signal.sosfiltfilt(sos, Channels,padlen=0)
    D_filtered = signal.sosfiltfilt(SOS, Channels, axis=0, padlen=11999)
    #D_filtered = signal.lfilter(b, a, Channels, axis=1)

    # Se utiliza la funcion CPSD para calcular la densidad espectral cruzada con todos los canales y a la vez se extraen las lineas de frecuencia
    F, CH_CPSD = CPSD(D_filtered,fs,nperseg=1000,noverlap=500,Time=Time)


    #Se calcula la descompisicion en valores singulares para los canales en cada linea de frecuencia
    S1, U1 = SVD_F(CH_CPSD)

    #Imprimir todos los SVD
    plt.figure(2)
    SVD = plt.subplot()
    SVD.set_title('Grafica de todos los SVD')
    SVD.set_ylabel('Singular Values [dB]')
    SVD.set_xlabel('Frequency (Hz)')
    #SVD.plot(F[11, 0, :], abs(CPSD_CH[11, 0, :]))
    for i in range(S1.shape[0]):
        SVD.plot(F[0,0,:],20*np.log10(S1[i,:]))


    #Crear un proceso alterno para imprimir los SVD y no interfiera con el script Principal
    parent_conn, child_conn = Pipe()   #Objeto para la comunicacion, por este pipe se reciven los rectangulos dibujados
    p_plot1 = Process(target=PS.S_modes, args=(SVD, child_conn,))
    p_plot1.start()
    coord_P = parent_conn.recv()

    #Se manda ha identificar los picos maximos de cada rectangulo dibujado
    P_Val, P_Ind = PS.freq_Max(coord_P, F[0,0,:],S1)
    PF = F[0, 0, P_Ind]
    SCAT = np.concatenate((PF.T, P_Val.T), axis=1)
    print('Las frecuencias identificadas fueron: ')
    print(SCAT)
    for x in range(SCAT.shape[0]):
        SVD.scatter(SCAT[x,0], 20*np.log10(SCAT[x,1]), marker='^', s=10 ** 2, c='r', alpha=0.5, edgecolors='blue')

    #Se manda a graficar los canales sin filtrar
    plt.figure(3)
    ax = plt.subplot(211)
    ax.set_title('Canal 1')
    ax.plot(Time, Channels[:,0])
    ax = plt.subplot(212)
    ax.set_title('Canal 2')
    ax.plot(Time, Channels[:,1])

# Resultado de toda la operacion mostrar los modos de vibracion:
    MS = np.array(U1[P_Ind[0,:],:], ndmin=2)  ## Tuvimos que poner los indices al array de P_Ind por que si no tomama la forma de 3 dimensiones
    print('Los modos de vibracion para los picos seleccionados son: ')
    print(MS.shape)
    MS_Normalizados = MS_Norm(MS)
    print(MS_Norm(MS))   ###Imprimir las formas modales ya normalizadas
    print('Modal Assurance Criterion ')
    Exp = np.empty([12,1], dtype=complex)
    Exp[:,0] = MS_Normalizados.T[:,0]
    print(MAC(Exp, Exp))



# Iniciar a graficar todas las graficas
    #Se volvio a poner la figura 2 pero en este caso se usa para mostrar los picos
    plt.show()
    print("with CPU:", timer() - start)

    # Mandar a imprimir todas las graficas pero cada grafica en un proceso diferente
    # p_plot2 = Process(target=PS.G_print, args=(Fig_Filtro,))
    # p_plot2.start()
    # p_plot3 = Process(target=PS.G_print, args=(ax,))
    # p_plot3.start()


if __name__ == '__main__':
    main()