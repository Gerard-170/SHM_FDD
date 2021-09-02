import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import scipy.signal as signal
import scipy.linalg as linalg

import PeakSelect as PS
import GraphPlot as GP
import MathFunc as MF

from multiprocessing import Process, Pipe
from timeit import default_timer as timer

def main():
# Inicio del Script se solicitan los datos para el diseño
    start = timer()
    # #Variables de muestreo
    low_fc = 0.05                                 #Frecuencia de corte (Bajo)
    high_fc = 4                                   #Frecuencia de corte (Alto)
    fs = 50                                       #Frecuencia de muestreo

    #El archivo de datos debe contener en su primera columna el tiempo y las demas columnas los canales
#    Data = pd.read_csv('Sample record 12 channels - sampling frequency 10 Hz.csv', header=None)
    Data = pd.read_csv('Data02_harvard_3_channels_50hz.csv', header=None)
    Time = np.array(Data.iloc[:, 0])
    per_segmen = int(len(Time) * 0.10)
    n_overlap = (per_segmen * 0.5)
    Channels = np.array(Data.iloc[:, 1:Data.shape[1]])    #Carga los datos de la siguiente forma: la primera columna es de Tiempo el resto son los acelerometros

#Se manda a diseñar el filtro pero esta ves usando la funcion creada
    SOS = MF.D_Filter(low_fc, high_fc, fs, o_filt=10)
    #Aplicacion de filtro con Zero Phase
    #D_filtered = signal.sosfiltfilt(sos, Channels,padlen=0)
    D_filtered = signal.sosfiltfilt(SOS, Channels, axis=0, padlen=len(Time)-1)
    #D_filtered = signal.lfilter(b, a, Channels, axis=1)

# Se utiliza la funcion CPSD para calcular la densidad espectral cruzada con todos los canales y a la vez se extraen las lineas de frecuencia
    F, CH_CPSD = MF.CPSD(D_filtered, fs, nperseg=per_segmen, noverlap=n_overlap, Time=Time)

# Se calcula la coherencia cuadratica entre la densidad espectral de los canales
    FCoh, Cohr = MF.Coherence(D_filtered, D_filtered, fs, nperseg=per_segmen, noverlap=n_overlap, Time=Time)

#Se calcula la descompisicion en valores singulares para los canales en cada linea de frecuencia
    S1, U1 = MF.SVD_F(CH_CPSD)
    print(S1.shape)
    print(U1.shape)
    print(D_filtered.shape)
    print(CH_CPSD.shape)

#Imprimir todos los SVD
    plt.figure(2)
    SVD = plt.subplot()
    SVD.set_title('Grafica de todos los SVD')
    SVD.set_ylabel('Singular Values [dB]')
    SVD.set_xlabel('Frequency (Hz)')
    #SVD.plot(F[11, 0, :], abs(CPSD_CH[11, 0, :]))
    for i in range(S1.shape[0]):
        SVD.plot(F[0, 0, :], 20*np.log10(S1[i, :]))    ##El primer SV lleva la mayor informacion S1[0,:]
        #SVD.plot(F[0, 0, :], S1[i, :])

#Crear un proceso alterno para imprimir los SVD y no interfiera con el script Principal
    parent_conn, child_conn = Pipe()   #Objeto para la comunicacion, por este pipe se reciven los rectangulos dibujados
    p_plot1 = Process(target=PS.S_modes, args=(SVD, child_conn,))
    p_plot1.start()
    coord_P = parent_conn.recv()

# Se manda ha identificar los picos maximos de cada rectangulo dibujado
# P_Val = el valor del pico seleccionado
# P_Ind = el valor del indice del pico seleccionado
    P_Val, P_Ind = PS.freq_Max(coord_P, F[0, 0, :], S1)
    PF = F[0, 0, P_Ind]
    SCAT = np.concatenate((PF.T, P_Val.T), axis=1)
    print('Las frecuencias identificadas  (Frecuencia, Amplitud): ')
    print(SCAT)   ##Nombre SCAT al vector que contiene las frecuencias
    for x in range(SCAT.shape[0]):
        SVD.scatter(SCAT[x, 0], 20*np.log10(SCAT[x, 1]), marker='^', s=10 ** 2, c='r', alpha=0.5, edgecolors='blue')

#Se manda a graficar los canales sin filtrar
    Plot_Canales = GP.Add_Buttons(ax=3, xbuffer=Time, ybuffer=Channels, name="Canales ")
    Plot_Canales.define_Cartesina()

# Resultado de toda la operacion mostrar los modos de vibracion:
    MS = np.array(U1[P_Ind[0, :], :], ndmin=2)  ## Tuvimos que poner los indices al array de P_Ind por que si no tomama la forma de 3 dimensiones MS es vector fila
    print('Los modos de vibracion para los picos seleccionados son: ')
#    print(MS.shape)
#    print(P_Ind)
    MS_Normalizados = MF.MS_Norm(MS)
    print(MF.MS_Norm(MS))   ###Imprimir las formas modales ya normalizadas

# Realizar el Modal Assurance Criterion
    print('Modal Assurance Criterion ')
    Exp = np.empty([MS_Normalizados.shape[1], 1], dtype=complex)
    Exp[:, 0] = MS_Normalizados.T[:, 0]
    #print(Exp.shape)
    valor_mac = MF.MAC(Exp, Exp)   # La funcion MAC devuelve una matriz 1x1
    print(valor_mac[0, 0])

## En las siguientes lineas se implementa el EFDD:
# Primer paso se consiguen los indices de la campana de SDOF
    SDOF_I = MF.SDOF_I(mac=0.80, MS_N=MS_Normalizados, U=U1, P_ind=P_Ind)
    print("SDOF Indices e Indices de los picos seleccionados")
    print(SDOF_I)
    print(P_Ind)
# Segundo paso se realiza la transformada de fourier inversa a los SDOF systems
    SDOF_T, SDOF_F = MF.SDOF_T(SDOF_I=SDOF_I, SV=S1)   # SDOF_T posee la mitad de datos de SDOF_F por la IFFT que devuelve la señal en forma de espejo

# Tercer paso Calcular la funcion: 1) Transformada de hilbert 2) decremento logaritmico y obtener el damping ratio
    h_crop = MF.hilbert_func(bell_t=SDOF_T, limit_min=0.005)
# Realizamos un ajuste de curva con Scipy utilizando y = A*exp(B*x)
# La funcion es y = A*exp(-B*x) funcion decremento logaritmico
    popt, pcov = MF.log_dec_fit(h_cr=h_crop)
    print("El damping ratio es: ")
    print(popt)
#Se grafica el SDOF en el tiempo y la curva ajustada a la transformada de hilbert del SDOF:
    Hilbert = GP.EFDDPlot(nfig=7, sdof_t=SDOF_T, h_cr=h_crop, popt=popt)
    Hilbert.enhanced_plot()


# Imprimir las formal modales en el plano complejo
    Plot_FormasModales = GP.Add_Buttons(ax=4, xbuffer=np.angle(MS_Normalizados), ybuffer=np.absolute(MS_Normalizados), name="Formas Modales ")   #### para polares se debe pasar en formato de angulo y magnitud los datos
    Plot_FormasModales.define_polar()

# Imprimir PSD y Coherencia entre los canales
    Plot_Coherencia = GP.Coherplot(ax=5, freq=F, cpsdata=CH_CPSD, cohdata=Cohr)
    Plot_Coherencia.define_Coherenceplot()

# Imprimir los Modal Assurance Criterion
    Modal_assurance = GP.Mac3dplot(nfig=6, ms1=MS_Normalizados, ms2=MS_Normalizados)
    Modal_assurance.macplot()

# Iniciar a graficar todas las graficas
#Se volvio a poner la figura 2 pero en este caso se usa para mostrar los picos
    plt.show()
    print("with CPU Time:", timer() - start)


if __name__ == '__main__':
    main()