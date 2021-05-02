import numpy as np
import matplotlib.pyplot as plt

import scipy.signal as signal
import scipy.linalg as linalg

### La siguiente funcion calcula la MAC (Modal Assurance Criterion)
def MAC(T, E):    ### Aqui T (Teorico) y E (experimental) deben ser numpy arrays y vectores columnas
    Teorico = np.mat(T)
    Experimental = np.mat(E)
    mac_value = np.square(np.absolute(np.conjugate(Teorico.T)*Experimental)) / ((np.conjugate(Teorico.T) * Teorico) * (np.conjugate(Experimental.T) * Experimental))
    return mac_value    ### Numpy Matrix 1x1

### la siguiente funcion estima la diferencia entre 2 frecuencias obtenidas por estimacion numerica o por OMA
def Diff_Freq(Fn, Fo):   ### Fn para frecuencia numerica y Fo para frecuencia operacional
    Delta_F = ((Fo - Fn) / Fn) * 100
    return Delta_F

### La siguiente funcion normaliza las formas modales dividiendo el vector de formas modales por la forma modal con maxima amplitud del vector
def MS_Norm(U):
    #print(U.shape)
    M = np.empty([U.shape[0],U.shape[1]], dtype=complex)
    for x in range(U.shape[0]):
        i = np.argmax(np.absolute(U[x,:]))
        M[x,:] = U[x,:]/U[x,i]
    return M     ###Vector Fila

### La siguiente funcion calcula la Densidad espectral de potencia cruzada entre todos los canales
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

    F = np.array(Fr, dtype=float, ndmin=3)
    CPSD_CH = np.array(C_PSD, dtype=complex, ndmin=3)

    return F, CPSD_CH

### La siguiente funcion calcula la descompisicion en valores singulares de la matriz espectral de potencia
def SVD_F(CPSD_CH):
    S1 = np.empty([CPSD_CH.shape[0],CPSD_CH.shape[2]], dtype=float)
    U1 = np.empty([CPSD_CH.shape[2],CPSD_CH.shape[0]],dtype=complex)
    for j in range(CPSD_CH.shape[2]):
        U, s, Vh = linalg.svd(CPSD_CH[:,:,j])
        U1[j, :] = U[:,0]     ### No estoy tomando valores distintos del U (por eso el 0) solo estoy tomando el primer vector y por lo tanto podemos decir que tomo de referencia al primer canal
        for i in range(CPSD_CH.shape[0]):
            S1[i,j] = s[i]
    return S1, U1                     #S1 Es un vector fila


### La siguiente funcion calcula la coherencia cuadratica media entre las matrices de densidad espectral
def Coherence(x, y, fs, nperseg, noverlap, Time):  ##### x, y son los datos filtrados, son time series
    Fcoh = []
    D_Cohr = []
    for i in range(x.shape[1]):
        Fcoh.append([])
        D_Cohr.append([])
        for j in range(x.shape[1]):
            f, c = signal.coherence(x[:,i], y[:,j], fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, nfft=len(Time))
            Fcoh[i].append([])
            D_Cohr[i].append([])
            Fcoh[i][j] = f
            D_Cohr[i][j] = c
    F = np.array(Fcoh, dtype=float, ndmin=3)
    Cohr = np.array(D_Cohr, dtype=float, ndmin=3)
    return F, Cohr

### Con la siguiente funcion se construye un filtro iir pasabanda
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