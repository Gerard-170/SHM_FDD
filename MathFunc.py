import numpy as np
import matplotlib.pyplot as plt

import scipy.signal as signal
from scipy.optimize import curve_fit
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

## La siguiente funcion calcula la Densidad espectral de potencia cruzada entre todos los canales  (Esta version es la mas exacta)
def CPSD(D_filtered,fs,nperseg,noverlap,Time):
    Fr = []
    C_PSD = []
    for i in range(D_filtered.shape[1]):
         Fr.append([])
         C_PSD.append([])
         for j in range(D_filtered.shape[1]):
            f, PSD = signal.csd(D_filtered[:,i], D_filtered[:,j], fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, nfft=len(Time))
            #f, PSD = signal.csd(D_filtered[:, i], D_filtered[:, j], fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, nfft=nperseg)
            Fr[i].append([])
            C_PSD[i].append([])
            Fr[i][j] = f
            C_PSD[i][j] = PSD

    F = np.array(Fr, dtype=float, ndmin=3)
    CPSD_CH = np.array(C_PSD, dtype=complex, ndmin=3)

    return F, CPSD_CH


# def CPSD(D_filtered,fs,nperseg,noverlap,Time):
#     Fr = []
#     C_PSD = []
#     for i in range(D_filtered.shape[1]):
#          Fr.append([])
#          C_PSD.append([])
#          for j in range(D_filtered.shape[1]):
#             if j > i or j == i:
#                 f, PSD = signal.csd(D_filtered[:,i], D_filtered[:,j], fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, nfft=len(Time))
#                 #f, PSD = signal.csd(D_filtered[:, i], D_filtered[:, j], fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, nfft=nperseg)
#                 Fr[i].append([])
#                 C_PSD[i].append([])
#                 Fr[i][j] = f
#                 C_PSD[i][j] = PSD
#             else:
#                 Fr[i].append([])
#                 C_PSD[i].append([])
#                 Fr[i][j] = Fr[j][i]
#                 C_PSD[i][j] = C_PSD[j][i]
#     F = np.array(Fr, dtype=float, ndmin=3)
#     CPSD_CH = np.array(C_PSD, dtype=complex, ndmin=3)
#
#     return F, CPSD_CH




### La siguiente funcion calcula la descompisicion en valores singulares de la matriz espectral de potencia
def SVD_F(CPSD_CH):
    S1 = np.empty([CPSD_CH.shape[0], CPSD_CH.shape[2]], dtype=float)
    U1 = np.empty([CPSD_CH.shape[2], CPSD_CH.shape[0]], dtype=complex)
    for j in range(CPSD_CH.shape[2]):
        U, s, Vh = linalg.svd(CPSD_CH[:,:,j])
        U1[j, :] = U[:,0]     ### No estoy tomando valores distintos del U (por eso el 0) solo estoy tomando el primer vector y por lo tanto podemos decir que tomo de referencia al primer canal
        S1[:,j] = s
        # for i in range(CPSD_CH.shape[0]):
        #     S1[i,j] = s[i]
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
    #b, a = signal.iirfilter(o_filt, Wn=[l_F/fs, h_F/fs], btype='bandpass', analog=False, ftype='butter', output='ba')   ### Diseño utilizando las frecuencias normalizadas, no se le especifica Frecuencias de sampling
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


#### La siguiente funcion identifica la campana de SDOF (single degree of freedom) de todos los modos
def SDOF_I(mac, MS_N, U, P_ind):  ## Modal Shape Normalizados Numpy array vector fila, U = complete data modal shape, P_ind Peak Index
    SDOF_B_I = np.empty([MS_N.shape[0], 2], dtype=int) ## MS_N
    valor_mac = 1.0
    y = 0
    for x in range(MS_N.shape[0]):
        Modal_p = np.empty([MS_N.shape[1], 1], dtype=complex)
        Modal_p[:, 0] = MS_N.T[:, x]
        while valor_mac > mac:
            y = y + 1
            Modal_c = np.empty([MS_N.shape[1], 1], dtype=complex)
            Modal_c[:, 0] = U[P_ind[0, x] + y, :]
            MAC_MAT = MAC(Modal_p, Modal_c)
            valor_mac = float(abs(MAC_MAT[0, 0]))
        SDOF_B_I[x, 1] = int(P_ind[0, x] + y - 1)
        y = 0
        valor_mac = 1
        while valor_mac > mac:
            y = y + 1
            Modal_c = np.empty([MS_N.shape[1], 1], dtype=complex)
            Modal_c[:, 0] = U[P_ind[0, x] - y, :]
            MAC_MAT = MAC(Modal_p, Modal_c)
            valor_mac = float(abs(MAC_MAT[0,0]))
        SDOF_B_I[x, 0] = int(P_ind[0, x] - y + 1)
        y = 0
        valor_mac = 1
    return SDOF_B_I

## La siguiente funcion toma los indices de la campana SDOF identificada, los paso por el primer SV y obtiene en el tiempo la correlacion de cada SDOF system

#             print(f"{i_max[0, 0]}, {i_min[0, 0]}")   # Ejemplo de cadenas f en python

def SDOF_T(SDOF_I, SV):
    Bell_F = []                 # Lista de nparrays de SDOF en frecuencia
    Bell_T = []                 # Lista de nparrays de SDOF en el Tiempo
    #print(Bell_T[0].shape[0])
    for x in range(SDOF_I.shape[0]):
        Bell_F.append(SV[0, SDOF_I[x, 0]:SDOF_I[x, 1]])
    for y in range(SDOF_I.shape[0]):   # Las condiciones son para partir a la mitad la cantidad de datos ya que la IFFT devuelve la señal espejo en el extremo derecho
        if ((SDOF_I[y, 1] - SDOF_I[y, 0]) % 2) != 0:
            Temp = np.fft.ifft(Bell_F[y])
            Bell_T.append(Temp[0:int((Temp.shape[0] + 1) / 2)])
        elif ((SDOF_I[y, 1] - SDOF_I[y, 0]) % 2) == 0:
            Temp = np.fft.ifft(Bell_F[y])
            Bell_T.append(Temp[0:int(Temp.shape[0] / 2)])
    return Bell_T, Bell_F

# La siguiente funcion es usada para el ajuste de curva del EFDD utilizando la curva de hilbert
# La funcion es y = A*exp(-B*x) funcion decremento logaritmico
# Limit_min indica el minimo valor de la curva de hilbert esto para eliminar la cola y solo quede el descenso o envolvente
# se utilizan listas de arreglos Numpy
def hilbert_func(bell_t, limit_min):
    h = []
    h_abs = []
    i_max = []
    i_min = []
    d_max = []
    d_min = []
    h_crop = []
    for x in range(len(bell_t)):
        h.append(signal.hilbert(bell_t[x].real))
        h_abs.append(np.abs(h[x]))
        i_max.append(np.argmax(h_abs[x]))
        d_max.append(h_abs[x][i_max[x]])
        d_min.append(d_max[x] * limit_min)
        for y in range(h_abs[x].shape[0]):
            if h_abs[x][y] > d_min[x]:
                temp = y
            else:
                break
        i_min.append(temp)
        h_crop.append(h_abs[x][i_max[x]:i_min[x]])
    return h_crop
#     h = signal.hilbert(SDOF_T[0].real)
#     h_abs = np.abs(h)
#     i_max = np.argmax(h_abs)
#     d_max = h_abs[i_max]
#     d_min = d_max * 0.005
#     for x in range(h_abs.shape[0]):
#         if h_abs[x] > d_min:
#             i_min = x
#         else:
#             break

# Calculo de la fase
    # ph = np.arctan(np.divide(h.imag, h.real))
    # ph_crop = ph[i_max:i_min]
    # Hilbert.plot(np.linspace(0, 1200, ph_crop.shape[0]), ph_crop, 'g--')
    # def fun_lin(xv, a, b):
    #     return (a * xv) + b
    # popt_lin, pcov_lin = curve_fit(fun_lin, np.linspace(0, 1, ph.shape[0]), np.abs(ph))
    # print(popt_lin)


# Realizamos un ajuste de curva con Scipy utilizando y = A*exp(B*x)
# La siguiente funcion es usada para el ajuste de curva del EFDD utilizando la curva de hilbert
# La funcion es y = A*exp(-B*x) funcion decremento logaritmico
def fun_exp(xv, a, b):
    return a * np.exp(-b * xv)

def log_dec_fit(h_cr): #hcrop como parametro de entrada y la funcion
    popt_weight = [] #son los pesos de la funcion exponencial (a, b), de la misma forma son listas de Numpy arreglos
    pcov_fit = []
    for x in range(len(h_cr)):
        popt, pcov = curve_fit(fun_exp, np.linspace(0, 1, h_cr[x].shape[0]), h_cr[x])
        popt_weight.append(popt)
        pcov_fit.append(pcov)
    return popt_weight, pcov_fit
