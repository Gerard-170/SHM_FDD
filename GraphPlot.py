import numpy as np

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.gridspec as gridspec
from matplotlib.widgets import MultiCursor
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.mplot3d import Axes3D

import MathFunc as MF


class Add_Buttons:
    def __init__(self, ax, xbuffer, ybuffer, name):   ####ax se refiere al numero de la figura
        self.index1 = 0
        self.index2 = 0
        self.figNumber = ax
        self.xdata = xbuffer
        self.ydata = ybuffer
        self.Name = name
        self.Nchannel = self.ydata.shape[1]

    def define_polar(self):
        plt.figure(self.figNumber)
        self.ax1 = plt.subplot(1, 2, 1, projection='polar')
        self.ax1.grid(b=True)
        self.ax1.set_title(self.Name + str(self.index1))
        self.ax1.set_xlabel("R")
        self.ax1.set_ylabel("Pha", horizontalalignment='right')
        for x in range (self.ydata.shape[1]):
            self.ax1.plot(self.xdata[self.index1,x], self.ydata[self.index1,x], 'o')    ####xdata serian los angulos, ydata la magnitud ###Plot en polar toma radianes como datos de entrada
        self.ax2 = plt.subplot(1, 2, 2, projection= 'polar')
        self.ax2.grid(b=True)
        self.ax2.set_title(self.Name + str(self.index2))
        self.ax2.set_xlabel("R")
        self.ax2.set_ylabel("Pha", horizontalalignment='right')
        for x in range(self.ydata.shape[1]):
            self.ax2.plot(self.xdata[self.index2,x], self.ydata[self.index2,x], 'o')    ####xdata serian los angulos, ydata la magnitud ###Plot en polar toma radianes como datos de entrada
        plt.subplots_adjust(bottom=0.18)

        self.ax1prev = plt.axes([0.30, 0.05, 0.060, 0.060])
        self.ax1next = plt.axes([0.38, 0.05, 0.060, 0.060])
        self.bnext1 = Button(self.ax1next, 'Next')
        self.bprev1 = Button(self.ax1prev, 'Prev')
        self.bnext1.on_clicked(self.nextpolar1)
        self.bprev1.on_clicked(self.prevpolar1)

        self.ax2prev = plt.axes([0.70, 0.05, 0.060, 0.060])
        self.ax2next = plt.axes([0.78, 0.05, 0.060, 0.060])
        self.bnext2 = Button(self.ax2next, 'Next')
        self.bprev2 = Button(self.ax2prev, 'Prev')
        self.bnext2.on_clicked(self.nextpolar2)
        self.bprev2.on_clicked(self.prevpolar2)

    def define_Cartesina(self):
        plt.figure(self.figNumber)
        self.ax1 = plt.subplot(1, 2, 1)
        self.ax1.grid(b=True)
        self.ax1.set_title(self.Name + str(self.index1))
        self.ax1.plot(self.xdata, self.ydata[:,self.index1])
        self.ax2 = plt.subplot(1, 2, 2)
        self.ax2.grid(b=True)
        self.ax2.set_title(self.Name + str(self.index2))
        self.ax2.plot(self.xdata, self.ydata[:,self.index2])
        plt.subplots_adjust(bottom=0.18)

        self.ax1prev = plt.axes([0.30, 0.05, 0.060, 0.060])
        self.ax1next = plt.axes([0.38, 0.05, 0.060, 0.060])
        self.bnext1 = Button(self.ax1next, 'Next')
        self.bprev1 = Button(self.ax1prev, 'Prev')
        self.bnext1.on_clicked(self.next1)
        self.bprev1.on_clicked(self.prev1)

        self.ax2prev = plt.axes([0.70, 0.05, 0.060, 0.060])
        self.ax2next = plt.axes([0.78, 0.05, 0.060, 0.060])
        self.bnext2 = Button(self.ax2next, 'Next')
        self.bprev2 = Button(self.ax2prev, 'Prev')
        self.bnext2.on_clicked(self.next2)
        self.bprev2.on_clicked(self.prev2)

    def next1(self, event):
        self.ax1.axes.clear()
        self.ax1.axes.grid(b=True)
        self.index1 += 1
        i = self.index1 % self.Nchannel
        self.ax1.set_title(self.Name + str(i))
        self.ax1.plot(self.xdata, self.ydata[:,i])
        plt.draw()

    def prev1(self, event):
        self.ax1.axes.clear()
        self.ax1.axes.grid(b=True)
        self.index1 -= 1
        i = self.index1 % self.Nchannel
        self.ax1.set_title(self.Name + str(i))
        self.ax1.plot(self.xdata, self.ydata[:,i])
        plt.draw()

    def next2(self, event):
        self.ax2.axes.clear()
        self.ax2.axes.grid(b=True)
        self.index2 += 1
        i = self.index2 % self.Nchannel
        self.ax2.set_title(self.Name + str(i))
        self.ax2.plot(self.xdata, self.ydata[:,i])
        plt.draw()

    def prev2(self, event):
        self.ax2.axes.clear()
        self.ax2.axes.grid(b=True)
        self.index2 -= 1
        i = self.index2 % self.Nchannel
        self.ax2.set_title(self.Name + str(i))
        self.ax2.plot(self.xdata, self.ydata[:,i])
        plt.draw()

    def nextpolar1(self, event):
        self.ax1.axes.clear()
        self.ax1.grid(b=True)
        self.index1 += 1
        i = self.index1 % self.ydata.shape[0]
        self.ax1.set_title(self.Name + str(i))
        for x in range (self.ydata.shape[1]):
            self.ax1.plot(self.xdata[i,x], self.ydata[i,x], 'o')
        plt.draw()

    def prevpolar1(self, event):
        self.ax1.axes.clear()
        self.ax1.grid(b=True)
        self.index1 -= 1
        i = self.index1 % self.ydata.shape[0]
        self.ax1.set_title(self.Name + str(i))
        for x in range (self.ydata.shape[1]):
            self.ax1.plot(self.xdata[i,x], self.ydata[i,x], 'o')
        plt.draw()

    def nextpolar2(self, event):
        self.ax2.axes.clear()
        self.ax2.grid(b=True)
        self.index2 += 1
        i = self.index2 % self.ydata.shape[0]
        self.ax2.set_title(self.Name + str(i))
        for x in range (self.ydata.shape[1]):
            self.ax2.plot(self.xdata[i,x], self.ydata[i,x], 'o')
        plt.draw()

    def prevpolar2(self, event):
        self.ax2.axes.clear()
        self.ax2.grid(b=True)
        self.index2 -= 1
        i = self.index2 % self.ydata.shape[0]
        self.ax2.set_title(self.Name + str(i))
        for x in range (self.ydata.shape[1]):
            self.ax2.plot(self.xdata[i,x], self.ydata[i,x], 'o')
        plt.draw()


class Coherplot:
    def __init__(self, ax, cpsdata, cohdata, freq):   ####ax se refiere al numero de la figura
        self.index1 = 0
        self.index2 = 0
        self.figNumber = ax
        self.cpsd = cpsdata
        self.coh = cohdata
        self.Freq = freq
        self.Nchannel = self.cpsd.shape[0]
        self.xdata = 0.00000
        self.ydata = 0.00000

    def define_Coherenceplot(self):
        fig = plt.figure(self.figNumber)
        gs = gridspec.GridSpec(2, 2)            ##Crea una rejilla para los subplots
        self.ax1 = fig.add_subplot(gs[0, :])
        self.ax2 = fig.add_subplot(gs[1,0])
        self.ax3 = fig.add_subplot(gs[1,1])
        self.ax1.grid(b=True)
        self.ax2.grid(b=True)
        self.ax3.grid(b=True)

        self.ax1.set_title("Coherencia Magnitud cuadratica entre Auto PSD Canales: " + str(self.index1) + " y " + str(self.index2))
        self.ax1.plot(self.Freq[0, 0, :], self.coh[0, 0, :])

        self.ax2.set_title("Auto Densidad Espectral canal %d" % self.index1)
        self.ax2.plot(self.Freq[0, 0, :], np.absolute(self.cpsd[0, 0, :]))
#        plt.subplots_adjust(bottom=0.18)

        self.ax3.set_title("Auto Densidad Espectral canal %d" % self.index2)
        self.ax3.plot(self.Freq[0, 0, :], np.absolute(self.cpsd[0, 0, :]))

        fig.subplots_adjust(bottom=0.13)
        self.ax1prev = plt.axes([0.20, 0.05, 0.040, 0.040])
        self.ax1next = plt.axes([0.25, 0.05, 0.040, 0.040])
        self.bnext1 = Button(self.ax1next, 'Next')
        self.bprev1 = Button(self.ax1prev, 'Prev')
        self.bnext1.on_clicked(self.next1)
        self.bprev1.on_clicked(self.prev1)

        self.ax2prev = plt.axes([0.60, 0.05, 0.040, 0.040])
        self.ax2next = plt.axes([0.65, 0.05, 0.040, 0.040])
        self.bnext2 = Button(self.ax2next, 'Next')
        self.bprev2 = Button(self.ax2prev, 'Prev')
        self.bnext2.on_clicked(self.next2)
        self.bprev2.on_clicked(self.prev2)

        self.multi = MultiCursor(fig.canvas, (self.ax1, self.ax2, self.ax3), color='b', lw=1)
        self.textbox()
        self.cid = fig.canvas.mpl_connect('button_release_event', self.onclick)

    def get_values(self):
        I = np.argmin(abs(self.Freq[0, 0, :] - self.xdata), axis=-1)
        return I


    def textbox(self):
        self.at1 = AnchoredText(("Freq %.5f\nAmp  %.5f" % (self.xdata, self.ydata)),
                                prop=dict(size=8), frameon=True, loc='upper right')
        self.at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        self.at2 = AnchoredText(("Freq %.5f\nAmp  %.5f" % (self.xdata, self.ydata)),
                                prop=dict(size=8), frameon=True, loc='upper right')
        self.at2.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        self.atc = AnchoredText(("Freq %.5f\nCoh  %.5f" % (self.xdata, self.ydata)),
                                prop=dict(size=8), frameon=True, loc='upper right')
        self.atc.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        self.artist1 = self.ax1.add_artist(self.atc)
        self.artist2 = self.ax2.add_artist(self.at1)
        self.artist3 = self.ax3.add_artist(self.at2)
        self.ax1.figure.canvas.draw_idle()
        self.ax2.figure.canvas.draw_idle()
        self.ax3.figure.canvas.draw_idle()

    def onclick(self, event):
        if event.xdata == None: return
        self.artist1.remove()
        self.artist2.remove()
        self.artist3.remove()
        self.xdata = event.xdata
        self.ydata = event.ydata
        ind = self.get_values()
        self.at1 = AnchoredText(("Freq1 %.5f\nAmp  %.5f" % (self.Freq[0, 0, ind], abs(self.cpsd[self.index1, self.index1, ind]))),
                                prop=dict(size=8), frameon=True, loc='upper right')
        self.at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        self.at2 = AnchoredText(("Freq2 %.5f\nAmp  %.5f" % (self.Freq[0, 0, ind], abs(self.cpsd[self.index2, self.index2, ind]))),
                                prop=dict(size=8), frameon=True, loc='upper right')
        self.at2.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        self.atc = AnchoredText(("Freq %.5f\nCoh  %.5f" % (self.Freq[0, 0, ind], abs(self.coh[self.index1, self.index2, ind]))),
                                prop=dict(size=8), frameon=True, loc='upper right')
        self.atc.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        self.artist1 = self.ax1.add_artist(self.atc)
        self.artist2 = self.ax2.add_artist(self.at1)
        self.artist3 = self.ax3.add_artist(self.at2)
        self.ax1.figure.canvas.draw_idle()
        self.ax2.figure.canvas.draw_idle()
        self.ax3.figure.canvas.draw_idle()

    def next1(self, event):
        self.ax1.axes.clear()
        self.ax2.axes.clear()
        self.ax2.axes.grid(b=True)
        self.index1 += 1
        if self.index1 == self.Nchannel: self.index1 = 0
        i = self.index1 % self.Nchannel
        i2 = self.index2 % self.Nchannel
        self.ax1.grid(b=True)
        self.ax1.set_title("Coherencia Magnitud cuadratica entre Auto PSD Canales: " + str(self.index1) + " y " + str(self.index2))
        self.ax1.plot(self.Freq[0, 0, :], self.coh[i, i2, :])

        self.ax2.set_title("Auto Densidad Espectral canal %d" % i)
        self.ax2.plot(self.Freq[i, i, :], np.absolute(self.cpsd[i, i, :]))
        plt.draw()

    def prev1(self, event):
        self.ax1.axes.clear()
        self.ax2.axes.clear()
        self.ax2.axes.grid(b=True)
        self.index1 -= 1
        if self.index1 == -self.Nchannel: self.index1 = 0
        i = self.index1 % self.Nchannel
        i2 = self.index2 % self.Nchannel
        self.ax1.grid(b=True)
        self.ax1.set_title("Coherencia Magnitud cuadratica entre Auto PSD Canales: " + str(self.index1) + " y " + str(self.index2))
        self.ax1.plot(self.Freq[0, 0, :], self.coh[i, i2, :])
        self.ax2.set_title("Auto Densidad Espectral canal %d" % i)
        self.ax2.plot(self.Freq[i, i, :], np.absolute(self.cpsd[i, i, :]))
        plt.draw()

    def next2(self, event):
        self.ax1.axes.clear()
        self.ax3.axes.clear()
        self.ax3.axes.grid(b=True)
        self.index2 += 1
        if self.index2 == self.Nchannel: self.index2 = 0
        i = self.index2 % self.Nchannel
        i2 = self.index1 % self.Nchannel
        self.ax1.grid(b=True)
        self.ax1.set_title("Coherencia Magnitud cuadratica entre Auto PSD Canales: " + str(self.index1) + " y " + str(self.index2))
        self.ax1.plot(self.Freq[0, 0, :], self.coh[i2, i, :])
        self.ax3.set_title("Auto Densidad Espectral canal %d" % i)
        self.ax3.plot(self.Freq[i, i, :], np.absolute(self.cpsd[i, i, :]))
        plt.draw()

    def prev2(self, event):
        self.ax1.axes.clear()
        self.ax3.axes.clear()
        self.ax3.axes.grid(b=True)
        self.index2 -= 1
        if self.index2 == -self.Nchannel: self.index2 = 0
        i = self.index2 % self.Nchannel
        i2 = self.index1 % self.Nchannel
        self.ax1.grid(b=True)
        self.ax1.set_title("Coherencia Magnitud cuadratica entre Auto PSD Canales: " + str(self.index1) + " y " + str(self.index2))
        self.ax1.plot(self.Freq[0, 0, :], self.coh[i2, i, :])
        self.ax3.set_title("Auto Densidad Espectral canal %d" % i)
        self.ax3.plot(self.Freq[i, i, :], np.absolute(self.cpsd[i, i, :]))
        plt.draw()



class Mac3dplot:
    def __init__(self, nfig, ms1, ms2):   ###ms tienen que ser un numpy array vector fila
        self.Nfig = nfig
        self.MS1 = ms1.T
        self.MS2 = ms2.T
        self.N = self.MS1.shape[1]

    def macplot(self):
        fig = plt.figure(self.Nfig)
        self.ax = fig.add_subplot(111, projection='3d')
        self.ax.set_title("Modal Assurance Criterion")
        self.ax.set_xlabel("Modo Num Numero")
        self.ax.set_ylabel("Modo Exp Numero")
        self.ax.set_zlabel("MAC")
        x2d = np.arange(self.N) + 1
        y2d = np.arange(self.N) + 1
        xm, ym = np.meshgrid(x2d - 0.25, y2d - 0.25, indexing="ij")
        x3d, y3d = xm.ravel(), ym.ravel()           ### ravel aplana el arreglo volviendolo de 1-D
        top1 = np.identity(n=x2d.shape[0], dtype="float")
        for i in range(top1.shape[0]):
            a1 = np.empty([12,1], dtype=complex)
            a2 = np.empty([12, 1], dtype=complex)
            a1[:, 0] = self.MS1[:, i]    ### Se debe crear un array e indicarle los indices que debe ocupar (forma)
            a2[:, 0] = self.MS2[:, i]
            top1[i, i] = np.absolute(MF.MAC(a1, a2))

        top = top1.ravel()
        width = depth = 0.5
        bottom = np.zeros_like(x3d)
        self.ax.bar3d(x3d, y3d, bottom, width, depth, top, shade=True, color='cyan')