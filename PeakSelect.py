import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from multiprocessing import Process, Pipe


def freq_Max(Rects, Freq, SVD):
    SV = SVD
    F = np.array(Freq, ndmin=2)
    print("Aqui es")
    print(F.shape)
    N = len(Rects)
    V_X_Min = []
    V_X_Max = []
    I_X_Min = []
    I_X_Max = []
    P_I = []
    P_V = []
    #print(Rects[0].get_width())

    for x in range(N):
        if Rects[x].get_width() < 0:
            V_X_Min.append(Rects[x].get_x() + Rects[x].get_width())
            V_X_Max.append(Rects[x].get_x())
            print(V_X_Min[x],V_X_Max[x])
            I_X_Min.append(np.argmin(abs(F-V_X_Min[x]), axis=1))
            I_X_Max.append(np.argmin(abs(F-V_X_Max[x]), axis=1))
            V_X_Min[x] = F[0,np.argmin(abs(F-V_X_Min[x]), axis=1)]
            V_X_Max[x] = F[0, np.argmin(abs(F - V_X_Max[x]), axis=1)]
            print(I_X_Min[x][0],I_X_Max[x][0])
            print(V_X_Min[x][0],V_X_Max[x][0])
            P_I.append(np.argmax(SV[0,I_X_Min[x][0]:I_X_Max[x][0]]) + I_X_Min[x][0])
            P_V.append(SV[0,(P_I[x])])
            #print(P_I[x],P_V[x])

        else:
            V_X_Min.append(Rects[x].get_x())
            V_X_Max.append(Rects[x].get_x() + Rects[x].get_width())
            print(V_X_Min[x], V_X_Max[x])
            I_X_Min.append(np.argmin(abs(F-V_X_Min[x]), axis=1))
            I_X_Max.append(np.argmin(abs(F-V_X_Max[x]), axis=1))
            V_X_Min[x] = F[0,np.argmin(abs(F-V_X_Min[x]), axis=1)]
            V_X_Max[x] = F[0, np.argmin(abs(F - V_X_Max[x]), axis=1)]
            print(I_X_Min[x][0],I_X_Max[x][0])
            print(V_X_Min[x][0],V_X_Max[x][0])
            P_I.append(np.argmax(SV[0,I_X_Min[x][0]:I_X_Max[x][0]]) + I_X_Min[x][0])
            P_V.append(SV[0,(P_I[x])])
            #print(P_I[x], P_V[x],F[0, P_I[x]])

    print(P_V, P_I)
    Peak_V = np.array(P_V, ndmin=2, dtype='float64')
    Peak_I = np.array(P_I, ndmin=2)
    I = np.argsort(Peak_I, axis=1)
    Peak_I = Peak_I[0,I]
    Peak_V = Peak_V[0,I]
    print(Peak_V, Peak_I, I)
    return Peak_V, Peak_I



class SelectPeaks:
    def __init__(self, ax, conn):
        self.figplot = ax
        self.Rects = []
        self.out = conn
        self.count = 0
        self.x_0 = None
        self.y_0 = None
        self.Button = None
        self.key = None
        self.dx = 0
        self.dy = 0

    def connect(self):
        self.cidpress = self.figplot.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.figplot.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.figplot.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cidkey = self.figplot.figure.canvas.mpl_connect('key_press_event', self.key_press)
        self.cidkeyrelease = self.figplot.figure.canvas.mpl_connect('key_release_event', self.key_release)

    def key_press(self, event):
        self.key = event.key

    def key_release(self, event):
        self.key = None

    def on_press(self, event):
         self.x_0 = event.xdata
         self.y_0 = event.ydata
         self.Button = event.button
         print('you pressed', self.Button, self.x_0, self.y_0,len(self.figplot.patches))
        # En este IF se manda a llamar la funcion remove rect que elimina el ultimo rectangulo dibujado
         if self.Button == 3:
                self.rm_rect()
         elif self.Button == 2:
                self.disconnect()

    def on_release(self,event):
        'on release we reset the press data'
        if ((self.dx or self.dy) != 0) and ((self.Button != 3) or (self.Button != 2)):
            self.Rects.append(patches.Rectangle((self.x_0, self.y_0), self.dx, self.dy, linewidth=2, edgecolor='r', linestyle='--',facecolor='none'))
            self.figplot.patches = []
            self.count = self.count + 1
            for x in self.Rects:
                self.figplot.add_patch(x)
                self.figplot.figure.canvas.draw_idle()
        print('you despressed', self.Button, self.x_0, self.y_0, self.dx, self.dy, len(self.Rects), len(self.figplot.patches))
        self.x_0 = None
        self.y_0 = None
        self.dx = 0
        self.dy = 0
        self.Button = None
        self.key = None


    def on_motion(self, event):
        if (self.x_0 or self.y_0) is None: return
        if event.inaxes != self.figplot.axes: return
        if (self.Button != 1) or (self.key != "control"): return
        print(self.key)
        #el self.figplot.patches es una lista de todos los patches creados por el usuario. A medida que se crean aqui se van guardando
        if (len(self.figplot.patches) > 0) and ((self.dx != 0) or (self.dy != 0)):                          #Aqui compruebo si existe algun patches en la figura o no
            self.figplot.patches.pop(len(self.figplot.patches)-1)
            self.figplot.figure.canvas.draw_idle()
            self.dx = event.xdata - self.x_0
            self.dy = event.ydata - self.y_0
            print(self.dx,self.dy)
            p = patches.Rectangle((self.x_0, self.y_0), self.dx, self.dy, linewidth=2, edgecolor='g', linestyle='--', facecolor='none')
            self.figplot.add_patch(p)
            self.figplot.figure.canvas.draw_idle()
        else:                                                     #si no existe que ejecuta el codigo por primera vez
            self.dx = event.xdata - self.x_0
            self.dy = event.ydata - self.y_0
            print(self.dx, self.dy)
            p = patches.Rectangle((self.x_0, self.y_0), self.dx, self.dy, linewidth=2, edgecolor='g', linestyle='--', facecolor='none')
            self.figplot.add_patch(p)
            self.figplot.figure.canvas.draw_idle()

    def rm_rect(self):
        if len(self.Rects) > 0:
            self.Rects.pop(len(self.Rects)-1)
            self.count = self.count - 1
            self.figplot.patches.pop(len(self.figplot.patches) - 1)
            self.figplot.figure.canvas.draw_idle()
            print('Seleccion eliminada',len(self.figplot.patches),len(self.Rects),self.count)
        else: return

    def disconnect(self):
        print('Disconnected')
        self.figplot.figure.canvas.mpl_disconnect(self.cidpress)
        self.figplot.figure.canvas.mpl_disconnect(self.cidmotion)
        self.figplot.figure.canvas.mpl_disconnect(self.cidrelease)
        self.figplot.figure.canvas.mpl_disconnect(self.cidkey)
        self.figplot.figure.canvas.mpl_disconnect(self.cidkeyrelease)
        self.out.send(self.Rects)
        self.out.close()


def S_modes(ax, conn):
    Peak = SelectPeaks(ax, conn)
    Peak.connect()
    print(Peak.cidpress)
    print(Peak.count)
    plt.show()
    # ax.figure.show()
    # ax.figure.canvas.start_event_loop()

def G_print(ax):
    ax.figure.show()
    ax.figure.canvas.start_event_loop()

def main():
    # Construir la señal
    t = np.arange(0.0, 2.0, 0.01)
    s = 1 + np.sin(2 * np.pi * t)
    fig = plt.figure(1)
    ax = plt.subplot()
    ax.plot(t, s)
    ax.set(xlabel='time (s)', ylabel='voltage (mV)', title='About as simple as it gets, folks')
    ax.grid()
    print("Por favor dibuje un cuadrado por cada pico que necesite analizar")
    print("Dibuje con click izquierdo y deshaga con click derecho")
    parent_conn, child_conn = Pipe()
    p_plot1 = Process(target=S_modes, args=(ax, child_conn,))
    p_plot1.start()
    coord_P = parent_conn.recv()
    print(coord_P[0].get_xy())


if __name__ == '__main__':
    main()
