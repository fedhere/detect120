import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets

class getBval:

    def __init__(self):
        figWH = (8,5) # in
        self.fig = plt.figure(figsize=figWH)
        plt.plot(range(10),range(10),'k--')
        self.ax = self.fig.get_axes()[0]
        self.x = [] # will contain 4 "x" values
        self.lines = [] # will contain 2D line objects for each of 4 lines            

        self.cursor = mwidgets.Cursor(self.ax, useblit=True, color='k')
        self.cursor.horizOn = False

        self.connect = self.ax.figure.canvas.mpl_connect
        self.disconnect = self.ax.figure.canvas.mpl_disconnect

        self.clickCid = self.connect("button_press_event",self.onClick)

    def onClick(self, event):
        if event.inaxes:
            self.x.append(event.xdata)
            if len(self.x)==4:
                self.cleanup()

    def cleanup(self):
        self.disconnect(self.clickCid)
        plt.close()


xvals = getBval()
plt.show()

print (xvals.x)
