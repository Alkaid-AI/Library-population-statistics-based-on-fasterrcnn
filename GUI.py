#
# # 创建GUI窗口打开图像 并显示在窗口中
#
# from PIL import Image, ImageTk # 导入图像处理函数库
# import tkinter as tk           # 导入GUI界面函数库
#
# # 创建窗口 设定大小并命名
# window = tk.Tk()
# window.title('图像显示界面')
# window.geometry('600x500')
# global img_png           # 定义全局变量 图像的
# var = tk.StringVar()    # 这时文字变量储存器
#
# # 创建打开图像和显示图像函数
# def Open_Img():
#     global img_png
#     var.set('已打开')
#     Img = Image.open('D:\AI_path\keras-frcnn-master\\results_images\0.png')
#     img_png = ImageTk.PhotoImage(Img)
#
# def Show_Img():
#     global img_png
#     var.set('已显示')   # 设置标签的文字为 'you hit me'
#     label_Img = tk.Label(window, image=img_png)
#     label_Img.pack()
#
# # 创建文本窗口，显示当前操作状态
# Label_Show = tk.Label(window,
#     textvariable=var,   # 使用 textvariable 替换 text, 因为这个可以变化
#     bg='blue', font=('Arial', 12), width=15, height=2)
# Label_Show.pack()
# # 创建打开图像按钮
# btn_Open = tk.Button(window,
#     text='打开图像',      # 显示在按钮上的文字
#     width=15, height=2,
#     command=Open_Img)     # 点击按钮式执行的命令
# btn_Open.pack()    # 按钮位置
# # 创建显示图像按钮
# btn_Show = tk.Button(window,
#     text='显示图像',      # 显示在按钮上的文字
#     width=15, height=2,
#     command=Show_Img)     # 点击按钮式执行的命令
# btn_Show.pack()    # 按钮位置
#
# # 运行整体窗口
# window.mainloop()










import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
class picture(QWidget):
    def __init__(self):
        super(picture, self).__init__()

        self.resize(1000, 600)
        self.setWindowTitle("图书馆区域画面及人数")

        self.label = QLabel(self)
        self.label.setText("                           显示图片")
        self.label.setFixedSize(600, 400)
        self.label.move(100, 160)

        self.label.setStyleSheet("QLabel{background:white;}"
                                 "QLabel{color:rgb(300,300,300,120);font-size:10px;font-weight:bold;font-family:宋体;}"
                                 )

        btn = QPushButton(self)
        btn.setText("打开图片")
        btn.move(10, 30)
        btn.clicked.connect(self.openimage)
    def openimage(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        jpg = QtGui.QPixmap(imgName).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(jpg)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    my = picture()
    my.show()
    sys.exit(app.exec_())





# import sys
# from PyQt5 import QtWidgets, QtCore, QtGui
# # 定义窗口函数window
# def window():
#     # 我事实上不太明白干嘛要这一句话，只是pyqt窗口的建立都必须调用QApplication方法
#     app = QtWidgets.QApplication(sys.argv)
#     # 新建一个窗口，名字叫做w
#     w = QtWidgets.QWidget()
#     # 定义w的大小
#     w.setGeometry(100, 100, 300, 200)
#     # 给w一个Title
#     w.setWindowTitle('lesson 2')
#     # 在窗口w中，新建一个lable，名字叫做l1
#     l1 = QtWidgets.QLabel(w)
#     # 调用QtGui.QPixmap方法，打开一个图片，存放在变量png中
#     png = QtGui.QPixmap('D:\AI_path\keras-frcnn-master\\results_images\0.png')
#     # 在l1里面，调用setPixmap命令，建立一个图像存放框，并将之前的图像png存放在这个框框里。
#     l1.setPixmap(png)
#
#     # 在窗口w中，新建另一个label，名字叫做l2
#     l2 = QtWidgets.QLabel(w)
#     # 用open方法打开一个文本文件，并且调用read命令，将其内容读入到file_text中
#     file = open('D:\AI_path\keras-frcnn-master\\txt\\test1.txt')
#     file_text = file.read()
#     # 调用setText命令，在l2中显示刚才的内容
#     l2.setText(file_text)
#
#     # 调整l1和l2的位置
#     l1.move(100, 20)
#     l2.move(140, 120)
#     # 显示整个窗口
#     w.show()
#     # 退出整个app
#     app.exit(app.exec_())
#
#
# # 调用window这个函数
# window()