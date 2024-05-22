import streamlit as st
import numpy as np
import pandas as pd
from numpy.linalg import inv
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import streamlit.components.v1 as components

g_colors = ['azure',
         'purple',
         'lightgreen',
         'honeydew',
         'goldenrod',
         'cadetblue',
         'paleturquoise',
         'orangered',
         'springgreen',
         'mediumturquoise',
         'navajowhite',
         'gray']

c = g_colors 



def find_exp(cou_from, cou_to, otr, X, Y, G, list_cous = [-1]):
    exp = 0
    if (len(list_cous)<2):
        exp = (X[cou_to * 35 + otr].sum() + Y[cou_to*35 + otr].sum())*G[cou_from*35 + otr][cou_to*35 + otr]
    else:
        for cou_i in list_cous:
            exp = exp + (X[cou_i * 35 + otr].sum() + Y[cou_i*35 + otr].sum())*G[cou_from*35 + otr][cou_i*35 + otr]
    return exp

def find_imp(cou_to, otr, X, Y, G):
    imp = (X[cou_to * 35 + otr].sum() + Y[cou_to*35 + otr].sum())*(1 - G[cou_to*35 + otr][cou_to*35 + otr])
    return imp

def custom_inv(Mat):#для нормального нахождения обратной матрицы
        Mat_x = 1.5 * Mat
        inv_Mat = np.linalg.inv(Mat_x)
        return inv_Mat/(2/3)

#сортировка - вспом функция
def sort(X, Y):
    #отсортируем по возрастанию
    for i in range(len(Y)-1):
        for j in range(len(Y)-i-1):
            if Y[j] > Y[j+1]:
                Y[j], Y[j+1] = Y[j+1], Y[j]
                X[j], X[j+1] = X[j+1], X[j]


def findOut (A, G, Y):#функция нахождения выпусков
    L = inv(custom_inv(G) - A)
    X = np.dot(L, Y.transpose())
    return X


def sort_mod(X, Y):
    #отсортируем по возрастанию модуля
    for i in range(len(Y)-1):
        for j in range(len(Y)-i-1):
            if abs(Y[j]) > abs(Y[j+1]):
                Y[j], Y[j+1] = Y[j+1], Y[j]
                X[j], X[j+1] = X[j+1], X[j]
                
def find_vvp(cou_n, VIP, A, list_cous = [-1]):#функция поиска ввп страны
    X = A * VIP
    vvp = 0
    if (len(list_cous)<=1):
        for i in range(len(ind_agr)):
            vvp = vvp + VIP[cou_n*len(ind_agr)+i] - X.transpose()[cou_n*len(ind_agr)+i].sum()
        return vvp
    else:
        k = 0
        list_vvp = np.zeros(len(list_cous))
        for cou in list_cous:
            for i in range(len(ind_agr)):
                vvp = vvp + VIP[cou*len(ind_agr)+i] - X.transpose()[cou*len(ind_agr)+i].sum()
                list_vvp[k] = list_vvp[k] + VIP[cou*len(ind_agr)+i] - X.transpose()[cou*len(ind_agr)+i].sum()
            k = k+1
        return vvp, list_vvp
    
def str_exp(cou_exp, otr, X, Y, G, list_cous = [-1]):

    from matplotlib.ticker import FuncFormatter
    if (len(list_cous)<=1):
        exports = np.zeros(len(cou_agr)-1)
        x = []
        k = 0
        for i in range(len(cou_agr)):
            if (i != cou_exp):
                exports[k] = find_exp(cou_exp, i, otr, X, Y, G)
                x.append(cou_agr[i])
                k=k+1
    else:
        exports = np.zeros(len(cou_agr)- len(list_cous))
        x = []
        for cou in list_cous:
            k = 0
            for i in range(len(cou_agr)):
                if (i not in list_cous):
                    exports[k] = find_exp(cou, i, otr, X, Y, G)
                    if len(x)< len(exports):
                        x.append(cou_agr[i])
                    k=k+1
        
    
    sort(x, exports)

    x1 = list(x[-10:])
    x1.append('others')

    
    y1 = list(exports[-10:])
    y1.append(sum(exports[:len(cou_agr) - 11]))
    
    sort(x1, y1)
    
    def currency(x, pos):
    #'Два аргумента - это значение и позиция отметки.'
        return '{:1.0f} млн.$'.format(x)
        
    
   

    fig, ax = plt.subplots(figsize=(15,15))
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.title("Структура экспорта продукта "+ ind_agr[otr]+ ' региона '+ cou_agr[cou_exp]+', млн. $', fontsize = 20) # заголовок

    bars = plt.bar(x1, y1)
    ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False) 

    #ax.spines[['right', 'top']].set_visible(False) 
    #ax.xaxis.set_visible(False)

    ax.bar_label(bars, padding=-13, color='white', 
             fontsize=12, label_type='edge', fmt='%1.0f',
            fontweight='bold')
    ax.grid(axis = 'y')
    #plt.show()
    formatter = FuncFormatter(currency)
    ax.yaxis.set_major_formatter(formatter)
    ax.text(0, np.max(y1)*0.9, "Всего экспорт на "+'%.1f'%(np.sum(y1)/1000)+ ' млрд. $', bbox = dict(boxstyle="square",
                   ec='black',
                   fc='black',
                   ), fontsize = 20, c = 'w')
    
    #print('proverka = ',y1[8]/np.sum(y1)*100 /10)
    
    for i in range(11):
        if (y1[i]>np.max(y1)/25):
            ax.text(i-0.14 if (y1[i]/np.sum(y1)*100 /10 < 0.95) else i-0.22, y1[i]*0.4,'%1.1f%%' %(y1[i]/np.sum(y1)*100), 
                fontsize = 10, c = 'w', fontweight='bold')
        else:
            ax.text(i-0.14, y1[i] + 300,'%1.1f%%' %(y1[i]/np.sum(y1)*100), 
                fontsize = 10, c = 'black', fontweight='bold')
    return y1, x1, plt  

def str_imp(cou_to, otr, X, Y, G, list_cous = [-1]):#функция нахождения структуры импорта (даёт график ещё)

    y_sum = np.zeros(len(cou_agr) - len(list_cous))
    y = np.zeros(len(cou_agr) - len(list_cous))
    x = []
    vnut = 0
    vnut_sum = 0
    #global cou_agr

    if (cou_to>=0):#одна страна
        k = 0
        for i in range(len(cou_agr)):
            if (i!=cou_to):
                y[k] = G[i*35 + otr][cou_to*35 + otr]
                k = k+1
                x.append(cou_agr[i])
            else:
                vnut = G[i*35 + otr][cou_to*35 + otr]
                
        imp = find_imp(cou_to, otr, X, Y, G)
                
    else:#несколько регионов
        for cou_i in list_cous:
            k = 0
            for i in range(len(cou_agr)):
                if (i in list_cous):
                    vnut_sum = vnut + find_exp(i, cou_i, otr, X, Y, G)
                else:
                    y_sum[k] = y_sum[k] + find_exp(i, cou_i, otr, X, Y, G)#def find_exp(cou_from, cou_to, otr, X, Y, G):
                    k = k+1
                    if (len(x)<len(y)):
                        x.append(cou_agr[i])
                        
        #print('all potreb = ',vnut_sum+ np.sum(y_sum))
        #print('sum(y) = ', np.sum(y_sum))
        imp = np.sum(y_sum)
        
        y = y_sum / (vnut_sum+ np.sum(y_sum))
        vnut = vnut_sum / (vnut_sum+ np.sum(y_sum))
        #print('sum(y)+vnut = ', np.sum(y)+vnut)    

    
    sort(x, y)
    #print('str_imp: y: ', y)
    #x1 = list(x[-10:])
    
    x1 = []
    y1 = []
    for i in range(len(y)):
        if (y[len(y)-1 - i]>(0.01*(1 - vnut))):
            y1.append(y[len(y)-1 - i])
            x1.append(x[len(y)-1 - i])
        else:
            break
    #y1 = list(y[-10:])
    y1.append(sum(y[:len(cou_agr) - len(y1) - len(list_cous)]))
    #print('raz = ', sum(y1) - sum(y))
    x1.append('others')

    
    for i in range(len(y1)):
        y1[i] = float(y1[i])/(1 - vnut)
        
    sort(x1, y1)

    fig, ax = plt.subplots(figsize = (15,15))
    plt.rc('xtick', labelsize=20)
    #plt.rc('pct', textsize=15)
    #print('find_imp = ',np.sum(y)*find_imp(cou_to, otr, X, Y, G)/1000)
    ax.pie(y1, labels=x1, autopct='%1.1f%%', textprops={'size': 'xx-large', 'c':'black'}, colors = c, pctdistance=0.85, labeldistance=1.1)
    ax.text(-1, -0.55, "Общая сумма импорта региона продукта "+ ind_agr[otr]+' %.1f'%(imp/1000)+ ' млрд. $', bbox = dict(boxstyle="square",
                   ec='black',
                   fc='black',
                   ), fontsize = 20, c = 'w')
    if (cou_to>=0):
        plt.title("Структура импорта "+ cou_agr[cou_to], fontsize = 30)
    else:
        plt.title("Структура импорта", fontsize = 30)
    #plt.pie(y1, labels = x1, colors = c)
    
    
    return y1, x1, plt

def str_ind(otr, X, Y, G):
    
    y = np.zeros(len(cou_agr))
    for cou_ex in range(len(cou_agr)):
        for cou_im in range(len(cou_agr)):
            if (cou_ex != cou_im):
                y[cou_ex] = y[cou_ex] + find_exp(cou_ex, cou_im, otr, X, Y, G)
    x = list(cou_agr)
    
    sort(x, y)
    
    x1 = list(x[-10:])
    x1.append('others')

    
    y1 = list(y[-10:])
    y1.append(sum(y[:len(cou_agr) - 11]))
    
    sort(x1, y1)
    
    def currency(x, pos):
    #'Два аргумента - это значение и позиция отметки.'
        return '{:1.0f} млн.$'.format(x)
        
    
   

    fig, ax = plt.subplots(figsize=(15,15))
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.title("Структура мирового экспорта продукта "+ ind_agr[otr]+', млн. $', fontsize = 20) # заголовок

    bars = plt.bar(x1, y1)

    ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False) 

    #ax.spines[['right', 'top']].set_visible(False) 
    #ax.xaxis.set_visible(False)

    ax.bar_label(bars, padding=-13, color='white', 
             fontsize=12, label_type='edge', fmt='%1.0f',
            fontweight='bold')
    
    ax.grid(axis = 'y')
    #plt.show()
    formatter = FuncFormatter(currency)
    ax.yaxis.set_major_formatter(formatter)
    ax.text(0, np.max(y1)*0.9, "Общий объём мировой торговли продуктом "+ind_agr[otr]+' %.1f'%(np.sum(y1)/1000)+ ' млрд. $', bbox = dict(boxstyle="square",
                   ec='black',
                   fc='black',
                   ), fontsize = 20, c = 'w')
    
    #for i, v in enumerate(y1):
    #    plt.text( i + 0.1, str(v), v - 3 if v > 10 else v + 3, color='black', fontweight='bold')
    for i in range(11):
        if (y1[i]>np.max(y1)/25):
            ax.text(i-0.14 if (y1[i]/np.sum(y1)*100 /10 < 0.95) else i-0.22, y1[i]*0.4,'%1.1f%%' %(y1[i]/np.sum(y1)*100), 
                fontsize = 10, c = 'w', fontweight='bold')
        else:
            ax.text(i-0.14, y1[i] + 300,'%1.1f%%' %(y1[i]/np.sum(y1)*100), 
                fontsize = 10, c = 'black', fontweight='bold')
            
    return y1, x1, plt

def vnut_iz(cou, delta_vip, mod = '1'):
    
    y1 = []
    x1 = []
    for i in range(35):#[44*35 + 2]
        y1.append(delta_vip[cou*35 + i])
        x1.append(ind_agr[i])  

    sort(x1,y1)
    
    if (mod != '%'):
        plt.rc('xtick', labelsize=14)
    else:
        plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    fig, ax = plt.subplots(figsize=(15,15))
    #if (mod != '%'):
    #    plt.rc('xtick', labelsize=10)
    #    #print("HERE 1")
    #else:
    #    plt.rc('xtick', labelsize=20)
    #    #print("HERE 2")
    
    
    def currency(x, pos):
        if (x>1000 or x<1000):
            return '{:1.0f} млрд. $'.format(x/1000)
    #'Два аргумента - это значение и позиция отметки.'
        return '{:1.0f} млн.$'.format(x)
    def currency2(x, pos):
    #'Два аргумента - это значение и позиция отметки.'
        return '{:1.0f} %'.format(x*100)
    
    bars = plt.barh(x1, y1)

    ax.spines[['right', 'top', 'left','bottom']].set_visible(False) 
    #ax.xaxis.set_visible(False)
    if (mod != '%'):
        if (max(bars.datavalues) > 1000 or min(bars.datavalues)<1000) :
            ax.bar_label(bars, padding=4, color='black', 
             fontsize=10, label_type='edge', labels = ['{:.2f}'.format(x/1000) for x in bars.datavalues],
            fontweight='bold')
        else:
            ax.bar_label(bars, padding=4, color='black', 
             fontsize=10, label_type='edge', labels = ['{:f}'.format(x) for x in bars.datavalues],
            fontweight='bold')
    else:
        ax.bar_label(bars, padding=4, color='black', 
             fontsize=10, label_type='edge', labels = [f'{x:.1%}' for x in bars.datavalues] ,
            fontweight='bold')
    ax.grid(axis = 'x')
    #plt.show()
    if (mod == '%'):
        formatter = FuncFormatter(currency2)
        #plt.rc('xtick', labelsize=20)
    else:
        formatter = FuncFormatter(currency)
        #plt.rc('xtick', labelsize=10)
    ax.xaxis.set_major_formatter(formatter)
    
    
    
    return y1, x1, plt 

def find_benef(VIP_0, VIP_1, A):
    list_delta_vvp = []
    x = []
    for i in range(len(cou_agr)):
        if (find_vvp(i, VIP_1, A) - find_vvp(i, VIP_0, A)>0):
            list_delta_vvp.append(find_vvp(i, VIP_1, A) - find_vvp(i, VIP_0, A))
            x.append(cou_agr[i])
    
    sort(x, list_delta_vvp)
    z = min(len(list_delta_vvp), 10)
    
    def currency(x, pos):
    #'Два аргумента - это значение и позиция отметки.'
        return '{:1.0f} млн.$'.format(x)
        
    
   

    fig, ax = plt.subplots(figsize=(15,15))
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.title("Главные бенефициары", fontsize = 30) # заголовок

    bars = plt.bar(x[-z:], list_delta_vvp[-z:])

    #ax.spines[['right', 'top']].set_visible(False) 
    #ax.xaxis.set_visible(False)

    ax.bar_label(bars, padding=6, color='black', 
             fontsize=12, label_type='edge', fmt='%1.0f',
            fontweight='bold')
    ax.grid(axis = 'y')
    #plt.show()
    formatter = FuncFormatter(currency)
    ax.yaxis.set_major_formatter(formatter)
    ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False) 
    
    return list_delta_vvp, x, plt

def big_ch(VIP_0, VIP_1, A):
    list_delta_vvp = []
    x = []
    for i in range(len(cou_agr)):
        list_delta_vvp.append(find_vvp(i, VIP_1, A) - find_vvp(i, VIP_0, A))
        x.append(cou_agr[i])
    
    sort_mod(x, list_delta_vvp)
    z = min(len(list_delta_vvp), 10)
    
    def currency(x, pos):
    #'Два аргумента - это значение и позиция отметки.'
        return '{:1.0f} млн.$'.format(x)
        
    
   

    fig, ax = plt.subplots(figsize=(15,15))
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.title("Топ 10 стран по изменению Выпусков", fontsize = 30) # заголовок

    bars = plt.bar(x[-z:], list_delta_vvp[-z:])

    ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False) 
    #ax.xaxis.set_visible(False)

    ax.bar_label(bars, padding=6, color='black', 
             fontsize=12, label_type='edge', fmt='%1.0f',
            fontweight='bold')
    ax.grid(axis = 'y')
    #plt.show()
    formatter = FuncFormatter(currency)
    ax.yaxis.set_major_formatter(formatter)
    
    return list_delta_vvp, x, plt



dfA0 = pd.read_csv('data/A3.csv')
A_0 = np.array(dfA0)
A_0 = A_0.transpose()[1:].transpose()


dfA1 = pd.read_csv('data/A4.csv')
A_1 = np.array(dfA1)
A_1 = A_1.transpose()[1:].transpose()

dfG0 = pd.read_csv('data/G3.csv')
G_0 = np.array(dfG0)
G_0 = G_0.transpose()[1:].transpose()

dfG1 = pd.read_csv('data/G4.csv')
G_1 = np.array(dfG1)
G_1 = G_1.transpose()[1:].transpose()


dfVIP0 = pd.read_csv('data/VIP3.csv')
VIP_0 = np.array(dfVIP0)
VIP_0 = VIP_0.transpose()[1:].transpose()
VIP_0 = VIP_0.reshape(-1)

dfVIP1 = pd.read_csv('data/VIP4.csv')
VIP_1 = np.array(dfVIP1)
VIP_1 = VIP_1.transpose()[1:].transpose()
VIP_1 = VIP_1.reshape(-1)


dfY0 = pd.read_csv('data/Y3.csv')
Y_0 = np.array(dfY0)
Y_0 = Y_0.transpose()[1:].transpose()
Y_0 = Y_0.reshape(-1)

dfY1 = pd.read_csv('data/Y4.csv')
Y_1 = np.array(dfY1)
Y_1 = Y_1.transpose()[1:].transpose()
Y_1 = Y_1.reshape(-1)

dfInd = pd.read_csv('data/ind.csv')
ind_agr = dfInd['0'].to_list()

dfCou = pd.read_csv('data/cou.csv')
cou_agr = dfCou['0'].to_list()


dfInd_rus = pd.read_csv('data/rus_ind.csv', header=None)
#dfInd_rus.columns
ind_agr_rus = dfInd_rus[0].to_list()

list_eu = [2,3,4,12,13,14,15,17,18,19,20,21,22,23,24,30,34,35,36,39,42,47,48,49,52,53,54]

list_CHN_IND = [9, 26]

list_G7 = [7, 14, 20, 21, 30, 31, 60]#Сша - 60, Канада - 7, Франция - 20, Германия - 14, Италия - 30, Япрония - 31, Британия - 21

list_G20 =[0, 1, 6, 14, 26, 25, 30, 7, 9, 21, 38, 50, 60, 57, 20, 33, 62, 31, 51] #0,1, 6, 14, 26, 25, 30, 7,  9, 21, 38, 50, 60, 57, 20, 33, 62, 31, 51

list_BRICS_ = [6, 16, 26, 9, 50, 62]#6, 16, 26, 9, 50, 62 без Ирана , Эфиопии, ОАЭ


def prognoz2(n_RUS, otr, delta, cou_to, year, scen, cou_zam = -1, val_zam = 0):

#Задание переменных
    col = 14#перемененная агрегирования (по странам)
    
    #n_RUS = cou_agr.index('RUS') # номер России в списке

    scen_otr = otr#пока по номеру

    scen_v = delta#изменение торгового коэф 

    scen_cou_to = cou_to#1 пока цифры в порядке стран (7 - Китай, 23 - Индия, 11 - Германия) / -10 - Европа / -20 Китай + Индия 
    
    list_scen_cou_to = [scen_cou_to]
    
    if (scen_cou_to == -10):
        list_scen_cou_to = list_eu
    if (scen_cou_to == -20):
        list_scen_cou_to = list_CHN_IND  
    if (scen_cou_to == -30):
        list_scen_cou_to = list_G7
    if (scen_cou_to == -40):
        list_scen_cou_to = list_G20
    if (scen_cou_to == -50):
        list_scen_cou_to = list_BRICS_

    
    scen_year = year# 1 - 2019, 2 - 2020 // 

    scen_n = scen#1 - уменьшение/увеличение внут торговых коэф импортёра пропорционально старым торговым коэф
                          #2 - уменьшение/увеличение имп из ост мира
        
    scen_cou_zam = cou_zam
    
    list_scen_cou_zam = [cou_zam]
    
    if (scen_cou_zam == -10):
        list_scen_cou_zam = list_eu
    if (scen_cou_zam == -20):
        list_scen_cou_zam = list_CHN_IND 
    if (scen_cou_zam == -30):
        list_scen_cou_zam = list_G7
    if (scen_cou_zam == -40):
        list_scen_cou_zam = list_G20
    if (scen_cou_zam == -50):
        list_scen_cou_zam = list_BRICS_
        
    #print('here')
    if (scen_year == 1):
        G_nach = np.array(G_0)
        G_new = np.array(G_0)
        A_scen = np.array(A_0)
        Y_scen = np.array(Y_0)
        VIP_scen = np.array(VIP_0)
    else:
        G_nach = np.array(G_1)
        G_new = np.array(G_1)
        A_scen = np.array(A_1)
        Y_scen = np.array(Y_1)
        VIP_scen = np.array(VIP_1)

    X_scen = A_scen * VIP_scen 

#Начало рассчётов
    raz = 0
    list_raz = []
    if (scen_cou_to >= 0):#один регион scen_cou_to
        raz = G_new[n_RUS*35 + scen_otr][scen_cou_to*35 + scen_otr] * (scen_v/100)
        G_new[n_RUS*35 + scen_otr][scen_cou_to*35 + scen_otr] = G_new[n_RUS*35 + scen_otr][scen_cou_to*35 + scen_otr] + raz
    else:
        l = 0
        for cou_i in list_scen_cou_to:
            list_raz.append(G_new[n_RUS*35 + scen_otr][cou_i*35 + scen_otr] * (scen_v/100))
            G_new[n_RUS*35 + scen_otr][cou_i*35 + scen_otr] = G_new[n_RUS*35 + scen_otr][cou_i*35 + scen_otr] + list_raz[l]
            l = l+1
#СЦЕНАРИИ:
    #1 - сценарий
    if (scen_n == 1):
        if (scen_cou_to>=0):
            G_new[scen_cou_to*35 + scen_otr][scen_cou_to*35 + scen_otr] = G_new[scen_cou_to*35 + scen_otr][scen_cou_to*35 + scen_otr] - raz
        else:
            l = 0
            for cou_i in list_scen_cou_to:
                G_new[cou_i*35 + scen_otr][cou_i*35 + scen_otr] = G_new[cou_i*35 + scen_otr][cou_i*35 + scen_otr] - list_raz[l]
                l=l+1
        res = findOut (A_scen, G_new, Y_scen)


    #2 - сценарий
    if (scen_n == 2):
        sum_ost = 0
        for i in range((77-col+1)*35):
            if (i == n_RUS*35 + scen_otr):
                sum_ost = sum_ost
            else:
                sum_ost = sum_ost + G_new[i][scen_cou_to*35 + scen_otr]
        
        for i in range((77-col+1)):
            if (i != n_RUS):
                G_new[i*35 + scen_otr][scen_cou_to*35 + scen_otr] =G_new[i*35 + scen_otr][scen_cou_to*35 + scen_otr] - raz * G_new[i*35 + scen_otr][scen_cou_to*35 + scen_otr] / sum_ost
    
        res = findOut (A_scen, G_new, Y_scen)
    
    #Доделать на несколько регионов
        #3 - сценарий оставляем без изменений внут производство меняем только импорт
    #'''
    #if (scen_n  == 3):
    #    sum_ost = 0
    #    for i in range((77-col+1)*35):
    #        if (i == n_RUS*35 + scen_otr or i == scen_cou_to*35 + scen_otr):
    #            sum_ost = sum_ost
    #        else:
    #            sum_ost = sum_ost + G_new[i][scen_cou_to*35 + scen_otr]
    #   
    #    for i in range((77-col+1)):
    #        if (i != n_RUS and i != scen_cou_to):
    #            G_new[i*35 + scen_otr][scen_cou_to*35 + scen_otr] = G_new[i*35 + scen_otr][scen_cou_to*35 + scen_otr] - raz * G_new[i*35 + scen_otr][scen_cou_to*35 + scen_otr] / sum_ost
    # 
    #    res = findOut (A_scen, G_new, Y_scen)
    #'''
    #4 - сценарий: Замещаем экспорт другой страны
    if (scen_n == 4):
        if (scen_cou_to >= 0):
            if (scen_cou_zam>=0):
                G_new[cou_zam*35 + scen_otr][cou_to*35 + scen_otr] = G_new[cou_zam*35 + scen_otr][cou_to*35 + scen_otr] - raz
            else:#в этом случае уменьшаем пропорционально их начальному значению
                sum =  0 
                for cou_zam_i in list_scen_cou_zam:
                    sum = sum + G_new[cou_zam_i*35 + scen_otr][cou_to*35 + scen_otr]
                
                for cou_zam_i in list_scen_cou_zam:
                    G_new[cou_zam_i*35 + scen_otr][cou_to*35 + scen_otr] = G_new[cou_zam_i*35 + scen_otr][cou_to*35 + scen_otr] - raz *  G_new[cou_zam_i*35 + scen_otr][cou_to*35 + scen_otr] / sum

        else:
            if (scen_cou_zam>=0):
                k = 0
                for cou_to_i in list_scen_cou_to:
                    G_new[cou_zam*35 + scen_otr][cou_to_i*35 + scen_otr] = G_new[cou_zam*35 + scen_otr][cou_to_i*35 + scen_otr] - list_raz[k] 
                    k = k + 1
            else:
                k = 0
                sum = 0
                for cou_to_i in list_scen_cou_to:

                    for cou_zam_i in list_scen_cou_zam:
                        sum = sum + G_new[cou_zam_i*35 + scen_otr][cou_to_i*35 + scen_otr]
                
                    for cou_zam_i in list_scen_cou_zam:
                        G_new[cou_zam_i*35 + scen_otr][cou_to_i*35 + scen_otr] = G_new[cou_zam_i*35 + scen_otr][cou_to_i*35 + scen_otr] - list_raz[k] *  G_new[cou_zam_i*35 + scen_otr][cou_to_i*35 + scen_otr] / sum
                    k = k + 1
        
        res = findOut (A_scen, G_new, Y_scen)
        
    #5 (+3)- сценарий: разворот из Европы на Восток (выбираем ресурс, выбираем изменение, выбираем где изменениея, выбираем сколько Россия смогла(в процентах) перенаправить из измения, выбираем куда перенаправила)
    if (scen_n == 5 or scen_n == 3):
        if (scen_n == 3):
            val_zam = 0
        delta_exp_to_cou1 = 0
        zamech_exp = 0
        exp_to_cou2_nach = 0
        raz2 = 0
        if (scen_cou_to >= 0 ):#одна отдельная страна1
            #изменение внутри импорта страны 1 
            sum_ost = 0
            for i in range((77-col+1)*35):
                if (i == n_RUS*35 + scen_otr or i == scen_cou_to*35 + scen_otr):
                    sum_ost = sum_ost
                else:
                    sum_ost = sum_ost + G_new[i][scen_cou_to*35 + scen_otr]
                    
            for i in range((77-col+1)):
                if (i != n_RUS and i != scen_cou_to):
                    G_new[i*35 + scen_otr][scen_cou_to*35 + scen_otr] = G_new[i*35 + scen_otr][scen_cou_to*35 + scen_otr] - raz * G_new[i*35 + scen_otr][scen_cou_to*35 + scen_otr] / sum_ost
        
            #изменение внутри страны 2 (куда идёт замещение) (нужно найти для начала велечину изменения экспорта)
            #def find_exp(cou_from, cou_to, otr, X, Y, G):
            delta_exp_to_cou1 = find_exp(n_RUS, scen_cou_to, scen_otr,X_scen, Y_scen, G_new) - find_exp(n_RUS, scen_cou_to, scen_otr, X_scen, Y_scen, G_nach)
            
            zamech_exp = -delta_exp_to_cou1 * val_zam / 100
            
            #exp_to_cou2_nach = find_exp(n_RUS, cou_zam, scen_otr, G_nach)
            
            #raz2 = 1 +  zamech_exp / exp_to_cou2_nach
            
        else: #группа стран1
            l = 0
            for cou_i in list_scen_cou_to:
                sum_ost = 0
                for i in range((77-col+1)*35):
                    if (i == n_RUS*35 + scen_otr or i == cou_i*35 + scen_otr):
                        sum_ost = sum_ost
                    else:
                        sum_ost = sum_ost + G_new[i][cou_i*35 + scen_otr]
                    
                for i in range((77-col+1)):
                    if (i != n_RUS and i != cou_i):
                        G_new[i*35 + scen_otr][cou_i*35 + scen_otr] = G_new[i*35 + scen_otr][cou_i*35 + scen_otr] - list_raz[l] * G_new[i*35 + scen_otr][cou_i*35 + scen_otr] / sum_ost
                l=l+1       
                delta_exp_to_cou1 = delta_exp_to_cou1 + find_exp(n_RUS, cou_i, scen_otr, X_scen, Y_scen, G_new) - find_exp(n_RUS, cou_i, scen_otr, X_scen, Y_scen, G_nach)
                
            zamech_exp = -delta_exp_to_cou1 * val_zam / 100
          
            #st.write('val_zam = ',  val_zam,  'val_zam / 100 = ', val_zam / 100)
            #st.write('zamech_exp',zamech_exp)
            #изменяем элементы матрицы G связанные с страной/странами 2
        if (scen_cou_zam>-1):#один замещающий регион
            exp_to_cou2_nach = find_exp(n_RUS, cou_zam, scen_otr, X_scen, Y_scen, G_nach)
            
            #print('exp_to_cou2_nach',exp_to_cou2_nach)
            k2 = (zamech_exp+exp_to_cou2_nach) / exp_to_cou2_nach 
            
            #raz = G_new[n_RUS*35 + scen_otr][scen_cou_to*35 + scen_otr] * (scen_v/100)
            
            G_new[n_RUS*35 + scen_otr][scen_cou_zam*35 + scen_otr] = G_new[n_RUS*35 + scen_otr][scen_cou_zam*35 + scen_otr]*k2
            raz2 = G_new[n_RUS*35 + scen_otr][scen_cou_zam*35 + scen_otr] - G_nach[n_RUS*35 + scen_otr][scen_cou_zam*35 + scen_otr]
            
            sum_ost = 0
            for i in range((77-col+1)*35):
                if (i == n_RUS*35 + scen_otr or i == cou_zam*35 + scen_otr):
                    sum_ost = sum_ost
                else:
                    sum_ost = sum_ost + G_new[i][cou_zam*35 + scen_otr]
                    
            for i in range((77-col+1)):
                if (i != n_RUS and i != cou_zam):
                    G_new[i*35 + scen_otr][cou_zam*35 + scen_otr] = G_new[i*35 + scen_otr][cou_zam*35 + scen_otr] - raz2 * G_new[i*35 + scen_otr][cou_zam*35 + scen_otr] / sum_ost
            
            #print('scen 5: np.transpose(G_new)[scen_cou_zam*35 + otr].sum() = ',np.transpose(G_new)[scen_cou_zam*35 + otr].sum())
            
            res = findOut (A_scen, G_new, Y_scen)
        else:
            #пропорциональное увеличение по списку стран (найдём изменения)
            list_exp_2 = []
            list_delta_exp_2 = []
            list_raz2 = []
            
            for cou_i_2 in list_scen_cou_zam:#ищем сумму всех импортов продукции и отдельно импорты
                exp_to_cou2_nach = exp_to_cou2_nach + find_exp(n_RUS, cou_i_2, scen_otr, X_scen, Y_scen, G_nach)
                list_exp_2.append(find_exp(n_RUS, cou_i_2, scen_otr, X_scen, Y_scen, G_nach))
            
            
            for i in range(len(list_scen_cou_zam)):# находим на сколько нужно увеличить импорт пропорционально
                list_delta_exp_2.append(zamech_exp * list_exp_2[i] / exp_to_cou2_nach)
                
            #st.write('list_delta_exp_2 = ', list_delta_exp_2)
            #st.write('list_delta_exp_2.sum() - zamech_exp = ', sum(list_delta_exp_2) - zamech_exp)
                                
            #меняем матрицу G в (list_scen_cou_zam) столбцах
            k = 0
            for cou_i_2 in list_scen_cou_zam:
                koef = (list_delta_exp_2[k] + list_exp_2[k])/ list_exp_2[k]
                #st.write(cou_agr[list_scen_cou_zam[k]]+': '+'koef = ', koef)
                list_raz2.append(G_new[n_RUS*35 + scen_otr][cou_i_2*35 + scen_otr] * (koef) - G_new[n_RUS*35 + scen_otr][cou_i_2*35 + scen_otr])
                G_new[n_RUS*35 + scen_otr][cou_i_2*35 + scen_otr] = G_new[n_RUS*35 + scen_otr][cou_i_2*35 + scen_otr] + list_raz2[k]
                
                
                sum_ost = 0
                for i in range((77-col+1)*35):
                    if (i == n_RUS*35 + scen_otr or i == cou_i_2*35 + scen_otr):
                        sum_ost = sum_ost
                    else:
                        sum_ost = sum_ost + G_new[i][cou_i_2*35 + scen_otr]
                    
                for i in range((77-col+1)):
                    if (i != n_RUS and i != cou_i_2):
                        G_new[i*35 + scen_otr][cou_i_2*35 + scen_otr] = G_new[i*35 + scen_otr][cou_i_2*35 + scen_otr] - list_raz2[k] * G_new[i*35 + scen_otr][cou_i_2*35 + scen_otr] / sum_ost
                k = k + 1
            res = findOut (A_scen, G_new, Y_scen)
    

    #1.Графики
        #1.1 - Мировая структура по продукту до и после (def str_ind(otr, X, Y, G))
    y_str_ind_nach, x_str_ind_nach, fig_str_ind_nach = str_ind(otr, X_scen, Y_scen, G_nach)
    fig_str_ind_nach.title('Начальная структура мировой торговли продуктом '+ind_agr[otr], fontsize = 30)
    #fig_str_ind_nach.savefig('data/'+'couTo-'+cou_agr[scen_cou_to]+'_'+'otr-'+str(otr)+'_'+'scen-'+str(scen_n)+'_'+'delta-'+str(scen_v)+'_'+'year-'+str(scen_year)+'_str_nach_ind.png')
    
    y_str_ind_after, x_str_ind_after, fig_str_ind_after = str_ind(otr, A_scen*res, Y_scen, G_new)
    fig_str_ind_after.title('Прогнозная структура мировой торговли продуктом '+ind_agr[otr], fontsize = 30)
    #fig_str_ind_after.savefig('data/'+'couTo-'+cou_agr[scen_cou_to]+'_'+'otr-'+str(otr)+'_'+'scen-'+str(scen_n)+'_'+'delta-'+str(scen_v)+'_'+'year-'+str(scen_year)+'_str_after_ind.png')
        
        #1.2 -  Начальная и конечная стр экспорта (пока тольк России/мб основной страны) def str_exp(cou_exp, otr, X, Y, G):
            #1.2.1 - Начальная и конечная стр экспорта России
    y_str_exp_nach, x_str_exp_nach, fig_str_exp_nach = str_exp(n_RUS, otr, X_scen, Y_scen, G_nach)
    fig_str_exp_nach.title('Начальная структура экспорта '+cou_agr[n_RUS], fontsize = 30)
    #fig_str_exp_nach.savefig('data/'+'couTo-'+cou_agr[scen_cou_to]+'_'+'otr-'+str(otr)+'_'+'scen-'+str(scen_n)+'_'+'delta-'+str(scen_v)+'_'+'year-'+str(scen_year)+'_str_nach_exp.png')
    
    y_str_exp_after, x_str_exp_after, fig_str_exp_after = str_exp(n_RUS, otr, A_scen*res, Y_scen, G_new)
    fig_str_exp_after.title('Прогнозная структура экспорта '+cou_agr[n_RUS], fontsize = 30)
    #fig_str_exp_after.savefig('data/'+'couTo-'+cou_agr[scen_cou_to]+'_'+'otr-'+str(otr)+'_'+'scen-'+str(scen_n)+'_'+'delta-'+str(scen_v)+'_'+'year-'+str(scen_year)+'_str_after_exp.png')
            #1.2.2 - Начальная и конечная стр экспорта замещающего региона
    y_str_exp_nach_zam = []
    x_str_exp_nach_zam = []
    y_str_exp_after_zam = []
    x_str_exp_after_zam = []
    if (scen_n == 4):
        y_str_exp_nach_zam, x_str_exp_nach_zam, fig_str_exp_nach_zam = str_exp(cou_zam, otr, X_scen, Y_scen, G_nach, list_scen_cou_zam)
        fig_str_exp_nach_zam.title('Начальная структура экспорта замещающего региона '+cou_agr[cou_zam], fontsize = 30)
        #fig_str_exp_nach_zam.savefig('data/'+'couTo-'+cou_agr[scen_cou_to]+'_'+'otr-'+str(otr)+'_'+'scen-'+str(scen_n)+'_'+'delta-'+str(scen_v)+'_'+'year-'+str(scen_year)+'_str_nach_exp_couZam.png')
    
        y_str_exp_after_zam, x_str_exp_after_zam, fig_str_exp_after_zam = str_exp(cou_zam, otr, A_scen*res, Y_scen, G_new, list_scen_cou_zam)
        fig_str_exp_after_zam.title('Прогнозная структура экспорта замещающего региона '+cou_agr[cou_zam], fontsize = 30)
        #fig_str_exp_after_zam.savefig('data/'+'couTo-'+cou_agr[scen_cou_to]+'_'+'otr-'+str(otr)+'_'+'scen-'+str(scen_n)+'_'+'delta-'+str(scen_v)+'_'+'year-'+str(scen_year)+'_str_after_exp_couZam.png')
        #1.3 - Структура импорта в регионах где что-то меняется: def str_imp(cou_to, otr, X, Y, G, list_cous = [-1])
            #1.3.1 - Структура импорта региона в котором происходят начальные изменения
    y_str_cou1_nach, x_str_cou1_nach, fig_str_cou1_nach = str_imp(scen_cou_to, otr, X_scen, Y_scen, G_nach, list_scen_cou_to)
    fig_str_cou1_nach.title('Начальная структура импорта региона '+(cou_agr[scen_cou_to] if (scen_cou_to >=0) else ''), fontsize = 30)
    #fig_str_cou1_nach.savefig('data/'+'couTo-'+cou_agr[scen_cou_to]+'_'+'otr-'+str(otr)+'_'+'scen-'+str(scen_n)+'_'+'delta-'+str(scen_v)+'_'+'year-'+str(scen_year)+'_str_after_cou1.png')

    y_str_cou1_after, x_str_cou1_after, fig_str_cou1_after = str_imp(scen_cou_to, otr, A_scen*res, Y_scen, G_new, list_scen_cou_to)
    fig_str_cou1_after.title('Прогнозная структура импорта региона '+(cou_agr[scen_cou_to] if (scen_cou_to >=0) else ''), fontsize = 30)
    #fig_str_cou1_after.savefig('data/'+'couTo-'+cou_agr[scen_cou_to]+'_'+'otr-'+str(otr)+'_'+'scen-'+str(scen_n)+'_'+'delta-'+str(scen_v)+'_'+'year-'+str(scen_year)+'_str_after_cou1.png')
            
            #1.3.2 - Структура импорта замещающего/альтернативного региона
    y_str_cou2_nach = []
    x_str_cou2_nach = []
    y_str_cou2_after = []
    x_str_cou2_after = []
    
    if (scen_n == 5):
        y_str_cou2_nach, x_str_cou2_nach, fig_str_cou2_nach = str_imp(scen_cou_zam, otr, X_scen, Y_scen, G_nach, list_scen_cou_zam)
        fig_str_cou2_nach.title('Начальная структура импорта замещающего региона '+(cou_agr[scen_cou_to] if (scen_cou_to >=0) else ''), fontsize = 30)
        #fig_str_cou2_nach.savefig('data/'+'couTo-'+cou_agr[scen_cou_to]+'_'+'otr-'+str(otr)+'_'+'scen-'+str(scen_n)+'_'+'delta-'+str(scen_v)+'_'+'year-'+str(scen_year)+'_str_after_cou2.png')

        y_str_cou2_after, x_str_cou2_after, fig_str_cou2_after = str_imp(scen_cou_zam, otr, A_scen*res, Y_scen, G_new, list_scen_cou_zam)
        fig_str_cou2_after.title('Прогнозная структура импорта замещающего региона '+(cou_agr[scen_cou_to] if (scen_cou_to >=0) else ''), fontsize = 30)
        #fig_str_cou2_after.savefig('data/'+'couTo-'+cou_agr[scen_cou_to]+'_'+'otr-'+str(otr)+'_'+'scen-'+str(scen_n)+'_'+'delta-'+str(scen_v)+'_'+'year-'+str(scen_year)+'_str_after_cou2.png')
        
        #1.4 - Изменения внутри Региона
             #1.4.1 - изменения в России
    delta = res - VIP_scen
    
    y_delta_cou_from_abs, x_delta_cou_from_abs, fig_delta_cou_from_abs = vnut_iz(n_RUS, delta)
    fig_delta_cou_from_abs.title('Абсолютное изменение выпусков '+cou_agr[n_RUS] , fontsize = 30)
    #fig_delta_cou_from_abs.savefig('data/'+'couTo-'+cou_agr[scen_cou_to]+'_'+'otr-'+str(otr)+'_'+'scen-'+str(scen_n)+'_'+'delta-'+str(scen_v)+'_'+'year-'+str(scen_year)+'_delta_couFrom_abs.png')

    delta_otn = (res - VIP_scen)/VIP_scen
    
    y_delta_cou_from_otn, x_delta_cou_from_otn, fig_delta_cou_from_otn = vnut_iz(n_RUS, delta_otn, '%')
    fig_delta_cou_from_otn.title('Относительное изменение выпусков '+ cou_agr[n_RUS] , fontsize = 30)
    #fig_delta_cou_from_otn.savefig('data/'+'couTo-'+cou_agr[scen_cou_to]+'_'+'otr-'+str(otr)+'_'+'scen-'+str(scen_n)+'_'+'delta-'+str(scen_v)+'_'+'year-'+str(scen_year)+'_delta_couFrom_otn.png')   
            #1.4.2 - изменения в замещающей стране
    y_delta_cou_zam_abs = []
    x_delta_cou_zam_abs = []
    y_delta_cou_zam_otn = []
    x_delta_cou_zam_otn = []
    if (scen_n == 4):
        y_delta_cou_zam_abs, x_delta_cou_zam_abs, fig_delta_cou_zam_abs = vnut_iz(scen_cou_zam, delta)
        fig_delta_cou_zam_abs.title('Абсолютное изменение выпусков '+cou_agr[scen_cou_zam] , fontsize = 30)
        #fig_delta_cou_from_abs.savefig('data/'+'couTo-'+cou_agr[scen_cou_to]+'_'+'otr-'+str(otr)+'_'+'scen-'+str(scen_n)+'_'+'delta-'+str(scen_v)+'_'+'year-'+str(scen_year)+'_delta_couZam_abs.png')

    
        y_delta_cou_zam_otn, x_delta_cou_zam_otn, fig_delta_cou_zam_otn = vnut_iz(scen_cou_zam, delta_otn, '%')
        fig_delta_cou_from_otn.title('Относительное изменение выпусков '+ cou_agr[scen_cou_zam] , fontsize = 30)
        #fig_delta_cou_zam_otn.savefig('data/'+'couTo-'+cou_agr[scen_cou_to]+'_'+'otr-'+str(otr)+'_'+'scen-'+str(scen_n)+'_'+'delta-'+str(scen_v)+'_'+'year-'+str(scen_year)+'_delta_couZam_otn.png')
        
        #1.5 - Главные бенефициары (def find_benef(VIP_0, VIP_1, A):)
    y_ben, x_ben, fig_ben = find_benef(VIP_scen, res, A_scen)
    #fig_ben.savefig('data/'+'couTo-'+cou_agr[scen_cou_to]+'_'+'otr-'+str(otr)+'_'+'scen-'+str(scen_n)+'_'+'delta-'+str(scen_v)+'_'+'year-'+str(scen_year)+'_benef.png')
        
        #1.6 - Основные изменения в мировой торговле
        
    y_top10, x_top10, fig_top10 = big_ch(VIP_scen, res, A_scen)
    #fig_top10.savefig('data/'+'couTo-'+cou_agr[scen_cou_to]+'_'+'otr-'+str(otr)+'_'+'scen-'+str(scen_n)+'_'+'delta-'+str(scen_v)+'_'+'year-'+str(scen_year)+'_top10.png')


    #2.Запись в файл
    
    filename = 'res_ras.txt' #"data/"+'couTo-'+(cou_agr[scen_cou_to] if (scen_cou_to>=0) else str(scen_cou_to))+'_'+'otr-'+str(otr)+'_'+'scen-'+str(scen_n)+'_'+'delta-'+str(scen_v)+'_'+'year-'+str(scen_year)+'results.txt'
    
    file = open(filename, "w")
    
    
    file.write("страна: "+ (cou_agr[scen_cou_to] if (scen_cou_to>=0) else str(scen_cou_to)) +'  отрасль: '+ str(otr) + '  сценарий: '+str(scen_n)+'  Изменение: '+ str(scen_v)
                + (' Замещающая страна:' + (cou_agr[scen_cou_zam] if (scen_cou_zam >= 0) else str(scen_cou_zam)) if (scen_n == 4) else '')+ 
                (' Альтернативный регион сбыта: ' + (cou_agr[scen_cou_zam] if (scen_cou_zam >= 0) else str(scen_cou_zam)) 
                + ' восстановление экспорта: ' +  str(val_zam)+'%' if (scen_n == 5) else '') +'%\n' )
    
    file.write('\n')
    
    #Начало
        #2.1 - Мировая структура по продукту до и после 
    file.write('#начальная структура мировой торговли продуктом '+ind_agr_rus[otr]+':\n')
    for i in range(len(x_str_ind_nach)):
        file.write(str(i+1)+'. '+x_str_ind_nach[-1-i]+': '+str(y_str_ind_nach[-1-i]) +'\n')
    
    file.write('\n')
         
    file.write('#прогнозная структура мировой торговли продуктом '+ind_agr_rus[otr]+':\n')
    for i in range(len(x_str_ind_after)):
        file.write(str(i+1)+'. '+x_str_ind_after[-1-i]+': '+str(y_str_ind_after[-1-i]) +'\n')
    
    file.write('\n')
    
        #2.2 - Начальная и конечная стр экспорта
    file.write('#начальная структура экспорта продукта '+ind_agr_rus[otr]+' '+cou_agr[n_RUS]+':\n')
    for i in range(len(x_str_exp_nach)):
        file.write(str(i+1)+'. '+x_str_exp_nach[-1-i]+': '+str(y_str_exp_nach[-1-i]) +'\n')
    
    file.write('\n')
    
    file.write('#прогнозная структура экспорта продукта '+ind_agr_rus[otr]+' '+cou_agr[n_RUS]+':\n')
    for i in range(len(x_str_exp_after)):
        file.write(str(i+1)+'. '+x_str_exp_after[-1-i]+': '+str(y_str_exp_after[-1-i]) +'\n')
        
    file.write('\n')
    
    if (scen_n == 4):
        file.write('#начальная структура экспорта продукта '+ind_agr_rus[otr]+' '+cou_agr[scen_cou_zam]+':\n')
        for i in range(len(x_str_exp_nach_zam)):
            file.write(str(i+1)+'. '+x_str_exp_nach_zam[-1-i]+': '+str(y_str_exp_nach_zam[-1-i]) +'\n')
    
        file.write('\n')
        
        file.write('#прогнозная структура экспорта продукта '+ind_agr_rus[otr]+' '+cou_agr[scen_cou_zam]+':\n')
        for i in range(len(x_str_exp_after_zam)):
            file.write(str(i+1)+'. '+x_str_exp_after_zam[-1-i]+': '+str(y_str_exp_after_zam[-1-i]) +'\n')
    
        file.write('\n')
        
        #2.3 - Структура импорта в регионах где что-то меняется: def str_imp(cou_to, otr, X, Y, G, list_cous = [-1])
    file.write('#Начальная структура импорта продукта '+ind_agr_rus[otr]+ ' в регионе ' + (cou_agr[scen_cou_to] if (scen_cou_to>=0) else 'Регион1')+ ':\n')
    for i in range(len(x_str_cou1_nach)):
        file.write(str(i+1)+'. '+x_str_cou1_nach[-1-i]+': '+str(y_str_cou1_nach[-1-i]) +'\n')
    
    file.write('\n')
    
    file.write('#Прогнозная структура импорта продукта '+ind_agr_rus[otr]+ ' в регионе ' + (cou_agr[scen_cou_to] if (scen_cou_to>=0) else 'Регион1')+ ':\n')
    for i in range(len(x_str_cou1_after)):
        file.write(str(i+1)+'. '+x_str_cou1_after[-1-i]+': '+str(y_str_cou1_after[-1-i]) +'\n')
        
    file.write('\n')
    
    if (scen_n == 5):
        file.write('#Начальная структура импорта продукта '+ind_agr_rus[otr]+ ' в регионе ' + (cou_agr[scen_cou_zam] if (scen_cou_zam>=0) else 'Регион2')+ ':\n')
        for i in range(len(x_str_cou2_nach)):
            file.write(str(i+1)+'. '+x_str_cou2_nach[-1-i]+': '+str(y_str_cou2_nach[-1-i]) +'\n')
    
        file.write('\n')
    
        file.write('#Прогнозная структура импорта продукта '+ind_agr_rus[otr]+ ' в регионе ' + (cou_agr[scen_cou_zam] if (scen_cou_zam>=0) else 'Регион2')+ ':\n')
        for i in range(len(x_str_cou2_after)):
            file.write(str(i+1)+'. '+x_str_cou2_after[-1-i]+': '+str(y_str_cou2_after[-1-i]) +'\n')
        
        file.write('\n')
        #2.4 - Изменения внутри Региона
    file.write('#Абсолютные изменения выпусков страны экспортёра '+cou_agr[n_RUS]+' \n')
    for i in range(len(x_delta_cou_from_abs)):
        file.write(str(i+1)+'. '+x_delta_cou_from_abs[-1-i]+': '+str(y_delta_cou_from_abs[-1-i]) +'\n')
    
    file.write('\n')
    
    file.write('#Относительные изменения выпусков страны экспортёра '+cou_agr[n_RUS]+'\n')
    for i in range(len(x_delta_cou_from_otn)):
        file.write(str(i+1)+'. '+x_delta_cou_from_otn[-1-i]+': '+str(y_delta_cou_from_otn[-1-i]) +'\n')
        
    file.write('\n')
               
    if (scen_n == 4):
        file.write('#Абсолютные изменения выпусков ' + cou_agr[scen_cou_zam] +'\n')
        for i in range(len(x_delta_cou_zam_abs)):
            file.write(str(i+1)+'. '+x_delta_cou_zam_abs[-1-i]+': '+str(y_delta_cou_zam_abs[-1-i]) +'\n')
    
        file.write('\n')
    
        file.write('#Относительные изменения выпусков ' + cou_agr[scen_cou_zam] +'\n')
        for i in range(len(x_delta_cou_zam_otn)):
            file.write(str(i+1)+'. '+x_delta_cou_zam_otn[-1-i]+': '+str(y_delta_cou_zam_otn[-1-i]) +'\n')
        
        file.write('\n')
        
        #2.5 - Главные бенефициары (def find_benef(VIP_0, VIP_1, A):)
    file.write('#Бенефициары изменений \n')
    for i in range(len(x_ben)):
        file.write(str(i+1)+'. '+x_ben[-1-i]+': '+str(y_ben[-1-i]) +'\n')      
    
    file.write('\n')
    
        #2.6 - Основные изменения в мировой торговле
    file.write('#Изменения ВВП\n')
    for i in range(len(y_top10)):
        file.write(str(i+1)+'. '+x_top10[-1-i]+': '+str(y_top10[-1-i]) +'\n')      

        #2.7 Запишем все новые выпуски  
    file.write('#Вектор прогнозируемых выпусков:\n')
    for i in range(len(res)):
        file.write(str(i+1)+'. '+cou_agr[i//35]+' - '+ind_agr[i%35]+': ' + str(res[i]) + '\n')
    
    file.write('\n') 
    file.close()
    
    if(scen_n == 4):
        return cou_ex,list_scen_cou_to, list_scen_cou_zam, cou_zam, scen_cou_to, X_scen, Y_scen, A_scen, VIP_scen, res, G_nach, G_new, y_str_ind_nach, x_str_ind_nach, fig_str_ind_nach, y_str_ind_after, x_str_ind_after, fig_str_ind_after, y_str_exp_nach, x_str_exp_nach, fig_str_exp_nach, y_str_exp_after, x_str_exp_after, fig_str_exp_after, y_str_exp_nach_zam, x_str_exp_nach_zam, fig_str_exp_nach_zam, y_str_exp_after_zam, x_str_exp_after_zam, fig_str_exp_after_zam, y_str_cou1_nach, x_str_cou1_nach, fig_str_cou1_nach, y_str_cou1_after, x_str_cou1_after, fig_str_cou1_after, y_delta_cou_from_abs, x_delta_cou_from_abs, fig_delta_cou_from_abs, y_delta_cou_from_otn, x_delta_cou_from_otn,  fig_delta_cou_from_otn, y_delta_cou_zam_abs, x_delta_cou_zam_abs,  fig_delta_cou_zam_abs, y_delta_cou_zam_otn, x_delta_cou_zam_otn,  fig_delta_cou_zam_otn, y_ben, x_ben,  fig_ben, y_top10, x_top10, fig_top10 

    if(scen_n == 5):
        return cou_ex,list_scen_cou_to, list_scen_cou_zam, cou_zam, scen_cou_to, X_scen, Y_scen, A_scen, VIP_scen, res, G_nach, G_new, y_str_ind_nach, x_str_ind_nach, fig_str_ind_nach, y_str_ind_after, x_str_ind_after, fig_str_ind_after, y_str_exp_nach, x_str_exp_nach, fig_str_exp_nach, y_str_exp_after, x_str_exp_after, fig_str_exp_after, y_str_exp_nach_zam, y_str_cou1_nach, x_str_cou1_nach, fig_str_cou1_nach, y_str_cou1_after, x_str_cou1_after, fig_str_cou1_after, y_str_cou2_nach, x_str_cou2_nach, fig_str_cou2_nach, y_str_cou2_after,  x_str_cou2_after, fig_str_cou2_after, y_delta_cou_from_abs, x_delta_cou_from_abs, fig_delta_cou_from_abs, y_delta_cou_from_otn, x_delta_cou_from_otn,  fig_delta_cou_from_otn, y_ben, x_ben,  fig_ben, y_top10, x_top10, fig_top10 
    
    return cou_ex,list_scen_cou_to, list_scen_cou_zam, cou_zam, scen_cou_to, X_scen, Y_scen, A_scen, VIP_scen, res, G_nach, G_new, y_str_ind_nach, x_str_ind_nach, fig_str_ind_nach, y_str_ind_after, x_str_ind_after, fig_str_ind_after, y_str_exp_nach, x_str_exp_nach, fig_str_exp_nach, y_str_exp_after, x_str_exp_after, fig_str_exp_after, y_str_cou1_nach, x_str_cou1_nach, fig_str_cou1_nach, y_str_cou1_after, x_str_cou1_after, fig_str_cou1_after, y_delta_cou_from_abs, x_delta_cou_from_abs, fig_delta_cou_from_abs, y_delta_cou_from_otn, x_delta_cou_from_otn,  fig_delta_cou_from_otn, y_ben, x_ben, fig_ben, y_top10, x_top10, fig_top10


FLAG = 0


# Add a selectbox to the sidebar:
st.title("Приложение - калькулятор сдвигов в мировой торговле")

with st.expander("Описание модели"):
    st.write("Вычисления проводятся на основе модели - MRIO. Модель MRIO является обобщением стандартной модели МОБ на $m$ регионов.\n")
    st.write("Основопологающим понятием модели MRIO являются торговые коэффициенты $g^{rs}_i$, которые отражают долю продукта $i$ из региона $r$ в общем (сумма промежуточного и конечного) потреблении региона $s$. Для торговых коэффициентов выполняются следующие соотношения:")
    st.latex(r'''\sum\limits_{r=1}^m g^{rs}_i = 1''')	
    st.write("Вычисляются торговые коэффициенты по следующей формуле:")
    st.latex(r'''g^{rs}_i = z^{rs}_i / z^s_i \\''')
	
    st.write("* $z^{rs}_i$ - объем продукции $i$-ой отрасли, поставляемый из региона $r$ в регион $s$ для промежуточного и конечного потребления.")
			
    st.write("* $z^{s}_i$ - весь объем продукции $i$-ой отрасли, используемый в регионе $s$ для промежуточного и конечного потребления.")

    st.write("Основное уравнение модели:") 
    st.latex(r'''X = G\times A\times X + G\times \bar Y''')
    st.write(", где $G$ - блочно-диагональная матрица торговых коэффициентов.")

cou_ex_name = st.selectbox(
    'Выберете страну экспортёра:',
    ('ARG',
    'AUS',
    'AUT',
    'BEL',
    'BGR',
    'BLR',
    'BRA',
    'CAN',
    'CHE',
    'CHN',
    'CIV',
    'CMR',
    'CYP',
    'CZE',
    'DEU',
    'DNK',
    'EGY',
    'ESP',
    'EST',
    'FIN',
    'FRA',
    'GBR',
    'GRC',
    'HRV',
    'HUN',
    'IDN',
    'IND',
    'IRL',
    'ISL',
    'ISR',
    'ITA',
    'JPN',
    'KAZ',
    'KOR',
    'LTU',
    'LUX',
    'LVA',
    'MAR',
    'MEX',
    'MLT',
    'MYS',
    'NGA',
    'NLD',
    'NOR',
    'NZL',
    'PER',
    'PHL',
    'POL',
    'PRT',
    'ROU',
    'RUS',
    'SAU',
    'SVK',
    'SVN',
    'SWE',
    'THA',
    'TUN',
    'TUR',
    'TWN',
    'UKR',
    'USA',
    'VNM',
    'ZAF',
    'ROW'), (50)
)

otr_name = st.selectbox(
    'Выберете отрасль:',
    ('A01_02, Сельское хозяйство, охота, лесное хозяйство', 
     'B05_06, Горнодобывающая промышленность, продукция для производства энергии',
     'B07_08, Горнодобывающая промышленность, неэнергетическая продукция',
     'B09, Деятельность по поддержке горнодобывающей промышленности',
     'C10T12, Пищевые продукты, напитки и табачные изделия',
     'C13T15, Текстиль, текстильные изделия, кожа и обувь',
     'C16, Древесина и изделия из дерева и пробки',
     'C19, Кокс и продукты нефтепереработки',
     'C20, Химия и химическая продукция',
     'C21, Фармацевтические препараты, лекарственные химические и растительные продукты',
     'C22, Резиновые и пластмассовые изделия',
     'C23, Прочие неметаллические минеральные продукты',
     'C24, Основные металлы',
     'C25, Готовые металлические изделия',
     'C26, Компьютерное, электронное и оптическое оборудование',
     'C27, Электрическое оборудование',
     'C28, Машины и оборудование, не включенные в другие категории',
     'C29, Автомобили, прицепы и полуприцепы',
     'C30, Другое транспортное оборудование',
     'C31T33, Производство, не включенное в другие категории; ремонт и монтаж машин и оборудования',
     'D, Электроснабжение, газ, пар и кондиционирование воздуха',
     'F, Строительство',
     'G, Оптовая и розничная торговля; ремонт автомобилей',
     'H49, Наземный транспорт и транспорт по трубопроводам',
     'H50, Водный транспорт',
     'H51, Воздушный транспорт',
     'H52, Складирование и вспомогательная деятельность по транспортировке',
     'I, Деятельность по размещению и питанию',
     'J61, Телекоммуникации',
     'J62_63, IT и другие информационные услуги',
     'K, Финансовая и страховая деятельность',
     'L, Деятельность в сфере недвижимости',
     'M, Профессиональная, научная и техническая деятельность',
     'O, Государственное управление и оборона; обязательное социальное обеспечение',
     'other')
)


# Add a slider to the sidebar:
#add_slider = st.sidebar.slider(
#    'Select a range of values',
#    0.0, 100.0, (25.0, 75.0)
#)
col1, col2 = st.columns([2,1])
u = 0
v = 0
#with col1:
#    u = st.slider("Изменение экспорта (страны экспортёра) в %:", -100, 1000, (v))
with col1:
    #st.number_input(label, min_value=None, max_value=None, value="min", step=None, format=None, key=None, help=None, on_change=None, args=None, kwargs=None, *, placeholder=None, disabled=False, label_visibility="visible")
    v = st.number_input(label = "Изменение экспорта (страны экспортёра) в %:", min_value=-100, max_value=10000, value = u, label_visibility="visible")
if (u != v):
    u = v
st.write("Изменение на " + str(v)+ "%")
scen_val = v


cou_to_name = st.selectbox(
    'Выберете страну импортёра:',
    ('EU',
    'CHN+IND',
    'G7',
    'G20',
    'BRICS',
    'ARG',
    'AUS',
    'AUT',
    'BEL',
    'BGR',
    'BLR',
    'BRA',
    'CAN',
    'CHE',
    'CHN',
    'CIV',
    'CMR',
    'CYP',
    'CZE',
    'DEU',
    'DNK',
    'EGY',
    'ESP',
    'EST',
    'FIN',
    'FRA',
    'GBR',
    'GRC',
    'HRV',
    'HUN',
    'IDN',
    'IND',
    'IRL',
    'ISL',
    'ISR',
    'ITA',
    'JPN',
    'KAZ',
    'KOR',
    'LTU',
    'LUX',
    'LVA',
    'MAR',
    'MEX',
    'MLT',
    'MYS',
    'NGA',
    'NLD',
    'NOR',
    'NZL',
    'PER',
    'PHL',
    'POL',
    'PRT',
    'ROU',
    'RUS',
    'SAU',
    'SVK',
    'SVN',
    'SWE',
    'THA',
    'TUN',
    'TUR',
    'TWN',
    'UKR',
    'USA',
    'VNM',
    'ZAF',
    'ROW')
)




scen_name = st.selectbox(
    'Сценарий:',
    ('Изменение за счёт внутреннего производства импортёра в данном регионе',
    'Изменение за счёт пропорционального изменения экспорта всего мира',
    'Изменение за счёт изменения экспорта конкретного (конкурирующего) региона',
    'Переориентации на альтернативный регион')
)



with st.expander("Подробнее о сценариях"):
    st.subheader("Изменение за счёт внутреннего производства импортёра в данном регионе:")
    st.write("Торговый коэффициент* региона** импортёра, отвечающий за внутренние поставки, изменятся на то значение, на которое меняется торговый коэффициент экспортёра.")
    st.subheader("Пример:")
    st.write("Увеличение торгового коэффициента, отвечающего за поставки энергоресурсов из России в Германию, на $\Delta g$ приведёт к уменьшению торгового коэффициента, отвечающего за поставки энергорессурсов из Германии в Германию (внутреннее производство), на $\Delta g$.")
    
    st.subheader("Изменение за счёт пропорционального изменения экспорта всего мира:")
    st.write("Торговые коэффициенты всех экспортёров (помимо того, за счёт которого происходят все изменения), отвечающие за поставки в данный регион импортёр, изменяются пропорционально их начальным значениям так, чтоб сбалансировать изменение торгового коэффициента страны экспортёра.")
    st.subheader("Пример:")
    st.write("Увеличение торгового коэффициента, отвечающего за поставки энергоресурсов из России в Германию, на $\Delta g$ приведёт к пропорциональному (относительно начальных значений) уменьшению торговых коэффициентов остальных регионов-экспортёров в Германию, отвечающих за поставки энергорессурсов из регионов-экспортёров в Германию, так, чтоб сумма всех изменений торговых коэффициентов остальных регионах-экспортёров была равна $-\Delta g$.")
    

    st.subheader("Изменение за счёт изменения экспорта конкретного (конкурирующего) региона:")
    st.write("Торговый коэффициент региона конкурента, отвечающий за поставки в регион импортёр, изменяются на то значение, на которое меняется торговый коэффициент экспортёра.")
    st.subheader("Пример:")
    st.write("Увеличение торгового коэффициента, отвечающего за поставки энергоресурсов из России в Германию, на $\Delta g$ приведёт к уменьшению торгового коэффициента, отвечающего за поставки энергорессурсов из региона-конкурента в Германию, на $\Delta g$.")

    st.subheader("Переориентации на альтернативный регион:")
    st.write("Торговые коэффициенты всех экспортёров (помимо того, за счёт которого происходят все изменения), отвечающие за поставки в данный регион импортёр, изменяются пропорционально их начальным значениям так, чтоб сбалансировать изменение торгового коэффициента страны экспортёра.\n")
    st.write("Вычисляется абсолютное изменение в торговле между регионом экспортёром и регионом импортёром, доля этого изменения (зависит от процента замещения альтернативным регионом) перекладывается на альтернативный регион.\n")
    st.write("Торговые коэффициенты всех экспортёров (помимо того, за счёт которого происходят все изменения), отвечающие за поставки в альтернативный регион, изменяются пропорционально их начальным значениям так, чтоб сбалансировать изменение торгового коэффициента страны экспортёра.")
    st.subheader("Пример:")
    st.write("Уменьшение торгового коэффициента, отвечающего за поставки энергоресурсов из России в Германию с переориентацией",
    "на алтернативный регион - Китай, с процентом замещения - 75% на $\Delta g_1$ приведёт к пропорциональному (относительно начальных",
    "значений) увелиичению торговых коэффициентов остальных регионов-экспортёров в Германию, отвечающих за поставки энергорессурсов из ",
    "регионов-экспортёров в Германию, так, чтоб сумма всех изменений торговых коэффициентов остальных регионов-экспортёров была равна", 
    "$-\Delta g_1$.\n Будет высчитано изменение экспорта энергоресурсов из России в Германию - $\Delta I$, на основе которого будет расчитано увеличение торгового коэффициента",
    "$\Delta g_2$, отвечающего за поставки энергоресурсов из России В Китай, а торговые коэффициенты остальных регионов-экспортёров",
    "в Китай, отвечающие за поставки энергоресурсов из регонов-экспортёров в Китай, будут пропорционально (относительно начальных значений)",
    "уменьшены так, чтоб сумма изменений торговых коэффициентов остальных регионов-экспортёров равнялась $-\Delta g_2$")



    st.divider()
    st.write("\*подробнее о торговых коэффициентах в описании модели")
    st.write("\*\* здесь регион - страна или группа стран (например: DEU - Германия, EU - страны входящие в Европейский союз).")
if (scen_name == 'Изменение за счёт изменения экспорта конкретного (конкурирующего) региона'):
    cou_zam_name = st.selectbox(
    'Конкурирующий регион:',
    ('EU',
    'CHN+IND',
    'G7',
    'G20',
    'BRICS',
    'ARG',
    'AUS',
    'AUT',
    'BEL',
    'BGR',
    'BLR',
    'BRA',
    'CAN',
    'CHE',
    'CHN',
    'CIV',
    'CMR',
    'CYP',
    'CZE',
    'DEU',
    'DNK',
    'EGY',
    'ESP',
    'EST',
    'FIN',
    'FRA',
    'GBR',
    'GRC',
    'HRV',
    'HUN',
    'IDN',
    'IND',
    'IRL',
    'ISL',
    'ISR',
    'ITA',
    'JPN',
    'KAZ',
    'KOR',
    'LTU',
    'LUX',
    'LVA',
    'MAR',
    'MEX',
    'MLT',
    'MYS',
    'NGA',
    'NLD',
    'NOR',
    'NZL',
    'PER',
    'PHL',
    'POL',
    'PRT',
    'ROU',
    'RUS',
    'SAU',
    'SVK',
    'SVN',
    'SWE',
    'THA',
    'TUN',
    'TUR',
    'TWN',
    'UKR',
    'USA',
    'VNM',
    'ZAF',
    'ROW')
)

if (scen_name == 'Переориентации на альтернативный регион'):
    cou_zam_name = st.selectbox(
    'Альтернативный регион:',
    ('EU',
    'CHN+IND',
    'G7',
    'G20',
    'BRICS',
    'ARG',
    'AUS',
    'AUT',
    'BEL',
    'BGR',
    'BLR',
    'BRA',
    'CAN',
    'CHE',
    'CHN',
    'CIV',
    'CMR',
    'CYP',
    'CZE',
    'DEU',
    'DNK',
    'EGY',
    'ESP',
    'EST',
    'FIN',
    'FRA',
    'GBR',
    'GRC',
    'HRV',
    'HUN',
    'IDN',
    'IND',
    'IRL',
    'ISL',
    'ISR',
    'ITA',
    'JPN',
    'KAZ',
    'KOR',
    'LTU',
    'LUX',
    'LVA',
    'MAR',
    'MEX',
    'MLT',
    'MYS',
    'NGA',
    'NLD',
    'NOR',
    'NZL',
    'PER',
    'PHL',
    'POL',
    'PRT',
    'ROU',
    'RUS',
    'SAU',
    'SVK',
    'SVN',
    'SWE',
    'THA',
    'TUN',
    'TUR',
    'TWN',
    'UKR',
    'USA',
    'VNM',
    'ZAF',
    'ROW')
)

    col1, col2 = st.columns([2,1])
    u = 75
    v = 75
    #with col1:
    #    u = st.slider("Процент замещения альтернативным регионом:", 0, 100, (v))
    with col1:
        v = st.number_input(label = "Процент замещения альтернативным регионом:", min_value=0, max_value=100, value = u, label_visibility="visible")

    val_zam = v
    st.write("Процент замещения: " + str(val_zam)+ "%")


year = st.selectbox(
    'Расчётный год:', ('2019', '2020')
)

flag_but = st.button("Выполнить расчёты", type="primary")

#словари

cou_slov = {'EU': -10,
    'CHN+IND': -20,
    'G7': -30,
    'G20': -40,
    'BRICS': -50,
    'ARG':0,
    'AUS':1,
    'AUT':2,
    'BEL':3,
    'BGR':4,
    'BLR':5,
    'BRA':6,
    'CAN':7,
    'CHE':8,
    'CHN':9,
    'CIV':10,
    'CMR':11,
    'CYP':12,
    'CZE':13,
    'DEU':14,
    'DNK':15,
    'EGY':16,
    'ESP':17,
    'EST':18,
    'FIN':19,
    'FRA':20,
    'GBR':21,
    'GRC':22,
    'HRV':23,
    'HUN':24,
    'IDN':25,
    'IND':26,
    'IRL':27,
    'ISL':28,
    'ISR':29,
    'ITA':30,
    'JPN':31,
    'KAZ':32,
    'KOR':33,
    'LTU':34,
    'LUX':35,
    'LVA':36,
    'MAR':37,
    'MEX':38,
    'MLT':39,
    'MYS':40,
    'NGA':41,
    'NLD':42,
    'NOR':43,
    'NZL':44,
    'PER':45,
    'PHL':46,
    'POL':47,
    'PRT':48,
    'ROU':49,
    'RUS':50,
    'SAU':51,
    'SVK':52,
    'SVN':53,
    'SWE':54,
    'THA':55,
    'TUN':56,
    'TUR':57,
    'TWN':58,
    'UKR':59,
    'USA':60,
    'VNM':61,
    'ZAF':62,
    'ROW':63}

otr_slov = {
    'A01_02, Сельское хозяйство, охота, лесное хозяйство':0, 
     'B05_06, Горнодобывающая промышленность, продукция для производства энергии':1,
     'B07_08, Горнодобывающая промышленность, неэнергетическая продукция':2,
     'B09, Деятельность по поддержке горнодобывающей промышленности':3,
     'C10T12, Пищевые продукты, напитки и табачные изделия':4,
     'C13T15, Текстиль, текстильные изделия, кожа и обувь':5,
     'C16, Древесина и изделия из дерева и пробки':6,
     'C19, Кокс и продукты нефтепереработки':7,
     'C20, Химия и химическая продукция':8,
     'C21, Фармацевтические препараты, лекарственные химические и растительные продукты':9,
     'C22, Резиновые и пластмассовые изделия':10,
     'C23, Прочие неметаллические минеральные продукты':11,
     'C24, Основные металлы':12,
     'C25, Готовые металлические изделия':13,
     'C26, Компьютерное, электронное и оптическое оборудование':14,
     'C27, Электрическое оборудование':15,
     'C28, Машины и оборудование, не включенные в другие категории':16,
     'C29, Автомобили, прицепы и полуприцепы':17,
     'C30, Другое транспортное оборудование':18,
     'C31T33, Производство, не включенное в другие категории; ремонт и монтаж машин и оборудования':19,
     'D, Электроснабжение, газ, пар и кондиционирование воздуха':20,
     'F, Строительство':21,
     'G, Оптовая и розничная торговля; ремонт автомобилей':22,
     'H49, Наземный транспорт и транспорт по трубопроводам':23,
     'H50, Водный транспорт':24,
     'H51, Воздушный транспорт':25,
     'H52, Складирование и вспомогательная деятельность по транспортировке':26,
     'I, Деятельность по размещению и питанию':27,
     'J61, Телекоммуникации':28,
     'J62_63, IT и другие информационные услуги':29,
     'K, Финансовая и страховая деятельность':30,
     'L, Деятельность в сфере недвижимости':31,
     'M, Профессиональная, научная и техническая деятельность':32,
     'O, Государственное управление и оборона; обязательное социальное обеспечение':33,
     'other':34
}

year_slov = {'2019': 1, '2020': 2}

scen_slov = {
    'Изменение за счёт внутреннего производства импортёра в данном регионе': 1,
     'Изменение за счёт пропорционального изменения экспорта всего мира': 3,
     'Изменение за счёт изменения экспорта конкретного (конкурирующего) региона': 4,
     'Переориентации на альтернативный регион': 5
}


if flag_but:
    FLAG = FLAG + 1




if FLAG > 0:
    cou_ex = cou_slov[cou_ex_name]
    otr = otr_slov[otr_name]
    scen_v = scen_val
    cou_to = cou_slov[cou_to_name]
    scen_n = scen_slov[scen_name]
    scen_cou_zam = cou_slov[cou_zam_name] if (scen_n == 5 or scen_n == 4) else -1
    zam_v = val_zam if scen_n == 5 else 0
    scen_year = year_slov[year]

    number_reg = len(cou_agr)

    #st.write(' otr: ', otr, ' scen_v: ', scen_v, ' cou_to: ', cou_to, ' scen_n: ', scen_n, ' scen_cou_zam: ', scen_cou_zam, ' zam_v: ', zam_v, ' scen_year: ', scen_year)

    n_RUS = cou_ex


    #применяем функции (разные возвраты при разных сценариях (вообще можно поубавить переменных, которые возвращаем))
    if scen_n == 4:
        cou_ex, list_scen_cou_to, list_scen_cou_zam, cou_zam, scen_cou_to, X_scen, Y_scen, A_scen, VIP_scen, res, G_nach, G_new, y_str_ind_nach, x_str_ind_nach, fig_str_ind_nach, y_str_ind_after, x_str_ind_after, fig_str_ind_after, y_str_exp_nach, x_str_exp_nach, fig_str_exp_nach, y_str_exp_after, x_str_exp_after, fig_str_exp_after, y_str_exp_nach_zam, x_str_exp_nach_zam, fig_str_exp_nach_zam, y_str_exp_after_zam, x_str_exp_after_zam, fig_str_exp_after_zam, y_str_cou1_nach, x_str_cou1_nach, fig_str_cou1_nach, y_str_cou1_after, x_str_cou1_after, fig_str_cou1_after, y_delta_cou_from_abs, x_delta_cou_from_abs, fig_delta_cou_from_abs, y_delta_cou_from_otn, x_delta_cou_from_otn,  fig_delta_cou_from_otn, y_delta_cou_zam_abs, x_delta_cou_zam_abs,  fig_delta_cou_zam_abs, y_delta_cou_zam_otn, x_delta_cou_zam_otn,  fig_delta_cou_zam_otn, y_ben, x_ben,  fig_ben, y_top10, x_top10, fig_top10 = prognoz2(cou_ex, otr, scen_v, cou_to, scen_year, scen_n, scen_cou_zam, zam_v)
        
    elif (scen_n == 5):
        cou_ex, list_scen_cou_to, list_scen_cou_zam, cou_zam, scen_cou_to, X_scen, Y_scen, A_scen, VIP_scen, res, G_nach, G_new, y_str_ind_nach, x_str_ind_nach, fig_str_ind_nach, y_str_ind_after, x_str_ind_after, fig_str_ind_after, y_str_exp_nach, x_str_exp_nach, fig_str_exp_nach, y_str_exp_after, x_str_exp_after, fig_str_exp_after, y_str_exp_nach_zam, y_str_cou1_nach, x_str_cou1_nach, fig_str_cou1_nach, y_str_cou1_after, x_str_cou1_after, fig_str_cou1_after, y_str_cou2_nach, x_str_cou2_nach, fig_str_cou2_nach, y_str_cou2_after,  x_str_cou2_after, fig_str_cou2_after, y_delta_cou_from_abs, x_delta_cou_from_abs, fig_delta_cou_from_abs, y_delta_cou_from_otn, x_delta_cou_from_otn,  fig_delta_cou_from_otn, y_ben, x_ben,  fig_ben, y_top10, x_top10, fig_top10 = prognoz2(cou_ex, otr, scen_v, cou_to, scen_year, scen_n, scen_cou_zam, zam_v)
        
    else:
        cou_ex, list_scen_cou_to, list_scen_cou_zam, cou_zam, scen_cou_to, X_scen, Y_scen, A_scen, VIP_scen, res, G_nach, G_new, y_str_ind_nach, x_str_ind_nach, fig_str_ind_nach, y_str_ind_after, x_str_ind_after, fig_str_ind_after, y_str_exp_nach, x_str_exp_nach, fig_str_exp_nach, y_str_exp_after, x_str_exp_after, fig_str_exp_after, y_str_cou1_nach, x_str_cou1_nach, fig_str_cou1_nach, y_str_cou1_after, x_str_cou1_after, fig_str_cou1_after, y_delta_cou_from_abs, x_delta_cou_from_abs, fig_delta_cou_from_abs, y_delta_cou_from_otn, x_delta_cou_from_otn,  fig_delta_cou_from_otn, y_ben, x_ben, fig_ben, y_top10, x_top10, fig_top10 = prognoz2(cou_ex,otr, scen_v, cou_to, scen_year, scen_n, scen_cou_zam, zam_v)

    st.write("sum G_nach = ", np.sum(G_nach), "sum G_new = ", np.sum(G_new))
        
    #0.Метрики
        #0.1 Сравнение торг коэф и экспорта:
    exp_rus_to_cou1 = round(find_exp(n_RUS, scen_cou_to, otr, A_scen*res, Y_scen, G_new, list_scen_cou_to))
    exp_rus_to_cou1_old = round(find_exp(n_RUS, scen_cou_to, otr, A_scen*VIP_scen, Y_scen, G_nach, list_scen_cou_to))


    g_rus = G_new[n_RUS*35 + otr][list_scen_cou_to[0]*35 + otr]
    g_rus_old = G_nach[n_RUS*35 + otr][list_scen_cou_to[0]*35 + otr]

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Торг. коэфф. "+cou_ex_name+"->"+cou_to_name, value= round(g_rus, 4), delta= '{:.1%}'.format(g_rus/g_rus_old - 1))
    with col2:
        st.metric(label="Эксп. "+ind_agr[otr]+" "+cou_ex_name+"->"+cou_to_name, value='{:1.0f} млн.$'.format(exp_rus_to_cou1), delta= '{:.1%} {:1.0f} млн. $'.format(exp_rus_to_cou1/exp_rus_to_cou1_old - 1, exp_rus_to_cou1 - exp_rus_to_cou1_old) )

    if (scen_n == 5):
        exp_rus_to_cou2 = round(find_exp(n_RUS, scen_cou_zam, otr, A_scen*res, Y_scen, G_new, list_scen_cou_zam))
        exp_rus_to_cou2_old = round(find_exp(n_RUS, scen_cou_zam, otr, A_scen*VIP_scen, Y_scen, G_nach, list_scen_cou_zam))

        g_rus = G_new[n_RUS*35 + otr][list_scen_cou_zam[0]*35 + otr]
        g_rus_old = G_nach[n_RUS*35 + otr][list_scen_cou_zam[0]*35 + otr]

        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Торг. коэфф. "+cou_ex_name+"->"+cou_zam_name, value= round(g_rus, 4), delta= '{:.1%}'.format(g_rus/g_rus_old - 1))
        with col2:
            st.metric(label="Эксп. "+ind_agr[otr]+" "+cou_ex_name+"->"+cou_zam_name, value='{:1.0f} млн.$'.format(exp_rus_to_cou2), delta= '{:.1%} {:1.0f} млн. $'.format(exp_rus_to_cou2/exp_rus_to_cou2_old - 1, exp_rus_to_cou2 - exp_rus_to_cou2_old) )

        #0.2 Изменение основных ВВП (мир, Россия, (конкурент, если сцен 4) )
    vvp_w_1, list_vvp_1 = find_vvp(-1, res, A_scen, np.arange(number_reg).tolist())
    vvp_w_0, list_vvp_0 = find_vvp(-1, VIP_scen, A_scen, np.arange(number_reg).tolist())

    vvp_rus_1 = round(find_vvp(n_RUS , res, A_scen)/1000, 1)
    vvp_rus_0 = round(find_vvp(n_RUS, VIP_scen, A_scen)/1000, 1)

    if (scen_n == 4):
        vvp_k_1 = round(find_vvp(cou_zam , res, A_scen)/1000, 1)# конкурент
        vvp_k_0 = round( find_vvp(cou_zam, VIP_scen, A_scen)/1000, 1 )

        col1, col2, col3 = st.columns(3)

        with col1:  
            st.metric(label="Мировой ВВП'", value= '{:1.1f} млрд. $'.format(vvp_w_1/1000), delta= '{:.2%} {:1.1f} млрд. $'.format(round(vvp_w_1/vvp_w_0 - 1,1), round(vvp_w_1/1000 - vvp_w_0/1000,1)))

        with col2:
            st.metric(label="ВВП " + cou_agr[n_RUS], value = '{:1.1f} млрд. $'.format(vvp_rus_1), delta =  '{:.2%} {:1.1f} млрд. $'.format(vvp_rus_1/vvp_rus_0 - 1, vvp_rus_1 - vvp_rus_0) )

        with col3:
            if (cou_zam<0):# пока это исключаем (сами сценарии не прописаны)
                #st.write('HI1')
                vvp_1, list_vvp_1 = find_vvp(-1, res, A_scen, np.arange(number_reg).tolist())
                vvp_0, list_vvp_0 = find_vvp(-1, VIP_scen, A_scen, np.arange(number_reg).tolist())
                st.metric(label="ввп " + cou_zam_name+', млрд. $', value=round(vvp_1/1000,1), delta= round(vvp_1/1000 - vvp_0/1000, 1))  
            else:
                #st.write('HI2')
                st.metric(label="ВВП " + cou_agr[cou_zam], value = '{:1.1f} млрд. $'.format(vvp_k_1), delta =  '{:.2%} {:1.1f} млрд. $'.format(vvp_k_1/vvp_k_0 - 1, vvp_k_1 - vvp_k_0) )

    else: #не 4 сцен
        col1, col2 = st.columns(2)

        with col1:  
            st.metric(label="Мировой ВВП", value= '{:1.1f} млрд. $'.format(vvp_w_1/1000), delta= '{:.2%} {:1.1f} млрд. $'.format(round(vvp_w_1/vvp_w_0 - 1,1), round(vvp_w_1/1000 - vvp_w_0/1000,1)))

        with col2:
            st.metric(label="ВВП " + cou_agr[n_RUS], value = '{:1.1f} млрд. $'.format(vvp_rus_1), delta =  '{:.2%} {:1.1f} млрд. $'.format(vvp_rus_1/vvp_rus_0 - 1, vvp_rus_1 - vvp_rus_0) )

        #0.3 ВВП различных альянсов
    with st.expander("ВВП различных экономических альянсов"):
        vvp_eu_1, list_vvp_eu_1 = find_vvp(-1, res, A_scen, list_eu)
        vvp_eu_0, list_vvp_eu_0 = find_vvp(-1, VIP_scen, A_scen, list_eu)

        vvp_G7_1, list_vvp_G7_1 = find_vvp(-1, res, A_scen, list_G7)
        vvp_G7_0, list_vvp_G7_0 = find_vvp(-1, VIP_scen, A_scen, list_G7)

        vvp_G20_1, list_vvp_G20_1 = find_vvp(-1, res, A_scen, list_G20)
        vvp_G20_0, list_vvp_G20_0 = find_vvp(-1, VIP_scen, A_scen, list_G20)

        vvp_BRICS_1, list_vvp_BRICS_1 = find_vvp(-1, res, A_scen, list_BRICS_)
        vvp_BRICS_0, list_vvp_BRICS_0 = find_vvp(-1, VIP_scen, A_scen, list_BRICS_)

        vvp_C_I_1, list_vvp_C_I_1 = find_vvp(-1, res, A_scen, list_CHN_IND)
        vvp_C_I_0, list_vvp_C_I_0 = find_vvp(-1, VIP_scen, A_scen, list_CHN_IND)


        #col1, col2, col3 = st.columns(3)
        

        #with col1:
        st.metric('ВВП EU', value = '{:1.1f} млрд. $'.format(vvp_eu_1/1000), delta = '{:.2%} {:1.1f} млрд. $'.format(vvp_eu_1/vvp_eu_0 - 1,1, vvp_eu_1/1000 - vvp_eu_0/1000 ) )

        col1, col2 = st.columns(2)     

        with col1:
            st.metric('ВВП G7', value = '{:1.1f} млрд. $'.format(vvp_G7_1/1000), delta = '{:.2%} {:1.1f} млрд. $'.format(vvp_G7_1/vvp_G7_0 - 1,1, vvp_G7_1/1000 - vvp_G7_0/1000 ) )

        with col2:
            st.metric('ВВП G20', value = '{:1.1f} млрд. $'.format(vvp_G20_1/1000), delta = '{:.2%} {:1.1f} млрд. $'.format(vvp_G20_1/vvp_G20_0 - 1,1, vvp_G20_1/1000 - vvp_G20_0/1000 ) )

        col1, col2 = st.columns(2)
        with col1:
            st.metric('ВВП BRICS(*)', value = '{:1.1f} млрд. $'.format(vvp_BRICS_1/1000), delta = '{:.2%} {:1.1f} млрд. $'.format(vvp_BRICS_1/vvp_BRICS_0 - 1,1, vvp_BRICS_1/1000 - vvp_BRICS_0/1000 ) )
                      
        with col2:
            st.metric('ВВП CHN и IND', value = '{:1.1f} млрд. $'.format(vvp_C_I_1/1000), delta = '{:.2%} {:1.1f} млрд. $'.format(vvp_C_I_1/vvp_C_I_0 - 1,1, vvp_C_I_1/1000 - vvp_C_I_0/1000 ) )

    #1.Графики
    with st.expander("Мировая структура по продукту до и после"):
        col1, col2 = st.columns(2)

        with col1:
                #1.1 - Мировая структура по продукту до и после (def str_ind(otr, X, Y, G))
            y_str_ind_nach, x_str_ind_nach, fig_str_ind_nach = str_ind(otr, X_scen, Y_scen, G_nach)
            fig_str_ind_nach.title('Начальная структура мировой торговли продуктом '+ind_agr[otr], fontsize = 30)
            #fig_str_ind_nach.savefig('data/'+'couTo-'+cou_agr[scen_cou_to]+'_'+'otr-'+str(otr)+'_'+'scen-'+str(scen_n)+'_'+'delta-'+str(scen_v)+'_'+'year-'+str(scen_year)+'_str_nach_ind.png')
            st.pyplot(fig_str_ind_nach)

        with col2:
            y_str_ind_after, x_str_ind_after, fig_str_ind_after = str_ind(otr, A_scen*res, Y_scen, G_new)
            fig_str_ind_after.title('Прогнозная структура мировой торговли продуктом '+ind_agr[otr], fontsize = 30)
            #fig_str_ind_after.savefig('data/'+'couTo-'+cou_agr[scen_cou_to]+'_'+'otr-'+str(otr)+'_'+'scen-'+str(scen_n)+'_'+'delta-'+str(scen_v)+'_'+'year-'+str(scen_year)+'_str_after_ind.png')
            st.pyplot(fig_str_ind_after)

        st.divider()
        st.write("Здесь и далее others - сумма показателей стран не вошедших отдельно в график (ROW выделен как отдельный регион - стран не вошедших отдельно в модель).")

        #1.2 -  Начальная и конечная стр экспорта (пока тольк России/мб основной страны) def str_exp(cou_exp, otr, X, Y, G):

    with st.expander("Начальная и конечная структура экспорта " + cou_agr[n_RUS]):
            #1.2.1 - Начальная и конечная стр экспорта России
        col1, col2 = st.columns(2)

        with col1:
            y_str_exp_nach, x_str_exp_nach, fig_str_exp_nach = str_exp(n_RUS, otr, X_scen, Y_scen, G_nach)
            fig_str_exp_nach.title('Начальная структура экспорта '+cou_agr[n_RUS], fontsize = 30)
            #fig_str_exp_nach.savefig('data/'+'couTo-'+cou_agr[scen_cou_to]+'_'+'otr-'+str(otr)+'_'+'scen-'+str(scen_n)+'_'+'delta-'+str(scen_v)+'_'+'year-'+str(scen_year)+'_str_nach_exp.png')
            st.pyplot(fig_str_exp_nach)

        with col2:
            y_str_exp_after, x_str_exp_after, fig_str_exp_after = str_exp(n_RUS, otr, A_scen*res, Y_scen, G_new)
            fig_str_exp_after.title('Прогнозная структура экспорта '+cou_agr[n_RUS], fontsize = 30)
            #fig_str_exp_after.savefig('data/'+'couTo-'+cou_agr[scen_cou_to]+'_'+'otr-'+str(otr)+'_'+'scen-'+str(scen_n)+'_'+'delta-'+str(scen_v)+'_'+'year-'+str(scen_year)+'_str_after_exp.png')
            st.pyplot(fig_str_exp_after)

            #1.2.2 - Начальная и конечная стр экспорта замещающего региона
    y_str_exp_nach_zam = []
    x_str_exp_nach_zam = []
    y_str_exp_after_zam = []
    x_str_exp_after_zam = []
    if (scen_n == 4):
        with st.expander("Начальная и конечная структура экспорта конкурирующего региона " + cou_zam_name):
            col1, col2 = st.columns(2)

            with col1:
                y_str_exp_nach_zam, x_str_exp_nach_zam, fig_str_exp_nach_zam = str_exp(cou_zam, otr, X_scen, Y_scen, G_nach, list_scen_cou_zam)
                fig_str_exp_nach_zam.title('Начальная структура экспорта замещающего региона '+cou_agr[cou_zam], fontsize = 30)
                #fig_str_exp_nach_zam.savefig('data/'+'couTo-'+cou_agr[scen_cou_to]+'_'+'otr-'+str(otr)+'_'+'scen-'+str(scen_n)+'_'+'delta-'+str(scen_v)+'_'+'year-'+str(scen_year)+'_str_nach_exp_couZam.png')
                st.pyplot(fig_str_exp_nach_zam)   

            with col2:
                y_str_exp_after_zam, x_str_exp_after_zam, fig_str_exp_after_zam = str_exp(cou_zam, otr, A_scen*res, Y_scen, G_new, list_scen_cou_zam)
                fig_str_exp_after_zam.title('Прогнозная структура экспорта замещающего региона '+cou_agr[cou_zam], fontsize = 30)
                #fig_str_exp_after_zam.savefig('data/'+'couTo-'+cou_agr[scen_cou_to]+'_'+'otr-'+str(otr)+'_'+'scen-'+str(scen_n)+'_'+'delta-'+str(scen_v)+'_'+'year-'+str(scen_year)+'_str_after_exp_couZam.png')
                st.pyplot(fig_str_exp_after_zam)

        #1.3 - Структура импорта в регионах где что-то меняется: def str_imp(cou_to, otr, X, Y, G, list_cous = [-1])
            #1.3.1 - Структура импорта региона в котором происходят начальные изменения
    with st.expander("Структура импорта региона "+ cou_to_name + ", в котором происходят начальные изменения"):
        col1, col2 = st.columns(2)

        with col1: 
            y_str_cou1_nach, x_str_cou1_nach, fig_str_cou1_nach = str_imp(scen_cou_to, otr, X_scen, Y_scen, G_nach, list_scen_cou_to)
            fig_str_cou1_nach.title('Начальная структура импорта региона '+(cou_agr[scen_cou_to] if (scen_cou_to >=0) else cou_to_name), fontsize = 30)
            #fig_str_cou1_nach.savefig('data/'+'couTo-'+cou_agr[scen_cou_to]+'_'+'otr-'+str(otr)+'_'+'scen-'+str(scen_n)+'_'+'delta-'+str(scen_v)+'_'+'year-'+str(scen_year)+'_str_after_cou1.png')
            st.pyplot( fig_str_cou1_nach)

        with col2:
            y_str_cou1_after, x_str_cou1_after, fig_str_cou1_after = str_imp(scen_cou_to, otr, A_scen*res, Y_scen, G_new, list_scen_cou_to)
            fig_str_cou1_after.title('Прогнозная структура импорта региона '+(cou_agr[scen_cou_to] if (scen_cou_to >=0) else cou_to_name), fontsize = 30)
            #fig_str_cou1_after.savefig('data/'+'couTo-'+cou_agr[scen_cou_to]+'_'+'otr-'+str(otr)+'_'+'scen-'+str(scen_n)+'_'+'delta-'+str(scen_v)+'_'+'year-'+str(scen_year)+'_str_after_cou1.png')
            st.pyplot(fig_str_cou1_after)

            #1.3.2 - Структура импорта замещающего/альтернативного региона
    y_str_cou2_nach = []
    x_str_cou2_nach = []
    y_str_cou2_after = []
    x_str_cou2_after = []
    
    if (scen_n == 5):
        with st.expander("Структура импорта региона "+ cou_zam_name + " (альтернативного)"):
            col1, col2 = st.columns(2)

            with col1:
                y_str_cou2_nach, x_str_cou2_nach, fig_str_cou2_nach = str_imp(scen_cou_zam, otr, X_scen, Y_scen, G_nach, list_scen_cou_zam)
                fig_str_cou2_nach.title('Начальная структура импорта альтернативного региона '+(cou_agr[scen_cou_to] if (scen_cou_to >=0) else ''), fontsize = 30)
                #fig_str_cou2_nach.savefig('data/'+'couTo-'+cou_agr[scen_cou_to]+'_'+'otr-'+str(otr)+'_'+'scen-'+str(scen_n)+'_'+'delta-'+str(scen_v)+'_'+'year-'+str(scen_year)+'_str_after_cou2.png')
                st.pyplot(fig_str_cou2_nach)

            with col2:
                y_str_cou2_after, x_str_cou2_after, fig_str_cou2_after = str_imp(scen_cou_zam, otr, A_scen*res, Y_scen, G_new, list_scen_cou_zam)
                fig_str_cou2_after.title('Прогнозная структура импорта альтернативного региона '+(cou_agr[scen_cou_to] if (scen_cou_to >=0) else ''), fontsize = 30)
                #fig_str_cou2_after.savefig('data/'+'couTo-'+cou_agr[scen_cou_to]+'_'+'otr-'+str(otr)+'_'+'scen-'+str(scen_n)+'_'+'delta-'+str(scen_v)+'_'+'year-'+str(scen_year)+'_str_after_cou2.png')
                st.pyplot(fig_str_cou2_after)      

        #1.4 - Изменения внутри Региона
             #1.4.1 - изменения в России
    delta = res - VIP_scen
    
    with st.expander("Изменения внутри экономики региона "+ cou_agr[n_RUS]):
        col1, col2 = st.columns(2)

        with col1:
            y_delta_cou_from_abs, x_delta_cou_from_abs, fig_delta_cou_from_abs = vnut_iz(n_RUS, delta)
            fig_delta_cou_from_abs.title('Абсолютное изменение выпусков '+cou_agr[n_RUS] , fontsize = 30)
            #fig_delta_cou_from_abs.savefig('data/'+'couTo-'+cou_agr[scen_cou_to]+'_'+'otr-'+str(otr)+'_'+'scen-'+str(scen_n)+'_'+'delta-'+str(scen_v)+'_'+'year-'+str(scen_year)+'_delta_couFrom_abs.png')
            st.pyplot(fig_delta_cou_from_abs)

        with col2:
            delta_otn = (res - VIP_scen)/VIP_scen
            
            y_delta_cou_from_otn, x_delta_cou_from_otn, fig_delta_cou_from_otn = vnut_iz(n_RUS, delta_otn, '%')
            fig_delta_cou_from_otn.title('Относительное изменение выпусков '+ cou_agr[n_RUS] , fontsize = 30)
            #fig_delta_cou_from_otn.savefig('data/'+'couTo-'+cou_agr[scen_cou_to]+'_'+'otr-'+str(otr)+'_'+'scen-'+str(scen_n)+'_'+'delta-'+str(scen_v)+'_'+'year-'+str(scen_year)+'_delta_couFrom_otn.png')   
            st.pyplot(fig_delta_cou_from_otn)

            #1.4.2 - изменения в замещающей стране
    y_delta_cou_zam_abs = []
    x_delta_cou_zam_abs = []
    y_delta_cou_zam_otn = []
    x_delta_cou_zam_otn = []
    if (scen_n == 4):
        with st.expander("Изменения внутри экономики региона "+ cou_zam_name+ '(конкурента)'):
            col1, col2 = st.columns(2)

            with col1:
                y_delta_cou_zam_abs, x_delta_cou_zam_abs, fig_delta_cou_zam_abs = vnut_iz(scen_cou_zam, delta)
                fig_delta_cou_zam_abs.title('Абсолютное изменение выпусков '+cou_agr[scen_cou_zam] , fontsize = 30)
                #fig_delta_cou_from_abs.savefig('data/'+'couTo-'+cou_agr[scen_cou_to]+'_'+'otr-'+str(otr)+'_'+'scen-'+str(scen_n)+'_'+'delta-'+str(scen_v)+'_'+'year-'+str(scen_year)+'_delta_couZam_abs.png')
                st.pyplot(fig_delta_cou_zam_abs)
        
            with col2:
                y_delta_cou_zam_otn, x_delta_cou_zam_otn, fig_delta_cou_zam_otn = vnut_iz(scen_cou_zam, delta_otn, '%')
                fig_delta_cou_from_otn.title('Относительное изменение выпусков '+ cou_agr[scen_cou_zam] , fontsize = 30)
                #fig_delta_cou_zam_otn.savefig('data/'+'couTo-'+cou_agr[scen_cou_to]+'_'+'otr-'+str(otr)+'_'+'scen-'+str(scen_n)+'_'+'delta-'+str(scen_v)+'_'+'year-'+str(scen_year)+'_delta_couZam_otn.png')
                st.pyplot(fig_delta_cou_from_otn)

        #1.5 - Главные бенефициары (def find_benef(VIP_0, VIP_1, A):)
    y_ben, x_ben, fig_ben = find_benef(VIP_scen, res, A_scen)
    fig_ben.title('Главные бенефициары по ВВП', fontsize = 30)
    #fig_ben.savefig('data/'+'couTo-'+cou_agr[scen_cou_to]+'_'+'otr-'+str(otr)+'_'+'scen-'+str(scen_n)+'_'+'delta-'+str(scen_v)+'_'+'year-'+str(scen_year)+'_benef.png')
    st.pyplot(fig_ben)

    st.divider()

        #1.6 - Основные изменения в мировой торговле
        
    y_top10, x_top10, fig_top10 = big_ch(VIP_scen, res, A_scen)
    fig_top10.title('Наибольшие изменения ВВП', fontsize = 30)
    #fig_top10.savefig('data/'+'couTo-'+cou_agr[scen_cou_to]+'_'+'otr-'+str(otr)+'_'+'scen-'+str(scen_n)+'_'+'delta-'+str(scen_v)+'_'+'year-'+str(scen_year)+'_top10.png')
    st.pyplot(fig_top10)


    col1, col2, col3, col4 = st.columns(4)
    with col4:
        file = open('res_ras.txt')
        st.download_button(label = 'скачать результаты .txt', data = file, file_name = 'calc_res.txt', type="primary")