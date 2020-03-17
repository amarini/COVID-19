import os,sys,re
import json
from datetime import datetime
import numpy as np
import math

population={
        'Lombardia':10.04 , #M
        'Veneto':4.905,
        'Abruzzo':1.315,
        'Calabria':1.957,
        'Toscana':3.737,
        'Molise':0.308493,
        'Liguria':1.557,
        'Sardegna':1.648,
        'Umbria':0.884640,
        'italia':60.48,
        'Basilicata':0.567118,
        'Valle d\'Aosta':.126202,
        'Marche':1.532,
        'Friuli Venezia Giulia':1.216,
        'Lazio':5.897,
        'Campania':5.827,
        'Sicilia':5.027,
        'Puglia':4.048 ,
        'Emilia Romagna':4.453,
        'P.A. Trento':0.538223,
        'P.A. Bolzano':0.520891,
        'Piemonte':4.376,
        }

#f=open("dati-json/dpc-covid19-ita-province.json")
f=open("dati-json/dpc-covid19-ita-regioni.json")
j=json.load(f)
# "denominazione_regione": "Abruzzo",

f=open("dati-json/dpc-covid19-ita-andamento-nazionale.json")
j2=json.load(f)
#        "data": "2020-02-24 18:00:00",
#        "stato": "ITA",
#        "ricoverati_con_sintomi": 101,
#        "terapia_intensiva": 26,
#        "totale_ospedalizzati": 127,
#        "isolamento_domiciliare": 94,
#        "totale_attualmente_positivi": 221,
#        "nuovi_attualmente_positivi": 221,
#        "dimessi_guariti": 1,
#        "deceduti": 7,
#        "totale_casi": 229,
#        "tamponi": 4324

zero=datetime.strptime("2020-02-24 18:00:00","%Y-%m-%d %H:%M:%S")

data={} #regione -> {"x"} list
keys=["terapia_intensiva","totale_attualmente_positivi","deceduti","totale_casi","ricoverati_con_sintomi","dimessi_guariti"]

for key in j:
    date=datetime.strptime(key["data"],"%Y-%m-%d %H:%M:%S")
    
    regione=key['denominazione_regione']
    if regione not in data: 
        data[regione]={"time":[]}
        for k in keys: data[regione][k] =[]

    if len(data[regione]['time']) >0 and (date-zero).days ==  data[regione]['time'][-1]:
        for k in data[regione]:
            data[regione][k] = data[regione][k][:-1]

    data[regione]['time'].append(  (date-zero).days   ) 
    for k in keys:
        try:
            data[regione][k].append( key[k])
        except KeyError:
            data[regione][k].append( 0.)

for key in j2:
    date=datetime.strptime(key["data"],"%Y-%m-%d %H:%M:%S")
    
    regione='italia'
    if regione not in data: 
        data[regione]={"time":[]}
        for k in keys: data[regione][k] =[]


    if len(data[regione]['time']) >0 and (date-zero).days ==  data[regione]['time'][-1]:
        for k in data[regione]:
            data[regione][k] = data[regione][k][:-1]

    data[regione]['time'].append(  (date-zero).days   ) 
    for k in keys:
        try:
            data[regione][k].append( key[k])
        except KeyError:
            data[regione][k].append( 0.)


## plot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages('summary.pdf')

if False:
    fig=plt.figure()
    plt.xlabel("time")
    plt.ylabel('totale_attualmente_positivi')
    plt.yscale('log')
    colors=["lightcoral","coral","salmon","indianred","red","brown","firebrick","darkred","maroon","lightskyblue","aqua","cyan","darkturquoise","lightskyblue","deepskyblue","steelblue","dodgerblue","royalblue","mediumblue","blue","darkblue","indigo"]
    for ir,reg in enumerate(data):
        print "reg",reg,"tl=",len(data[reg]['time']),len(data[reg]['totale_attualmente_positivi']),ir,len(colors)
        plt.plot(data[reg]['time'],data[reg]['totale_attualmente_positivi'],label=reg,color=colors[ir])
    plt.legend()
    plt.savefig(pp, format='pdf')
 
    plt.show()

if False:
    fig=plt.figure()
    plt.xlabel("time")
    plt.ylabel('totale_attualmente_positivi/1M')
    plt.yscale('log')
    #ylim top bottom
    plt.xlim( (-5,len(data['italia']['time'])+1) )

    import matplotlib.colors as colors
    import matplotlib.cm as cmx
    cNorm  = colors.Normalize(vmin=0, vmax=len(data))
    #scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=plt.get_cmap('tab20'))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=plt.get_cmap('coolwarm'))
   
    # color regions for value at the last day
    reglist= []
    for ir,reg in enumerate(data):
        reglist.append([reg,data[reg]['totale_attualmente_positivi'][-1]/population[reg] ])
    reglist.sort(key=lambda x: x[1])
    reglist2=[ r[0] for r in reglist ]

    for ir,reg in enumerate(reglist2):
        col=scalarMap.to_rgba(ir)
        if reg=='italia': col='black'
        print "reg",reg,"tl=",len(data[reg]['time']),len(data[reg]['totale_attualmente_positivi']),ir,col#,len(colors)
        #plt.plot(data[reg]['time'],data[reg]['totale_attualmente_positivi']/population[reg],label=reg,color=colors[ir])
        plt.plot(data[reg]['time'],np.array(data[reg]['totale_attualmente_positivi'])/population[reg],label=reg,color=col)

    xlomb=(datetime.strptime("2020-03-07 18:00:00","%Y-%m-%d %H:%M:%S")-zero).days
    xit=(datetime.strptime("2020-03-10 18:00:00","%Y-%m-%d %H:%M:%S")-zero).days

    plt.plot([xlomb,xlomb],[0,20000],label='chiusura lomb',color='grey',linestyle='--')
    plt.plot([xit,xit],[0,20000],label='chiusura it',color='black',linestyle='--')

    plt.legend()
    plt.savefig(pp, format='pdf')
    plt.show()

if True:
    xlomb=(datetime.strptime("2020-03-07 18:00:00","%Y-%m-%d %H:%M:%S")-zero).days
    xit=(datetime.strptime("2020-03-10 18:00:00","%Y-%m-%d %H:%M:%S")-zero).days

    fig=plt.figure()
    plt.xlabel("days from 24th Feb")
    plt.ylabel('numbers')
    plt.yscale('log')
    colors=["lightcoral","coral","salmon","indianred","red","brown","firebrick","darkred","maroon","lightskyblue","aqua","cyan","darkturquoise","lightskyblue","deepskyblue","steelblue","dodgerblue","royalblue","mediumblue","blue","darkblue","indigo"]
    
    plt.scatter(data['italia']['time'],data['italia']['totale_attualmente_positivi'],label='positivi',color='indianred')
    plt.scatter(data['italia']['time'],data['italia']['deceduti'],label='deceduti',color='black',marker='s')
    plt.scatter(data['italia']['time'],data['italia']['terapia_intensiva'],label='T.I.',color='deepskyblue',marker='^')
    plt.scatter(data['italia']['time'],data['italia']['dimessi_guariti'],label='guariti',color='forestgreen',marker='v')

    plt.plot([xlomb,xlomb],[0,20000],label='chiusura lomb',color='grey',linestyle='--')
    plt.plot([xit,xit],[0,20000],label='chiusura it',color='black',linestyle='--')
    
    if True:
        ## expected == N e^xc
        def exponential(r,time):
            exp = r[1]*np.exp((time)*r[0])
            #for x in exp:
            #    if np.isnan(x) or np.isinf(x):
            #        print "NAN OR INF"
            #        print "params=",r
            #        print "x=",time
            #        print "e^x=",exp
            return exp

        def nll(r):
            global y, time 
            return np.sum(- poisson.logpmf(y,exponential(r,time)) ) 
            ## -Sum log P(y, exp time)

        def estimate():
            ''' Linear regression on logy and time'''
            global time, y
            l = np.log(y)
            N=len(time)
            Sxy= np.sum(time*l)
            Sx = np.sum(time)
            Sy = np.sum(l)
            Sx2= np.sum(time*time)
            a= (N*Sxy - Sx*Sy) /  (N *Sx2 - Sx*Sx)
            b= (Sy*Sx2-Sx*Sxy) / (N*Sx2 -Sx*Sx)
            #logY= at+b
            # y =e^{at} * e^b
            return np.array([a,np.exp(b)])

        def jac(r):
            global y , time
            #raise ValueError("WRONG")
            #sum _i 
            l=exponential(r,time)
            b=  np.sum( (np.ones(y.shape)-np.divide(y,l)) * np.exp(time*r[0])*r[1]* time ) 
            a=  np.sum( (np.ones(y.shape)-np.divide(y,l)) * np.exp(time*r[0])) 
            return np.array([b,a])

        def jac_num(r,e=0.0001):
            f2=np.array( [nll(r+np.array([e,0.])), nll(r+np.array([0.,e])) ])
            f0=np.array( [nll(r-np.array([e,0.])), nll(r-np.array([0.,e])) ])
            return np.divide((f2-f0), np.array([2.*e,2.*e]))

        def logistic(r,time):
            return r[1]/(1+np.array(np.exp(-(time-r[2])*r[0])))

        def nll_logistic(r):
            global y, time  ## logistic fit L/(1+e^kx)
            exp=logistic(r,time)
            return np.sum(- poisson.logpmf(y,exp) ) 

        def nll_logistic2(r):
            return nll_logistic(np.append(r,np.array([17])))
            
        def nll3(r):
            global y, time ## Gomperetz ## N e^{A - C e^(-B (t-t0))}
            N=r[0]
            A=r[1]
            B=r[2]
            C=r[3]
            t0=r[4]
            exp=N*np.exp(A-C*np.exp(-(time - t0)*B))
            return np.sum(- poisson.logpmf(y,exp) ) 

        def chi2(y,x):
            return np.sum(np.divide(np.power(y-x,2),x))

        def Wald(y,x):
            Np=0
            Nm=0
            Nr=0
            lastP=0
            for i in range(0,len(y)):
                y1=y[i]
                y0=x[i]
                if (y1>=y0) :
                    Np +=1
                    if lastP<=0: Nr+=1
                    lastP=1
                else :
                    Nm +=1
                    if lastP>=0: Nr+=1
                    lastP=-1
            return Nr,Np,Nm

        def WaldProb(Nr,Np,Nm):
            N=Np+Nm
            mu=2*Np*Nm/N+1
            sigma=math.sqrt((mu-1)*(mu-2)/(N-1))
            z=(Nr-mu)/sigma
            if z>0:
                pOne=1.-1./2.*(1+math.erf(z/math.sqrt(2)))
            else:
                pOne=1./2.*(1+math.erf(z/math.sqrt(2)))
            return pOne*2 ## two sided

        def get_toy(bf):
            y=(np.random.poisson(bf) + np.random.normal(0,bf*0.03) ) * np.random.lognormal(0,0.03) 
            #y=(np.random.poisson(bf) ) * np.random.lognormal(0,0.05)
            #y=np.random.poisson(bf)
            return y
        


        from scipy.optimize import minimize
        from scipy.stats import poisson
        keys=['totale_attualmente_positivi', 'deceduti', 'terapia_intensiva', 'dimessi_guariti']
        colors=['darkred','black','darkblue','darkgreen']
        for key,col in zip(keys,colors):
            y=np.array(data['italia'][key][xit:])  ## FIT
            time=np.array(data['italia']['time'][xit:]) ## FIT
            r0=np.array([.5,100.])
            r0=estimate()
            rbf=minimize(nll,r0,method='SLSQP',jac=jac);
            bf = exponential(rbf.x,time)
            plt.plot(time,bf, color=col, label="tau=%.1f (t2=%.1f)"%(1./rbf.x[0],math.log(2.)/rbf.x[0]))

            c2= chi2(y,bf)

            timeplt=np.array(data['italia']['time'][xit-5:xit+1])
            bf = exponential(rbf.x,timeplt)
            plt.plot(timeplt,bf, color=col, ls='--')
            
            #timeplt=np.array(data['italia']['time'][xit-5:]) #T
            timeplt=np.append(np.array(data['italia']['time'][xit-5:]) ,np.arange(data['italia']['time'][-1],data['italia']['time'][-1]+5,1)[1:])
            print "BF = ",rbf.x,r0


            #raw_input("ok?")
            if True: ##error
                alpha=0.6814
                ytot=[[] for t in timeplt] # 
                c2tot=[]
                for itoy in range(0,1000):
                    time=np.array(data['italia']['time'][xit:])  ## FIT
                    bf = exponential(rbf.x,time)
                    y = get_toy(bf)
                    #r0=np.array([.5,1000.])
                    r0=estimate()
                    rt=minimize(nll,r0,method='SLSQP',jac=jac);
                    yt= exponential(rt.x,timeplt)
                    #yt= exponential(r0,timeplt)

                    for i in range(0,len(timeplt)):
                        if np.isnan(yt[i]): 
                            continue
                        ytot[i].append(yt[i])

                    c2_= chi2(y,exponential(rt.x,time) )
                    c2tot.append(c2_)
                ## y

                yup = [] #T
                ydn = [] #T
                quantminusone = 0.5*(1.0+ math.erf(-1.0/math.sqrt(2)));
                quantplusone  = 0.5*(1.0+ math.erf(1.0/math.sqrt(2)));
                for i in range(0,len(timeplt)):
                    ytot[i].sort()
                    n=len(ytot[i])
                    yup.append(ytot[i][int(n*quantplusone)])
                    ydn.append(ytot[i][int(n*quantminusone)])

                c2Prob=float(np.sum(c2tot>=c2) )/ len(c2tot)
                print "CHI2",c2, c2Prob
                plt.fill_between(timeplt, ydn,yup,color=col,alpha=0.3, label="prob(stat+syst)=%.1f"%(c2Prob))
            #####################
            ### logistic      ###
            #####################
            time=np.array(data['italia']['time'][xlomb:]) 
            y=np.array(data['italia'][key][xlomb:])
            #r0=np.array([.2,2000.,17])
            r0=np.array([.2,2000.])
            if key == 'totale_attualmente_positivi': 
                r0[0]=.1
                r0[1]=20000.
                #r0[2]=15
            rbf2=minimize(nll_logistic2,r0,method='SLSQP',jac=False);
            timeplt=np.append( np.array(data['italia']['time'][xlomb:]) , np.arange(data['italia']['time'][-1],data['italia']['time'][-1]+5,1)[1:])
            param=np.append(rbf2.x,np.array([17]))
            bf2 = logistic( param,timeplt)
            print key,rbf2.x
            plt.plot(timeplt,bf2, color=col, ls=':')

            if False: ##error -> understand fit stability
                alpha=0.6814
                ytot=[[] for t in timeplt] # 
                c2tot=[]
                for itoy in range(0,1000):
                    time=np.array(data['italia']['time'][xlomb:])  ## FIT
                    bf = logistic(param,time)
                    y = get_toy(bf)
                    rt=minimize(nll_logistic2,rbf2.x,method='SLSQP',jac=False);
                    param2= np.append(rt.x,np.array([17]))
                    yt= logistic(param2,timeplt)

                    for i in range(0,len(timeplt)):
                        if np.isnan(yt[i]): 
                            print "NAN find in logistic"
                            continue
                        ytot[i].append(yt[i])

                    c2_= chi2(y,logistic(param2,time) )
                    c2tot.append(c2_)
                ## y

                yup = [] #T
                ydn = [] #T
                quantminusone = 0.5*(1.0+ math.erf(-1.0/math.sqrt(2)));
                quantplusone  = 0.5*(1.0+ math.erf(1.0/math.sqrt(2)));
                for i in range(0,len(timeplt)):
                    ytot[i].sort()
                    n=len(ytot[i])
                    print "N=",n
                    yup.append(ytot[i][int(n*quantplusone)])
                    ydn.append(ytot[i][int(n*quantminusone)])

                c2Prob=float(np.sum(c2tot>=c2) )/ len(c2tot)
                print "CHI2-LOGISTIC",c2, c2Prob
                plt.fill_between(timeplt, ydn,yup,color=col,alpha=0.5, label="prob-Logistic=%.1f"%(c2Prob))
   
            #gamperez?
            #r0=np.array([1000,10,0.7,1,0])
            # totale_attualmente_positivi
            if False:
                r0=np.array([879.7425436907399 , 6.474677155296394 , 0.04330746910612633 , 3.7121608178773573 , 16.535239670897592])
                if key == 'totale_attualmente_positivi':
                    r0=np.array([880.3569199109379 , 8.86646599051443 , 0.026048683498941263 , 5.5158218616191 , 20.466067885819722])
                if key == 'deceduti':
                    r0=np.array([1187.8264508026455 , 9.050013555293576 , 0.020319834530314294 , 6.06787428781581 , 36.40767523799071])
                if key == 'terapia_intensiva':
                    r0=np.array([879.8038335642318 , 12.59582478503967 , 0.009473594738933982 , 12.524185412524016 , 13.969584460375554])
                if key == 'dimessi_guariti':
                    r0=np.array([873.6299083436106 , 27.55628645846774 , 0.007383765050153899 , 28.174028449540998 , 11.186499210670052])

                ## Gomperez
                rbf3=minimize(nll3,r0,method='SLSQP',jac=False);
                timeplt=np.append( np.array(data['italia']['time'][xlomb:]) , np.arange(data['italia']['time'][-1],data['italia']['time'][-1]+5,1)[1:])
                N=rbf3.x[0]
                A=rbf3.x[1]
                B=rbf3.x[2]
                C=rbf3.x[3]
                t0=rbf3.x[4]
                bf3=N*np.exp(A-C*np.exp(-(timeplt - t0)*B))
                plt.plot(timeplt,bf3, color=col, ls=':')
                print key
                print "N",N,"A",A,"B",B,"C",C,"t0",t0

    plt.legend(loc='lower right',ncol=3)
    plt.savefig(pp, format='pdf')
    plt.show()
pp . close()
