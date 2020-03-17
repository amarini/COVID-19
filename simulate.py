import os, sys
import numpy as np
import random
random.seed()



class Person:
    def __init__(self,w=0):
        ## SIR
        self.isI=False
        self.isS=True
        self.isR=False
        ##
        self.pI=0.00015 ## counts as multiplied by NN infected 0.0001
        self.pR=0.07 ## 1./t is the average infection time 0.05
        ## location and contact
        self.where=w

    def infect(self,certain=False):
        if not self.isS: return self
        if certain or random.random()< self.pI:
            self.isI=True
            self.isS=False
        return self

    def recover(self):
        if not self.isI: return self
        if random.random()< self.pR:
            self.isI=False
            self.isR=True
        return self

    def fallback(self):
        return self
        # if random.random()< self.pX:
        #     self.R=False
        #     self.S=True

class Word:
    def __init__(self,nplaces=10,people_per_place=1000,pshort=.001,plong=0.0002):
        self.nplaces=nplaces
        self.people_per_place=people_per_place

        self.m = None ## this is the number of infected in each place
        self.word=None ## this is where the people are

        self.totI=0
        self.totS=0
        self.totR=0

        self.population = self.people_per_place*self.nplaces
        self.people = [ Person(i%self.nplaces) for i in range(0,self.population)]

        # infect part of the population. I0
        [self.people[self.people_per_place/2+i*self.nplaces].infect(True) for i in range(0,10)]

        self.p_short=pshort
        self.p_long=plong

        self.update_map()
        self.update_word() 

    def update_map(self):
        self.m = [ 0 for i in range(0,self.nplaces)] ## this is the number of infected in each place
        for p in self.people:
            if p.isI: self.m[p.where] += 1
        #self.totI=np.sum(np.array(self.m))
        self.totI=np.sum(np.array([ 1 for p in self.people if p.isI ]))
        self.totS=np.sum(np.array([ 1 for p in self.people if p.isS ]))
        self.totR=np.sum(np.array([ 1 for p in self.people if p.isR ]))
        return self

    def update_word(self):
        # split people across the map
        self.word=[[] for i in range(0,self.nplaces)]
        for ip,p in enumerate(self.people):
            self.word[p.where].append(ip)
        return self

    def update_position(self):
        ## short term mobility
        for p in self.people:
            if self.p_short != None and random.random()< self.p_short:
                p.where += random.choice([-1,1])
                if p.where <0 : p.where += self.nplaces
                if p.where >= self.nplaces: p.where -= self.nplaces
            if self.p_long != None and random.random()< self.p_long:
                p.where = random.randint(0,self.nplaces-1)

        self.update_map()
        self.update_word() 

        return self

    def update_infect (self):
        self.update_map()   ## if consistent these two calls can be avoided
        self.update_word()  ##

        for il,loc in enumerate(self.word):
            for ip in loc:
                for i in range(self.m[il]): self.people[ip].infect() ## for each infected
                self.people[ip].recover()
                self.people[ip].fallback()

        self.update_map()
        return self

    def update_sequence(self):
        self.update_infect()
        self.update_position()
        return self
#########################################
t=[]
totI=[]
totS=[]
totR=[]
totP=[]
totM=[] ## list of list

#def __init__(self,nplaces=10,people_per_place=1000,pshort=.001,plong=0.0002):
w=Word() # scenario 0

w=Word(50,1000,0.001,0.0002)
## second population  pI-> vecchi x2
w.second_population=0.1
for p in w.people:
    if random.random()< w.second_population: p.pI = 0.00030

w.realize=0

print "INIT",0,w.m
for it in range(0,1000):
    #print ".",

    w.update_sequence()

    ## save info
    t.append(it)
    totI.append(w.totI)
    totS.append(w.totS)
    totR.append(w.totR)

    ## total map
    if it%10 ==1: 
        print "TIME",it,w.m
        totM.append(w.m[:])
    
    ## total of infected places
    totP.append(np.sum(np.array([ 1 for m in w.m if m>0 ])))

    #if max(w.m) > 100 and w.realize==0:
    #    w.p_short*=.1
    #    w.p_long*=.1
    #    w.realize=it
    #    print "REALIZE"

    if w.totI <1: 
        print " ending at epoch",it,"no more I"
        break
    if w.totS <1: 
        print " ending at epoch",it, "no more S"
        break
print "."

#########################################
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages('summary.pdf')

if True:
    fig=plt.figure()
    rows=[
        ("prob infect", w.people[0].pI),
        ("prob recover",w.people[0].pR),
        ("nearby move",w.p_short),
        ("long move",w.p_long),
        ("nplaces",w.nplaces),
        ("npeople",w.population),
        ("realize travel/10 (100)",w.realize),
        ("2nd population",w.second_population),
        ]
    
    x,y=0.1,.8
    for r in rows:
        plt.text(x,y,r[0]+"  "+str(r[1]))
        y-=0.1
    plt.savefig(pp, format='pdf')
    plt.show()

if True:
    fig=plt.figure()
    plt.xlabel("time")
    plt.ylabel("people")
    plt.plot(np.array(t),np.array(totI),color="coral",label="infected")
    plt.plot(np.array(t),np.array(totS),color="lightskyblue",label="susceptible")
    plt.plot(np.array(t),np.array(totR),color="green",label="recovered")
    plt.legend()
    plt.savefig(pp, format='pdf')
    plt.show()
    #raw_input("ok?")

if True:
    fig=plt.figure()
    plt.xlabel("time")
    plt.ylabel("infected places")
    plt.plot(np.array(t),np.array(totP),color="coral",label="infected")
    plt.legend()
    plt.savefig(pp, format='pdf')
    plt.show()
    #raw_input("ok?")

if True:
    fig=plt.figure()
    plt.xlabel("time")
    plt.ylabel("places")
    x = []
    y = []
    s = []
    for it in range(0,len(totM)):
        for ip in range(0,len(totM[it])):
            x.append(it)
            y.append(ip)
            s.append(totM[it][ip])
    plt.scatter(x,y,s=s)
    plt.savefig(pp, format='pdf')
    plt.show()
pp.close()
