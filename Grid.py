import numpy as np
import re
import os

dictname="dictDB2.txt"
dict={}
mol=[]

def makeDick():
    dictfp = open(dictname, "rb")
    i=1
    for line in dictfp.readlines():
        list1=list(line)
        del list1[-1]
        line="".join(list1)
        dict.setdefault(line)
        dict[line]=i
        i+=1
    dictfp.close()

def scan(fp):
    p1 = "-?\d+\.{1}\d+"
    ptrn = re.compile(p1)
    for line in fp.readlines():
        if(line.split()[0]=="ATOM"):
            match = re.findall(ptrn, line)
            x_site=float(match[0])
            y_site=float(match[1])
            z_site=float(match[2])
            scrap = line.split()[2]
            atom=""
            for ch in scrap:
                if(ch>="A" and ch<="Z"):
                    atom+=ch
            info=[x_site,y_site,z_site,atom]
            mol.append(info)
    fp.close()

def findCenter():
    max = [mol[0][0],mol[0][1],mol[0][2]]
    min = [mol[0][0],mol[0][1],mol[0][2]]
    for item in mol:
        for i in range(3):
            if(item[i]>max[i]):
                max[i]=item[i]
            if(item[i]<min[i]):
                min[i]=item[i]
    c=[(max[0]+min[0])/2.,(max[1]+min[1])/2.,(max[2]+min[2])/2.]
    return c

def makeTensor():#shape[depth,height,width,channel]
    cube=np.zeros((48,48,48,32))
    key=1
    center=findCenter()
    for item in mol:
        for i in range(3):
            if not(item[i]<center[i]+12 and item[i]>center[i]-12):
                key=0
            if not dict.has_key(item[3]):
                key=0
                print "DON'T HAVE "+item[3]
        if(key==0):
            continue
        chanel=dict[item[3]]
        x=int(((item[0]-center[0]+12)*2))
        y=int(((item[1]-center[1]+12)*2))
        z=int(((item[2]-center[2]+12)*2))
        cube[z][y][x][chanel]=cube[z][y][x][chanel]+1
    return cube

def makeCube(ligand,pocket):
    makeDick()
    fp = open(ligand, "rb")
    scan(fp)
    fp = open(pocket,"rb")
    scan(fp)
    cube=makeTensor()
    return cube

def makeBatch(path,n,batch):
    fp = open(path, "rb")
    featset=[]
    labelset=[]
    i=0
    count=0
    sham =n*batch
    imbreak=1
    again=0
    while(imbreak):
        again+=1
        for line in fp.readlines():
            scrap= line.split(",")
            pos  = scrap[0]
            name = scrap[1]
            id   = "".join((list(scrap[2])[:-1]))
            exist= "/home/wyd/sets/%s"%name
            exist= "/Users/cynsier/sets/%s"%name
            if(not os.path.isdir(exist)):
                continue
            count+=1
            if(count<=sham):
                continue
            ligand = "/home/wyd/sets/%s/%s_ligand_%s.pdbqt" % (name,name,id)
            pocket = "/home/wyd/sets/%s/%s_receptor.pdbqt" % (name,name)
            ligand = "/Users/cynsier/sets/%s/%s_ligand_%s.pdbqt" % (name,name,id)
            pocket = "/Users/cynsier/sets/%s/%s_receptor.pdbqt" % (name,name)
            #featset.append(makeCube(ligand,pocket))
            print ligand
            if(pos=="0"):
                labelset.append([1,0])
            else:
                labelset.append([0,1])
            i+=1
            if(i==batch):
                imbreak=0
                break
        fp.close()
        fp=open(path,"rb")
    fp.close()
    return featset, labelset, again

if __name__=="__main__":
    makeBatch("sample-list.txt",1,10)