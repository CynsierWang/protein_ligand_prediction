'''
this is used to select the top 32 atoms of all .pdbqt files
'''
import os

rootDir ="/Users/cynsier/sets/set0"
dict={}

for lists in os.listdir(rootDir):
    road = os.path.join(rootDir, lists)
    path = road+"/ligand_0.pdbqt"
    path2= road+"/receptor.pdbqt"
    file = open(path,"r")
    for line in file.readlines():
        if(line.split()[0]=="ATOM"):
            scrap = line.split()[2]
            atom=""
            for ch in scrap:
                if(ch>="A" and ch<="Z"):
                    atom+=ch
            if not dict.has_key(atom):
                dict.setdefault(atom)
                dict[atom]=1
            else:
                dict[atom]+=1
    file.close()
    file = open(path2,"r")
    for line in file.readlines():
        if(line.split()[0]=="ATOM"):
            scrap = line.split()[2]
            atom=""
            for ch in scrap:
                if(ch>="A" and ch<="Z"):
                    atom+=ch
            if not dict.has_key(atom):
                dict.setdefault(atom)
                dict[atom]=1
            else:
                dict[atom]+=1
    file.close()

'''READ A DICTIONARY
for (k,v) in dict.items():
   print(k,v)
'''

count =0
sort = sorted(dict.items(),key=lambda e:e[1],reverse=True)
for items in sort:
    print items[0]
    count+=1
    if count==32:
        break
