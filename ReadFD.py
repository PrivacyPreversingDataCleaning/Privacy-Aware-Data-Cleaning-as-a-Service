

def reFD(filepath):

    file=open(filepath)
    Fd_Dic={}
    for row in file:
        fd=row.split(';')
        Fd_Dic[fd[0]]=fd[1]

    return Fd_Dic

# print(reFD('../data/sTest/testFD.csv'))











