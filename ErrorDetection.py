def ErrorDetect_perFD(data, X, Y):
    '''
    对每一组FD(X->Y)找violations，
    计算的思想是：
    1:将tuple的X部分拼成str
    :param data: list
    :param X: str
    :param Y: str
    :return: list of violations
    '''
    from src.OntologyNode import getFamily
    from src.OntologyNode import findNodes
    from src.ReadJson import reJson

    print('*'*20+' FD ['+X+'--->'+Y+' ]'+'*'*20)
    X, Y = X.split(','), Y.split(',')
    dic = {}
    violations = []

    for i in range(0, len(data)):
        a = data[i]
        # print('---::current tuple::---')
        # print(a)
        strX = ''
        listY = []
        b = ''
        for x in X:
            strX = strX + str(a[int(x)])
        if strX not in dic:
            i = 0
            y_dic = {}
            for y in Y:
                y = y.split('*')
                if len(y) > 1:
                    y[1] = reJson('../data/lTest/city.json')
                    # print(y[1])
                    listY = getFamily(y[1], findNodes(y[1], a[int(y[0])]))
                    y_dic[i] = listY
                else:
                    listY.append(a[int(y[0])])
                    y_dic[i] = listY
                i = i + 1
            dic[strX] = y_dic
        else:
            i = 0
            for y in Y:
                y = y.split('*')
                if a[int(y[0])] not in dic[strX][i]:
                    ori = []
                    ori.append(strX)
                    ori.append(dic[strX])
                    if ori not in violations:
                        violations.append(ori)

                    violations.append(a)
                i = i + 1
        # print('::current dic::')
        # print(dic)
        # print('::current violations::')
        # print(violations)

    return violations


def ErrorDetect(data, fd_dic):
    violations = {}
    for fd in fd_dic:
        X = fd
        Y = fd_dic[fd]
        vio_key = str(X) + '-' + str(Y)
        violations[vio_key] = ErrorDetect_perFD(data, X, Y)
    return violations


data=[
    ['quzhi','94','mac','cs','Dieppe'],
    ['quzhi','93','mac','cas','Dieppe'],
    ['huangyu','86','mac','cs','Toronto'],
    ['huangyu','86','mac','cs','ON'],
    ['huangyu','86','mac','cas','Dieppe'],
    ['zhengzheng','89','mac','cs','Windsor'],
    ['zhengzheng','89','mac','cs','Windsor']]

fd_dic={
    '0':'1,3',
    '0,1': '4*city',
     }
#ErrorDetect(data,fd_dic)

print(ErrorDetect(data, fd_dic))
