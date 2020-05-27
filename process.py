import copy
import csv
from datetime import datetime
from random import randrange
import random
import time

def generate_repeat(elements, size, replace=True):
    '''
    args:
        elements, the given list
        size, the size of output
        replace, whether repeatly select
    return:
        a list of random elements which is picked from given elements (repeatly)
    '''

    import numpy as np
    return np.random.choice(elements, size, replace)


def read_to_list(path):

    with open(path) as f:
        lines = f.readlines()
        # remove whitespace charactor
        lines = [x.strip() for x in lines]
        return lines


def random_date(start,l):
   current = start
   while l >= 0:
    current = current + datetime.timedelta(minutes=randrange(10))
    yield current
    l-=1

# startDate = datetime.datetime(2013, 9, 20,13,00)
#
# for m in reversed(list(random_date(startDate,10))):
#     print (m.strftime("%d/%m/%y %H:%M"))

def strTimeProp(start, end, format, prop):
    """Get a time at a proportion of a range of two formatted times.

    start and end should be strings specifying times formated in the
    given format (strftime-style), giving an interval [start, end].
    prop specifies how a proportion of the interval to be taken after
    start.  The returned time will be in the specified format.
    """

    stime = time.mktime(time.strptime(start, format))
    etime = time.mktime(time.strptime(end, format))
    ptime = stime + prop * (etime - stime)

    return time.strftime(format, time.localtime(ptime))


def randomDate(start, end, prop):
    return strTimeProp(start, end, '%m/%d/%Y %I:%M %p', prop)


def generate_time(size):
    l = []
    for i in range(size):
        year = random.randint(2000, 2017)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        hour = random.randint(1, 23)
        minite1 = random.randint(0, 59)
        # sec = random.randint(0, 59)
        t = datetime(year, month, day, hour, minite1)
        l.append(t)
    return l


def generate_date(size):
    l = []
    for i in range(size):
        year = random.randint(2013, 2017)
        month = random.randint(1, 9)
        day = random.randint(1, 28)
        t = datetime(year, month, day)
        '{:%m/%d/%Y}'.format(t)
        l.append(t)
    return l


def initialize_columns_list(file_path):

    initial_columns_list=[]

    with open(file_path, "r") as csvfile:
        reader = csv.reader(csvfile)  # 读取csv文件，返回的是迭代类型
        initial_columns_list=[]
        for item in reader:
            columns=item
            for col in columns:
                new_column=[]
                initial_columns_list.append(new_column)
            break
    csvfile.close()
    return initial_columns_list


def read_csv_file(file_path):

    initial_columns_list=initialize_columns_list(file_path)
    return initial_columns_list


def get_elements(file_path,initial_columns_list):

    ref_initial_columns_list=initial_columns_list

    with open(file_path,"r") as csvfile:
        reader = csv.reader(csvfile)
        index=0
        for item in reader:
            if index != 0 :
                len_item=len(item)
                for i in range(0,len_item):
                    ref_initial_columns_list[i].append(item[i])
            index=index+1
    csvfile.close()
    return ref_initial_columns_list


def deduplicate(columns_list):

    ref_columns_list=[]
    for column in columns_list:
        col = list(set(column))
        ref_columns_list.append(col)

    return ref_columns_list


def generate_csv_file(elements_list,size):

    ref_elements_list=elements_list
    generating_file = open('generated_admissions.csv', 'w', newline='')  # 设置newline，否则两行之间会空一行
    writer = csv.writer(generating_file)
    for i in range(size):
        record=[]
        for x in range(len(ref_elements_list)):
            record.append(random.choice(ref_elements_list[x]))
        writer.writerow(record)
    generating_file.close()


# initial_columns_list=read_csv_file("ADMISSIONS.csv")
# columns_list=get_elements("ADMISSIONS.csv",initial_columns_list)
# columns_list=deduplicate(columns_list)
#
# generate_csv_file(columns_list,50000)
#
# generating_file = open('5_Time.csv', 'w', newline='')  # 设置newline，否则两行之间会空一行
# writer = csv.writer(generating_file)
# for i in range(50000):
#     record = []
#     for x in range(5):
#         t = generate_time(1)
#         record.append(t[0])
#     record = sorted(record)
#
#     fomat_record=[]
#     for r in record:
#         fomat_record.append(str(r))
#         print(str(r))
#
#     writer.writerow(fomat_record)
# generating_file.close()

langs=['English','English','English','English','English','English','English','English','English','English','English','Spanish','Russian','Spanish','Slovenian','Spanish','Swedish','Taiwanese','Thai','Turkish','Ukrainian','Vietnamese','Yiddish']
generating_file = open('lang.csv', 'w', newline='')  # 设置newline，否则两行之间会空一行
writer = csv.writer(generating_file)
for i in range(50000):
    record = []

    record.append(random.choice(langs))

    writer.writerow(record)
generating_file.close()








