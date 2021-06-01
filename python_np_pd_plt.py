# -*- coding: utf-8 -*-
"""
Created on Mon May 31 13:12:16 2021

@author: Lenovo
"""

a =3
print(a)
a="abc"
print(a)
a = 4
b = 5
print(a+b)

my_list=[10,20,30,40]
print(my_list[-1])
print(my_list[-2])

my_list_2=[12,"abc",20,40]
print(my_list_2[-3])

def calculateSum(a,b):
    return a+b, a/b
var1, var2 = calculateSum(10,2)
print(var1)
print(var2)

with open("my_file_1.txt", "w") as f: #mode write w 
    f.write("sample content 1")
    
with open("my_file_1.txt", "a") as f: #mode add a
    f.write("more content")
    
import numpy as np

sample_list=[10,20,30,40,50,60]
sample_numpy_1d_array= np.array(sample_list)
sample_numpy_2d_array= np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
sample_numpy_2d_array.reshape(2,6)
