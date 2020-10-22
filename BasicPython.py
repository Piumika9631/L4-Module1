import cv2

#printing the opencv version
print(cv2.__version__)

#Python Basics

#Python Assignment (String)
data = 'Hello World'
print(data[0])
print(len(data))
print(data)
#Python Assignment (Number)
value = 123.1
print(value)
value = 10
print(value)
#Python Assignment (Boolean)
a = True
b = False
print(a, b)
#Python Assignment (Multiple)
a, b, c = 1, 2, 3
print(a, b, c)
#Python Assignment (None)
a = None
print(a)

#Flow Control (If)
value = 99
if value == 99:
    print('That is fast')
elif value >200:
    print('That is too fast')
else:
    print('That is safe')
#Flow Control (For)
for i in range(10):
    print(i)
#Flow Control (While)
i = 0
while i<10:
    print(i)
    i += 1

#Data Structures (Tuples)
a = (1, 2, 3)
print(a)
#Data Structures (Lists)
mylist = [1, 2, 3]
print('Zeroth value: %d' % mylist[0])
mylist.append(4)
print('List length: %d' % len(mylist))
for val in mylist:
    print(val)
#Data Structures (Dictionary)
mydict = {'a':1, 'b':2, 'c':3}
print('A value: %d' % mydict['a'])
mydict['a'] = 11
print('A value: %d' % mydict['a'])
print('Keys %s' % mydict.keys())
print('Values %s' % mydict.values())
for key in mydict.keys():
    print(mydict[key])

#Functions
def mysum(x,y):
    return x+y

result = mysum(5, 3)
print(result)
