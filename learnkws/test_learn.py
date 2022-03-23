'''
Date: 2022-03-04 18:10:52
LastEditors: Cyan
LastEditTime: 2022-03-07 10:21:34
'''

if __name__ == '__main__':
    a = [1,2,3,4,5,6,7]
    for i in range(len(a)):
        print('i = ', i)
        if a[i] >= 3:
            i += 2
        # print('a[i] = ' , a[i])
