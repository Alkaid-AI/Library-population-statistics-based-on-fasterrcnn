# for i in range(30):
#     print(i+1)
#     if (i+1) % 2 == 0:
#         print('even number')
#     if (i+1) % 3 == 0:
#         print('multiply of 3')


# def search_binery(list, item):
#     low=0
#     high=len(list)-1
#     while low <= high:
#         mid = (low+high)//2
#         guess=list[mid]
#         if item == guess:
#             return mid
#         if guess < item:
#             low = mid+1
#         else:
#             high = mid-1
#     return None
#
# list = [3, 5, 7, 9, 11, 13, 15, 17, 19]
#
# print(search_binery(list, 19))

# def findsmallest(array):
#     smallest_index = 0
#     smallest = array[smallest_index]
#     for i in range(1,len(array)):
#         if array[i] < smallest:
#             smallest_index = i
#             smallest = array[smallest_index]
#     return smallest
#
# def SelectionSort(array):
#     new_array = []
#     length = len(array)
#     for i in range(length):

# data = [1,3,5,7,9]
# data.insert(1,'2')
# print(data)


# same = []
# for x in M:
#     if x in N:
#         print(x)
#
# # print(same)


M = list(input("please input M:"))
N = list(input("please input N:"))

def findthesame(M, N):
    m = len(M)
    n = len(N)

    maximum = 0         # 最长匹配的长度
    p = 0

    record = [[0 for i in range(m+1)] for j in range(n+1)]        # 生成0矩阵，为方便后续计算，比字符串长度多了一列
    for i in range(m):
        for j in range(n):
            if M[i] == N[j]:
                record[i+1][j+1] = record[i][j] + 1
                if record[i+1][j+1] > maximum:
                    maximum = record[i+1][j+1]     # 获取最大匹配长度
                    p = i+1                        # 记录最大匹配长度的终止位置
    # print(record)
    # print(p)
    return M[p-maximum:p]

a=''
print(a.join(findthesame(M,N)))


# def find_lcsubstr(s1, s2):
#     m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]  # 生成0矩阵，为方便后续计算，比字符串长度多了一列
#     mmax = 0  # 最长匹配的长度
#     p = 0  # 最长匹配对应在s1中的最后一位
#     for i in range(len(s1)):
#         for j in range(len(s2)):
#             if s1[i] == s2[j]:
#                 m[i + 1][j + 1] = m[i][j] + 1
#                 if m[i + 1][j + 1] > mmax:
#                     mmax = m[i + 1][j + 1]
#                     p = i + 1
#     return s1[p - mmax:p], mmax  # 返回最长子串及其长度
#
#
#
# print(find_lcsubstr('abcdfg', 'abdfg'))
