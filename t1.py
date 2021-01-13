#
# voted = {}
# def check_voter(name):
#     if voted.get(name):
#         print('kick them out')
#     else:
#         voted[name] = True
#         print('let them vote')
#
#
# check_voter("tom")
# check_voter("tom")
# check_voter("tom")
# check_voter("jack")
# print(voted)


nums = [2, 7, 11, 15]
target = 9
class Solution:
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        hashmap = {}
        for index, num in enumerate(nums):
            another_num = target - num
            if another_num in hashmap:
                return [hashmap[another_num], index]
            hashmap[num] = index
        return None

# print(twoSum(nums, target))

