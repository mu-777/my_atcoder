


# ---------------------------------------------------------------
# https://atcoder.jp/contests/abc134/tasks/abc134_e
import bisect

n = int(input())
arr = [int(input()) for _ in range(n)]
colors = []

colors.append(arr[-1])
cnum = len(colors)
for a in arr[-2::-1]:
  i = bisect.bisect_right(colors, a)
  if i < cnum:
    colors[i] = a
  else:
    colors.append(a)
    cnum += 1

print(cnum)


# TLE
colors = {}
cnum = 0

colors[cnum] = arr[-1]
for a in arr[-2::-1]:
  for c in range(cnum + 1):
    if colors[c] > a:
      colors[c] = a
      break
  else:
    cnum += 1
    colors[cnum] = a

print(cnum+1)



# ---------------------------------------------------------------
# https://atcoder.jp/contests/abc134/tasks/abc134_d

n = int(input())
arr = [int(x) for x in input().split()]
balls = {}
result = []
reslen = 0

for i in range(n, n // 2, -1):
  if arr[i - 1] == 1:
    balls[i - 1] = 1
    result.append(str(i))
    reslen += 1
  else:
    balls[i - 1] = 0

for i in range(n // 2, 0, -1):
  s = sum([balls[i * m - 1] for m in range(2, n // i + 1)])
  if arr[i - 1] == 1:
    if (s % 2) == 1:
      balls[i - 1] = 0
    else:
      balls[i - 1] = 1
      result.append(str(i))
      reslen += 1
  if arr[i - 1] == 0:
    if (s % 2) == 0:
      balls[i - 1] = 0
    else:
      balls[i - 1] = 1
      result.append(str(i))
      reslen += 1

print(reslen)
if reslen != 0:
  print(' '.join(result))

# ---------------------------------------------------------------
# https://atcoder.jp/contests/abc134/tasks/abc134_c

n = int(input())
arr = [int(input()) for _ in range(n)]
sorted_arr = sorted(arr, reverse=True)
result = []

for a in arr:
  multi_same_num_flag = False
  for m in sorted_arr:
    if m > a:
      result.append(str(m))
      break
    elif m == a:
      if not multi_same_num_flag:
        multi_same_num_flag = True
        continue
      else:
        result.append(str(m))
        break
    elif m < a:
      result.append(str(m))
      break

print('{}'.format('\n'.join(result)))

# ---------------------------------------------------------------
# https://atcoder.jp/contests/abc134/tasks/abc134_b
import math

inputs = [int(x) for x in input().split()]
n, d = inputs[0], inputs[1]

print(math.ceil(n / (2 * d + 1)))

# ---------------------------------------------------------------
# https://atcoder.jp/contests/abc134/tasks/abc134_a

r = int(input())
print(3 * r * r)

# ---------------------------------------------------------------
# https://atcoder.jp/contests/abc136/tasks/abc136_e

inputs = [int(x) for x in input().split()]
n, k = inputs[0], inputs[1]
ls = [int(x) for x in input().split()]
s = sum(ls)

# https://qiita.com/LorseKudos/items/9eb560494862c8b4eb56
divisors = []
for i in range(1, int(s ** 0.5) + 1):
  if s % i == 0:
    divisors.append(i)
    if i != s // i:
      divisors.append(s // i)

divisors.sort(reverse=True)
ans = 1
for d in divisors:
  mods = sorted([l % d for l in ls if l % d != 0])
  opcnt = 0
  is_success = False
  # print("{}: {}".format(d, ", ".join([str(m) for m in mods])))
  while True:
    if opcnt > k or len(mods) == 1:
      # print("opcnt: {}".format(opcnt))
      is_success = False
      break

    if len(mods) == 0:
      is_success = True
      ans = d
      # print("ans: {}, opcnt: {}".format(ans, opcnt))
      break

    if mods[0] + mods[-1] < d:
      mods[-1] += mods[0]
      opcnt += mods[0]
      mods.pop(0)
    elif mods[0] + mods[-1] == d:
      opcnt += mods[0]
      mods.pop(0)
      mods.pop(-1)
    else:
      opcnt += (d - mods[-1])
      mods[0] -= (d - mods[-1])
      mods.pop(-1)
    # print("{}: {}".format(d, ", ".join([str(m) for m in mods])))

  if is_success:
    break

print(ans)

# ---------------------------------------------------------------
# https://atcoder.jp/contests/abc136/tasks/abc136_d

ss = input()
lr = [1]
ans = [0] * len(ss)

for s in ss[1:]:
  if lr[-1] > 0:
    if s == "R":
      lr[-1] += 1
    else:
      lr.append(-1)
  else:
    if s == "L":
      lr[-1] -= 1
    else:
      lr.append(1)

i = 0
for num in lr:
  is_r = (num > 0)
  num = abs(num)
  half = num // 2
  if is_r:
    i += num
    ans[i - 1] += num - half
    ans[i] += half
  else:
    ans[i - 1] += half
    ans[i] += num - half
    i += num

print(" ".join([str(c) for c in ans]))

# TLE

ss = input()
l = [0 for i in range(len(ss))]

for i, s in enumerate(ss):
  p = i
  while ss[p] == ss[i]:
    p = p + (1 if ss[p] == "R" else -1)

  if (p - i) % 2 == 0:
    l[p] = l[p] + 1
  elif ss[p] == "L":
    l[p - 1] = l[p - 1] + 1
  else:
    l[p + 1] = l[p + 1] + 1

print(" ".join([str(c) for c in l]))

# ---------------------------------------------------------------
# https://atcoder.jp/contests/abc136/tasks/abc136_c
n = int(input())
hs = [int(x) for x in input().split()]

for i in range(1, n):
  if hs[i - 1] < hs[i]:
    hs[i] = hs[i] - 1

for i in range(1, n):
  if hs[i - 1] > hs[i]:
    print("No")
    break
else:
  print("Yes")

# ---------------------------------------------------------------
# https://atcoder.jp/contests/abc136/tasks/abc136_b
n = int(input())
cnt = 0
for i in range(n + 1):
  if len(str(i)) % 2 == 1:
    cnt += 1
print(cnt)

# ---------------------------------------------------------------
# https://atcoder.jp/contests/abc136/tasks/abc136_a

inputs = [int(x) for x in input().split()]
a, b, c = inputs[0], inputs[1], inputs[2]

ans = b + c - a
print("{}".format(abs(ans) if ans < 0 else 0))

# ---------------------------------------------------------------
# https://atcoder.jp/contests/abc137/tasks/abc137_e

inputs = [int(x) for x in input().split()]
n, m, p = inputs[0], inputs[1], inputs[2]
inputs = [[int(x) for x in input().split()] for _ in range(m)]
graphs, scores = {}, {}
for x, y, c in inputs:
  if y in graphs.keys():
    graphs[y].append([x, c - p])
  else:
    graphs[y] = [[x, c - p]]
    scores[y] = [None, False]
for k, v in graphs.items():
  graphs[k] = sorted(v, key=lambda x: x[0])

has_loop_in_the_path = False
scores[1] = [0, has_loop_in_the_path]

for goal in range(2, n + 1):
  for start, score in graphs[goal]:
    # 1につながっていない
    if scores[start][0] is None:
      continue
    has_loop_in_the_path = (goal == start) or scores[start][1]
    total = scores[start][0] + score
    if (scores[goal][0] is None) or (total > scores[goal][0]):
      scores[goal] = [total, has_loop_in_the_path]

# print(scores)
if scores[n][0] < 0:
  print(0)
elif scores[n][1] == True:
  print(-1)
else:
  print(scores[n][0])

# ---------------------------------------------------------------
# https://atcoder.jp/contests/abc137/tasks/abc137_d
import sys

sys.setrecursionlimit(10 ** 6)

import collections
import heapq

inputs = [int(x) for x in input().split()]
n, m = inputs[0], inputs[1]
conditions = collections.deque([])
for i in range(n):
  conditions.append([int(x) for x in input().split()])

conditions = sorted(conditions)

cnt, ans = 0, 0
candidates = []
for reversed_day in range(1, m + 1):
  while cnt < len(conditions) and conditions[cnt][0] <= reversed_day:
    heapq.heappush(candidates, -conditions[cnt][1])
    cnt += 1

  if len(candidates) == 0:
    continue
  ans -= heapq.heappop(candidates)

print(ans)

# import itertools
#
# inputs = [int(x) for x in input().split()]
# n, m = inputs[0], inputs[1]
# # conf = [[int(x) for x in input().split()] for _ in range(n)]
# conf = sorted([[int(x) for x in input().split()] for _ in range(n)], reverse=True)
#
# maxres = 0
# for cc in itertools.permutations(conf, min(m, len(conf))):
#   res = 0
#   for i, c in enumerate(cc):
#     if i + c[0] <= m:
#       res = res + c[1]
#     else:
#       break
#   if res > maxres:
#     maxres = res
# print(maxres)

# ---------------------------------------------------------------
# https://atcoder.jp/contests/abc137/tasks/abc137_c

n = int(input())
strs = [sorted(str(input())) for _ in range(n)]

cache = {}
cnt = 0
for s in strs:
  if s in cache.keys():
    cnt = cnt + cache[s]
    cache[s] = cache[s] + 1
  else:
    cache[s] = 1
print(cnt)

# アナグラム総当りでは時間が厳しい
import itertools

n = int(input())
strs = [str(input()) for _ in range(n)]

cnt = 0
while len(strs) > 1:
  cache = []
  for anag_tuple in itertools.permutations(strs.pop()):
    anag = "".join(anag_tuple)
    if (anag not in cache) and (anag in strs):
      cnt += 1
      cache.append(anag)

print(cnt)

# ---------------------------------------------------------------
# https://atcoder.jp/contests/abc137/tasks/abc137_b

inputs = [int(x) for x in input().split()]
k, x = inputs[0], inputs[1]

absmax = 1000000

high = x + k - 1 if x + k - 1 <= absmax else absmax
low = x - k + 1 if x - k + 1 >= -absmax else -absmax

from functools import reduce

print(reduce(lambda prev, curr: "{} {}".format(prev, str(curr)), range(low, high + 1), "")[1:])

# ---------------------------------------------------------------
# https://atcoder.jp/contests/abc137/tasks/abc137_a

inputs = [int(x) for x in input().split()]
a, b = inputs[0], inputs[1]

print(max(max(a + b, a - b), a * b))

# ---------------------------------------------------------------
# https://atcoder.jp/contests/abs/tasks/arc089_a
n = int(input())
trag = [[int(x) for x in input().split()] for _ in range(n)]


def can_access(length, stepnum):
  for possible_length in range(stepnum + 1)[::-1][::2]:
    if possible_length == length:
      return True
  return False


res = False
t_prev, x_prev, y_prev = 0, 0, 0
for t, x, y in trag:
  # print("{}: {},{}".format(t, x, y ))
  diff_t, diff_x, diff_y = t - t_prev, abs(x - x_prev), abs(y - y_prev)
  for i in range(diff_t + 1):
    if can_access(diff_x, i) and can_access(diff_y, diff_t - i):
      # print("step:{},{}  goal:{},{}".format(i, diff_t - i, x, y))
      break
  else:
    break
  t_prev, x_prev, y_prev = t, x, y
else:
  res = True

print("Yes" if res else "No")

# ---------------------------------------------------------------
# https://atcoder.jp/contests/abs/tasks/arc065_a

s = str(input())
candidates = ["dream", "dreamer", "erase", "eraser"]


def findstrinstr(target_str):
  ret = []
  for c in candidates:
    if target_str.find(c) == 0:
      ret.append(target_str[len(c):])
  return ret


ret = [s]
while True:
  next_ret = []
  for s in ret:
    next_ret = next_ret + findstrinstr(s)
  if len(next_ret) == 0:
    print("NO")
    break
  elif "" in next_ret:
    print("YES")
    break
  ret = next_ret

# 再帰で解くとメモリが大きいっぽくてアウトっぽい
# https://qiita.com/lethe2211/items/38303f4120bfd963679a#%E5%86%8D%E5%B8%B0%E6%B7%B1%E5%BA%A6%E3%81%AE%E5%A4%89%E6%9B%B4
#
# s = str(input())
# candidates = ["dream", "dreamer", "erase", "eraser"]
#
# def findstrinstr(target_str):
#   if len(target_str) == 0:
#     return True
#   for c in candidates:
#     if target_str.find(c) == 0:
#       if findstrinstr(target_str[len(c):]):
#         return True
#   return False
#
# if findstrinstr(s):
#   print("YES")
# else:
#   print("NO")

# ---------------------------------------------------------------
# https://atcoder.jp/contests/abs/tasks/abc085_c

inputs = [int(x) for x in input().split()]
n, total = inputs[0], inputs[1]

for x in range(n + 1):
  for y in range(n - x + 1):
    z = n - x - y
    if (10000 * x + 5000 * y + 1000 * z == total):
      print("{} {} {}".format(x, y, z))
      break
  else:
    continue
  break
else:
  print("-1 -1 -1")
