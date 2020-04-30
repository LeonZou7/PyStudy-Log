# ---Version: python 3.7---


# list_1 = A 和 list_1[:] = A 的不同之处：
# 在Python3中，对象是一个盒子，有具体的地址，而变量名相当于是 "标签"，可以贴在盒子上。

list_1 = A	# 更改 list_1 这一变量名所指向的对象。让 list_1 变量指向 A 所指向的对象
list_1[:] = A	# 对 list_1 指向的对象赋值。把 A 变量指向的对象的值逐个 copy 到 list_1 指向的对象中并覆盖 list_1 指向的对象的原来值。

A[:2] = [0,1]	# 改变 A 所指向的 list 对象的前两个值。
A = [0,1]	# A这一变量名指向新的 list 对象 [0,1]