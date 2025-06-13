import random
from collections import defaultdict, deque

# 假设有8000个单词，索引从0到7999
total_words = 8000
words_per_group = 50
total_groups = total_words // words_per_group

# 复习周期
review_intervals = [1, 2, 4, 7, 15, 30, 60, 90]

# 打乱顺序
indices = list(range(total_words))
random.shuffle(indices)

# 分组
groups = [indices[i:i + words_per_group] for i in range(0, total_words, words_per_group)]

# 规划每天的学习计划
schedule = []  # 每天的学习计划
review_queue = defaultdict(deque)  # 每天复习的组

# 已学习的组数
learned_groups = 0

for day in range(365):  # 模拟一年
    today_plan = []

    # 添加需要复习的组，最多2组
    if day in review_queue:
        while len(today_plan) < 2 and review_queue[day]:
            today_plan.append(review_queue[day].popleft())

    # 如果今天复习的组少于3个，加入1个新的组
    if len(today_plan) < 3 and learned_groups < total_groups:
        today_plan.append(learned_groups)

        # 将新学的组加入未来的复习计划
        for interval in review_intervals:
            review_day = day + interval
            if review_day < 365:
                review_queue[review_day].append(learned_groups)

        learned_groups += 1

    schedule.append(today_plan)

# 输出结果
ibreak = False
for day, plan in enumerate(schedule[:]):
    print(f"Day {day + 1}:")
    for group_index in plan:
        print(f"  Group {group_index}: {groups[group_index]}")
        # if group_index >= 100:
        #     ibreak = True
        #     break
    # if ibreak:
        # break