import re

def normalize_author_name_for_generated(name):
    """
    Normalize author names to the format: [Last Name] [First Initial(s)] (e.g., 'Smith J' or 'Li XK')
    """
    # 去除名字中的多余空格
    name = name.strip()
    # print(name)
    # 将姓氏和名字分开
    parts = name.split(',')
    
        
    # 如果格式是 "姓氏, 名字"
    if len(parts) == 2:
        surname = parts[0].strip()
        # print("parts[1]",parts[1])
        if parts[1].strip() != "":
            first_names = parts[1].split()[0].strip()
        else:
            first_names = ""
    else:
        # 如果没有逗号的情况（例如 "名字 姓氏"）
        parts = name.split()
        # print(parts)
        surname = parts[-1]  # 姓氏是最后一个部分
        first_names = ' '.join(parts[:-1])  # 名字部分

    # 提取名字的首字母
    # print("first_names",first_names)
    first_names = first_names.strip()
    if first_names != "":
        initials = first_names.split()[0][0].upper()
    else:
        initials = ""
    # else:
    #     initials = first_names[0].upper()
    return surname.lower(), initials

def normalize_author_name_for_original_paper(name):
    """
    将作者名字规范化，返回姓氏和名字首字母的形式。
    """
    # 去除名字中的多余空格
    name = name.strip()
    # print(name)
    # 将姓氏和名字分开
    parts = name.split(',')
    # 如果格式是 "姓氏, 名字"
    if len(parts) == 2:
        surname = parts[0].strip()
        first_names = parts[1].strip()
        initials = ''.join([p[0].upper() for p in first_names.split()])
    else:
        parts = name.replace(".", "").split()
        if len(parts) >= 2:
            surname = parts[0].strip()
            first_names = ''.join(parts[1::]).strip()
            initials = first_names
        else:
            surname = ""
            initials = parts[0]
            # print("please check name in the original paper", name)

    return surname.lower(), initials

def compare_author_lists_for_t1_external(list1, list2):
    """
    比较两个作者列表，返回相同作者的名字。
    """
    # 使用字典保存标准化后的作者
    list1_ = list1
    list1=  []
    for a in list1_:
        if a.strip() != "":
            list1.append(a)
    normalized_list1 = {normalize_author_name_for_generated(author): author for author in list1}
    list2_ = list2
    list2 = []
    for a in list2_:
        if a.strip() != "":
            list2.append(a)
    normalized_list2 = {normalize_author_name_for_generated(author): author for author in list2}
    
    # 找出交集
    # print(list1)
    # print(list2)
    # print(normalized_list1)
    # print(normalized_list2)
    # # # exit()
    common_authors = set(normalized_list1.keys()).intersection(set(normalized_list2.keys()))
    # print(common_authors)
    # exit()
    return [(normalized_list1[author], normalized_list2[author]) for author in common_authors]


def compare_author_lists_for_t1_internal(list1, list2):
    """
    比较两个作者列表，返回相同作者的名字。
    """
    # 使用字典保存标准化后的作者
    list1_ = list1
    list1=  []
    for a in list1_:
        if a.strip() != "":
            list1.append(a)
    normalized_list1 = {normalize_author_name_for_original_paper(author): author for author in list1}
    list2_ = list2
    list2 = []
    for a in list2_:
        if a.strip() != "":
            list2.append(a)
    normalized_list2 = {normalize_author_name_for_generated(author): author for author in list2}
    
    # 找出交集
    # print(normalized_list1)
    # print(normalized_list2)
    # exit()
    common_authors = set(normalized_list1.keys()).intersection(set(normalized_list2.keys()))
    
    # 返回相同作者的原始名字
    return [(normalized_list1[author], normalized_list2[author]) for author in common_authors]

# 示例列表
# list1 = ["Smith, John", "Doe, Jane", "Brown, B.", "Johnson, A. K."]
# list2 = ["John Smith", "J. Doe", "A. K. Johnson", "Brown, Bob"]

# # 找出相同的作者
# common_authors = compare_author_lists(list1, list2)
# print(common_authors)
# if __name__=="__main__":
#     list1 = ["John Michael Smith"]
#     list2 = ["Smith, J. M."]
#     print(compare_author_lists_for_t1_external(list1, list2))