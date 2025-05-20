import re

# 常见的期刊名称缩写及其全称
abbreviation_dict = {
    "j.": "journal",
    "rev.": "review",
    "phys.": "physics",
    "chem.": "chemistry",
    "biol.": "biology",
    "natl.": "national",
    "acad.": "academy",
    "sci.": "science",
    "proc.": "proceedings",
    "med.": "medicine",
    "exp.": "experimental",
    "am.": "american",
    "int.": "international",
    "ann.": "annals",
    "eng.": "engineering",
    "eur.": "european",
    # 你可以根据需要扩展更多缩写
}
stop_words = {"journal", "the", "of", "and", "an", "a", "for", "in", "&"}

def standardize_journal_name(name):
    """
    标准化期刊名称：去除标点符号、转换为小写，并扩展常见的缩写形式。
    """
    # 去除标点符号
    
    # name = re.sub(r'[.,]', '', name)
    
    # 将期刊名称转换为小写并拆分为单词
    words = name.lower().split()

    # 将常见缩写转换为全称
  
    standardized_words = [abbreviation_dict.get(word, word) for word in words]
    standardized_words = [ word for word in standardized_words if word not in stop_words]
    standardized_words = re.sub(r'[.,]', '',  ' '.join(standardized_words))
    # 返回标准化后的期刊名称
    return standardized_words.strip()

def compare_journals(journal1, journal2):
    """
    比较两个期刊名称是否一致。
    """
    # 标准化两个期刊名称
    std_journal1 = standardize_journal_name(journal1)
    std_journal2 = standardize_journal_name(journal2)
    std_journal1 = std_journal1.split(" ")
    std_journal2 = std_journal2.split(" ")
    # print(std_journal2)
    # print(std_journal1)

    if set(std_journal1).issubset(set(std_journal2)) or set(std_journal2).issubset(set(std_journal1)):
        return 1        
    else:
        return 0
        # return std_journal1 == std_journal2

# 示例期刊名称
# journal1 = "J. Chem. Phys."
# journal2 = "Journal of Chemical Physics"

# # 比较期刊名称是否一致
# is_same = compare_journals(journal1, journal2)
# print(f"Are the journals the same? {is_same}")
