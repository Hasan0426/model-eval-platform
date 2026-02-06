import pandas as pd
import numpy as np

def generate_structured_data(filename="train_data.csv", rows=10000):
    np.random.seed(42)
    
    # 1. 生成特征 (Features)
    # ------------------------------------------------
    # 部门：给每个部门设定一个“性别倾向系数”
    depts = ["Sales", "Engineering", "HR", "Marketing"]
    dept_bias = {"Sales": 0.2, "Engineering": 0.8, "HR": 0.1, "Marketing": 0.4}
    
    department = np.random.choice(depts, size=rows)
    
    # 年龄 & 工资
    age = np.random.randint(22, 60, size=rows)
    # 让工资和年龄挂钩（越老工资越高），稍微真实一点
    base_salary = 40000 + (age - 20) * 2000 
    salary = base_salary + np.random.normal(0, 10000, size=rows)
    
    # 2. 生成目标 (Target: Gender) - 核心修改！！！
    # ------------------------------------------------
    # 我们制定一个“上帝规则”：
    # 性别得分 = 部门系数 + (工资是否高于平均线) * 0.3 + 随机噪音
    
    gender_list = []
    for i in range(rows):
        # 基础分来自部门（比如 Engineering 更容易是 Male）
        score = dept_bias[department[i]]
        
        # 额外分：如果工资极高，增加是 Male 的概率（仅为模拟数据规律，无刻板印象之意）
        if salary[i] > 80000:
            score += 0.3
            
        # 增加一点随机扰动，不让模型达到 100% 准确（防止过拟合）
        noise = np.random.normal(0, 0.1)
        final_score = score + noise
        
        # 根据得分划分性别
        if final_score > 0.6:
            gender_list.append("Male")
        elif final_score < 0.3:
            gender_list.append("Female")
        else:
            # 中间模糊地带随机分配
            gender_list.append(np.random.choice(["Male", "Female"]))

    # 3. 组装 DataFrame
    data = {
        "user_id": [f"U{i:04d}" for i in range(rows)],
        "age": age,
        "salary": salary.round(2),
        "department": department,
        "city": np.random.choice(["New York", "London", "Paris", "Tokyo"], size=rows),
        "gender": gender_list  # 🔥 这里不再是随机的，而是由上面逻辑生成的
    }
    
    df = pd.DataFrame(data)
    
    # 模拟一些缺失值（只在特征上，不在 Target 上，方便训练）
    df.loc[np.random.choice(rows, size=50), "age"] = None 
    
    df.to_csv(filename, index=False)
    print(f"✅ Generated {filename} with logical patterns.")

if __name__ == "__main__":
    generate_structured_data()