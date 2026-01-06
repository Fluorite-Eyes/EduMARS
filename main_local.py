import os 
import json 
import sys
sys.path.append("/data1/zhaoxuan/ai+education/ai_education_code")
"""
data 输入格式：
{
    "questions":
    "standard_answers":
    "student_answers":(img or text)
    "rubric":
    "prompt":
}
输出格式：
1.(score)(理由)
2.(score)(理由)
...
overall_score:
"""

"""
验证 3种验证
acc mae 离散+sperman

中间验证：
llm_base
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
# 假设 call_api 是一个已经定义的函数，用于调用模型 API
# 它的签名是 call_api(prompt, model_name)
# 请确保你的环境中已经定义了这个函数
from utils import call_api

def get_rubric(single_data, model_name):
    
    prompt = f"""
        请你充当一位专业的、严格遵循评分规则的阅卷老师。你的任务是根据提供的【题目总分】、【标准答案】、【学生作答】和【评价规则（Rubric）】，对学生作答进行逐项批阅，并给出详细的点评和最终得分。

        关键要求：

        逐项评分： 评分必须严格依据【评价规则】中**“阅卷关键步骤”**的顺序进行。

        连锁扣分（仅适用于理科/逻辑题）： 如果学生在某个核心逻辑步骤（通常是数学、物理等学科中作为后续计算基础的步骤）出现本质错误，导致后续步骤完全依赖于错误的结果，那么后续步骤的标准分值将直接记为 0 分（即使其步骤和计算形式正确，但结果不对）。

        输出格式： 输出必须是一个单一的字符串，点评部分和总分部分使用固定分隔符 <<<SCORE_BREAK>>> 连接。

        具体的输出内容
        1. 点评部分（[详细的点评内容]）
        请严格按照上述要求进行输出，依据输入的rubric中的需要批阅的点输出学生作答中对应的内容获得的得分和理由
        2. 总分部分（[最终计算出的总分数]）
        总分必须是根据逐项批阅后，将所有步骤的得分相加得出的最终分数（仅输出数值）。

        格式要求
        请严格使用以下固定格式标志词进行连接：<<<SCORE_BREAK>>>
        在<<<SCORE_BREAK>>>之后只需要输出一个数字就可以了
        学生作答：{single_data["std_ans"]}
        标准答案：{single_data["standard_answer"]}
        题目总分：{single_data["full_score"]}
        可以稍微放宽一点评测标准
        """
    single_data["pred"]=call_api(prompt, model_name)
    return single_data


def get_all_rubrics_parallel(pp, model_name, max_workers):
    
    results = []
   
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务到线程池，并保持对 Future 对象的引用
        future_to_ans = {executor.submit(get_rubric, ans, model_name): ans for ans in pp}
        
        # 使用 tqdm 包装 as_completed，以便在等待结果时显示进度条
        for future in tqdm(as_completed(future_to_ans), total=len(pp), desc="Processing API Calls"):
            try:
                # 获取 get_rubric 函数返回的结果 (std_ans, api_response)
                original_ans = future.result()
                results.append(original_ans)
            except Exception as exc:
                # 记录或处理在 API 调用或处理过程中发生的异常
                print(f'Standard answer processing generated an exception: {exc}')
                # 可以选择将失败的答案标记为 None 或记录错误信息
                results.append({})

    return results

if __name__ == "__main__":
    model_name="DeepSeek-Math-V2"
    
    print(f"call {model_name}")
    # 1. 读取数据
    try:
        with open("/data1/zhaoxuan/ai+education/ai_education_code/dataset/data.json", "r") as f:
            data = json.load(f)
        print(f"total num:{len(data)}")
    except FileNotFoundError:
        print("Error: data__.json not found at the specified path.")
        exit()
    with open("/data1/zhaoxuan/ai+education/ai_education_code/rubric.json","r") as f:
        rubrics=json.load(f)
    data=[x for x in data if x["subject"]=="数学"]
    data=[x for x in data if x["img_path"]=="367022202416_姜楠楠_16_9分.jpg" or x["img_path"]=="367012202303_陈智成_17_9分.jpg"]
    for single_data in data:
        rubric=rubrics[single_data["standard_answer"]]
        single_data["rubric"]=rubric
    pp_to_process = data
 
    # 4. 并行调用 API
    print(f"Starting parallel call {model_name} API calls for {len(pp_to_process)} answers...")
    # 可以根据需要调整 max_workers 的值
    ppp = get_all_rubrics_parallel(pp_to_process, model_name=model_name, max_workers=12)
    print("All API calls completed.")

    # 5. 写入结果
    try:
        with open(os.path.join("/data1/zhaoxuan/ai+education/ai_education_code/results",f"{model_name}_text.json"), "w", encoding='utf-8') as f:
            json.dump(ppp, f, indent=4, ensure_ascii=False)
        print(f"save to {model_name}_text_sample.json")
    except Exception as e:
        print(f"Error saving results: {e}")

    



