import json

def statics(datas, total_nums):
    effe_sample_nums = len(datas)
    correct_sample_nums = 0
    effe_total_time_cost = 0.0
    effe_total_input_tokens = 0
    effe_total_output_tokens = 0

    for data in datas:
        # 假设 'success' 字段表示是否正确
        is_correct = data.get('success', False)
        time_cost = data.get('time_cost', 0.0)
        prompt_tokens = data.get('prompt_tokens_cost', 0)
        completion_tokens = data.get('completion_tokens_cost', 0)

        effe_total_time_cost += time_cost
        effe_total_input_tokens += prompt_tokens
        effe_total_output_tokens += completion_tokens

        if is_correct:
            correct_sample_nums += 1

    all_accuracy = (correct_sample_nums / total_nums) * 100 if total_nums else 0
    effective_accuracy = (correct_sample_nums / effe_sample_nums) * 100 if effe_sample_nums else 0
    avg_time_cost = effe_total_time_cost / effe_sample_nums if effe_sample_nums else 0
    avg_input_tokens = effe_total_input_tokens / effe_sample_nums if effe_sample_nums else 0
    avg_output_tokens = effe_total_output_tokens / effe_sample_nums if effe_sample_nums else 0

    print("有效个数：", effe_sample_nums)
    print("全部准确率：{:.2f}%".format(all_accuracy))
    print("有效准确率：{:.2f}%".format(effective_accuracy))
    print("有效 Avg Time Cost：{:.2f}".format(avg_time_cost))
    print("有效 Avg Input Tokens：{:.2f}".format(avg_input_tokens))
    print("有效 Avg Output Tokens：{:.2f}".format(avg_output_tokens))

def read_data(file_path):
    datas = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    datas.append(data)
                except json.JSONDecodeError as e:
                    print(f"JSON解析错误: {e}")
    return datas

if __name__ == "__main__":
    file_path = 'logs/websearch_messages_20250428_180520.jsonl'
    TOTAL_NUMS = 2
    datas = read_data(file_path)
    statics(datas, TOTAL_NUMS)
