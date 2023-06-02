# -*- coding: utf-8 -*-

import torch
import time
import re
# import fasttext
import logging
import pandas as pd
import tritonclient.grpc as grpcclient
import tiktoken

triton_client = grpcclient.InferenceServerClient(
    url="10.200.0.2:8001",  # ---端口
    verbose=False,  # ---日志
    ssl=False,
    root_certificates=None,
    private_key=None,
    certificate_chain=None,
    creds=None,
    keepalive_options=None,
    channel_args=None,
)

# 批量输入检测长度，大于max_length截断，小于max_length padding 0
max_length = 1024
# 模型地址
model_dir = "/home/jovyan/syj_work/finetune_model/0406_binary_classify_100w/"
# 多分类模型地址
multi_model_dir = "/home/jovyan/syj_work/code/type_classification/fasttext_model_cooking_V5.bin"
# 多分类模型
model_back = None
# 多分类模型引入，不打印错误信息或调试信息
# fasttext.FastText.eprint = lambda x: None
# 多分类参数
type_transformer = {"__label__1": "java_inject",
                    "__label__2": "cmd_inject",
                    "__label__3": "java_deser",
                    "__label__4": "php_inject",
                    "__label__5": "sql_inject",
                    "__label__6": "webshell_upload",
                    "__label__7": "xss",
                    }

black_threshold = {"__label__1": 0.25,
                   "__label__2": 0.6,
                   "__label__3": 0.6,
                   "__label__4": 0.25,
                   "__label__5": 0.6,
                   "__label__7": 0.25,
                   }

""" 未知攻击 1;  java_inject 2;  cmd_inject 3;  java_deser 4;  
php_inject 5;  sql_inject 6;  webshell_upload 7;  xss 8;"""
convert_attack_to_index = {
    "other_attack_type": 1,
    "java_inject": 2,
    "cmd_inject": 3,
    "java_deser": 4,
    "php_inject": 5,
    "sql_inject": 6,
    "webshell_upload": 7,
    "xss": 8
}

# 标识GPU ID
all_log_num = 0
detect_black_nums = 0
tokenizer_new = None
all_detect_nums = 0
model_infer_time = 0
# secgpt检测逻辑入口
data_list = []


# 模型加载初始化
def sec_gpt_init(input_id):
    global tokenizer_new, model_back
    tokenizer_new = tiktoken.get_encoding("gpt2")
    # 多分类模型加载初始化
    # model_back = fasttext.load_model(multi_model_dir)

# 多分类预测
def multi_class_predict(payload):
    payload = re.sub('[\r\n ]', '', payload)
    final_payload = " ".join([x for x in payload])
    out = model_back.predict(final_payload, k=1)
    top_1_out = out[0][0]  # key值
    top_1_value = out[1][0]
    if "multipart/form-data" in payload and "filename=" in payload:
        return "webshell_upload"
    if top_1_value > black_threshold[top_1_out]:
        return type_transformer[top_1_out]
    return "other_attack_type"


# tensor长度统一
def deal_input_tensor(input_ids, attention_mask, input_max_length):
    # 升维度
    input_ids = input_ids.unsqueeze(0)
    attention_mask = attention_mask.unsqueeze(0)
    length = len(input_ids[0])
    if len(input_ids) != len(attention_mask):
        logging.info("secgpt_detects input_ids and attention_mask length error")
        return 0, input_ids, attention_mask
    # 小于max_length, 填充0
    if length < input_max_length:
        padding_tensor = torch.zeros((input_max_length - length,), dtype=torch.long)
        # input_ids做50256填充, 标示词汇表中的最后一个编号+1
        padding_tensor_value = torch.full((input_max_length - length,), 50256, dtype=torch.long)
        padding_tensor = padding_tensor.unsqueeze(0)
        padding_tensor_value = padding_tensor_value.unsqueeze(0)
        input_ids = torch.cat([input_ids, padding_tensor_value], dim=1)
        attention_mask = torch.cat([attention_mask, padding_tensor], dim=1)
    # 转换成功 1024
    if len(input_ids[0]) == input_max_length and len(attention_mask[0]) == input_max_length:
        return input_ids, attention_mask
    else:
        # 转换失败
        return input_ids, attention_mask


# 超长token处理
def remove_consecutive_strings(str_list):
    min_len = 17
    index_start = 0
    new_list = []
    for index in range(len(str_list)):
        if len(str_list[index]) <= 2:
            continue
        if index - index_start < min_len:
            new_list += str_list[index_start:index]
        index_start = index
    if index - index_start < min_len:
        new_list += str_list[index_start:index + 1]
    return new_list


# 批量检测检测逻辑
def detect_log_in_triton(batch_input_ids, batch_attention_masks, batch_max_length, input_index, result_dicts):
    global detect_black_nums, model_infer_time, all_detect_nums

    batch_max_length = 512
    # 按照最大长度裁剪
    batch_input_ids_detect = batch_input_ids.narrow(1, 0, batch_max_length)
    batch_attention_masks_detect = batch_attention_masks.narrow(1, 0, batch_max_length)
    batch_length = (batch_input_ids_detect.shape)[0]
    # print("batch_length ", batch_length)
    # print("batch_input_ids.shape ", batch_input_ids_detect.shape)
    input_ids = batch_input_ids_detect.numpy()
    attention_mask = batch_attention_masks_detect.numpy()
    batch_size = 106
    input_ids = input_ids[:batch_size]
    attention_mask = attention_mask[:batch_size]
    print("input_ids : ", input_ids.shape)
    print("input_ids : ", type(input_ids))

    # print("input_ids.shape : ", input_ids.shape)
    inputs = []
    outputs = []
    inputs.append(grpcclient.InferInput(name='input_ids', shape=input_ids.shape, datatype="INT32"))
    inputs.append(grpcclient.InferInput(name='attention_mask', shape=attention_mask.shape, datatype="INT32"))
    outputs.append(grpcclient.InferRequestedOutput('logits'))
    input_ids = input_ids.astype('int32')
    attention_mask = attention_mask.astype('int32')
    # Initialize the data
    inputs[0].set_data_from_numpy(input_ids)
    inputs[1].set_data_from_numpy(attention_mask)
    start_time = time.time()
    try:
        # Test with outputs     ---第一种是有指定outputs
        results = triton_client.infer(
            model_name="plan_model",  # ---模型名称
            # model_version="1",      # ---请求变量
            inputs=inputs,  # ---版本名称
            outputs=outputs,  # ---请求输出对象的列表，可以为空
            client_timeout=1000  # ---每个请求的超时值，以微秒为单位
        )  # ---允许请求占用的最大端到端时间，以秒为单位
    except Exception as e:
        print("error", e)
        print("input_ids : ", input_ids)
    end_time = time.time()
    elapsed_time_s = (end_time - start_time)
    model_infer_time += elapsed_time_s
    logits = results.as_numpy('logits')
    for i, logit in enumerate(logits):
        all_detect_nums += 1
        predicted_class_id = logit.argmax().item()
        if predicted_class_id:
            detect_black_nums += 1
        index = input_index[i]
        if predicted_class_id == 1:
            result_dicts[index]["attackType"] = 1
        elif predicted_class_id == 0:
            result_dicts[index]["attackType"] = 0
    print(f"本批次 : {batch_size / elapsed_time_s:.2f} it/s \n")
    # print("result_dicts : ", result_dicts)
    return result_dicts


# 按照token长度对流量进行分类(0~512, 512~1024)，再根据不同长度进行分批检测
def detect_log_batch(httpLogs):
    start_time = time.time()
    global max_length
    batch_input_ids_1024 = torch.empty((0, max_length), dtype=torch.int)
    batch_attention_masks_1024 = torch.empty((0, max_length), dtype=torch.int)
    result_dicts = []
    # 日志流量遍历
    for i, data_per in enumerate(httpLogs):
        encoded_input_new = tokenizer_new.encode(data_per)
        # 超长token处理
        encoded_input_length = len(encoded_input_new)
        if encoded_input_length > max_length:
            tokens = [tokenizer_new.decode_single_token_bytes(token) for token in encoded_input_new]
            new_tokens = remove_consecutive_strings(tokens)
            encoding_res = [tokenizer_new.encode_single_token(token) for token in new_tokens]
            ff = encoding_res[:max_length]  # 最大裁剪到1024
            input_ids = torch.tensor(ff).int()
        else:
            input_ids = torch.tensor(encoded_input_new)
        attention_mask = torch.ones_like(input_ids)
        # 保存token长度，用于排序分批
        length = (input_ids.shape)[0]
        # 升维
        input_ids, attention_mask = deal_input_tensor(input_ids, attention_mask, max_length)
        # 把所有流量统一到1024，保存到tensor里面
        batch_input_ids_1024 = torch.cat((batch_input_ids_1024, input_ids), dim=0)
        batch_attention_masks_1024 = torch.cat((batch_attention_masks_1024, attention_mask), dim=0)
        # 保存检测结果{索引, 预测值}
        log_dict = {"id": i, "length": length, "attackType": -1}
        result_dicts.append(log_dict)

    #  按照长度进行长度排序
    sorted_dict_list = sorted(result_dicts, key=lambda x: x['length'])
    batch_num = 0                         # 统计本批次送入triton的batch大小
    leave_num = len(sorted_dict_list)     # 统计剩余流量数量
    detect_index = []
    for dict_tmp in sorted_dict_list:    # 对sorted_dict_list进行遍历
        # 对检测结果进行处理
        if batch_num == 0:
            batch_input_ids_detect = torch.empty((0, max_length), dtype=torch.int)
            batch_attention_masks_detect = torch.empty((0, max_length), dtype=torch.int)
            detect_index = []

        index = dict_tmp["id"]  # index里面保存初始索引i
        detect_index.append(index)  # 保存本批次索引i
        batch_input_ids_detect = torch.cat((batch_input_ids_detect, batch_input_ids_1024[index].unsqueeze(0)), dim=0)
        batch_attention_masks_detect = torch.cat((batch_attention_masks_detect, batch_attention_masks_1024[index].unsqueeze(0)), dim=0)
        batch_num += 1
        leave_num -= 1
        batch_max_length = dict_tmp["length"]
        # if leave_num == 0:
        #     result_dicts = detect_log_in_triton(batch_input_ids_detect, batch_attention_masks_detect, batch_max_length, detect_index, result_dicts)
        #     batch_num = 0
        #     continue
        # if leave_num <= 12 and batch_num < 106 - 12:
        #     continue
        # 判断准则
        if batch_num >= 106:
            result_dicts = detect_log_in_triton(batch_input_ids_detect, batch_attention_masks_detect, batch_max_length, detect_index, result_dicts)
            batch_num = 0
            continue
        # if batch_num >= 76 and batch_max_length > 408:
        #     result_dicts = detect_log_in_triton(batch_input_ids_detect, batch_attention_masks_detect, batch_max_length, detect_index, result_dicts)
        #     batch_num = 0
        #     continue
        # if batch_num >= 60 and batch_max_length > 512:
        #     result_dicts = detect_log_in_triton(batch_input_ids_detect, batch_attention_masks_detect, batch_max_length, detect_index, result_dicts)
        #     batch_num = 0
        #     continue
        # if batch_num >= 48 and batch_max_length > 796:
        #     result_dicts = detect_log_in_triton(batch_input_ids_detect, batch_attention_masks_detect, batch_max_length, detect_index, result_dicts)
        #     batch_num = 0
        #     continue
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("elapsed_time s : ", elapsed_time)
    print("length : ", len(sorted_dict_list))
    logging.info(f"本批次 : {len(sorted_dict_list) / elapsed_time:.2f} it/s")
    # 消息队列返回
    return result_dicts


# log_dict = {"id": i, "length": length, "attackType": -1}
"""对外api接口, 只输出判黑流量"""
def secgpt_detect_outapi(resultQueue_datas):
    start_time = time.time()
    global all_log_num
    # 保存输入流量
    httpLogs = []
    this_input_length = len(resultQueue_datas)  # 本批次数据数量
    all_log_num += this_input_length
    for httpLog in resultQueue_datas:
        request = httpLog["requestData"]
        httpLogs.append(request)
    out_datas = detect_log_batch(httpLogs)

    # out_data = {"id": i, "length": length, "attackType": -1} {输入httpLogs索引, token长度, 攻击类型}
    out_results_list = []

    for i, out_data in enumerate(out_datas):
        if out_data["attackType"] == 1:
            index = out_data["id"]   # 索引
            out_attack_type = multi_class_predict(httpLogs[index])
            attack_type = convert_attack_to_index[out_attack_type] # 攻击类型转换为数字
            out_result = {"id": index, "attackType": attack_type}
            out_results_list.append(out_result)

    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info("secgpt输入流量数量 : {}".format(this_input_length))
    logging.info(f"secgpt_detects 该批次代码执行时间为 {elapsed_time:.2f} 秒")
    logging.info(f"secgpt检测 eps : {this_input_length / elapsed_time:.2f} it/s")
    logging.info("secgpt_detects all_log_num : {}".format(all_log_num))
    return out_results_list


# 测试接口
def secgpt_detect_test(resultQueue_datas):
    global data_list, all_log_num, out_data
    start_time = time.time()
    # 保存输入流量
    httpLogs = []
    this_input_length = len(resultQueue_datas)  # 本批次数据数量
    logging.info("secgpt_detects input length: {}".format(this_input_length))
    for httpLog in resultQueue_datas:
        request = httpLog["requestData"]
        httpLogs.append(request)
    out_datas = detect_log_batch(httpLogs)
    # for out_data in out_datas:
    #     index = out_data["id"]
    #     predict_id = out_data["attackType"]
    #     data = {'http_log': httpLogs[index], "predicted_class_id": predict_id}
    #     data_list.append(data)
    # 目前检测到的所有流量条数
    all_log_num += len(out_datas)
    logging.info("secgpt_detects all_log_num : {}".format(all_log_num))
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"secgpt_detects 该批次代码执行时间为 {elapsed_time:.2f} 秒")
    logging.info(f"secgpt_detects eps : {this_input_length / elapsed_time:.2f} it/s")
    return out_datas


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s - %(filename)s - [func:%(funcName)s] - [line:%(lineno)d] - %(levelname)s: %(message)s',
        level=logging.INFO)
    gpu_id = 0
    # 初始化模
    logging.info("开始加载模型初始化")
    sec_gpt_init(gpu_id)
    logging.info("初始化成功")
    file_name = 'http_log_2500_add0601.csv'
    # file_name = 'gpt_dataset_http_black_decode17026.csv'
    df = pd.read_csv(file_name, encoding='ISO-8859-1')
    num_index = df.shape[0]
    http_log_list = []
    start_time = time.time()

    for row in range(0, num_index):
        http_log = df.loc[row, 'http_log']  # get post
        http_log_dict = {"requestData": http_log}
        http_log_list.append(http_log_dict)
        # 开始检测
        if (row + 1) % 512 == 0 or row == num_index - 1:
            secgpt_detect_test(http_log_list)
            http_log_list = []
    print("输入数据总数量 : ", num_index)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("总共判黑流量 : ", detect_black_nums)
    print("GPU总共检测到的数量 : ", all_detect_nums)
    print("GPU推理耗费时间 : ", model_infer_time)
    print("总共检测时间 elapsed_time : ", elapsed_time)
    print("GPU qps : ", all_detect_nums / model_infer_time)
    print("总共 qps: ", all_detect_nums / elapsed_time)
    df_save = pd.DataFrame(data_list)
    filename = "./result_0529/{}".format("triton_detect_result_" + file_name)
    df_save.to_csv(filename, index=True)
