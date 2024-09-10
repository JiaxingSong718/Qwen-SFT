from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = snapshot_download('qwen/Qwen-1_8B-Chat-Int4')

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="auto",
    trust_remote_code=True
).eval()

# Q = '4月6号青岛天气预报?'
# Q = '我打算6月1号去北京旅游，请问天气怎么样？'
Q = '2020年4月16号三亚下雨吗？'
# Q = '你们打算1月3号坐哪一趟航班去上海？'
# Q = '一起去哈尔滨看冰雕么，大概是1月15号左右，我们3个人一起'

prompt_template='''
给定一句话：“%s”，请你按步骤要求工作。

步骤1：识别这句话中的城市和日期共2个信息
步骤2：根据城市和日期信息，生成JSON字符串，格式为{"city":城市,"date":日期}

请问，这个JSON字符串是：
'''
prompt = prompt_template%(Q,)
respone,history = model.chat(tokenizer,prompt,history=None)
print(respone)