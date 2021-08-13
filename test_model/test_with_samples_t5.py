from transformers import BertTokenizer, T5EncoderModel, T5ForConditionalGeneration, Text2TextGenerationPipeline
import torch
import random 
lines = [
'我是学西班牙语的 老师上课的时候老跟我们说西班牙的海鲜饭特好吃 还说他在西班牙的时候老吃 上礼拜就去了 海鲜饭128快 有点贵',
'每次去来福士逛街都会进去逛下这家店，鞋子都是非常的有设计感，种类很多，样式都十分的漂亮。名字也满好听的，穿起来很舒服。', 
'CRISPR到底是怎样的一种技术？是否真的能推动基因治疗？',
'新能源车2017项目开始申报 新机遇或来临导语科技部发布了“新能源汽车”试点专项2017年度项目申报指南',
'如何看待异乡好居老板娘控（wu）告程序员删代码？',
'针对这个问题，宋国君表示，如果开征环保税，而收税之后又去补贴污染企业，则明显违背国际上通行的污染者付费原则。',
'其实是个挺年轻品牌的地方，东西不太贵，也没有太大的品牌，都是流行服饰的品牌，不过男装的品牌不错',
'带着点评的优惠券去的，一进门感觉就有点冷清，也不知道是去早了还是什么关系',
'永磁高铁和上海磁悬浮有什么本质区别？今天的新闻：中国首辆永磁高铁下线试车 最快3年商业化运营。',
'杨绛先生的文字很平实温婉但是却包含着浓浓的感情一字就是一句字字都饱含深意。三联的书设计、印刷真是没得说非常好很精致。',
'朋友生日，家里规矩，他自己又号这口，要吃打卤面。上网查来查去，都没有正宗的打卤面。最后敲定这家。',
'步骤比较详细都能看明白比较适合我这样的新手作品比较实用做出来一个包包还是挺有成就感的'
]

punctuations = [' ', ',', '.', '，', '。']

drop_rate = 0.1
test_lines = []
label_lines = []
for line in lines:
    test_line = []
    t_count = 0
    label_line = []
    l_count = 0
    last_l_extra = False
    last_t_extra = False
    for c in line:
        if(c in punctuations):
            test_line.append(c)
            if(not last_l_extra):
                label_line.append("extra"+str(l_count))
                last_l_extra = True
                l_count+=1
            last_t_extra = False
            continue
        prob = random.random()
        if(prob<drop_rate):
            label_line.append(c)
            if(not last_t_extra):
                test_line.append("extra"+str(t_count))
                last_t_extra = True
                t_count+=1
            last_l_extra = False
        else:
            test_line.append(c)
            if(not last_l_extra):
                label_line.append("extra"+str(l_count))
                last_l_extra = True
                l_count+=1
            last_t_extra = False

    test_lines.append(''.join(test_line))
    label_lines.append(''.join(label_line))
print(test_lines)
print(label_lines)

tokenizer = BertTokenizer.from_pretrained('uer/t5-small-chinese-cluecorpussmall')
model = T5ForConditionalGeneration.from_pretrained('uer/t5-small-chinese-cluecorpussmall')
input_ids = tokenizer(test_lines, return_tensors='pt', padding=True).input_ids
labels = tokenizer(label_lines, return_tensors='pt', padding=True).input_ids

outputs = model(input_ids=input_ids, labels=labels)
loss = outputs.loss
logits = outputs.logits

print("loss", loss) #9.4

#input_ids = tokenizer("总结：研究表明养狗对人", return_tensors="pt").input_ids  # Batch size 1
outputs = model.generate(input_ids) 
generated = tokenizer.batch_decode(outputs)#, skip_special_tokens=True

for index, gen in enumerate(generated):
    print("test line", test_lines[index])
    print("gen line", gen)