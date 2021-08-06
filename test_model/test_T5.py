from transformers import BertTokenizer, T5EncoderModel, T5ForConditionalGeneration, Text2TextGenerationPipeline

tokenizer = BertTokenizer.from_pretrained('uer/t5-small-chinese-cluecorpussmall')
model = T5ForConditionalGeneration.from_pretrained('uer/t5-small-chinese-cluecorpussmall')
input_ids = tokenizer('一只extra0在extra1里走', return_tensors='pt').input_ids
labels = tokenizer('extra0狗extra1公园extra2</s>', return_tensors='pt').input_ids
outputs = model(input_ids=input_ids, labels=labels)
loss = outputs.loss
logits = outputs.logits

print(loss, logits)

input_ids = tokenizer("总结：研究表明养狗对人", return_tensors="pt").input_ids  # Batch size 1
outputs = model.generate(input_ids) 
generated = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(generated) #output ['extra0']

input_ids = tokenizer("总结：研究表明养狗对人extra0", return_tensors="pt").input_ids  # Batch size 1
outputs = model.generate(input_ids) 
generated = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(generated) #output ['extra0 好 extra1 extra2 extra3 extra4 extra5 extra6 extra7 extra8']

text2text_generator = Text2TextGenerationPipeline(model, tokenizer) 
output = text2text_generator("一只extra0在extra1里走", max_length=50, do_sample=False)
print(output) #output [{'generated_text': 'extra0 猫 extra1 哪 extra2 extra3 extra4 ...'}]