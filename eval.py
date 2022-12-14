# +
import pandas as pd 
from tqdm import tqdm
import math
import torch
import argparse
import nltk
from transformers import (
  BertTokenizerFast,
  AutoModel,
  GPT2LMHeadModel,
  GPT2Tokenizer, 
)
import numpy as np
import math
import os
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import tensorflow as tf

### score.py
import nltk
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu
nltk.download('punkt')

# emotion
# fin = open('cvaw3.csv', 'r')
# lines = fin.readlines()
# emotion_dict = dict()
# onegram_emotion_dict = dict()
# for line in lines[1:]:
#     _, word, v, _, a, _, _ = line.split(',')
#     v, a = float(v), float(a)
#     emotion_dict[word] = (v, a)
#     if len(word) == 1:
#         onegram_emotion_dict[word] = (v, a)

word_dict = {}
# +
def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model', type=str,
    #                 help='pretrained model name or path to local checkpoint')
    parser.add_argument('--filename', type=str,
                    help='txt file for evaluation',
                    required=True)
    parser.add_argument('--outputfilename', type=str, default='result.csv',
                    help='output txt file for evaluation')
    parser.add_argument('--topic', default='food', help='topic of specific word')
    # parser.add_argument('--outputfile', type=str)

    args = parser.parse_args()
    
    # fin = open(f'wordlist/{args.topic}.txt')
    # lines = fin.readlines()
    
    # for line in lines:
    #     word = line.strip()
    #     word_dict[word] = 1

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    sep = tokenizer.sep_token_id
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    # model.load_state_dict(torch.load("models/trad10epoch_v4_cont/model-5746730.pkl"))
    model = model.to(device)
    model = model.eval()

    with open(args.filename) as f:
        table = f.readlines()
    context = []
    human = []
    reply = []
    print("loading file...")
    for i in tqdm(range(0, len(table), 4)):

        context.append(table[i].strip())
        reply.append(table[i+1].strip())
        human.append(table[i+2].strip())
    '''
    inputs_format:
    context
    reply
    human
    ----------------------------------------------------------------
    context
    ....
    '''
    ppl_loss = 0
    cppl_loss = 0
    score = 0
    nan_sens = []
    print("calculating PPL, CPPL...")
    with torch.no_grad():

        for sentence in tqdm(reply):
            if sentence == "":
                nan_sens.append("")
                continue
            inputs = torch.LongTensor(tokenizer.encode(sentence)).to(device)
            outputs = model(inputs, labels=inputs, return_dict=True)
            
            if torch.isnan(outputs.loss):
                nan_sens.append(sentence)
            else:
                ppl_loss += outputs.loss.item()
        # # a good chinese chatbot
        # model = GPT2LMHeadModel.from_pretrained('ckiplab/gpt2-base-chinese')
        # model.load_state_dict(torch.load("models/trad10epoch_v4_cont/model-5746730.pkl"))
        # model = model.to(device)
        # model = model.eval()
        for i in tqdm(range(len(reply))):
            if reply[i] == "":
                nan_sens.append("")
                continue
            r = reply[i]
            c = context[i]
            inputs = (torch.LongTensor(tokenizer.encode(c) + tokenizer.encode(r)[1:])).to(device)
              
            # mask = [[0 for x in range(len(tokenizer.encode(c))+1)] + [1 for x in range(len(tokenizer.encode(r))+1)]]
            # mask = torch.LongTensor(mask).to(device)
            # inputs['attention_mask'] = mask
            label = []
            for j in range(len(tokenizer.encode(c))):
                label.append(-100)
            label += tokenizer.encode(r)[1:]
            label = torch.LongTensor([label]).to(device)

            outputs = model(inputs, labels=label, return_dict=True)
            cppl_loss += outputs.loss.item()
    
    print('PPL:', math.exp(ppl_loss/ len(reply)))
    print('CPPL:', math.exp(cppl_loss/len(reply)))
    print("skip {} sentences".format(len(nan_sens)))
    print(nan_sens)
    
    
    ##### score.py ######
    
    print("calculating emotion and length...")
    
    total_valence_score = 0
    total_gt4_freq = 0
    total_emotional_freq = 0
    total_repeat_freq = 0

    total_length = 0

    total_specific_word = 0

    total_onegram_emotion_freq = 0
    
    for sent in human:
        total_length += len(sent)
        valence_score = 0
        gt4_freq = 0
        emotional_freq = 0
        repeat_freq = 0
        onegram_emotion_freq = 0
        # for word in emotion_dict:
        #     if word in sent:
        #         valence_score += (emotion_dict[word][0])
        #         emotional_freq += 1
        #         if len(word) == 1: onegram_emotion_freq += 1
        #         repeat_freq += sent.count(word)
        #         if emotion_dict[word][0] > 4:
        #             gt4_freq += 1

        # for word in word_dict:
        #     if word in sent:
        #         total_specific_word += 1

            
        total_valence_score += valence_score
        total_gt4_freq += gt4_freq
        total_emotional_freq += emotional_freq
        total_repeat_freq += repeat_freq
        total_onegram_emotion_freq += onegram_emotion_freq

    

        
    # loss_1 = 0
    # loss_2 = 0
    # loss_3 = 0
    # loss_4 = 0
    # print("calculating self-bleu...")
    # reply = reply[:1500] ###
    #a_1 = SelfBleu(pool=pool, gram=1)
    #return a_1.get_bleu_parallel()
    #a_2 = SelfBleu(pool=pool, gram=2)
    #a_3 = SelfBleu(pool=pool, gram=3)
    #a_4 = SelfBleu(pool=pool, gram=4)
    #return (a_1.get_bleu_parallel(), a_2.get_bleu_parallel(), a_3.get_bleu_parallel(), a_4.get_bleu_parallel())
    # for i in range(len(reply)):
    #     reply[i] = nltk.word_tokenize(reply[i])
    # for i in tqdm(range(len(reply))):
    #     sen = reply[i]
    #     #print(sen)
    #     other = reply[:i] + reply[i+1:]
    #   #  loss_1 += sentence_bleu(other, sen, weights=(1, 0, 0, 0), smoothing_function=SmoothingFunction().method1)
    #     loss_2 += sentence_bleu(other, sen, weights=(0.5, 0.5, 0, 0), smoothing_function=SmoothingFunction().method1)
    #     loss_3 += sentence_bleu(other, sen, weights=(1/3, 1/3, 1/3, 0), smoothing_function=SmoothingFunction().method1)
    #   #  loss_4 += sentence_bleu(other, sen, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=SmoothingFunction().method1)
    # total = len(reply)
    # print("self-bleu:", loss_2/total, loss_3/total)
    
    # fout = open(args.outputfile, 'w')
    # fout.write("filaname,PPL,CPPL,#NAN,length,valence,positive freq,emo freq,emo count,bleu-2,bleu-3\n")
    # fout.write(f"{args.filename},{math.exp(ppl_loss/len(reply))},{math.exp(cppl_loss/len(reply))},{len(nan_sens)},{total_length/len(reply)},{total_valence_score/len(reply)},{total_gt4_freq/len(reply)},{total_emotional_freq/len(reply)},{total_repeat_freq/len(reply)},{loss_2/total},{loss_3/total}\n")
    # fout.close()
    # fout = open(args.outputfilename, 'a')
    # fout.write(f"filaname,PPL,CPPL,#NAN,length,valence,positive emo word freq,emo word freq,emo word count,bleu-2,bleu-3, {args.topic} specific word freq, one gram emo word freq\n")
    # fout.write(f"{args.filename},{math.exp(ppl_loss/len(reply))},{math.exp(cppl_loss/len(reply))},{len(nan_sens)},{total_length/len(reply)},{total_valence_score/len(reply)},{total_gt4_freq/len(reply)},{total_emotional_freq/len(reply)},{total_repeat_freq/len(reply)},{loss_2/total},{loss_3/total},{total_specific_word/len(reply)},{total_onegram_emotion_freq/len(reply)}\n")
    # fout.close()

if __name__ == '__main__':
    main()
