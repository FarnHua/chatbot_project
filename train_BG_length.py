import os
import numpy as np
import random
import json
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter
from os.path import join
import re
from argparse import ArgumentParser

#from Emo_detector.detect_emotion import re_emo_score, prepare_model
from transformers import GPT2LMHeadModel, GPT2Tokenizer
# from transformers import BlenderbotSmallTokenizer, BlenderbotSmallForCausalLM
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
# from chat_load import post_set
from lsp_model.optim import Adam
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable

import string
from tqdm import tqdm

import wandb

wandb.login()
torch.manual_seed(100)
emo_dict = {
                "<afraid>": [-0.12, 0.79], 
                "<angry>": [-0.42, 0.79], 
                "<annoyed>": [-0.44, 0.66], 
                "<anticipating>": [0.32, 0.06], # expectant
                "<anxious>":[-0.72, -0.8] , 
                    "<apprehensive>": [-0.77, -0.6], 
                "<ashamed>": [-0.45, -0.5], 
                    "<caring>": [0.25, -0.5], # not worried 
                "<confident>": [0.51, -0.2], 
                "<content>": [0.82, -0.55], 
                    "<devastated>": [-0.8, -0.5], 
                "<disappointed>": [-0.8, -0.03], 
                "<disgusted>": [-0.67, 0.49], 
                "<embarrassed>": [-0.32, -0.6], # caring
                "<excited>": [0.7, 0.72], 
                    "<faithful>": [0.6, 0.2], 
                    "<furious>": [-0.7, 0.85], 
                    "<grateful>": [0.8, -0.08], 
                "<guilty>": [-0.4, -0.43], # feel guilt
                "<hopeful>": [0.62, -0.3], 
                "<impressed>": [0.38, -0.07], 
                "<jealous>": [-0.08, 0.56], 
                "<joyful>": [0.85, 0.15], 
                    "<lonely>": [-0.2, -0.8], 
                    "<nostalgic>": [-0.5, -0.2],  # sentimental
                    # "prepared": , # confident
                    "<proud>": [0.45, 0.07],
                "<sad>": [-0.82, -0.4], 
                    # "sentimental": ,# nostalgic
                "<surprised>": [0.42, 0.79],  # astonished
                    # "terrified": , # afraid
                    "<trusting>": [0.3, 0.2]
            }

# detect_model, detect_processor, emotion_tokenizer = prepare_model()
# bad_word = ["4r5e", "5h1t", "5hit", "a55", "anal", "anus", "ar5e", "arrse", "arse", "ass", "ass-fucker", "asses", "assfucker", "assfukka", "asshole", "assholes", "asswhole", "a_s_s", "b!tch", "b00bs", "b17ch", "b1tch", "ballbag", "balls", "ballsack", "bastard", "beastial", "beastiality", "bellend", "bestial", "bestiality", "bi+ch", "biatch", "bitch", "bitcher", "bitchers", "bitches", "bitchin", "bitching", "bloody", "blow job", "blowjob", "blowjobs", "boiolas", "bollock", "bollok", "boner", "boob", "boobs", "booobs", "boooobs", "booooobs", "booooooobs", "breasts", "buceta", "bugger", "bum", "bunny fucker", "butt", "butthole", "buttmuch", "buttplug", "c0ck", "c0cksucker", "carpet muncher", "cawk", "chink", "cipa", "cl1t", "clit", "clitoris", "clits", "cnut", "cock", "cock-sucker", "cockface", "cockhead", "cockmunch", "cockmuncher", "cocks", "cocksuck", "cocksucked", "cocksucker", "cocksucking", "cocksucks", "cocksuka", "cocksukka", "cok", "cokmuncher", "coksucka", "coon", "cox", "crap", "cum", "cummer", "cumming", "cums", "cumshot", "cunilingus", "cunillingus", "cunnilingus", "cunt", "cuntlick", "cuntlicker", "cuntlicking", "cunts", "cyalis", "cyberfuc", "cyberfuck", "cyberfucked", "cyberfucker", "cyberfuckers", "cyberfucking", "d1ck", "damn", "dick", "dickhead", "dildo", "dildos", "dink", "dinks", "dirsa", "dlck", "dog-fucker", "doggin", "dogging", "donkeyribber", "doosh", "duche", "dyke", "ejaculate", "ejaculated", "ejaculates", "ejaculating", "ejaculatings", "ejaculation", "ejakulate", "f u c k", "f u c k e r", "f4nny", "fag", "fagging", "faggitt", "faggot", "faggs", "fagot", "fagots", "fags", "fanny", "fannyflaps", "fannyfucker", "fanyy", "fatass", "fcuk", "fcuker", "fcuking", "feck", "fecker", "felching", "fellate", "fellatio", "fingerfuck", "fingerfucked", "fingerfucker", "fingerfuckers", "fingerfucking", "fingerfucks", "fistfuck", "fistfucked", "fistfucker", "fistfuckers", "fistfucking", "fistfuckings", "fistfucks", "flange", "fook", "fooker", "fuck", "fucka", "fucked", "fucker", "fuckers", "fuckhead", "fuckheads", "fuckin", "fucking", "fuckings", "fuckingshitmotherfucker", "fuckme", "fucks", "fuckwhit", "fuckwit", "fudge packer", "fudgepacker", "fuk", "fuker", "fukker", "fukkin", "fuks", "fukwhit", "fukwit", "fux", "fux0r", "f_u_c_k", "gangbang", "gangbanged", "gangbangs", "gaylord", "gaysex", "goatse", "God", "god-dam", "god-damned", "goddamn", "goddamned", "hardcoresex", "hell", "heshe", "hoar", "hoare", "hoer", "homo", "hore", "horniest", "horny", "hotsex", "jack-off", "jackoff", "jap", "jerk-off", "jism", "jiz", "jizm", "jizz", "kawk", "knob", "knobead", "knobed", "knobend", "knobhead", "knobjocky", "knobjokey", "kock", "kondum", "kondums", "kum", "kummer", "kumming", "kums", "kunilingus", "l3i+ch", "l3itch", "labia", "lmfao", "lust", "lusting", "m0f0", "m0fo", "m45terbate", "ma5terb8", "ma5terbate", "masochist", "master-bate", "masterb8", "masterbat*", "masterbat3", "masterbate", "masterbation", "masterbations", "masturbate", "mo-fo", "mof0", "mofo", "mothafuck", "mothafucka", "mothafuckas", "mothafuckaz", "mothafucked", "mothafucker", "mothafuckers", "mothafuckin", "mothafucking", "mothafuckings", "mothafucks", "mother fucker", "motherfuck", "motherfucked", "motherfucker", "motherfuckers", "motherfuckin", "motherfucking", "motherfuckings", "motherfuckka", "motherfucks", "muff", "mutha", "muthafecker", "muthafuckker", "muther", "mutherfucker", "n1gga", "n1gger", "nazi", "nigg3r", "nigg4h", "nigga", "niggah", "niggas", "niggaz", "nigger", "niggers", "nob", "nob jokey", "nobhead", "nobjocky", "nobjokey", "numbnuts", "nutsack", "orgasim", "orgasims", "orgasm", "orgasms", "p0rn", "pawn", "pecker", "penis", "penisfucker", "phonesex", "phuck", "phuk", "phuked", "phuking", "phukked", "phukking", "phuks", "phuq", "pigfucker", "pimpis", "piss", "pissed", "pisser", "pissers", "pisses", "pissflaps", "pissin", "pissing", "pissoff", "poop", "porn", "porno", "pornography", "pornos", "prick", "pricks", "pron", "pube", "pusse", "pussi", "pussies", "pussy", "pussys", "rectum", "retard", "rimjaw", "rimming", "s hit", "s.o.b.", "sadist", "schlong", "screwing", "scroat", "scrote", "scrotum", "semen", "sex", "sh!+", "sh!t", "sh1t", "shag", "shagger", "shaggin", "shagging", "shemale", "shi+", "shit", "shitdick", "shite", "shited", "shitey", "shitfuck", "shitfull", "shithead", "shiting", "shitings", "shits", "shitted", "shitter", "shitters", "shitting", "shittings", "shitty", "skank", "slut", "sluts", "smegma", "smut", "snatch", "son-of-a-bitch", "spac", "spunk", "s_h_i_t", "t1tt1e5", "t1tties", "teets", "teez", "testical", "testicle", "tit", "titfuck", "tits", "titt", "tittie5", "tittiefucker", "titties", "tittyfuck", "tittywank", "titwank", "tosser", "turd", "tw4t", "twat", "twathead", "twatty", "twunt", "twunter", "v14gra", "v1gra", "vagina", "viagra", "vulva", "w00se", "wang", "wank", "wanker", "wanky", "whoar", "whore", "willies", "willy", "xrated", "xxx"]
# bad_dict = {}
# for w in bad_word:
#     bad_dict[w] = 1

word_dict = {}

class post_set(Dataset):
    def __init__(self, post, n_tokens, tokenizer):
        #eos = [tokenizer.encoder["<|endoftext|>"]]
        with open(post) as f:
            table = f.readlines()
        # new_table = []
        # eos = [2]
        # for t in table:
        #   input_id = tokenizer.encode(t)
        #   if len(input_id) < 20:
        #     new_table.append(t)
        # self.table = new_table
        temp = []
        m = []
        self.ll = []
        for l in table:
            srcs, tgt = l.strip().split('\t')
            temp_token = tokenizer.encode(srcs)
            temp_mask = [1 for i in range(len(temp_token))]
            if len(temp_token) >= 20: continue
            self.ll.append(len(temp_token))
            ## pad tokens
            temp_token = torch.cat((torch.full((1,n_tokens), 2).squeeze(0), torch.LongTensor(temp_token)), 0)
            temp_mask = torch.cat((torch.full((1,n_tokens), 1).squeeze(0), torch.LongTensor(temp_mask)), 0)
            ##
            temp.append(temp_token[:])
            m.append(temp_mask)
            
           # print(srcs)
        # print(len(temp))
        self.post = pad_sequence([x for x in temp], batch_first=True, padding_value=0)
        self.mask = pad_sequence([x for x in m], batch_first=True, padding_value=0)
    def __getitem__(self, index):
        # return self.table[index]
        return self.post[index], self.mask[index], self.ll[index]

    def __len__(self):
        return len(self.post)

class SoftEmbedding(nn.Module):
    def __init__(self, 
                wte: nn.Embedding,
                n_tokens: int = 10, 
                random_range: float = 0.5,
                initialize_from_vocab: bool = True,
                commonWords_id: list = None):
        """appends learned embedding to 
        Args:
            wte (nn.Embedding): original transformer word embedding
            n_tokens (int, optional): number of tokens for task. Defaults to 10.
            random_range (float, optional): range to init embedding (if not initialize from vocab). Defaults to 0.5.
            initialize_from_vocab (bool, optional): initalizes from default vocab. Defaults to True.
        """
        super(SoftEmbedding, self).__init__()
        self.wte = wte
        self.n_tokens = n_tokens
        self.learned_embedding = nn.parameter.Parameter(self.initialize_embedding(wte, n_tokens, random_range, initialize_from_vocab, commonWords_id))
        ## (10,768)
        
            
    def initialize_embedding(self, 
                             wte: nn.Embedding,
                             n_tokens: int = 10, 
                             random_range: float = 0.5, 
                             initialize_from_vocab: bool = True,
                             commonWords_id: list = None):
        """initializes learned embedding
        Args:
            same as __init__
        Returns:
            torch.float: initialized using original schemes
        """
        
        if initialize_from_vocab:
            output = self.wte.weight[commonWords_id].clone().detach()
            # print(output.size())
            return output
        return torch.FloatTensor(n_tokens, wte.weight.size(1)).uniform_(-random_range, random_range)
            
    def forward(self, tokens):
        """run forward pass
        Args:
            tokens (torch.long): input tokens before encoding
        Returns:
            torch.float: encoding of text concatenated with learned task specifc embedding
        """
        # input_embedding = self.wte(tokens[:, self.n_tokens:]) ## (1,4,768)
        if tokens.size()[1] > 1:
            # print('here')
            input_embedding = self.wte(tokens[:, self.n_tokens:]) ## (1,4,768)
            learned_embedding = self.learned_embedding.repeat(input_embedding.size(0), 1, 1) #torch.Size([1, 20, 768])
            return torch.cat([learned_embedding, input_embedding], 1) #torch.Size([1, 24, 768])
        else:
            return self.wte(tokens)



def top_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
    """
    # batch support!
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
       # print(values.shape)
        min_values = values[:, -1].unsqueeze(1).repeat(1, logits.shape[-1])
        logits = torch.where(logits < min_values, 
                             torch.ones_like(logits, dtype=logits.dtype) * -float('Inf'), 
                             logits)
    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        sorted_logits = sorted_logits.masked_fill_(sorted_indices_to_remove, filter_value)
        logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)
    
    return logits
temperature = 1 #2.2
top_k = 50        #50
top_p = 0.95
device_0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def make_response(model, sentences, tokenizer, first_input):
    with torch.no_grad():
        print(sentences)
        sentences = [tokenizer.encode(x) for x in sentences]
        print(sentences)
        assert(0)
        t = []
        for i in range(len(sentences)):
            t_0 = [0 for i in range(len(list(first_input[i])))]
            t_1 = [1 for i in range(len(list(sentences[i]))-1)]
            sentences[i] = list(first_input[i]) + list(sentences[i])

            t.append(t_0[:] + t_1[:])
        mask= []


        sentences = [x[:-1] for x in sentences]
        for i in range(len(sentences)):
            mask.append([1 for x in range(len(sentences[i]))])
        eos = [tokenizer.encoder["<|endoftext|>"]]

        prev_input = pad_sequence([torch.LongTensor(x) for x in sentences], batch_first=True, padding_value=0).to(device_1)
        mask = pad_sequence([torch.LongTensor(x) for x in mask], batch_first=True, padding_value=0).to(device_1)
        #_, past = model(prev_input, past=None, attention_mask=mask)
        output = model(prev_input, past_key_values=None, attention_mask=mask)
        past = output['past_key_values']
        prev_input = torch.LongTensor([[eos] * len(sentences)]).to(device_1)
        temp_sentence = [[] for i in range(len(sentences))]
        for i in range(128):
            #prev_input, past = model(prev_input, past=past)
            output = model(prev_input, past_key_values=past)
            prev_input, past = output['logits'], output['past_key_values']
            prev_input = prev_input.squeeze(0).squeeze(1)
            prev_input = prev_input / 0.7
            prev_input = torch.softmax(prev_input, dim=-1)

            prev_input = torch.multinomial(prev_input, num_samples=1)

            if i == 0:
                for j in range(len(sentences)):
                    temp_sentence[j].append(prev_input[j].item())
                continue
            flag = 1
            
            for j in range(len(sentences)):
                if temp_sentence[j][-1] != eos[0]: 
                    flag = 0
                    temp_sentence[j].append(prev_input[j].item())
            if flag == 1: break
    return [[tokenizer.decode(x).replace('<|endoftext|>', '')] for x in temp_sentence]

table_data = []

def train(model_train, inputs_id, mask, model_2, model_bot, tokenizer, tokenizer_gpt2, ll, args, batch_size, n_tokens,  batch, reward):
    

    loss = 0
    inputs_id = inputs_id.to(device_0) ## 8*29
    if args.emotion : 
        emo_embed = emo_dict['<'+args.emotion+'>']
    bos = [1]
    eos = [2] ## [50256]
    decoder_input_ids = torch.LongTensor([[bos] * inputs_id.shape[0]]).squeeze(0).to(device_0)
    
    mask = mask.to(device_0)
    prev_input = torch.LongTensor([[bos] * inputs_id.shape[0]]).squeeze(0).to(device_0) # (8,1)



    ######### all (8,) 
    temp_sentence = [[] for i in range(inputs_id.shape[0])]
    emotion_loss = [0 for i in range(inputs_id.shape[0])]
    coherence_loss = [0 for i in range(inputs_id.shape[0])]
    test_reward = [1 for i in range(inputs_id.shape[0])]
    #########


    append = torch.tensor([[1] for i in range(len(inputs_id))]).to(device_0)
    # mask = torch.cat((mask, append), 1) 
    coh_score = 0
    past = None
    past_co = None
    for i in range(40): # 40 words
        output = model_train(inputs_id, attention_mask=mask, decoder_input_ids = prev_input, past_key_values=past)
        logits, past = output['logits'], output['past_key_values']
        prev_input = prev_input.to(device_1)
        

        with torch.no_grad():
            output = model_2(inputs_id, attention_mask=mask, decoder_input_ids = prev_input, past_key_values=past_co)
            logits_co, past_co = output['logits'], output['past_key_values']

        logits = logits.squeeze(0).squeeze(1)

        logits = logits / temperature

        logits = torch.softmax(logits, dim=-1)
        with torch.no_grad():
            logits_co = torch.softmax(logits_co.squeeze(0).squeeze(1) / temperature, dim=-1)
        prev_input = torch.multinomial(logits[:], num_samples=1) #(8,1)
        

        probs = []
        for j in range(inputs_id.shape[0]):
            if i != 0 and temp_sentence[j][-1] == eos[0]: continue
            probs.append(logits_co[j][prev_input[j][0].item()].item()) ## compute conditional prob
            test_reward[j] *= logits_co[j][prev_input[j][0].item()].item()
        if len(probs) == 0:
            avg_prob = 0
        else:
            avg_prob = sum(probs) / len(probs)
            coh_score += avg_prob

        for j in range(inputs_id.shape[0]):
            if i != 0 and temp_sentence[j][-1] == eos[0]: continue
            ## prev_input.view(-1) ## size=(8)
            temp_loss = F.cross_entropy(logits[j].unsqueeze(0), prev_input.view(-1)[j].unsqueeze(0))
            coherence_loss[j] += (logits_co[j][prev_input[j][0].item()].item() - avg_prob) * temp_loss
            emotion_loss[j] += temp_loss

        if i == 0:
            ## if first word of chatbot
            for j in range(inputs_id.shape[0]):
                temp_sentence[j].append(prev_input[j].item())
            continue ## jump to second words
        flag = 1 ## to ascertain whether all sentence complete
        
        for j in range(inputs_id.shape[0]):
            if temp_sentence[j][-1] != eos[0]: 
                flag = 0
                temp_sentence[j].append(prev_input[j].item())
        if flag == 1: break
    decode_temp_sentence = tokenizer.batch_decode(temp_sentence, skip_special_tokens=True) # list[str]
    input_sentences = tokenizer.batch_decode(inputs_id[:, n_tokens:], skip_special_tokens=True)
    
    # eos = [2]
    eos = [tokenizer_gpt2.encoder["<|endoftext|>"]]
    # first_input = list(inputs_id.cpu().detach().numpy())
    first_input = [tokenizer_gpt2.encode(x) for x in input_sentences]
    first_input = [x + eos for x in first_input]
    # for j in range(inputs_id.shape[0]):
    #     l = ll[j]
    #     first_input[j] = first_input[j][n_tokens:n_tokens+l+1]
    #     first_input[j][-1] = eos[0]
    inter_response = []
    if 'gpt' in args.inter:
      inter_response.extend(make_response(model_bot, decode_temp_sentence, tokenizer_gpt2, first_input))
    #   for i in range(inputs_id.shape[0]):
    #     NEXT = input_sentences[i][1:] + '</s> <s>' + decode_temp_sentence[i]
    #     inter_reply = model_bot.generate(torch.tensor([tokenizer.encode(NEXT)]).to(device_0), max_new_tokens = 120, min_length = 1)
    #     inter_response.append(tokenizer.batch_decode(inter_reply, skip_special_tokens=True)[0])
    if batch % 100 == 0:
        # print(f'batch: {batch}')
        inpu_t = [tokenizer.decode(x[n_tokens:]) for x in inputs_id]
        my_table = wandb.Table(columns=['batch','input', 'chatbot', 'inter']) 
        for i in range(inputs_id.shape[0]):
            table_data.append([batch, input_sentences[i] , decode_temp_sentence[i], inter_response[i]])
        for t in table_data:
            my_table.add_data(*t)
        # print('************** save to table **********')
        wandb.log({'generation table': my_table}, commit=False)
    # if 'google' in args.inter:
    #     #k = []
    #     for j in range(inputs_id.shape[0]):
    #         k.append([jack.daemonPredict(sentence=a[j].replace('<|endoftext|>', ''))])
    # if 'retrieve' in args.inter:
    #     ii = []
    #     for j in range(inputs_id.shape[0]): 
    #         # ii = [tokenizer.decode(x[:-1]) for x in first_input]
    #         ii.append([tokenizer.decode(first_input[j][:-1]), a[j].replace('<|endoftext|>', '')])
    #     rps = ret_model.get_response(ii)
    #     k.extend([[x] for x in rps])

    #test_score += avg_prob

#################################################################################
#                                                                               #
#                   Reward                                                      #
#                                                                               #
#################################################################################
    score = None
    if reward == 'emotion': 
        sent_input = []

        for j in range(inputs_id.shape[0]*len(args.inter)):
            l = ll[j%inputs_id.shape[0]]
            sent_input.append([tokenizer.decode(inputs_id[j%inputs_id.shape[0]][n_tokens:]), decode_temp_sentence[j%inputs_id.shape[0]].replace('<|endoftext|>', ''), inter_response[j][0]])
        emo, embans = re_emo_score(detect_model, detect_processor, emotion_tokenizer, sent_input, len(inter_response))      
        temp_score = []
        for e in embans:
            temp_score.append(np.sum((e - emo_embed)**2))

        score = [0 for i in range(len(temp_score) // len(args.inter))]

        for j in range(len(temp_score) // len(args.inter)):
            for k in range(len(args.inter)):
                score[j] += temp_score[j + batch_size*k]

    elif reward == 'sw':
        score = np.array([0 for w in range(inputs_id.shape[0])])
        for j in range(inputs_id.shape[0]*len(args.inter)):
            for word in word_dict.keys():
                if re.search(r"\b{}\b".format(word.lower()), inter_response[j][0].lower().strip()):
                    score[j%8] += 1

    elif reward == 'length':
        sent_input = []
        for j in range(inputs_id.shape[0]*len(args.inter)):
            l = ll[j%inputs_id.shape[0]]
            # sent_input.append([tokenizer.decode(inputs_id[j%inputs_id.shape[0]][n_tokens:].tolist()), decode_temp_sentence[j%inputs_id.shape[0]], inter_response[j][0]])
            sent_input.append([input_sentences[j%inputs_id.shape[0]], decode_temp_sentence[j%inputs_id.shape[0]], inter_response[j][0]])
        score = []
        for sens in sent_input: 
            # print(sens[2])          
            # sen = (sens[0] + sens[1] + sens[2]).replace('[SEP]', '').replace('[CLS]', '').replace(' ', '')
            # print(sens[2].split())
            score.append(len(sens[2].split()))
        test_len = [len(s) for s in temp_sentence]
        test_reward = [test_reward[i] ** (1/test_len[i]) for i in range(inputs_id.shape[0])]

    score = np.array(score) / len(args.inter)
    # score = score - np.mean(score)
    mean_score = np.mean(score)
    for j in range(inputs_id.shape[0]):
        if reward == 'length' or 'sw':
            loss += (score[j] - mean_score) * emotion_loss[j] * (1 - args.ra)
        else:
            loss -= (score[j] - mean_score) * emotion_loss[j] #/ len(temp_sentence[j])
        loss += coherence_loss[j] * args.ra #/ len(temp_sentence[j])
    
    if reward == 'sw':
        return loss, sum(score), coh_score
    elif reward == 'emotion':
        return loss, sum(temp_score), coh_score
    elif reward == 'length':
        test_reward = np.mean(test_reward)
        return loss, sum(score), test_reward

def main():
    parser = ArgumentParser()
    parser.add_argument("--emotion", type=str, default=None)
    parser.add_argument("--writer", type=str, default="")
    parser.add_argument("--save", type=str, default="model/save/")
    parser.add_argument("--model", type=str, default="facebook/blenderbot-400M-distill")
    parser.add_argument("--ra", type=float, default=.5)
    parser.add_argument("--inter", type=str, default="gpt", nargs='+', required=True)
    parser.add_argument("--n_tokens", type=int, default=10)
    parser.add_argument("--sw", type=str, default=None)
    parser.add_argument("--len", type=str, default=None)
    parser.add_argument("--initial", type=str, default='vocab')
    parser.add_argument("--gpt", type=str, default="microsoft/DialoGPT-medium")
    args = parser.parse_args()

    if not args.emotion and not args.sw and not args.len :
        print('Please enter emotion or specific word')
        assert(0)
    else:
        if args.sw : 
            reward = 'sw'
        elif args.emotion : 
            reward = 'emotion'
        else :
            reward = 'length'
        # reward = 'sw' if args.sw else 'emotion' 

    wandb.init(
      # Set the project where this run will be logged
      project="chatbot_softprompt", 
      # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
      name=f"{args.save}"
      #,entity="chatbot_ntu"
      )
    # Track hyperparameters and run metadata
    wandb.config.update(args)
    wandb.config.update({"lr": 5e-4, 'epoch':1, "seed":100, 'batch_size':4, 
        'init_from_vocab': True if args.initial == 'vocab' else False})

    

    config = wandb.config
    os.makedirs('model/' + args.model, exist_ok=True)
    #test
    

    np.random.seed(config.seed)
    torch.random.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    mname = "facebook/blenderbot-400M-distill"
    model_train = BlenderbotForConditionalGeneration.from_pretrained(mname)
    model_2 = BlenderbotForConditionalGeneration.from_pretrained(mname)
    tokenizer = BlenderbotTokenizer.from_pretrained(mname)
    # model_train = GPT2LMHeadModel.from_pretrained(args.model) 
    # model_2 = GPT2LMHeadModel.from_pretrained(args.model)
    tokenizer_gpt2 = GPT2Tokenizer.from_pretrained(args.gpt)
    
    

    if args.sw:
        with open('specific_word.json') as f:
            data = json.load(f)
            data = data[args.sw]
            for w in data:
                word_dict[w] = 1

    ### setting  softprompt
    n_tokens = args.n_tokens
    initialize_from_vocab = config.init_from_vocab
    if initialize_from_vocab:
      random.seed(config.seed)
      with open("commonWords.txt") as f:
          words = f.read().splitlines()
          # print(args.n_tokens)
          random_num = random.sample(range(0, 3000), n_tokens)
          commonWords = words[random_num[0]]
          for i in range(1, n_tokens):
            commonWords = commonWords + ' ' + words[random_num[i]]
          commonWords_id = tokenizer.encode(commonWords)
          commonWords_id = commonWords_id[:n_tokens]
    else:
      commonWords_id = None
    s_wte = SoftEmbedding(model_train.get_input_embeddings(), 
                      n_tokens=n_tokens, 
                      initialize_from_vocab=initialize_from_vocab,
                      commonWords_id = commonWords_id)
    model_train.set_input_embeddings(s_wte)

    parameters = list(model_train.parameters())
    # parameters_check = list(model_train.parameters())

    for x in parameters[1:]:
        x.requires_grad = False
    
    ###


    if 'gpt' in args.inter:
        model_bot = GPT2LMHeadModel.from_pretrained(args.gpt)
        # model_bot = BlenderbotForConditionalGeneration.from_pretrained(mname)
        model_bot.to(device_1)
        model_bot.eval()
    #
    # if 'google' in args.inter:
    #     from main1 import chatbot
    #     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1) 
    #     sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    #     jack = chatbot.Chatbot()
    #     jack.main(['--test', 'daemon', '--rootDir', 'deepqa', '--maxLength', '20'])
    # if 'retrieve' in args.inter:
    #     with torch.no_grad():
    #         from retrieval_model.retrieval_chatbot import Retrievalchatbot
    #         ret_model = Retrievalchatbot()
    # writer = SummaryWriter('runs/'+args.writer+'/')

    
    wandb.config.update({'total_update_param': sum(p.numel() for p in model_train.parameters() if p.requires_grad)})
    optimizer = Adam([s_wte.learned_embedding], config.lr,
                     max_grad_norm=1.0)
    ##

    model_train = model_train.to(device_0)
    model_2.to(device_1)
    model_2.eval()
    batch_size = config.batch_size

        
    

    
    post = post_set('data/train_raw.tsv', args.n_tokens, tokenizer)
    train_dataloader = DataLoader(post, batch_size=batch_size, shuffle=True, num_workers=1)

    batch = 0
    temp_score = 0
    loss = 0
   
    test_score = 0
    name = 'model.shared.learned_embedding'
    for global_step in range(config.epoch):
        model_train.train()
        for inputs_id, mask, ll in tqdm(train_dataloader):
            batch += 1
            batch_loss, score, coh_score = train(model_train, inputs_id, mask, model_2, model_bot, tokenizer, tokenizer_gpt2, ll, args, batch_size, n_tokens,  batch, reward)
            loss += batch_loss

            test_score += coh_score
            temp_score += score


            if batch % 8 == 0:
                loss.backward()
                optimizer.step()
                # writer.add_scalar('loss', loss, batch) 
                wandb.log({"loss": loss})
                optimizer.zero_grad()  
                loss = 0
            if batch % 40 == 0:
                # writer.add_scalar('reward', temp_score/batch_size/20, batch)
                # writer.add_scalar('coherence', test_score/20, batch) # corherence
                
                
                wandb.log({"reward":  temp_score/batch_size/40, 'coherence': test_score/40})
                print("Reward:%.2f,    test:%.6f   "%(temp_score/batch_size/40, test_score/40))
                test_score = 0
                temp_score = 0
            if batch % 1000 == 0:
                # name = 'model.shared.learned_embedding' 
                # idx = random.randint(1, len(parameters_check) - 1)
                # param = list(model_train.parameters())
                # check_valid(param[idx], parameters_check[idx])
                torch.save(
                    {name: (model_train.state_dict()[name].cpu())},
                    join(f'model/save/',
                            f'{args.save}-{batch}.pkl'))
    wandb.finish()

if __name__ == "__main__":
    main()