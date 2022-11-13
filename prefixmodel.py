from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
from torch import nn

_ = AutoConfig.from_pretrained('microsoft/DialoGPT-medium')

class PrefixTuning(nn.Module):
    """Classification Head for  transformer encoders"""
    def __init__(self, preseqlen=5, use_infix=False, deep_param=False, config=_, device=None):
        super().__init__()
        print('under the PrefixTuning model')

        self.match_n_layer = config.n_layer
        self.match_n_head = config.n_head
        self.match_n_embd = config.n_embd // config.n_head
        self.n_embd = config.n_embd
        self.preseqlen = preseqlen
        self.device = device

        self.gpt2_model = GPT2LMHeadModel.from_pretrained('microsoft/DialoGPT-medium').to(self.device)

        for n, p in self.gpt2_model.named_parameters() :
            p.requires_grad = False
        # if hasattr(config, 'optim_prefix'):
        #     self.optim_prefix = config.optim_prefix
        # else:
        #     self.optim_prefix = optim_prefix


        # config_prefix.init_random = model_args.init_random
        # config_prefix.mid_dim = model_args.mid_dim

        self.mid_dim = 128


        if True:
            self.mode_para = 0
            print('PrefixTuning')
            print('preseqlen is {}, optimizing the prefix directly'.format(self.preseqlen))


            # DIFFERENT PARAMETRIZATION:
            if True:
                low_data_init = 0
                print('[Full prefix-tuning Setting :) ]')
                self.input_tokens = torch.arange(self.preseqlen).long()
                self.wte = nn.Embedding(self.preseqlen, config.n_embd)

            # 512 * 768 + 512 * 12 * 2 * 768
                self.control_trans = nn.Sequential(
                    nn.Linear(config.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, config.n_layer * 2 * config.n_embd))
                
                self.get_prompt = self.get_prompt_p5

        self.dropout = nn.Dropout(0.3)

        ###### NUM PARAMS #########
        total_param = 0
        for name, param in self.named_parameters():
            # print(param.shape)
            total_param += param.numel()
        print('total param is {}'.format(total_param))


    def get_prompt_p5(self, control_code=None, gpt2=None, bsz=None):
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(self.device)
        temp_control = self.wte(input_tokens)
        past_key_values = self.control_trans(temp_control) #bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values


    def forward(self, input_ids, past_key_values, attention_mask):

        #{"input_ids": batch, "labels": labels, 'src_attn': src_attn, 'tgt_attn':tgt_attn, 'src':src}

        bsz = input_ids.shape[0]


        if past_key_values == None :
            prefix_attention_mask = torch.ones(bsz, self.preseqlen + input_ids.shape[1]).to(self.device)
            past_key_values = self.get_prompt(bsz=bsz)
            output = self.gpt2_model(input_ids, past_key_values, prefix_attention_mask)
       
        else :
            mask = torch.cat((torch.ones(bsz, self.preseqlen).to(self.device), attention_mask.to(self.device)), dim=1).to(self.device)
            output = self.gpt2_model(input_ids, past_key_values, mask)
       
        return output


    

