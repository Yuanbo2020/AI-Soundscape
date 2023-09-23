import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import framework.config as config


def move_data_to_gpu(x, cuda):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)

    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)

    else:
        raise Exception("Error!")

    if cuda:
        x = x.cuda()

    return x


def init_layer(layer):
    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width

    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """

    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weights()

    def init_weights(self):

        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        else:
            raise Exception('Incorrect argument!')

        return x


class ConvBlock_rms(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock_rms, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 1), stride=(1, 1),
                               padding=(1, 0), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.init_weights()

    def init_weights(self):

        init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, input, pool_size=(2, 1), pool_type='avg'):

        x = input
        # print(x.size()) torch.Size([64, 1, 482, 1])
        x = F.relu_(self.bn1(self.conv1(x)))
        # print(x.size())  torch.Size([64, 64, 482, 1])

        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        else:
            raise Exception('Incorrect argument!')

        return x





from framework.selfattention import ScaledDotProductAttention_DCNN_CaF
class DCNN_CaF(nn.Module):
    def __init__(self, event_class, batchnormal=True):

        super(DCNN_CaF, self).__init__()

        self.batchnormal = batchnormal
        if batchnormal:
            self.bn0 = nn.BatchNorm2d(config.mel_bins)
            self.bn0_rms = nn.BatchNorm2d(1)

        self.conv_block1_rms = ConvBlock_rms(in_channels=1, out_channels=64)
        self.conv_block2_rms = ConvBlock_rms(in_channels=64, out_channels=128)
        self.conv_block3_rms = ConvBlock_rms(in_channels=128, out_channels=256)
        self.conv_block4_rms = ConvBlock_rms(in_channels=256, out_channels=512)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, 512, bias=True)

        self.fc_final_event = nn.Linear(512, event_class, bias=True)

        d_model = 512
        self.self_attention = ScaledDotProductAttention_DCNN_CaF(d_model=d_model,
                                                        d_k=64,
                                                        d_v=64, h=8)
        self.self_attention2 = ScaledDotProductAttention_DCNN_CaF(d_model=d_model,
                                                        d_k=64,
                                                        d_v=64, h=8)

        each_time = 32
        self.fusion_layer = nn.Linear(d_model*2, 32, bias=True)
        self.fc_rate_final = nn.Linear(30*each_time, 1, bias=True)

        self.init_weight()

    def init_weight(self):
        if self.batchnormal:
            init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_final_event)
        init_layer(self.fc_rate_final)

    def forward(self, input, rms, event_emb = False):
        (_, seq_len, mel_bins) = input.shape
        (_, rms_len, rms_bins) = rms.shape

        x = input.view(-1, 1, seq_len, mel_bins)
        x_rms = rms.view(-1, 1, rms_len, rms_bins)
        '''(samples_num, feature_maps, time_steps, freq_num)'''

        # print(rms.size())
        # print(x.size())
        # # torch.Size([4, 482, 1])
        # # torch.Size([4, 1, 480, 64])

        if self.batchnormal:
            x = x.transpose(1, 3)
            x = self.bn0(x)
            x = x.transpose(1, 3)

            x_rms = x_rms.transpose(1, 3)
            x_rms = self.bn0_rms(x_rms)
            x_rms = x_rms.transpose(1, 3)
            # print(x_rms.size())
            # torch.Size([64, 1, 482, 1])

        ##################################################################
        pool_size = (2, 1)
        x_rms = self.conv_block1_rms(x_rms, pool_size=pool_size, pool_type='avg')
        # print(x_rms.size())
        # torch.Size([64, 64, 241, 1])  torch.Size([64, 64, 160, 1])
        x_rms = self.conv_block2_rms(x_rms, pool_size=pool_size, pool_type='avg')
        # torch.Size([64, 128, 120, 1]) torch.Size([64, 128, 52, 1])
        # print(x_rms.size())

        x_rms = self.conv_block3_rms(x_rms, pool_size=pool_size, pool_type='avg')
        # torch.Size([64, 256, 60, 1]) torch.Size([64, 256, 16, 1])
        # print(x_rms.size())

        x_rms = self.conv_block4_rms(x_rms, pool_size=pool_size, pool_type='avg')
        # # torch.Size([64, 512, 30, 1]) 5
        # print(x_rms.size())  #v torch.Size([64, 512, 4, 1])
        x_rms = torch.mean(x_rms, dim=3)
        # print(x_rms.size())  # torch.Size([64, 512, 30])  torch.Size([64, 256, 16])

        #####################################################################

        # print(x.size())  # torch.Size([64, 1, 480, 64])
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        # print(x.size())  # torch.Size([64, 64, 240, 32])

        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        # print(x.size()) # torch.Size([64, 128, 120, 16])

        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        # print(x.size())  # torch.Size([64, 256, 60, 8])

        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x_512 = F.dropout(x, p=0.2, training=self.training)
        # print(x.size()) # torch.Size([64, 512, 30, 4])

        ####################### event ############################################

        x_event = torch.mean(x_512, dim=3)
        # print(x.size()) # torch.Size([64, 2048, 15])

        (x_event1, _) = torch.max(x_event, dim=2)
        x_event2 = torch.mean(x_event, dim=2)
        x_event = x_event1 + x_event2
        # print(x.size()) # torch.Size([64, 2048])
        x_event = F.relu_(self.fc1(x_event))
        # print(x_event.size())
        event = self.fc_final_event(x_event)
        # print(event.size())
        ##############################################################################

        x_mel_mean = torch.mean(x_512, dim=3)  # # torch.Size([64, 512, 30, 4])
        # (x_mel_max, _) = torch.max(x_512, dim=3)
        # print(x_mel_mean.size())

        x_mel = x_mel_mean

        x_rms = torch.transpose(x_rms, 1, 2)
        x_mel = torch.transpose(x_mel, 1, 2)
        # print(x_rms.size(), x_mel.size())  # torch.Size([64, 30, 512]) torch.Size([64, 30, 512])
        q_rms_kv_mel = self.self_attention(x_rms, x_mel, x_mel)
        # print(q_rms_kv_mel.size())  # torch.Size([64, 30, 512])

        q_mel_kv_rms = self.self_attention2(x_mel, x_rms, x_rms)
        # print(q_mel_kv_rms.size())

        using = torch.cat([q_mel_kv_rms, q_rms_kv_mel], dim=-1)
        # print(using.size())

        x = F.relu_(self.fusion_layer(using))  # torch.Size([64, 30, 32])
        # print(x.size())
        x = x.flatten(1)
        rate = self.fc_rate_final(x)
        # print(rate.size())

        if event_emb:
            return rate, event, x_event
        else:
            return rate, event



#################################### mha ########################################################
import numpy as np
# transformer
d_model = 512  # Embedding Size
d_ff = 2048  # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_heads = 8  # number of heads in Multi-Head Attention

class ScaledDotProductAttention_nomask(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention_nomask, self).__init__()

    def forward(self, Q, K, V, d_k=d_k):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention_nomask(nn.Module):
    def __init__(self, d_model=d_model, d_k=d_k, d_v=d_v, n_heads=n_heads,
                 output_dim=d_model):
        super(MultiHeadAttention_nomask, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.layernorm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(n_heads * d_v, output_dim)

    def forward(self, Q, K, V, d_model=d_model, d_k=d_k, d_v=d_v, n_heads=n_heads):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)

        context, attn = ScaledDotProductAttention_nomask()(q_s, k_s, v_s)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        output = self.fc(context)
        x = self.layernorm(output + residual)
        return x, attn


class EncoderLayer(nn.Module):
    def __init__(self, output_dim=d_model):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention_nomask(output_dim=output_dim)

    def forward(self, enc_inputs):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs)
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self, input_dim, n_layers, output_dim=d_model):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(output_dim) for _ in range(n_layers)])
        self.mel_projection = nn.Linear(input_dim, d_model)

    def forward(self, enc_inputs):
        # print(enc_inputs.size())  # torch.Size([64, 54, 8, 8])
        size = enc_inputs.size()
        enc_inputs = enc_inputs.reshape(size[0], size[1], -1)
        enc_outputs = self.mel_projection(enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask, d_k=d_k):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn
#################################################################################################


class CNN_Transformer(nn.Module):
    def __init__(self, event_class, batchnormal=True):

        super(CNN_Transformer, self).__init__()

        self.batchnormal = batchnormal
        if batchnormal:
            self.bn0 = nn.BatchNorm2d(config.mel_bins)
            self.bn0_rms = nn.BatchNorm2d(1)

        out_channels = 32
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=out_channels,
                               kernel_size=(7, 7), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        out_channels = 64
        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=out_channels,
                               kernel_size=(7, 7), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        ################################ rms ###############################################
        out_channels = 32
        self.conv1_rms = nn.Conv2d(in_channels=1,
                               out_channels=out_channels,
                               kernel_size=(7, 1), stride=(1, 1),
                               padding=(0, 0), bias=False)
        self.bn1_rms = nn.BatchNorm2d(out_channels)

        out_channels = 64
        self.conv2_rms = nn.Conv2d(in_channels=32,
                               out_channels=out_channels,
                               kernel_size=(7, 1), stride=(1, 1),
                               padding=(0, 0), bias=False)
        self.bn2_rms = nn.BatchNorm2d(out_channels)
        ####################################################################################
        encoder_layers = 1
        self.mha = Encoder(input_dim=17+18, n_layers=encoder_layers, output_dim=d_model)

        units = 256
        self.fc1 = nn.Linear(32768, units, bias=True)
        self.fc_final = nn.Linear(units, 1, bias=True)

        self.fc1_event = nn.Linear(32768, units, bias=True)
        self.fc_final_event = nn.Linear(units, event_class, bias=True)

        self.my_init_weight()

    def my_init_weight(self):
        init_layer(self.fc1)
        init_layer(self.fc_final)
        init_layer(self.fc1_event)
        init_layer(self.fc_final_event)

    def forward(self, input, rms):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (16, 481, 64)

        (_, seq_len, mel_bins) = input.shape

        x = input.view(-1, 1, seq_len, mel_bins)
        '''(samples_num, feature_maps, time_steps, freq_num)'''

        (_, seq_len_rms, rms_bins) = rms.shape
        x_rms = rms.view(-1, 1, seq_len_rms, rms_bins)

        if self.batchnormal:
            x = x.transpose(1, 3)
            x = self.bn0(x)
            x = x.transpose(1, 3)

            x_rms = x_rms.transpose(1, 3)
            x_rms = self.bn0_rms(x_rms)
            x_rms = x_rms.transpose(1, 3)

        # print(x.size())  # torch.Size([64, 1, 480, 64])
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=(5, 5))
        x = F.dropout(x, p=0.3, training=self.training)
        # print(x.size())  # torch.Size([64, 32, 95, 12])

        x = F.relu_(self.bn2(self.conv2(x)))
        # print(x.size())  # torch.Size([64, 64, 91, 8])
        x = F.max_pool2d(x, kernel_size=(5, 5))
        x = F.dropout(x, p=0.3, training=self.training)
        # print(x.size())  # torch.Size([64, 64, 18, 1])

        ########################## rms #########################################################
        # print(x_rms.size())  # torch.Size([64, 1, 482, 1])
        x_rms = F.relu_(self.bn1_rms(self.conv1_rms(x_rms)))
        # print(x_rms.size())  # torch.Size([64, 32, 476, 1])
        x_rms = F.max_pool2d(x_rms, kernel_size=(5, 1))
        x_rms = F.dropout(x_rms, p=0.3, training=self.training)
        # print(x_rms.size())  # torch.Size([64, 32, 95, 1])

        x_rms = F.relu_(self.bn2_rms(self.conv2_rms(x_rms)))
        # print(x_rms.size())  # torch.Size([64, 64, 89, 1])
        x_rms = F.max_pool2d(x_rms, kernel_size=(5, 1))
        x_rms = F.dropout(x_rms, p=0.3, training=self.training)
        # print(x_rms.size())  # torch.Size([64, 64, 17, 1])

        #####################
        x_rms = torch.mean(x_rms, dim=-1)
        x = torch.mean(x,  dim=-1)
        x = torch.cat([x, x_rms], dim=-1)  # torch.Size([64, 64, 17+18])
        x_com, x_scene_self_attns = self.mha(x)  # already have reshape
        # print(x_com.size())  # torch.Size([64, 64, 512])
        #################

        x_com = x_com.view(x_com.size()[0], -1)  # torch.Size([64, 32768])
        x_com = F.dropout(x_com, p=0.3, training=self.training)

        x_embed = F.relu_(self.fc1(x_com))
        x_rate_linear = self.fc_final(x_embed)

        x_event = F.relu_(self.fc1_event(x_com))
        x_event_linear = self.fc_final_event(x_event)

        return x_rate_linear, x_event_linear



class PANN(nn.Module):
    def __init__(self, event_num):

        super(PANN, self).__init__()

        self.event_num = event_num

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        pann_dim = 2048

        self.fc1 = nn.Linear(pann_dim, pann_dim, bias=True)

        self.fc1_event = nn.Linear(pann_dim, event_num, bias=True)


    def forward(self, input):

        (_, seq_len, mel_bins) = input.shape
        x = input[:, None, :, :]

        x_clip = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x_clip = F.dropout(x_clip, p=0.2, training=self.training)
        x_clip = self.conv_block2(x_clip, pool_size=(2, 2), pool_type='avg')
        x_clip = F.dropout(x_clip, p=0.2, training=self.training)
        x_clip = self.conv_block3(x_clip, pool_size=(2, 2), pool_type='avg')
        x_clip = F.dropout(x_clip, p=0.2, training=self.training)
        x_clip = self.conv_block4(x_clip, pool_size=(2, 2), pool_type='avg')
        x_clip = F.dropout(x_clip, p=0.2, training=self.training)
        x_clip = self.conv_block5(x_clip, pool_size=(2, 2), pool_type='avg')
        x_clip = F.dropout(x_clip, p=0.2, training=self.training)
        x_clip = self.conv_block6(x_clip, pool_size=(1, 1), pool_type='avg')
        x_clip = F.dropout(x_clip, p=0.2, training=self.training)

        x_clip = torch.mean(x_clip, dim=3)
        (x1_clip, _) = torch.max(x_clip, dim=2)
        x2_clip = torch.mean(x_clip, dim=2)
        x_clip = x1_clip + x2_clip
        # print('x_clip: ', x_clip.size())  # 10s clip: torch.Size([128, 2048])

        x_clip = F.dropout(x_clip, p=0.5, training=self.training)

        x_clip = F.relu_(self.fc1(x_clip))

        linear_each_events = self.fc1_event(x_clip)

        return linear_each_events



class DCNN_CaF_SSC(nn.Module):
    def __init__(self, event_class, batchnormal=False):

        super(DCNN_CaF_SSC, self).__init__()

        self.batchnormal = batchnormal
        if batchnormal:
            self.bn0 = nn.BatchNorm2d(config.mel_bins)
            self.bn0_rms = nn.BatchNorm2d(1)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, 512, bias=True)

        self.fc_final_event = nn.Linear(512, event_class, bias=True)


        self.init_weight()

    def init_weight(self):
        if self.batchnormal:
            init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_final_event)

    def forward(self, input):
        (_, seq_len, mel_bins) = input.shape

        x = input.view(-1, 1, seq_len, mel_bins)

        if self.batchnormal:
            x = x.transpose(1, 3)
            x = self.bn0(x)
            x = x.transpose(1, 3)

        ##################################################################

        # print(x.size())  # torch.Size([64, 1, 480, 64])
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        # print(x.size())  # torch.Size([64, 64, 240, 32])

        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        # print(x.size()) # torch.Size([64, 128, 120, 16])

        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        # print(x.size())  # torch.Size([64, 256, 60, 8])

        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x_512 = F.dropout(x, p=0.2, training=self.training)
        # print(x.size()) # torch.Size([64, 512, 30, 4])

        ####################### event ############################################
        x_event = torch.mean(x_512, dim=3)
        # print(x.size()) # torch.Size([64, 2048, 15])

        (x_event1, _) = torch.max(x_event, dim=2)
        x_event2 = torch.mean(x_event, dim=2)
        x_event = x_event1 + x_event2
        # print(x.size()) # torch.Size([64, 2048])
        x_event = F.relu_(self.fc1(x_event))
        # print(x_event.size())
        event = self.fc_final_event(x_event)
        # print(event.size())
        return event



class CNN(nn.Module):
    def __init__(self, event_class, batchnormal=True):

        super(CNN, self).__init__()

        self.batchnormal = batchnormal
        if batchnormal:
            self.bn0 = nn.BatchNorm2d(config.mel_bins)
            self.bn0_rms = nn.BatchNorm2d(1)

        out_channels = 32
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=out_channels,
                               kernel_size=(7, 7), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        out_channels = 64
        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=out_channels,
                               kernel_size=(7, 7), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # out_channels = 128
        # self.conv3 = nn.Conv2d(in_channels=64,
        #                        out_channels=out_channels,
        #                        kernel_size=(7, 7), stride=(1, 1),
        #                        padding=(1, 1), bias=False)
        # self.bn3 = nn.BatchNorm2d(out_channels)

        # out_channels = 256
        # self.conv4 = nn.Conv2d(in_channels=128,
        #                        out_channels=out_channels,
        #                        kernel_size=(7, 7), stride=(1, 1),
        #                        padding=(1, 1), bias=False)
        # self.bn4 = nn.BatchNorm2d(out_channels)

        ################################ rms ###############################################
        out_channels = 32
        self.conv1_rms = nn.Conv2d(in_channels=1,
                               out_channels=out_channels,
                               kernel_size=(7, 1), stride=(1, 1),
                               padding=(0, 0), bias=False)
        self.bn1_rms = nn.BatchNorm2d(out_channels)

        out_channels = 64
        self.conv2_rms = nn.Conv2d(in_channels=32,
                               out_channels=out_channels,
                               kernel_size=(7, 1), stride=(1, 1),
                               padding=(0, 0), bias=False)
        self.bn2_rms = nn.BatchNorm2d(out_channels)

        # out_channels = 128
        # self.conv3_rms = nn.Conv2d(in_channels=64,
        #                        out_channels=out_channels,
        #                        kernel_size=(7, 7), stride=(1, 1),
        #                        padding=(1, 1), bias=False)
        # self.bn3_rms = nn.BatchNorm2d(out_channels)

        # out_channels = 256
        # self.conv4_rms = nn.Conv2d(in_channels=128,
        #                        out_channels=out_channels,
        #                        kernel_size=(7, 7), stride=(1, 1),
        #                        padding=(1, 1), bias=False)
        # self.bn4_rms = nn.BatchNorm2d(out_channels)
        ####################################################################################

        units = 256
        self.fc1 = nn.Linear(1152+1088, units, bias=True)
        self.fc_final = nn.Linear(units, 1, bias=True)

        # self.fc2 = nn.Linear(1152 + 1088, 512, bias=True)

        self.fc1_event = nn.Linear(1152+1088, units, bias=True)
        self.fc_final_event = nn.Linear(units, event_class, bias=True)

        self.my_init_weight()

    def my_init_weight(self):
        init_layer(self.fc1)
        init_layer(self.fc_final)
        init_layer(self.fc1_event)
        init_layer(self.fc_final_event)

    def forward(self, input, rms):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (16, 481, 64)

        (_, seq_len, mel_bins) = input.shape

        x = input.view(-1, 1, seq_len, mel_bins)
        '''(samples_num, feature_maps, time_steps, freq_num)'''

        (_, seq_len_rms, rms_bins) = rms.shape
        x_rms = rms.view(-1, 1, seq_len_rms, rms_bins)

        if self.batchnormal:
            x = x.transpose(1, 3)
            x = self.bn0(x)
            x = x.transpose(1, 3)

            x_rms = x_rms.transpose(1, 3)
            x_rms = self.bn0_rms(x_rms)
            x_rms = x_rms.transpose(1, 3)

        # print(x.size())  # torch.Size([64, 1, 480, 64])
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=(5, 5))
        x = F.dropout(x, p=0.3, training=self.training)
        # print(x.size())  # torch.Size([64, 32, 95, 12])

        x = F.relu_(self.bn2(self.conv2(x)))
        # print(x.size())  # torch.Size([64, 64, 91, 8])
        x = F.max_pool2d(x, kernel_size=(5, 5))
        x = F.dropout(x, p=0.3, training=self.training)
        # print(x.size())  # torch.Size([64, 64, 18, 1])

        ########################## rms #########################################################
        # print(x_rms.size())  # torch.Size([64, 1, 482, 1])
        x_rms = F.relu_(self.bn1_rms(self.conv1_rms(x_rms)))
        # print(x_rms.size())  # torch.Size([64, 32, 476, 1])
        x_rms = F.max_pool2d(x_rms, kernel_size=(5, 1))
        x_rms = F.dropout(x_rms, p=0.3, training=self.training)
        # print(x_rms.size())  # torch.Size([64, 32, 95, 1])

        x_rms = F.relu_(self.bn2_rms(self.conv2_rms(x_rms)))
        # print(x_rms.size())  # torch.Size([64, 64, 89, 1])
        x_rms = F.max_pool2d(x_rms, kernel_size=(5, 1))
        x_rms = F.dropout(x_rms, p=0.3, training=self.training)
        # print(x_rms.size())  # torch.Size([64, 64, 17, 1])

        x = x.view(x.size()[0], -1)
        # print(x.size())  # torch.Size([64, 1152])
        x = F.dropout(x, p=0.3, training=self.training)

        x_rms = x_rms.view(x_rms.size()[0], -1)  # torch.Size([64, 1088])
        x_rms = F.dropout(x_rms, p=0.3, training=self.training)

        x = torch.cat([x, x_rms], dim=-1)

        x_embed = F.relu_(self.fc1(x))
        x_rate_linear = self.fc_final(x_embed)

        x_event = F.relu_(self.fc1_event(x))
        x_event_linear = self.fc_final_event(x_event)

        return x_rate_linear, x_event_linear



class DNN(nn.Module):
    def __init__(self, event_class):

        super(DNN, self).__init__()

        self.fc_64 = nn.Linear(64, 64, bias=True)
        self.fc_128 = nn.Linear(64, 128, bias=True)
        self.fc_256 = nn.Linear(128, 256, bias=True)
        self.fc_512 = nn.Linear(256, 512, bias=True)

        self.fc_64_rms = nn.Linear(1, 64, bias=True)
        self.fc_128_rms = nn.Linear(64, 128, bias=True)
        self.fc_256_rms = nn.Linear(128, 256, bias=True)
        self.fc_512_rms = nn.Linear(256, 512, bias=True)

        self.fc_event = nn.Linear(512*2, event_class, bias=True)
        self.fc_rate = nn.Linear(512*2, 1, bias=True)


    def forward(self, input, rms):

        (_, seq_len, mel_bins) = input.shape

        x = input

        # print(x.size())  # torch.Size([64, 480, 64])
        x = F.relu_(self.fc_64(x))
        x = F.max_pool2d(x, kernel_size=(4, 1))
        # print(x.size())  # torch.Size([64, 160, 64])

        x = F.relu_(self.fc_128(x))
        x = F.max_pool2d(x, kernel_size=(4, 1))

        x = F.relu_(self.fc_256(x))
        x = F.max_pool2d(x, kernel_size=(4, 1))

        x = F.relu_(self.fc_512(x))
        x = F.max_pool2d(x, kernel_size=(4, 1))
        # print(x.size())
        # torch.Size([64, 1, 512])

        # print(rms.size())  # torch.Size([64, 482, 1])
        x_rms = F.relu_(self.fc_64_rms(rms))
        x_rms = F.max_pool2d(x_rms, kernel_size=(4, 1))
        # print(x.size())  # torch.Size([64, 160, 64])

        x_rms = F.relu_(self.fc_128_rms(x_rms))
        x_rms = F.max_pool2d(x_rms, kernel_size=(4, 1))

        x_rms = F.relu_(self.fc_256_rms(x_rms))
        x_rms = F.max_pool2d(x_rms, kernel_size=(4, 1))

        x_rms = F.relu_(self.fc_512_rms(x_rms))
        x_rms = F.max_pool2d(x_rms, kernel_size=(4, 1))
        # print(x_rms.size())
        # torch.Size([64, 1, 512])

        x = torch.cat([x, x_rms], dim=1)
        # print(x.size())
        # torch.Size([64, 2, 512])

        x_all = torch.flatten(x, start_dim=1)

        x_rate_linear = self.fc_rate(x_all)
        x_event_linear = self.fc_event(x_all)

        return x_rate_linear, x_event_linear


#######################################################################################################################
from framework.Yamnet_params import YAMNetParams

class Conv2d_tf(nn.Conv2d):
    """
    Conv2d with the padding behavior from TF Slim
    """
    def __init__(self, *args, **kwargs):
        # remove padding argument to avoid conflict
        padding = kwargs.pop("padding", "SAME")
        # initialize nn.Conv2d
        super().__init__(*args, **kwargs)
        self.padding = padding
        assert self.padding == "SAME"
        self.num_kernel_dims = 2
        self.forward_func = lambda input, padding: F.conv2d(
            input, self.weight, self.bias, self.stride,
            padding=padding, dilation=self.dilation, groups=self.groups,
        )

    def tf_SAME_padding(self, input, dim):
        input_size = input.size(dim + 2)
        filter_size = self.kernel_size[dim]

        dilate = self.dilation
        dilate = dilate if isinstance(dilate, int) else dilate[dim]
        stride = self.stride
        stride = stride if isinstance(stride, int) else stride[dim]

        effective_kernel_size = (filter_size - 1) * dilate + 1
        out_size = (input_size + stride - 1) // stride
        total_padding = max(
            0, (out_size - 1) * stride + effective_kernel_size - input_size
        )
        total_odd = int(total_padding % 2 != 0)
        return total_odd, total_padding

    def forward(self, input):
        if self.padding == "VALID":
            return self.forward_func(input, padding=0)
        odd_1, padding_1 = self.tf_SAME_padding(input, dim=0)
        odd_2, padding_2 = self.tf_SAME_padding(input, dim=1)
        if odd_1 or odd_2:
            # NOTE: F.pad argument goes from last to first dim
            input = F.pad(input, [0, odd_2, 0, odd_1])

        return self.forward_func(
            input, padding=[ padding_1 // 2, padding_2 // 2 ]
        )


class CONV_BN_RELU(nn.Module):
    def __init__(self, conv):
        super().__init__()
        self.conv = conv
        self.bn = nn.BatchNorm2d(
            conv.out_channels, eps=YAMNetParams.BATCHNORM_EPSILON
        )  # NOTE: yamnet uses an eps of 1e-4. This causes a huge difference
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv(nn.Module):
    def __init__(self, kernel, stride, input_dim, output_dim):
        super().__init__()
        self.fused = CONV_BN_RELU(
            Conv2d_tf(
                in_channels=input_dim, out_channels=output_dim,
                kernel_size=kernel, stride=stride,
                padding='SAME', bias=False
            )
        )

    def forward(self, x):
        return self.fused(x)


class SeparableConv(nn.Module):
    def __init__(self, kernel, stride, input_dim, output_dim):
        super().__init__()
        self.depthwise_conv = CONV_BN_RELU(
            Conv2d_tf(
                in_channels=input_dim, out_channels=input_dim, groups=input_dim,
                kernel_size=kernel, stride=stride,
                padding='SAME', bias=False,
            ),
        )
        self.pointwise_conv = CONV_BN_RELU(
            Conv2d_tf(
                in_channels=input_dim, out_channels=output_dim,
                kernel_size=1, stride=1,
                padding='SAME', bias=False,
            ),
        )

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class YAMNet(nn.Module):
    def __init__(self, event_num=521):
        super().__init__()
        net_configs = [
            # (layer_function, kernel, stride, num_filters)
            (Conv,          [3, 3], 2,   32),
            (SeparableConv, [3, 3], 1,   64),
            (SeparableConv, [3, 3], 2,  128),
            (SeparableConv, [3, 3], 1,  128),
            (SeparableConv, [3, 3], 2,  256),
            (SeparableConv, [3, 3], 1,  256),
            (SeparableConv, [3, 3], 2,  512),
            (SeparableConv, [3, 3], 1,  512),
            (SeparableConv, [3, 3], 1,  512),
            (SeparableConv, [3, 3], 1,  512),
            (SeparableConv, [3, 3], 1,  512),
            (SeparableConv, [3, 3], 1,  512),
            (SeparableConv, [3, 3], 2, 1024),
            (SeparableConv, [3, 3], 1, 1024)
        ]

        input_dim = 1
        self.layer_names = []
        for (i, (layer_mod, kernel, stride, output_dim)) in enumerate(net_configs):
            name = 'layer{}'.format(i + 1)
            self.add_module(name, layer_mod(kernel, stride, input_dim, output_dim))
            input_dim = output_dim
            self.layer_names.append(name)

        self.classifier = nn.Linear(input_dim, event_num, bias=True)

    def forward(self, x):
        # print(x.size())
        # torch.Size([64, 480, 64])
        x = x.unsqueeze(1)  # torch.Size([64, 1, 480, 64])

        for name in self.layer_names:
            mod = getattr(self, name)
            x = mod(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x



############################################### AST ################################################################

from timm.models.layers import to_2tuple, trunc_normal_
from functools import partial


class PatchEmbed(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        self.proj = torch.nn.Conv2d(1, embed_dim, kernel_size=(16, 16), stride=(10, 10))

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here

        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # x = x + self.drop_path(self.attn(self.norm1(x)))
        # x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ASTModel(nn.Module):
    def __init__(self, label_dim=10):

        super(ASTModel, self).__init__()

        embed_dim = 768
        depth = 12
        num_heads = 12
        mlp_ratio = 4.
        qkv_bias = True
        qk_scale = None
        drop_rate = 0.
        attn_drop_rate = 0.
        drop_path_rate = 0.

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.patch_embed = PatchEmbed(embed_dim=embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate, inplace=True)
        trunc_normal_(self.cls_token, std=.02)

        self.apply(self._init_weights)

        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        num_patches = 1212  # self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, embed_dim))

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)

        ########################################################################################

        self.fc_norm = nn.LayerNorm(embed_dim)
        self.fc_final_aec = nn.Linear(embed_dim, label_dim, bias=True)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # print(x.size())
        # torch.Size([64, 480, 64])

        x = torch.cat([x, x, x[:, :64]], dim=1)

        # print(x.size())
        # torch.Size([64, 1024, 64])

        x = torch.cat([x, x], dim=-1)

        # print(x.size())
        # torch.Size([64, 1024, 128])

        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        # print(x.size())
        x = x.transpose(2, 3)
        # print(x.size())

        # print(x.size())  # torch.Size([64, 1, 1024, 128])
        # print(x.size())  # torch.Size([64, 1, 128, 1024])

        B = x.shape[0]

        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        x = (x[:, 0] + x[:, 1]) / 2

        x = self.fc_norm(x)
        x = self.fc_final_aec(x)

        return x





