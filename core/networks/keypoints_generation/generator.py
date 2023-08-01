import torch
from torch import nn

from ..building_blocks import ConvNormRelu
from core.utils.selfAttention import ScaledDotProductAttention

from transformers import Wav2Vec2Processor, Wav2Vec2Model
import librosa
import torch.nn.functional as F


class AudioEncoder(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()

        # Load the pre-trained language model
        self.model_name = "facebook/wav2vec2-base-960h"
        self.model = Wav2Vec2Model.from_pretrained(self.model_name)
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)

        leaky = cfg.VOICE2POSE.GENERATOR.LEAKY_RELU
        norm = cfg.VOICE2POSE.GENERATOR.NORM

        down_sample_block_1 = nn.Sequential(
            ConvNormRelu('2d', 1, 64, downsample=False, norm=norm, leaky=leaky),
            ConvNormRelu('2d', 64, 64, downsample=True, norm=norm, leaky=leaky),
        )
        down_sample_block_2 = nn.Sequential(
            ConvNormRelu('2d', 64, 128, downsample=False, norm=norm, leaky=leaky),
            ConvNormRelu('2d', 128, 128, downsample=True, norm=norm, leaky=leaky),  # downsample
        )
        down_sample_block_3 = nn.Sequential(
            ConvNormRelu('2d', 128, 256, downsample=False, norm=norm, leaky=leaky),
            ConvNormRelu('2d', 256, 256, downsample=True, norm=norm, leaky=leaky),  # downsample
        )
        down_sample_block_4 = nn.Sequential(
            ConvNormRelu('2d', 256, 256, downsample=False, norm=norm, leaky=leaky),
            ConvNormRelu('2d', 256, 256, kernel_size=(6, 3), stride=1, padding=0, norm=norm, leaky=leaky),  # downsample
        )

        self.specgram_encoder_2d = nn.Sequential(
            down_sample_block_1,
            down_sample_block_2,
            down_sample_block_3,
            down_sample_block_4
        )

    def extract_audio_features(self, x):
        original_features = x.clone()
        original_batch, feat_size, sequence_length = x.shape
        # If the tensor is not 1D, flatten it along all dimensions except for the batch dimension (dim=0)
        if len(x.shape) > 1:
            x = x.reshape(-1)

        max_length = 200000
        x_segments = [x[i:i+max_length] for i in range(0, len(x), max_length)]
        features_list = []

        # Process each small fragment and stitch the result back
        for x_segment in x_segments:
            with torch.no_grad():
                input_values = self.processor(x_segment, sampling_rate=16000, padding=True, return_tensors="pt").input_values
                outputs = self.model(input_values.cuda())
                semantic_features = outputs.last_hidden_state.transpose(1, 2)
                # print("semantic_features: ", semantic_features.shape)

                x_np = x_segment.cpu().numpy()
                rhythmic_features = torch.tensor(librosa.feature.mfcc(y=x_np, sr=16000, n_mfcc=13))
                rhythmic_features = rhythmic_features.unsqueeze(0)
                # print("rhythmic_features: ", rhythmic_features.shape)

                semantic_features_pooled = F.interpolate(semantic_features.unsqueeze(0),
                                                         size=(feat_size, semantic_features.shape[2]),
                                                         mode='nearest').squeeze(0)
                rhythmic_features_pooled = F.interpolate(rhythmic_features.unsqueeze(0),
                                                         size=(feat_size, rhythmic_features.shape[2]),
                                                         mode='nearest').squeeze(0)

                # print("semantic_features_pooled: ", semantic_features_pooled.shape)
                # print("rhythmic_features_pooled: ", rhythmic_features_pooled.shape)

                rhythmic_features_pooled = rhythmic_features_pooled.to(semantic_features_pooled.device)

                combined_features = torch.cat((semantic_features_pooled, rhythmic_features_pooled), dim=-1)
                # print("1combined_features.shape", combined_features.shape)

                features_list.append(combined_features)

        # Stitch all feature fragments back together
        combined_features = torch.cat(features_list, dim=2)
        # print("2combined_features.shape", combined_features.shape)

        _, num_features, original_samples = combined_features.size()
        # Trim the combined_features to a multiple of 32 to ensure compatibility with the model
        trim_length = original_samples // original_batch * original_batch
        combined_features = combined_features[:, :, :trim_length].reshape(original_batch, num_features, -1)
        # print("3combined_features: ", combined_features.shape)
        # print("4original_features: ", original_features.shape)

        result = torch.cat((combined_features, original_features), dim=-1)
        print("5result_features: ", result.shape)

        # attention = nn.MultiheadAttention(embed_dim=input_feature.shape[-1], num_heads=1)
        #
        # # Calculate self-attention weights
        # attention_output, _ = attention(input_feature, input_feature, input_feature)
        #
        # # 最终的融合特征为加权平均
        # fused_feature = input_feature.cpu() + attention_output
        # print("fused_feature: ", fused_feature.shape)

        # Clear GPU cache
        torch.cuda.empty_cache()

        return result

    def forward(self, x, num_frames):
        # print("1. ===============x.shape: ", x.shape, x.unsqueeze(1).shape)
        x = self.extract_audio_features(x)
        # print("2. ===============x.shape: ", x.shape, x.unsqueeze(1).shape)
        x = self.specgram_encoder_2d(x.unsqueeze(1))
        x = F.interpolate(x, (1, num_frames), mode='bilinear')
        x = x.squeeze(2)
        # print("===============x.shape: ", x.shape)

        # Add attention
        # sa = ScaledDotProductAttention(d_model=x.shape[2], d_k=256, d_v=256, h=8)
        # x = sa(x.cuda(), x.cuda(), x.cuda())
        return x


class UNet_1D(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()

        leaky = cfg.VOICE2POSE.GENERATOR.LEAKY_RELU
        norm = cfg.VOICE2POSE.GENERATOR.NORM

        if cfg.VOICE2POSE.GENERATOR.CLIP_CODE.DIMENSION is not None:
            self.e0 = ConvNormRelu('1d', 256 + cfg.VOICE2POSE.GENERATOR.CLIP_CODE.DIMENSION, 256, downsample=False,
                                   norm=norm, leaky=leaky)
        else:
            self.e0 = ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky)

        self.e1 = ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky)
        self.e2 = ConvNormRelu('1d', 256, 256, downsample=True, norm=norm, leaky=leaky)
        self.e3 = ConvNormRelu('1d', 256, 256, downsample=True, norm=norm, leaky=leaky)
        self.e4 = ConvNormRelu('1d', 256, 256, downsample=True, norm=norm, leaky=leaky)
        self.e5 = ConvNormRelu('1d', 256, 256, downsample=True, norm=norm, leaky=leaky)
        self.e6 = ConvNormRelu('1d', 256, 256, downsample=True, norm=norm, leaky=leaky)

        self.d5 = ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky)
        self.d4 = ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky)
        self.d3 = ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky)
        self.d2 = ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky)
        self.d1 = ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky)

    def forward(self, x):
        # Attention
        # sa = ScaledDotProductAttention(d_model=x.shape[2], d_k=256, d_v=256, h=8)
        # x = sa(x.cuda(), x.cuda(), x.cuda())

        e0 = self.e0(x)
        e1 = self.e1(e0)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)

        d5 = self.d5(F.interpolate(e6, e5.size(-1), mode='linear') + e5)
        d4 = self.d4(F.interpolate(d5, e4.size(-1), mode='linear') + e4)
        d3 = self.d3(F.interpolate(d4, e3.size(-1), mode='linear') + e3)
        d2 = self.d2(F.interpolate(d3, e2.size(-1), mode='linear') + e2)
        d1 = self.d1(F.interpolate(d2, e1.size(-1), mode='linear') + e1)

        return d1


class SequenceGeneratorCNN(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()

        self.cfg = cfg

        leaky = cfg.VOICE2POSE.GENERATOR.LEAKY_RELU
        norm = cfg.VOICE2POSE.GENERATOR.NORM

        self.audio_encoder = AudioEncoder(cfg)
        self.unet = UNet_1D(cfg)
        self.decoder = nn.Sequential(
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky),
            nn.Conv1d(256, cfg.DATASET.NUM_LANDMARKS * 2, kernel_size=1, bias=True)
        )

    def forward(self, x, num_frames, code=None):
        x = self.audio_encoder(x, num_frames).cuda()  # (B, C, num_frame)

        if self.cfg.VOICE2POSE.GENERATOR.CLIP_CODE.DIMENSION is not None:
            code = code.unsqueeze(2).repeat([1, 1, x.shape[-1]])
            # print("==== code.shape: ", code.shape, x.shape)
            x = torch.cat([x, code], 1).cuda()

        # Attention
        # sa = ScaledDotProductAttention(d_model=x.shape[2], d_k=256, d_v=256, h=8)
        # x = sa(x.cuda(), x.cuda(), x.cuda())

        x = self.unet(x)
        x = self.decoder(x)

        x = x.permute([0, 2, 1]).reshape(-1, num_frames, 2, self.cfg.DATASET.NUM_LANDMARKS)
        return x
