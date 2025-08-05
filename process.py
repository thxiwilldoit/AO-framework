# instances code:

# Model parameter initialization
attention_fig = Attention(40, 128).to(self.device)
Pnet = Pnet(40, 128, 40).to(self.device)
Cnet = Cnet(40, 128).to(self.device)
AngleNN = AngleNN(40, 128).to(self.device)

# img_feature is obtained by calculating image features through ResNet network.
# Candidate_feature represents the final feature obtained after fusion with image information (without changing the original feature dimension)
img_mean = img_feature.mean(dim=1, keepdim=True)
img_std = img_feature.std(dim=1, keepdim=True)
img_std = img_std + 1e-8
normalized_img_tensor = (img_feature - img_mean) / img_std
pic_shape = normalized_img_tensor.shape
img_feature = normalized_img_tensor.reshape(pic_shape[0], int(pic_shape[1] * pic_shape[2] * pic_shape[3] / 40), 40)
img_personality = Pnet(img_feature)
common_feature = attention_fig(img_feature, candidate_feature)
common_feature_rotate = Cnet(common_feature)
common_feature_rotate = AngleNN(common_feature_rotate)
proj_fir = orthogonal_projection(img_personality, common_feature_rotate)
img_feature_proj = orthogonal_projection(img_personality, img_personality - proj_fir)
candidate_feature = attention_fig(img_feature_proj, candidate_feature)