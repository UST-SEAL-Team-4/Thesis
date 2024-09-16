import torch
import torch.nn as nn
import numpy as np

class PatchEmbedding(nn.Module):
  def __init__(self, d_model, img_size, patch_size, n_channels):
    super().__init__()
    
    self.d_model = d_model
    self.patch_size = patch_size
    self.n_channels = n_channels
    
    # self.linear_project = nn.Conv2d(
    #   in_channels=self.n_channels,
    #   out_channels=self.d_model,
    #   kernel_size=self.patch_size,
    #   stride=self.patch_size
    # )

    # self.linear_project = nn.Conv2d(
    #   in_channels=self.n_channels,
    #   out_channels=self.d_model,
    #   kernel_size=1,
    #   stride=1
    # )

    self.linear_project = nn.Linear(n_channels * 16 * 16, d_model)
    
  # def forward(self, x):
  #   x = x.unsqueeze(0)
    
  #   x = torch.mean(input=x, dim=2)
  #   x = self.linear_project(x)
    
  #   x = x.flatten(2)
  #   x = x.transpose(1, 2)
    
  #   return x
  def forward(self, x):
    print(x.shape)
    # x = torch.stack(x_list, dim=0)
        
    # x = x_list.unsqueeze(0)
    # x = torch.mean(input=x_list, dim=2)
    # x = self.linear_project(x)
    # x = x.flatten(2)
    # x = x.transpose(1, 2)

    batch_size, channels, num_patches, patch_height, patch_width = x.shape

    x = x.view(batch_size, num_patches, -1)

    x = self.linear_project(x)
    print(x.shape)
        
    return x
  
class PositionalEncoding(nn.Module):
  def __init__(self, d_model, max_seq_length):
    super().__init__()
    
    self.cls_token = nn.Parameter(torch.randn(size=(1, 1, d_model)))
    
    pe = torch.zeros(size=(max_seq_length, d_model))
    
    for pos in range(max_seq_length):
      for i in range(d_model):
        if i % 2 == 0:
          pe[pos][i] = np.sin(pos/(10000 ** (i/d_model)))
        else:
          pe[pos][i] = np.cos(pos/(10000 ** ((i-1)/d_model)))
          
    self.register_buffer('pe', pe.unsqueeze(0))
    
  def forward(self, x):
    batch_size = x.size(0)

    tokens_batch = self.cls_token.expand(size=(batch_size, -1, -1)).to(x.device)
    # pe = self.pe[:, :x.size(1), :].to(x.device)
    x = torch.cat(tensors=(tokens_batch, x), dim=1)
    
    x = x + self.pe[:, :x.size(1), :].to(x.device)
    # x = x + pe
    
    return x

class AttentionHead(nn.Module):
  def __init__(self, d_model, head_size):
    super().__init__()
    self.head_size = head_size
    
    self.query = nn.Linear(
      in_features=d_model,
      out_features=head_size
    )
    self.key = nn.Linear(
      in_features=d_model,
      out_features=head_size
    )
    self.value = nn.Linear(
      in_features=d_model,
      out_features=head_size
    )
  
  def forward(self, x):
    Q = self.query(x)
    K = self.key(x)
    V = self.value(x)
    
    attention = Q @ K.transpose(-2, -1)
    
    attention = attention / (self.head_size ** 0.5)
    attention = torch.softmax(attention, dim=-1)
    
    attention_output = attention @ V
    return attention_output
  
class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, n_heads):
    super().__init__()
    self.head_size = d_model // n_heads
    
    self.W_o = nn.Linear(
      in_features=d_model,
      out_features=d_model
    )
    
    self.heads = nn.ModuleList([
      AttentionHead(
        d_model=d_model,
        head_size=self.head_size
      ) for _ in range(n_heads)
    ])
    
  def forward(self, x):
    out = torch.cat(
      tensors=[head(x) for head in self.heads],
      dim=-1
    )
    
    out = self.W_o(out)
    return out
  
class TransformerEncoder(nn.Module):
  def __init__(self, d_model, n_heads, r_mlp=4):
    super().__init__()
    self.d_model = d_model
    self.n_heads = n_heads
    
    self.ln1 = nn.LayerNorm(
      normalized_shape=d_model
    )
    
    self.mha = MultiHeadAttention(
      d_model=d_model,
      n_heads=n_heads
    )
    
    self.ln2 = nn.LayerNorm(
      normalized_shape=d_model
    )
    
    self.mlp = nn.Sequential(
      nn.Linear(
        in_features=d_model,
        out_features=d_model * r_mlp
      ),
      nn.GELU(),
      nn.Linear(
        in_features=d_model * r_mlp,
        out_features=d_model
      )
    )
    
  def forward(self, x):
    out = x + self.mha(self.ln1(x))
    out = out + self.mlp(self.ln2(out))
    
    return out
  
# class SegmentationHead(nn.Module):
#   def __init__(self, n_channels, n_classes, img_size, patch_size, d_model, device):
#     super(SegmentationHead, self).__init__()
#     self.img_size = img_size
#     self.patch_size = patch_size
#     self.n_channels = n_channels
#     self.d_model = d_model
#     self.device = device
#     self.n_classes = n_classes
    
#     self.conv = nn.Conv2d(
#       in_channels=self.n_channels,
#       out_channels=self.n_classes,
#       kernel_size=1,
#       stride=1,
#       padding=(1,1),
#       dilation=(1,1)
#     )
    
#   def forward(self, x, masks=None):
#     batch_size, embed_dim, sige_length = x.shape
    
#     x = x.unsqueeze(1)
#     print("shape of x before conv: ", x.shape)
#     x = self.conv(x)
#     print("shape of x after conv: ", x.shape)
    
#     if masks is not None:
#       x = x.to(self.device)
#       masks = masks.to(self.device)
      
#       x = nn.functional.interpolate(
#         input=x,
#         size=(masks.shape[-2], masks.shape[-1]),
#         mode='bilinear'
#       )
      
#       masks = masks.long()
#       print("shape of x: ", x.shape)
#       print("shape of mask: ", masks.shape)

#       # # Flatten tensors for loss calculation
#       # x = x.permute(0, 2, 3, 1).contiguous()  # [batch_size, height, width, n_classes]
#       # x = x.view(-1, x.size(3))  # [batch_size * height * width, n_classes]
#       # masks = masks.view(-1)  # [batch_size * height * width]
            
#       loss_fn = nn.CrossEntropyLoss()
#       # loss = loss_fn(x, masks)
      
#       # print("Loss: ", loss)
#       total_loss = 0
#       for i in range(masks.size(1)):
#         slice_pred = x[:, 0, :] # will still be checked
#         slice_mask = masks[:, i, :, :].float()

#         loss = loss_fn(
#           input=slice_pred,
#           target=slice_mask
#         )

#         total_loss += loss
      
#       avg_loss = total_loss / masks.size(1)  # Average loss across slices
#       print("Average Loss: ", avg_loss)
#       return x, avg_loss
#     else:
#       return x

class SegmentationHead(nn.Module):
  def __init__(self, D_model, num_classes, image_size, Patch_size, num_patches, device):
      super(SegmentationHead, self).__init__()
      self.image_size = image_size
      self.Patch_size = Patch_size
      self.device = device

      # Compute the number of patches along each dimension
      # self.h_patches = image_size[0] // Patch_size[0]
      # self.w_patches = image_size[1] // Patch_size[1]
      # self.num_patches = self.h_patches * self.w_patches

      self.h_patches = Patch_size[0]
      self.w_patches = Patch_size[1]
      self.num_patches = num_patches

      # Linear layer to project from the embedding dimension to the number of classes
      self.classifier = nn.Linear(D_model, num_classes).to(device)

      # ConvTranspose2d to upsample the patch-based output back to the original image size
      self.upsample = nn.ConvTranspose2d(
          in_channels=num_classes,
          out_channels=num_classes,
          kernel_size=1,
          stride=1
      ).to(device)

  def forward(self, x, masks):
      batch_size, num_patches, D_model = x.shape
      # masks = torch.stack(masks)
      print(f"Input shape before classification: {x.shape}")
      
      # Remove the class token (if present)
      # if num_patches == self.num_patches + 1:
      x = x[:, 1:, :]
      num_patches -= 1
      x = x.to(self.device)
      x = self.classifier(x)

      assert num_patches == self.num_patches, f"Mismatch in the number of patches: {num_patches} vs {self.num_patches}"
      print(x.shape)
      x = x.transpose(1, 2).reshape(batch_size, -1, self.h_patches, self.w_patches )
      print(f"Shape after reshaping: {x.shape}")

      x = self.upsample(x)
      print(f"Shape after upsampling: {x.shape}")
          
          
      masks = masks.long()
      print("Shape of mask in segmentation:", masks.shape)
      loss_fn = nn.CrossEntropyLoss()
      loss = loss_fn(
        input=x,
        target=masks
      )
        
      print("Loss: ", loss)
      return x, loss
      
class VisionTransformer(nn.Module):
  def __init__(
    self,
    d_model,
    n_classes,
    img_size,
    patch_size,
    n_channels,
    n_heads,
    n_layers,
    device
  ):
    super().__init__()
    
    assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, "img_size dimensions must be divisible by patch_size dimensions"
    assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
    
    self.d_model = d_model
    self.n_classes = n_classes
    self.img_size = img_size
    self.patch_size = patch_size
    self.n_channels = n_channels
    self.n_heads = n_heads
    self.device = device
    
    # self.n_patches = (self.img_size[0] * self.img_size[1]) // (self.patch_size[0] * self.patch_size[1])
    # self.max_seq_length = self.n_patches + 1
    
    self.patch_embedding = PatchEmbedding(
      d_model=self.d_model,
      img_size=self.img_size,
      patch_size=self.patch_size,
      n_channels=self.n_channels
    )
    
    self.positional_encoding = PositionalEncoding(
      d_model=self.d_model,
      max_seq_length=1000, # Placeholder only
    )
    
    self.transformer_encoder = nn.Sequential(
      *[
        TransformerEncoder(
          d_model=self.d_model,
          n_heads=self.n_heads
        ) for _ in range(n_layers)
      ] 
    )
    
    self.segmentation_head = SegmentationHead(
      # n_channels=self.n_channels,
      num_classes=self.n_classes,
      image_size=self.img_size,
      Patch_size=self.patch_size,
      D_model=self.d_model,
      num_patches=0, # placeholder
      device=self.device
    )
    
  def forward(self, images, mask=None):
    images = images.to(self.device)

    # Compute number of patches
    batch_size, channels, num_patches, patch_height, patch_width = images.shape
    max_seq_length = num_patches + 1

    if mask is not None:
      mask = mask.to(self.device)
    
    print('Shape before patch:', images.shape)

    x = self.patch_embedding(images)

    self.positional_encoding = PositionalEncoding(
        d_model=self.d_model,
        max_seq_length=max_seq_length
    )
    print('Shape after patch:', images.shape)
    x = self.positional_encoding(x)
    print('Shape after positional:', images.shape)
    x = self.transformer_encoder(x)
    print('Shape after transformer:', images.shape)
    
    self.segmentation_head = SegmentationHead(
      num_classes=self.n_classes,
      image_size=self.img_size,
      Patch_size=self.patch_size,
      D_model=self.d_model,
      num_patches=num_patches,
      device=self.device
    )
    x = self.segmentation_head(x, mask)
    print('Shape after segmentation:', images.shape)

    return x














