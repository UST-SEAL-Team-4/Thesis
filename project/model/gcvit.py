import torch
import torch.nn as nn
import numpy as np

class PatchEmbedding(nn.Module):
  def __init__(self, d_model, img_size, patch_size, n_channels):
    super().__init__()
    
    self.d_model = d_model
    self.patch_size = patch_size
    self.n_channels = n_channels
    
    self.linear_project = nn.Conv2d(
      in_channels=self.n_channels,
      out_channels=self.d_model,
      kernel_size=self.patch_size,
      stride=self.patch_size
    )
    
  def forward(self, x):
    x = x.unsqueeze(0)
    
    x = torch.mean(input=x, dim=2)
    x = self.linear_project(x)
    
    x = x.flatten(2)
    x = x.transpose(1, 2)
    
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
    tokens_batch = self.cls_token.expand(size=(batch_size, -1, -1))
    
    x = torch.cat(tensors=(tokens_batch, x),
                  dim=1)
    
    x = x + self.pe[:, :x.size(1), :]
    # x = x + self.pe
    
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
  
class SegmentationHead(nn.Module):
  def __init__(self, n_channels, n_classes, img_size, patch_size, d_model, device):
    super(SegmentationHead, self).__init__()
    self.img_size = img_size
    self.patch_size = patch_size
    self.n_channels = n_channels
    self.d_model = d_model
    self.device = device
    
    self.conv = nn.Conv2d(
      in_channels=self.n_channels,
      out_channels=self.d_model,
      kernel_size=1,
      stride=self.patch_size,
      padding=(1,1),
      dilation=(1,1)
    )
    
  def forward(self, x, masks=None):
    batch_size, embed_dim, sige_length = x.shape
    
    x = x.unsqueeze(1)
    x = self.conv(x)
    
    if masks is not None:
      x = x.to(self.device)
      masks = masks.to(self.device)
      
      x = nn.functional.interpolate(
        input=x,
        size=(masks.shape[1], masks.shape[2]),
        mode='bilinear'
      )
      
      masks = masks.long()
      loss_fn = nn.CrossEntropyLoss()
      loss = loss_fn(
        input=x,
        target=masks
      )
      
      print("Loss: ", loss)
      return x, loss
    else:
      return x
    
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
    
    self.n_patches = (self.img_size[0] * self.img_size[1]) // (self.patch_size[0] * self.patch_size[1])
    self.max_seq_length = self.n_patches + 1
    
    self.patch_embedding = PatchEmbedding(
      d_model=self.d_model,
      img_size=self.img_size,
      patch_size=self.patch_size,
      n_channels=self.n_channels
    )
    
    self.positional_encoding = PositionalEncoding(
      d_model=self.d_model,
      max_seq_length=self.max_seq_length,
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
      n_channels=self.n_channels,
      n_classes=self.n_classes,
      img_size=self.img_size,
      patch_size=self.patch_size,
      d_model=self.d_model,
      device=self.device
    )
    
  def forward(self, images, mask=None):
    images = images.to(self.device)
    if mask is not None:
      mask = mask.to(self.device)

    x = self.patch_embedding(images)
    x = self.positional_encoding(x)
    x = self.transformer_encoder(x)
    x = self.segmentation_head(x, mask)

    return x














