import torch
import torch.nn as nn
import numpy as np

class PatchEmbedding(nn.Module):
  def __init__(self, D_model, Img_size, Patch_size, N_channels):
    super().__init__()
    
    self.D_model = D_model
    self.Patch_size = Patch_size
    self.N_channels = N_channels
    
    self.linear_project = nn.Linear(N_channels * Patch_size[0] * Patch_size[1], D_model)
    

  def forward(self, x):
    batch_size, channels, num_patches, patch_height, patch_width = x.shape
    print("before", x.shape)
    x = x.view(batch_size, num_patches, patch_height * patch_width)
    print("after view", x.shape)
    x = self.linear_project(x)
    print("after project", x.shape)
        
    return x
  
class PositionalEncoding(nn.Module):
    def __init__(self, D_model, max_seq_length):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(size=(1, 1, D_model)))
        pe = torch.zeros(size=(max_seq_length, D_model))
        
        for pos in range(max_seq_length):
            for i in range(D_model):
                if i % 2 == 0:
                    pe[pos][i] = np.sin(pos/(10000 ** (i/D_model)))
                else:
                    pe[pos][i] = np.cos(pos/(10000 ** ((i-1)/D_model)))
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
    def __init__(self, D_model, head_size):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(in_features=D_model, out_features=head_size)
        self.key = nn.Linear(in_features=D_model, out_features=head_size)
        self.value = nn.Linear(in_features=D_model, out_features=head_size)
  
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
    def __init__(self, D_model, N_heads):
        super().__init__()
        self.head_size = D_model // N_heads
        self.W_o = nn.Linear(in_features=D_model, out_features=D_model)
        self.heads = nn.ModuleList([AttentionHead(D_model=D_model, head_size=self.head_size) for _ in range(N_heads)])
    
    def forward(self, x):
        out = torch.cat(tensors=[head(x) for head in self.heads], dim=-1)
        out = self.W_o(out)
        return out
  
class TransformerEncoder(nn.Module):
    def __init__(self, D_model, N_heads, r_mlp=4):
        super().__init__()
        self.D_model = D_model
        self.N_heads = N_heads
        
        self.ln1 = nn.LayerNorm(normalized_shape=D_model)
        self.mha = MultiHeadAttention(D_model=D_model, N_heads=N_heads)
        self.ln2 = nn.LayerNorm(normalized_shape=D_model)
        
        self.mlp = nn.Sequential(
            nn.Linear(in_features=D_model, out_features=D_model * r_mlp),
            nn.GELU(),
            nn.Linear(in_features=D_model * r_mlp, out_features=D_model)
        )
    
    def forward(self, x):
        out = x + self.mha(self.ln1(x))
        out = out + self.mlp(self.ln2(out))
        return out

# class SegmentationHead(nn.Module):
#     def __init__(self, D_model, num_classes, image_size, Patch_size):
#         super(SegmentationHead, self).__init__()
#         self.image_size = image_size
#         self.Patch_size = Patch_size

#         # Compute the number of patches along each dimension
#         self.h_patches = image_size[0] // Patch_size[0]
#         self.w_patches = image_size[1] // Patch_size[1]
#         self.num_patches = self.h_patches * self.w_patches

#         # Linear layer to project from the embedding dimension to the number of classes
#         self.classifier = nn.Linear(D_model, num_classes)

#         # ConvTranspose2d to upsample the patch-based output back to the original image size
#         self.upsample = nn.ConvTranspose2d(
#             in_channels=num_classes,
#             out_channels=num_classes,
#             kernel_size=Patch_size,
#             stride=Patch_size
#         )

#     def forward(self, x, masks):
#         batch_size, num_patches, D_model = x.shape
#         print(f"Input shape before classification: {x.shape}")

#         # Remove the class token (if present)
#         if num_patches == self.num_patches + 1:
#             x = x[:, 1:, :]
#             num_patches -= 1
#         x = self.classifier(x)

#         assert num_patches == self.num_patches, f"Mismatch in the number of patches: {num_patches} vs {self.num_patches}"

#         x = x.transpose(1, 2).reshape(batch_size, -1, self.h_patches, self.w_patches)
#         print(f"Shape after reshaping: {x.shape}")

#         x = self.upsample(x)
#         print(f"Shape after upsampling: {x.shape}")
        
#         masks = masks.long()
#         loss_fn = nn.CrossEntropyLoss()
#         loss = loss_fn(
#           input=x,
#           target=masks
#         )
      
#         print("Loss: ", loss)
#         return x, loss

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
      self.num_slices = num_patches
      # self.total_elements = self.h_patches * self.w_patches * self.num_patches

      # Linear layer to project from the embedding dimension to the number of classes
      self.classifier = nn.Linear(D_model, num_classes).to(device)

      # ConvTranspose2d to upsample the patch-based output back to the original image size
      self.upsample = nn.ConvTranspose2d(
          in_channels=num_classes,
          out_channels=num_classes,
          kernel_size=Patch_size,
          stride=Patch_size
      ).to(device)

  def forward(self, x, masks):
    batch_size, num_patches, D_model = x.shape
    print(f"Input shape before classification: {x.shape}")

    # Remove the class token (if present)
    if num_patches == self.num_slices + 1:
        x = x[:, 1:, :]
        num_patches -= 1
    x = self.classifier(x)

    assert num_patches == self.num_slices, f"Mismatch in the number of patches: {num_patches} vs {self.num_slices}"

    # Since num_patches = num_slices, reshape accordingly
    x = x.transpose(1, 2).reshape(batch_size, -1, self.num_slices, 1)
    print(f"Shape after reshaping: {x.shape}")

    x = self.upsample(x)

    masks = masks.view(batch_size, self.h_patches * self.num_slices, self.w_patches)
    masks = masks.long()
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(input=x, target=masks)
        
    print("Loss: ", loss)
    return x, loss



class VisionTransformer(nn.Module):
    def __init__(
        self,
        D_model,
        N_classes,
        Img_size,
        Patch_size,
        N_channels,
        N_heads,
        N_layers,
        device
    ):
        super().__init__()
        
        assert Img_size[0] % Patch_size[0] == 0 and Img_size[1] % Patch_size[1] == 0, "Img_size dimensions must be divisible by Patch_size dimensions"
        assert D_model % N_heads == 0, "D_model must be divisible by N_heads"
        
        self.D_model = D_model
        self.N_classes = N_classes
        self.Img_size = Img_size
        self.Patch_size = Patch_size
        self.N_channels = N_channels
        self.N_heads = N_heads
        self.device = device
        
        # self.n_patches = (self.Img_size[0] * self.Img_size[1]) // (self.Patch_size[0] * self.Patch_size[1])
        # self.max_seq_length = self.n_patches + 1
        
        self.patch_embedding = PatchEmbedding(
            D_model=self.D_model,
            Img_size=self.Img_size,
            Patch_size=self.Patch_size,
            N_channels=self.N_channels
        )
        
        # self.positional_encoding = PositionalEncoding(
        #     D_model=self.D_model,
        #     max_seq_length=self.max_seq_length,
        # )
        
        self.transformer_encoder = nn.Sequential(
            *[
                TransformerEncoder(
                    D_model=self.D_model,
                    N_heads=self.N_heads
                ) for _ in range(N_layers)
            ] 
        )
        
        # self.segmentation_head = SegmentationHead(
        #     D_model=self.D_model,
        #     num_classes=self.N_classes,
        #     image_size=self.Img_size,
        #     Patch_size=self.Patch_size
        # )

        self.segmentation_head = SegmentationHead(
        # n_channels=self.n_channels,
        num_classes=self.N_classes,
        image_size=self.Img_size,
        Patch_size=self.Patch_size,
        D_model=self.D_model,
        num_patches=0, # placeholder
        device=self.device
        )
    
    def forward(self, images, mask=None, current_slice=None):
        images = images.to(self.device)

        # x = self.patch_embedding(images)
        # x = self.positional_encoding(x)
        # x = self.transformer_encoder(x)
        # x, loss = self.segmentation_head(x, mask)

        # Compute number of patches
        batch_size, channels, num_patches, patch_height, patch_width = images.shape
        max_seq_length = num_patches + 1

        if mask is not None:
          mask = mask.to(self.device)
    
        print('Shape before patch:', images.shape)

        x = self.patch_embedding(images)

        self.positional_encoding = PositionalEncoding(
            D_model=self.D_model,
            max_seq_length=max_seq_length
        )
        print('Shape after patch:', x.shape)
        x = self.positional_encoding(x)
        print('Shape after positional:', x.shape)
        x = self.transformer_encoder(x)
        print('Shape after transformer:', x.shape)
        if current_slice != None:
            print(x[0, 0, current_slice].shape)
            temp = x[0, 0, current_slice]
            temp = temp.view(1, 1, patch_height * patch_width)
            x = self.transformer_encoder(temp)
            print('Shape after with current slice:', x.shape)
    
        self.segmentation_head = SegmentationHead(
          num_classes=self.N_classes,
          image_size=self.Img_size,
          Patch_size=self.Patch_size,
          D_model=self.D_model,
          num_patches=num_patches,
          device=self.device
        )
        x, loss = self.segmentation_head(x, mask)
        print('Shape after segmentation:', x.shape)

        return x
