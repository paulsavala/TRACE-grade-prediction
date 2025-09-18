import torch
from torch import nn
import math

from data_prep import dataset


# class PositionalEncoding(nn.Module):
#     def __init__(self, device, d_model, max_len, dropout=0.1):
#         super(PositionalEncoding, self).__init__()
#         self.encoding = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         # div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         div_term = torch.exp(- torch.arange(0, d_model, 2) * math.log(10000) / d_model)
#         self.encoding[:, 0::2] = torch.sin(position * div_term)
#         self.encoding[:, 1::2] = torch.cos(position * div_term)
#         self.encoding = self.encoding.to(device)

#         self.dropout = nn.Dropout(dropout)
#         self.register_buffer('pos_embedding', self.encoding)

#     def forward(self, x, custom_positions):
#         """
#         x: Tensor, shape [batch_size, seq_len, d_model]
#         custom_positions: Tensor, shape [batch_size, seq_len], integer values indicating the custom absolute positions
#         """
#         return self.dropout(x + self.encoding[custom_positions])

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        div_term = torch.exp(- torch.arange(0, d_model, 2) * math.log(10000) / d_model)
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', self.encoding)

    def forward(self, x, custom_positions):
        """
        x: Tensor, shape [batch_size, seq_len, d_model]
        custom_positions: Tensor, shape [batch_size, seq_len], integer values indicating the custom absolute positions
        """
        pos_embedding = self.encoding.to(x.device)
        return self.dropout(x + pos_embedding[custom_positions.to(x.device)]).to(x.device)

class TransformerModel(nn.Module):
    def __init__(self, device, n_courses, n_majors, d_model=128, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=512, max_len=70):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.n_courses = n_courses
        self.device = device

        # Embedding layer for the students major
        self.major_embedding = nn.Embedding(n_majors, d_model)

        # Embedding layer to map courses to model dimension
        self.embedding = nn.Embedding(n_courses, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(device, d_model, max_len)
        
        # Transformer model
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, batch_first=True)

        # Linear layer mapping to outputs
        self.linear = nn.Linear(d_model, n_courses)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, major, src, tgt, src_custom_positions, tgt_custom_positions, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # Embed the source and target
        src_embed = self.embedding(src)
        tgt_embed = self.embedding(tgt)

        # Embed the major. Insert a dimension where the sequence length would be (used for concatenating later)
        major_embed = self.major_embedding(major).unsqueeze(1)

        # Extend the src custom positions to include the major (first semester)
        src_major_custom_positions = torch.ones_like(major).unsqueeze(1)
        src_custom_positions = torch.cat((src_major_custom_positions, src_custom_positions), dim=1)

        # Extend the source key padding mask to include the major (first semester)
        src_major_key_padding_mask = torch.zeros_like(major).bool().unsqueeze(1)
        src_key_padding_mask = torch.cat((src_major_key_padding_mask, src_key_padding_mask), dim=1)
        
        # Concatenate the major and course sequence embeddings
        src_embed = torch.cat((major_embed, src_embed), dim=1)

        # Add positional encoding to the source and target
        src_enc = self.pos_encoder(src_embed, src_custom_positions)
        tgt_enc = self.pos_encoder(tgt_embed, tgt_custom_positions)

        # Run through the transformer
        output = self.transformer(src_enc, tgt_enc, 
                                  src_key_padding_mask=src_key_padding_mask, 
                                  tgt_key_padding_mask=tgt_key_padding_mask, 
                                  memory_key_padding_mask=memory_key_padding_mask)
        
        # Pass through linear layer
        output = self.linear(output).squeeze()
        
        # Reshape output into size expected by cross entropy loss: [batch size, number of classes, sequence length]
        output = output.permute(0, 2, 1)

        # Collapse down into a probability distribution over all courses (sequence elements)
        output_lsm = self.softmax(output.sum(dim=2))

        return output_lsm, output
    

class TransformerModelWithGrades(nn.Module):
    def __init__(self, n_courses, n_majors, max_len, config, PAD_IDX=0):
        super(TransformerModelWithGrades, self).__init__()
        self.d_model = config['d_model']
        self.major_embedding_dim = config['major_embedding_dim']
        self.course_embedding_dim = config['course_embedding_dim']
        self.gpa_embedding_dim = config['gpa_embedding_dim']
        self.nhead = config['nhead']
        self.num_encoder_layers = config['num_encoder_layers']
        self.num_decoder_layers = config['num_decoder_layers']
        self.dim_feedforward = config['dim_feedforward']
        self.dropout = config['dropout']

        self.n_courses = n_courses
        self.device = None # Assigned once data is first received, to handle multiple GPUs
        self.PAD_IDX = PAD_IDX

        # Embedding layer for the students major
        self.major_embedding = nn.Embedding(n_majors, self.major_embedding_dim)
        self.major_projection = nn.Linear(self.major_embedding_dim, self.d_model)

        # GPA embedding
        self.gpa_embedding = nn.Linear(1, self.gpa_embedding_dim)
        self.gpa_projection = nn.Linear(self.gpa_embedding_dim, self.d_model)

        # Embedding layer to map courses to model dimension
        self.course_embedding = nn.Embedding(n_courses, self.course_embedding_dim)
        self.course_projection = nn.Linear(self.course_embedding_dim, self.d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model, max_len, self.dropout)
        
        # Transformer model
        self.transformer = nn.Transformer(self.d_model, self.nhead, self.num_encoder_layers, self.num_decoder_layers, self.dim_feedforward, batch_first=True)

        # Linear layer mapping to course outputs
        self.linear_courses = nn.Linear(self.d_model, n_courses)

        # Linear layer mapping to grade outputs
        self.linear_gpa = nn.Linear(self.d_model, 1)

        # Softmax layer for course outputs (log-scale expected for KLDivLoss)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def embed(self, major=None, src_courses=None, tgt_courses=None, src_gpas=None, tgt_gpas=None):
        src_embed = None
        tgt_embed = None

        if src_courses is not None:
            # Embed the source
            src_courses_embed = self.course_projection(self.course_embedding(src_courses))
            
            # Embed the gpas
            src_gpas_embed = self.gpa_projection(self.gpa_embedding(src_gpas.unsqueeze(2)))
        
            # Embed the major. Insert a dimension where the sequence length would be (used for concatenating later)
            major_embed = self.major_projection(self.major_embedding(major)).unsqueeze(1)
            """
            major_embed: [batch size, 1, d_model]
            """

            # Concatenate the major, course sequence and grade embeddings
            src_embed = torch.cat((major_embed, src_courses_embed, src_gpas_embed), dim=1)
        
        if tgt_courses is not None:
            # Embed the target
            tgt_courses_embed = self.course_projection(self.course_embedding(tgt_courses))
            """
            src_courses_embed: [batch size, src sequence length, d_model]
            tgt_courses_embed: [batch size, tgt sequence length, d_model]
            """
            
            # Embed the gpas
            tgt_gpas_embed = self.gpa_projection(self.gpa_embedding(tgt_gpas.unsqueeze(2)))
            """
            src_gpas_embed: [batch size, src sequence length, d_model]
            tgt_gpas_embed: [batch size, tgt sequence length, d_model]
            """
            
            # Concatenate the course sequence and grade embeddings
            tgt_embed = torch.cat((tgt_courses_embed, tgt_gpas_embed), dim=1)
            """
            src_embed: [batch size, 1 + 2*src sequence length, d_model]
            tgt_embed: [batch size, 2*tgt sequence length, d_model]
            """
        
        if src_embed is not None and tgt_embed is not None:
            return src_embed, tgt_embed
        elif src_embed is not None:
            return src_embed
        elif tgt_embed is not None:
            return tgt_embed
        else:
            return None
    
    def position(self, major=None, src_embed=None, tgt_embed=None, src_positions=None, tgt_positions=None):
        src_pos = None
        tgt_pos = None

        if src_embed is not None:
            assert src_positions is not None, "src_positions must be provided if src_embed is provided"
            assert major is not None, "major must be provided if src_embed is provided"
            # Extend the src_courses custom positions to include the major (first semester) and gpas
            src_major_custom_positions = torch.ones_like(major).unsqueeze(1)
            src_gpas_custom_positions = src_positions
            src_custom_positions = torch.cat((src_major_custom_positions, src_positions, src_gpas_custom_positions), dim=1)
            """
            src_custom_positions: [batch size, 1 + 2*src sequence length, d_model]
            """

            # Add positional encoding to the source and target
            src_pos = self.pos_encoder(src_embed, src_custom_positions)
            """
            src_enc: [batch size, 1 + 2*src sequence length, d_model]
            """

        if tgt_embed is not None:
            assert tgt_positions is not None, "tgt_positions must be provided if tgt_embed is provided"
            # Extend the target custom positions to include the gpas
            tgt_gpas_custom_positions = tgt_positions
            tgt_custom_positions = torch.cat((tgt_positions, tgt_gpas_custom_positions), dim=1)
            """
            tgt_custom_positions: [batch size, 2*tgt sequence length, d_model]
            """

            # Add positional encoding to the source and target
            tgt_pos = self.pos_encoder(tgt_embed, tgt_custom_positions)
            """
            tgt_enc: [batch size, 2*tgt sequence length, d_model]
            """

        if src_pos is not None and tgt_pos is not None:
            return src_pos, tgt_pos
        elif src_pos is not None:
            return src_pos
        elif tgt_pos is not None:
            return tgt_pos
        else:
            return None

    def generator(self, logits, seq_len):
        # Split the output into course and gpa outputs
        output_courses, output_gpas = logits.split(seq_len, dim=1)

        # Pass through linear layers
        output_courses = self.linear_courses(output_courses)
        output_gpas = self.linear_gpa(output_gpas)
        # if output_courses.size(0) != 1:
            # output_courses = output_courses.squeeze()
            # output_gpas = output_gpas.squeeze()
        
        # Reshape output into size expected by cross entropy loss: [batch size, number of classes, sequence length].
        # output_courses will also be used later predicting the course.
        output_courses = output_courses.permute(0, 2, 1)

        return output_courses, output_gpas

    # def forward(self, major, src_courses, tgt_courses, src_gpas, tgt_gpas, src_courses_positions, tgt_courses_positions, src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask):
    def forward(self, major, src_courses, tgt_courses, src_gpas, tgt_gpas, src_courses_positions, tgt_courses_positions):
        print_sizes = False
        """
        major: [batch size, 1]
        src_courses: [batch size, src sequence length]
        tgt_courses: [batch size, tgt sequence length]
        src_gpas: [batch size, src sequence length]
        tgt_gpas: [batch size, tgt sequence length]
        src_courses_positions: [batch size, src sequence length]
        tgt_courses_positions: [batch size, tgt sequence length]
        src_courses_key_padding_mask: [batch size, src sequence length]
        tgt_courses_key_padding_mask: [batch size, tgt sequence length]
        memory_key_padding_mask: [batch size, src sequence length]
        """
        self.device = major.device

        assert src_courses.isnan().sum().item() == 0, "nan values in src_courses"
        assert tgt_courses.isnan().sum().item() == 0, "nan values in tgt_courses"
        assert src_gpas.isnan().sum().item() == 0, "nan values in src_gpas"
        assert tgt_gpas.isnan().sum().item() == 0, "nan values in tgt_gpas"

        # Move everything to the device of the model
        # major = major.to(self.device)
        # src_courses = src_courses.to(self.device)
        # tgt_courses = tgt_courses.to(self.device)
        # src_gpas = src_gpas.to(self.device)
        # tgt_gpas = tgt_gpas.to(self.device)
        # src_courses_positions = src_courses_positions.to(self.device)
        # tgt_courses_positions = tgt_courses_positions.to(self.device)

        # Source and target masks
        # Add one for the major
        src_seq_len = 1 + src_courses.size(1) + src_gpas.size(1)
        tgt_input_courses_seq_len = tgt_gpas.size(1)

        src_mask = dataset.generate_src_mask(src_seq_len).to(self.device)
        tgt_mask = dataset.generate_tgt_mask(tgt_input_courses_seq_len).to(self.device)

        # Padding masks
        src_key_padding_mask = dataset.generate_src_padding_masks(src_courses, major, self.PAD_IDX).to(self.device)
        tgt_key_padding_mask = dataset.generate_tgt_padding_masks(tgt_courses, self.PAD_IDX).to(self.device)

        # Embed and positionally encode the courses, gpas and major
        src_embed, tgt_embed = self.embed(major, src_courses, tgt_courses, src_gpas, tgt_gpas)
        src_pos_embed, tgt_pos_embed = self.position(major, src_embed, tgt_embed, src_courses_positions, tgt_courses_positions)

        assert src_embed.isnan().sum().item() == 0, "nan values in src_embed"
        assert src_pos_embed.isnan().sum().item() == 0, "nan values in src_pos_embed"
        assert tgt_embed.isnan().sum().item() == 0, "nan values in tgt_embed"
        assert tgt_pos_embed.isnan().sum().item() == 0, "nan values in tgt_pos_embed"

        # Run through the transformer
        output = self.transformer(src_pos_embed, tgt_pos_embed,
                                  src_mask=src_mask, 
                                  tgt_mask=tgt_mask,
                                  src_key_padding_mask=src_key_padding_mask, 
                                  tgt_key_padding_mask=tgt_key_padding_mask, 
                                  memory_key_padding_mask=src_key_padding_mask)
        if print_sizes:
            print(f'output: {output.size()}')

        # Split the output into course and gpa outputs and pass through the appropriate forward layers
        output_courses, output_gpas = self.generator(output, tgt_courses.size(1))

        # Log softmax for NLLLoss
        output_courses_lsm = self.log_softmax(output_courses)
        return output_courses_lsm, output_gpas, output_courses
    
    # Separate helper functions to run just the encoder/decoder layers of our model.
    # This is useful for inference where we want to encode our input sequence and
    # get the memory to be used for our decoder.
    def encode(self, major, src_courses, src_gpas, src_courses_positions, src_mask, src_key_padding_mask):
        # Embed and positionally encode the courses, gpas and major
        src_embed = self.embed(major=major, 
                               src_courses=src_courses, 
                               src_gpas=src_gpas)
        src_pos_embed = self.position(major=major, 
                                      src_embed=src_embed, 
                                      src_positions=src_courses_positions)
        memory = self.transformer.encoder(src=src_pos_embed, 
                                          mask=src_mask, 
                                          src_key_padding_mask=src_key_padding_mask)

        return memory

    def decode(self, memory, tgt_courses, tgt_gpas, tgt_courses_positions, tgt_mask, tgt_key_padding_mask):
        # Embed and positionally encode the courses and gpas
        tgt_embed = self.embed(tgt_courses=tgt_courses, 
                               tgt_gpas=tgt_gpas)
        tgt_pos_embed = self.position(tgt_embed=tgt_embed, 
                                      tgt_positions=tgt_courses_positions)
        
        out = self.transformer.decoder(tgt=tgt_pos_embed, 
                                       memory=memory, 
                                       tgt_mask=tgt_mask, 
                                       tgt_key_padding_mask=tgt_key_padding_mask)

        return out