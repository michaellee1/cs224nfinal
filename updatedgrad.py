
import ujson as json
import numpy as np
import spacy 
import setup
from args import get_setup_args
from collections import Counter
import time
import torch

args = get_setup_args()


# nlp = spacy.load('en_core_web_lg')
# word_counter = Counter()
# word_emb_mat, word2idx_dict = setup.get_embedding(word_counter, 'word', emb_file=args.glove_file, vec_size=args.glove_dim, num_vectors=args.glove_num_vecs)
# tokens = nlp(words) 
  
# for token in tokens: 
#     # Printing the following attributes of each token. 
#     # text: the word string, has_vector: if it contains 
#     # a vector representation in the model,  
#     # vector_norm: the algebraic norm of the vector, 
#     # is_oov: if the word is out of vocabulary. 
#     print(token.text, token.has_vector, token.vector_norm, token.is_oov) 
  
# token1, token2 = tokens[0], tokens[1] 
  
# print("Similarity:", token1.similarity(token2)) 


# word_embeddings_path = './data/word_emb.json'
# EXAMPLE_INDICES = range(100) #This is the word 'performance'
# index = 753
# n_to_compare
example_gradient = torch.load('grad_example.pt')
print(example_gradient)

# with open(word_embeddings_path, 'r') as fh:
#     embeddings = torch.Tensor(json.load(fh))

    

    # start = time.time()
    # target_embedding = torch.unsqueeze(embeddings[index], -1)
    # unnormalized_similarities = torch.mm(embeddings, target_embedding).squeeze()
    # norm_constants = torch.norm(embeddings, dim=1)
    # similarities = torch.div(unnormalized_similarities, norm_constants)
    # max_indices = np.argpartition(similarities, -10)[-10:]
    # print(max_indices)
    # end = time.time()
    # print(end - start)
    # 

    # for index in EXAMPLE_INDICES:
    #     

    



print(embeddings.shape)