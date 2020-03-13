
import ujson as json
import numpy as np
import spacy 
import setup
from args import get_setup_args
from collections import Counter
import time
import torch

args = get_setup_args()


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
# word2idx_path = './data/word2idx.json'
# word2idx_file = open(word2idx_path, 'r')
# word2idx = json.load(word2idx_file)
# idx2word = {v: k for k, v in word2idx.items()}
# EXAMPLE_INDICES = range(100) #This is the word 'performance'
# index = 753
# n_to_compare

eps = 0.0000001
THRESHOLD = 0.6

word_embeddings_path = './data/word_emb.json'
fh = open(word_embeddings_path, 'r')
embeddings = torch.Tensor(json.load(fh)) # (V, 300)

start = time.time()
embeddings_norm_factor = torch.unsqueeze(torch.norm(embeddings, dim=1), 1) # (V)

example_gradient = torch.load('grad_example.pt')[0] # (V, 300)

# find nonzero row indices
sums = torch.sum(example_gradient, dim=1)
row_numbers = torch.nonzero(sums).squeeze()

# get corresponding embeddings, and calculate similarity matrix
embeddings_with_grad = embeddings[row_numbers].t()  # (300, k)
similarities = torch.mm(embeddings, embeddings_with_grad) # (V, k)

# normalize similarities
embeddings_with_grad_norm = torch.unsqueeze(torch.norm(embeddings_with_grad, dim=0), 0) # (1, k)
similarities = (similarities / (embeddings_norm_factor + eps)) / (embeddings_with_grad_norm + eps) # (V, k)

# remove same embeddings
for i, row in enumerate(row_numbers):
    similarities[row, i] = 0

# new collateral grad algorithm faster
new_grad = example_gradient.clone().detach()
thresholded_similarities = torch.where(similarities > THRESHOLD, similarities, torch.zeros(similarities.size()))
gradient_addition = torch.mm(thresholded_similarities, example_gradient[row_numbers])
new_grad += gradient_addition

# get one sample
# N_SAMPLE = 993
# print(idx2word[int(row_numbers[N_SAMPLE])])
# sims = similarities[:, N_SAMPLE]
# values, indices = torch.topk(sims, 100)
# words = [idx2word[int(x)] for x in indices]
# print(values)
# print(words)


# print(torch.max(similarities))
# print(similarities.size())

end = time.time()
print('Time', end - start)

fh.close()
# word2idx_file.close()

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

    



# print(embeddings.shape)