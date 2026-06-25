import torch
import torch.nn.functional as F


class GATAggregator(torch.nn.Module):
    '''
    Aggregator class
    Mode in ['sum', 'concat', 'neighbor']
    '''
    
    def __init__(self, batch_size, dim, aggregator):
        super(GATAggregator, self).__init__()
        self.batch_size = batch_size
        self.dim = dim
        self.softmax = torch.nn.Softmax(dim=-1)
        self.att_weights = torch.nn.Linear(2 * dim, 1)

        if aggregator == 'concat':
            self.weights = torch.nn.Linear(2 * dim, dim, bias=True)
            self.weights2 = torch.nn.Linear(dim, dim, bias=True)
        else:
            self.weights = torch.nn.Linear(dim, dim, bias=True)
        self.aggregator = aggregator
        
    def forward(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, act):
        batch_size = user_embeddings.size(0)
        if batch_size != self.batch_size:
            self.batch_size = batch_size
        neighbors_agg = self._mix_neighbor_vectors(self_vectors, neighbor_vectors, neighbor_relations, user_embeddings)
        
        if self.aggregator == 'sum':
            output = (self_vectors + neighbors_agg).view((-1, self.dim))
            
        elif self.aggregator == 'concat':
            output = torch.cat((self_vectors, neighbors_agg), dim=-1)
            output = output.view((-1, 2 * self.dim))
            
        else:
            output = neighbors_agg.view((-1, self.dim))
            
        output = self.weights(output)
        return act(output.view((self.batch_size, -1, self.dim)))
        
    def _mix_neighbor_vectors(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings):
        '''
        This aims to aggregate neighbor vectors
        '''
        # [batch_size, 1, dim] -> [batch_size, 1, 1, dim]
        user_embeddings = user_embeddings.view((self.batch_size, 1, 1, self.dim))
        
        # [batch_size, -1, n_neighbor, dim] -> [batch_size, -1, n_neighbor]
        user_relation_scores = (user_embeddings * neighbor_relations).sum(dim = -1)
        user_relation_scores_normalized = F.softmax(user_relation_scores, dim = -1)
        
        attentions = torch.empty(neighbor_vectors.size(dim=0), 1, neighbor_vectors.size(dim=2), 2 * self.dim).to(device='cuda:0')
        # [batch_size,() -1, n_neighbor] -> [batch_size, -1, n_neighbor, 1]
        user_relation_scores_normalized = user_relation_scores_normalized.unsqueeze(dim = -1)
        
        if self.aggregator == 'concat':
            neighbor_vectors2, self_vectors2 = self.weights2(neighbor_vectors), self.weights2(self_vectors)
        else:    
            neighbor_vectors2, self_vectors2 = self.weights(neighbor_vectors), self.weights(self_vectors)

        attentions = torch.cat((neighbor_vectors2, self_vectors2.unsqueeze(dim=1).expand(neighbor_vectors2.size())), dim=-1)
        att_neighbor_vectors = F.softmax(self.att_weights(attentions), dim=2) * neighbor_vectors
        
        # [batch_size, -1, n_neighbor, 1] * [batch_size, -1, n_neighbor, dim] -> [batch_size, -1, dim]
        neighbors_aggregated = (user_relation_scores_normalized * att_neighbor_vectors).sum(dim = 2)
        
        return neighbors_aggregated