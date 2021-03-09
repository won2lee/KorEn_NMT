import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from tqdm import tqdm

def get_nn_avg_dist(emb, query, knn):
    """
    Compute the average distance of the `knn` nearest neighbors
    for a given set of embeddings and queries.
    Use Faiss if available.
    """
    FAISS_AVAILABLE = False
    
    if FAISS_AVAILABLE:
        emb = emb.cpu().numpy()
        query = query.cpu().numpy()
        if hasattr(faiss, 'StandardGpuResources'):
            # gpu mode
            res = faiss.StandardGpuResources()
            config = faiss.GpuIndexFlatConfig()
            config.device = 0
            index = faiss.GpuIndexFlatIP(res, emb.shape[1], config)
        else:
            # cpu mode
            index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        distances, _ = index.search(query, knn)
        return distances.mean(1)
    else:
        bs = 512 #1024
        all_distances = []
        emb = emb.transpose(0, 1).contiguous()
        for i in range(0, query.shape[0], bs):
            distances = query[i:i + bs].mm(emb)
            best_distances, _ = distances.topk(knn, dim=1, largest=True, sorted=True)
            all_distances.append(best_distances.mean(1)) #.cpu()
        all_distances = torch.cat(all_distances)
        return all_distances


def get_candidates(emb1, emb2, params, device):
    """
    Get best translation pairs candidates.
    """

    bs = 128

    all_scores = []
    all_targets = []

    # number of source words to consider
    n_src = emb1.size(0)
    if params['dico_max_rank'] > 0 and not params['dico_method'].startswith('invsm_beta_'):
        n_src = min(params['dico_max_rank'], n_src)


    # contextual dissimilarity measure
    if params['dico_method'].startswith('csls_knn_'):

        knn = params['dico_method'][len('csls_knn_'):]
        assert knn.isdigit()
        knn = int(knn)

        # average distances to k nearest neighbors
        average_dist1 = get_nn_avg_dist(emb2, emb1, knn)
        average_dist2 = get_nn_avg_dist(emb1, emb2, knn)
        #average_dist1 = torch.from_numpy(get_nn_avg_dist(emb2, emb1, knn))
        #average_dist2 = torch.from_numpy(get_nn_avg_dist(emb1, emb2, knn))
        #print('check_point_1')
        #average_dist1 = average_dist1.type_as(emb1)
        #print('check_point_2')
        #average_dist2 = average_dist2.type_as(emb2)

        # for every source word
        #for i in tqdm(range(0, n_src, bs)):
        for i in range(0, n_src, bs):

            # compute target words scores
            scores = emb2.mm(emb1[i:min(n_src, i + bs)].transpose(0, 1)).transpose(0, 1)
            scores.mul_(2)
            scores.sub_(average_dist1[i:min(n_src, i + bs)].unsqueeze(1).expand_as(scores) 
                        + average_dist2.unsqueeze(0).expand_as(scores))
            best_scores, best_targets = scores.topk(2, dim=1, largest=True, sorted=True)

            # update scores / potential targets
            all_scores.append(best_scores) #.cpu()
            all_targets.append(best_targets) #.cpu()

        all_scores = torch.cat(all_scores, 0)
        all_targets = torch.cat(all_targets, 0)

    all_pairs = torch.cat([
        torch.arange(0, all_targets.size(0)).long().unsqueeze(1).to(device),
        all_targets[:, 0].unsqueeze(1)
    ], 1)

    # sanity check
    assert all_scores.size() == all_pairs.size() == (n_src, 2)

    # sort pairs by score confidence
    diff = all_scores[:, 0] - all_scores[:, 1]
    #reordered = diff.sort(0, descending=True)[1]
    reordered = (all_scores[:, 0]+diff).sort(0, descending=True)[1]
    all_scores = all_scores[:, 0][reordered]
    all_pairs = all_pairs[reordered]
    """
    # max dico words rank
    if params['dico_max_rank'] > 0:
        selected = all_pairs.max(1)[0] <= params['dico_max_rank']
        mask = selected.unsqueeze(1).expand_as(all_scores).clone()
        all_scores = all_scores.masked_select(mask).view(-1, 2)
        all_pairs = all_pairs.masked_select(mask).view(-1, 2)

    # max dico size
    if params['dico_max_size'] > 0:
        all_scores = all_scores[:params['dico_max_size']]
        all_pairs = all_pairs[:params['dico_max_size']]
    
    # min dico size
    diff = all_scores[:, 0] - all_scores[:, 1]
    if params['dico_min_size'] > 0:
        diff[:params['dico_min_size']] = 1e9

    # confidence threshold
    if params['dico_threshold'] > 0:
        mask = diff > params['dico_threshold']
        logger.info("Selected %i / %i pairs above the confidence threshold." % (mask.sum(), diff.size(0)))
        mask = mask.unsqueeze(1).expand_as(all_pairs).clone()
        all_pairs = all_pairs.masked_select(mask).view(-1, 2)
    """
    return all_pairs, all_scores


def build_dictionary(src_emb, tgt_emb, params, s2t_candidates=None, t2s_candidates=None):
    """
    Build a training dictionary given current embeddings / mapping.
    """
    #logger.info("Building the train dictionary ...")
    s2t = 'S2T' in params['dico_build']
    t2s = 'T2S' in params['dico_build']
    assert s2t or t2s

    if s2t:
        if s2t_candidates is None:
            s2t_candidates = get_candidates(src_emb, tgt_emb, params)
    if t2s:
        if t2s_candidates is None:
            t2s_candidates = get_candidates(tgt_emb, src_emb, params)
        t2s_candidates = torch.cat([t2s_candidates[:, 1:], t2s_candidates[:, :1]], 1)

    if params['dico_build'] == 'S2T':
        dico = s2t_candidates
    elif params['dico_build'] == 'T2S':
        dico = t2s_candidates
    else:
        s2t_candidates = set([(a, b) for a, b in s2t_candidates.numpy()])
        t2s_candidates = set([(a, b) for a, b in t2s_candidates.numpy()])
        if params['dico_build'] == 'S2T|T2S':
            final_pairs = s2t_candidates | t2s_candidates
        else:
            assert params['dico_build'] == 'S2T&T2S'
            final_pairs = s2t_candidates & t2s_candidates
            if len(final_pairs) == 0:
                #logger.warning("Empty intersection ...")
                return None
        dico = torch.LongTensor(list([[int(a), int(b)] for (a, b) in final_pairs]))

    #logger.info('New train dictionary of %i pairs.' % dico.size(0))
    return dico   #.cuda() if params['cuda else dico']

def dict_merge(d1,d2):
    for k,v in d1.items():
        if k in d2.keys():
            d2[k] += v
        else:
            d2[k] = v

        
def save_best(self, to_log, metric):
    """
    Save the best model for the given validation metric.
    """
    # best mapping for the given validation criterion
    if to_log[metric] > self.best_valid_metric:
        # new best mapping
        self.best_valid_metric = to_log[metric]
        logger.info('* Best value for "%s": %.5f' % (metric, to_log[metric]))
        # save the mapping
        W = self.mapping.weight.data.cpu().numpy()
        path = os.path.join(self.params['exp_path'], 'best_mapping.pth')
        logger.info('* Saving the mapping to %s ...' % path)
        torch.save(W, path)
