import torch
import torch.nn.functional as F
import heapq

class DecodingStrategy:
    def __init__(self, logits, vocab, num_predictions, max_length):
        self.logits = logits 
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.num_predictions = num_predictions
        self.max_length = max_length

        assert logits.shape == (num_predictions, self.vocab_size, max_length), \
            f"Expected logits shape ({num_predictions}, {self.vocab_size}, {max_length}), got {logits.shape}"

    def decode(self, strategy_type='greedy', **kwargs):
        if strategy_type == 'greedy':
            return self._greedy_search()
        elif strategy_type == 'beam':
            return self._beam_search(**kwargs)
        elif strategy_type == 'sample':
            return self._sample(**kwargs)
        else:
            raise ValueError(f"Unsupported strategy_type: {strategy_type}")

    def _greedy_search(self):
        decoded = []
        for b in range(self.num_predictions):
            sequence = []
            for t in range(self.max_length):
                token_id = torch.argmax(self.logits[b, :, t]).item()
                token = self.vocab[token_id]  # Convert index to token
                sequence.append(token)
            decoded.append(sequence)
        return decoded

    def _beam_search(self, beam_width=3):
        decoded = []

        for b in range(self.num_predictions):
            beams = [(0.0, [])]
            for t in range(self.max_length):
                next_beams = []

                for log_prob, seq in beams:
                    logit = self.logits[b, :, t]
                    log_probs = F.log_softmax(logit, dim=-1)

                    k = min(beam_width, self.vocab_size)
                    topk_vals, topk_idx = torch.topk(log_probs, k)

                    for i in range(k):
                        token_id = topk_idx[i].item()
                        token = self.vocab[token_id]  # Convert index to token
                        score = topk_vals[i].item()
                        next_beams.append((log_prob + score, seq + [token]))

                beams = heapq.nlargest(beam_width, next_beams, key=lambda x: x[0])

            best_seq = max(beams, key=lambda x: x[0])[1]
            decoded.append(best_seq)

        return decoded

    def _sample(self, temperature=1.0, top_k=None, top_p=None):
        decoded = []

        for b in range(self.num_predictions):
            sequence = []
            for t in range(self.max_length):
                logits_t = self.logits[b, :, t] / temperature

                if top_k is not None:
                    logits_t = self._top_k_filter(logits_t, top_k)

                if top_p is not None:
                    logits_t = self._top_p_filter(logits_t, top_p)

                probs = F.softmax(logits_t, dim=-1)

                if probs.sum() == 0 or torch.isnan(probs).any():
                    probs = torch.ones_like(probs) / probs.size(-1)

                token_id = torch.multinomial(probs, 1).item()
                token = self.vocab[token_id]
                sequence.append(token)

            decoded.append(sequence)
        return decoded

    def _top_k_filter(self, logits, k):
        k = min(k, logits.size(-1))
        if k <= 0:
            return torch.full_like(logits, float('-inf'))

        topk_vals, topk_idx = torch.topk(logits, k)
        mask = torch.full_like(logits, float('-inf'))
        mask.scatter_(0, topk_idx, topk_vals)
        return mask

    def _top_p_filter(self, logits, p):
        if p <= 0.0:
            top_val, top_idx = torch.max(logits, dim=-1)
            mask = torch.full_like(logits, float('-inf'))
            mask[top_idx] = top_val
            return mask

        if p >= 1.0:
            return logits

        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cum_probs = torch.cumsum(probs, dim=-1)

        cutoff = (cum_probs >= p).nonzero(as_tuple=True)[0]
        cutoff_idx = cutoff[0] + 1 if len(cutoff) > 0 else len(logits)

        mask = torch.full_like(logits, float('-inf'))
        keep_idx = sorted_idx[:cutoff_idx]
        mask[keep_idx] = logits[keep_idx]
        return mask

