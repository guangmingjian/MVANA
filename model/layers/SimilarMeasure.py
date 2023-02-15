
import torch


class SimilarMeasure():
    """"""

    def __init__(self, agg="sum"):
        super(SimilarMeasure, self).__init__()
        """"""
        self.agg = agg

    def correlation_loss(self, z: torch.tensor):
        """"""
        # sz_c,d = z.size(0),z.size(-1)
        z_ = z.mean(dim=-1, keepdim=True)
        z_stds = z.std(dim=-1)
        all_t = None
        us = None
        for m in (z - z_):
            if all_t is None:
                all_t = m
            else:
                all_t = all_t * m
        for u in z_stds:
            if us is None:
                us = u
            else:
                us = us * u
        sim = all_t.sum(dim=-1) / (z.size(-1) - 1) / us
        # sim_dis = 1 - torch.abs(sim)
        # return sim
        if self.agg == 'sum':
            return torch.sum(sim ** 2)
        else:
            return torch.mean(sim ** 2)


if __name__ == '__main__':
    sm = SimilarMeasure()
    import torch
    _ = torch.manual_seed(0)
    z = torch.randn([2,2,4])
    print(sm.correlation_loss(z))
    from audtorch.metrics.functional import pearsonr
    print(pearsonr(z[0],z[1]))
