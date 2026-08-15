# Paper-to-Code Map

| Paper element | Equation/setting | Implementation |
| --- | --- | --- |
| Multimodal representation | Eq. 1 | `ft_ncfm.models.MiniVLAPolicy.encode` and future dataset adapters |
| Base influence | Eq. 2 | `ft_ncfm.influence.InfluenceEngine.score_samples` |
| LiSSA IHVP | depth 50, damping 0.01, batch 32 | `ft_ncfm.influence.lissa_ihvp` |
| Original/counterexample gradient alignment | Eqs. 3-4 | `ft_ncfm.contrastive.gradient_dot_score` |
| Contrastive refinement | Eq. 5, beta 1.0 | `ft_ncfm.contrastive.refine_influence_weights` |
| Uniform NCFM | Eq. 6 | upstream `NCFM.NCFM.CFLossFunc` |
| Influence-weighted NCFM | Eq. 7 | weighted path in `NCFM.NCFM.CFLossFunc` |
| Influence-aware sampling | Algorithm 1 line 24 | `ft_ncfm.sampling.InfluenceWeightedSampler` |
| Implicit generator | learnable synthetic tensors | `ft_ncfm.distillation.WeightedNCFMDistiller` |
| Auxiliary network | three-layer frequency MLP | upstream `NCFM.SampleNet.SampleNet` and the small proxy equivalent |

The production integration intentionally adds arguments rather than replacing the
uniform NCFM behavior. This makes uniform-vs-weighted ablations use the same code.
