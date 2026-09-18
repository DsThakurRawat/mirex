"""OC-Softmax semantics and SAM optimizer mechanics."""
import torch

from losses import OCSoftmax, SAM


def test_ocsoftmax_margin_semantics():
    oc = OCSoftmax(emb_dim=8, m_real=0.9, m_fake=0.2, alpha=20.0)
    with torch.no_grad():
        center = torch.nn.functional.normalize(oc.center, dim=0)
        aligned = center[None].repeat(4, 1) * 5.0           # on-center reals
        opposite = -aligned                                  # far-away fakes
    loss_good, score_good = oc(aligned, torch.zeros(4))
    loss_bad_real, _ = oc(opposite, torch.zeros(4))
    assert loss_good < loss_bad_real          # reals near center = low loss
    loss_good_fake, score_fake = oc(opposite, torch.ones(4))
    assert loss_good_fake < 1e-3              # fakes far away = ~no loss
    # Scores: higher = more anomalous/fake.
    assert score_fake.mean() > score_good.mean()


def test_ocsoftmax_trains_toy_separation():
    """Few gradient steps on separable toy embeddings must push fake scores
    above real scores."""
    torch.manual_seed(0)
    enc = torch.nn.Linear(16, 8)
    oc = OCSoftmax(emb_dim=8)
    opt = torch.optim.Adam(list(enc.parameters()) + list(oc.parameters()),
                           lr=5e-2)
    real_x = torch.randn(64, 16) + torch.tensor([2.0] * 8 + [0.0] * 8)
    fake_x = torch.randn(64, 16) - torch.tensor([2.0] * 8 + [0.0] * 8)
    x = torch.cat([real_x, fake_x])
    y = torch.cat([torch.zeros(64), torch.ones(64)])
    for _ in range(200):
        opt.zero_grad()
        loss, _ = oc(enc(x), y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        _, scores = oc(enc(x), y)
    real_s, fake_s = scores[:64], scores[64:]
    # Full separation on this toy problem.
    assert fake_s.min() > real_s.max(), \
        f"no separation: fake_min={fake_s.min():.3f} real_max={real_s.max():.3f}"


def test_sam_two_step_updates_params():
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 1)
    opt = SAM(model.parameters(), torch.optim.SGD, rho=0.05, lr=0.1)
    x, y = torch.randn(8, 4), torch.randn(8, 1)
    before = [p.clone() for p in model.parameters()]

    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    opt.first_step()
    perturbed = [p.clone() for p in model.parameters()]
    assert any(not torch.allclose(a, b)
               for a, b in zip(before, perturbed))       # ascent perturbation
    torch.nn.functional.mse_loss(model(x), y).backward()
    opt.second_step()
    after = [p.clone() for p in model.parameters()]
    assert any(not torch.allclose(a, b) for a, b in zip(before, after))
    # e_w must be consumed (no stale perturbation left behind).
    assert all("e_w" not in opt.state[p] for group in opt.param_groups
               for p in group["params"])
