import torch
import torch.nn.functional as F

def sample_p_base_x_t_cond_x_t_prev_x_t_next(
    sde,
    x_t_prev,
    x_t_next,
    t,
    t_prev,
    t_next,
):
    assert len(x_t_prev.shape) == 4
    assert x_t_prev.shape == x_t_next.shape
    # calc mu
    mu_prior = sde.phi(start=t_prev, end=t).reshape(-1, 1, 1, 1) * x_t_prev
    mu_gain = sde.C(start=t_prev, t_a=t, t_b=t_next) / sde.C(start=t_prev, t_a=t_next, t_b=t_next)
    mu_innovation = x_t_next - sde.phi(start=t_prev, end=t_next).reshape(-1, 1, 1, 1) * x_t_prev
    mu = mu_prior + mu_gain.reshape(-1, 1, 1, 1) * mu_innovation

    # calc sigma
    sigma_sq_first_term = sde.C(start=t_prev, t_a=t, t_b=t)
    sigma_sq_second_term = sde.C(start=t_prev, t_a=t, t_b=t_next)**2 / sde.C(start=t_prev, t_a=t_next, t_b=t_next)
    sigma_sq = (sigma_sq_first_term - sigma_sq_second_term).reshape(-1, 1, 1, 1)

    # sample
    return mu + torch.sqrt(sigma_sq) * torch.randn_like(x_t_prev)

def grad_wrt_x_t_log_p_base_x_t_next_cond_x_t(
    sde,
    x_t,
    t,
    x_t_next,
    t_next,
):
    assert len(x_t.shape) == 4
    assert x_t.shape == x_t_next.shape

    shrink = sde.phi(start=t, end=t_next)
    first_term = shrink / sde.C(start=t, t_a=t_next, t_b=t_next)
    second_term = x_t_next - shrink.reshape(-1, 1, 1, 1) * x_t
    return first_term.reshape(-1, 1, 1, 1) * second_term

def dsm_loss(
    model,
    sde, 
    # x_0, # (N, C, H, W)
    x_t, # (N, C, H, W)
    # x_t_prev, # (N, C, H, W)
    x_t_next, # (N, C, H, W)
    x_t_history, # (N, max_cond_images, C, H, W)
    t, # (N,)
    # t_prev, # (N,)
    t_next, # (N,)
    t_history, # (N, max_cond_images)
    cond_masks, # (N, max_cond_images)
    y, # (N,) class labels
    logvar_net=None, # log variance predictor
    t_is_physical=True, # whether t is the physical time or an auxiliary time variable
):
    target = grad_wrt_x_t_log_p_base_x_t_next_cond_x_t(
        sde=sde,
        x_t=x_t,
        t=t,
        x_t_next=x_t_next,
        t_next=t_next if t_is_physical else torch.ones_like(t_next),
    )
    pred = model(
        x=x_t,
        t=t,
        t_next=t_next,
        y=y, 
        cond_images=x_t_history,
        cond_times=t_history,
        cond_masks=cond_masks,
    )
    sim_t_next = t_next if t_is_physical else torch.ones_like(t_next)
    weighting = sde.C(start=t, t_a=sim_t_next, t_b=sim_t_next) / sde.phi(start=t, end=sim_t_next)
    assert t.dtype == sim_t_next.dtype == weighting.dtype == torch.float32
    loss = (pred - target) ** 2 * weighting.reshape(-1, 1, 1, 1)

    if logvar_net is not None:
        logvar = logvar_net(t=t, t_next=t_next, cond_masks=cond_masks)
        assert logvar.shape == (loss.shape[0], 1)
        logvar = logvar.reshape(-1, 1, 1, 1)
        assert len(logvar.shape) == len(loss.shape)
        loss = torch.exp(-logvar) * loss + logvar

    return loss.mean()

# TODO: now need to figure out how to create these inputs