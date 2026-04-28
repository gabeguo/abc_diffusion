import torch
import math

class SDE:
    def __init__(self, A, score_network):
        self.A = A
        self.score_network = score_network
        return
        
    def sigma(self, t):
        pass

    def phi(self, start, end):
        pass

    def C(self, start, t_a, t_b):
        pass

    def dX_t(
        self, 
        x_t, 
        t, 
        t_next, 
        x_t_history,
        t_history,
        cond_masks,
        y,
        dt, 
        return_all=False,
    ):
        dB_Q = torch.sqrt(torch.tensor(dt, device=x_t.device)) * torch.randn_like(x_t)
        score = self.score_network(
            x=x_t, 
            t=t, 
            t_next=t_next, 
            y=y,
            cond_images=x_t_history,
            cond_times=t_history,
            cond_masks=cond_masks,
        )
        dB_P = dB_Q + self.sigma(t).reshape(-1, 1, 1, 1) * score * dt
        dX_t = -self.A * x_t * dt + self.sigma(t).reshape(-1, 1, 1, 1) * dB_P
        if return_all:
            return dX_t, score, dB_Q
        return dX_t


class DecayingVolatilitySDE(SDE):
    def __init__(self, A, B, K, score_network):
        super().__init__(A=A, score_network=score_network)
        self.B = B
        self.K = K
        return

    def sigma(self, t):
        return self.K * torch.exp(-self.B * t)
        
    def phi(self, start, end):
        return torch.exp(-self.A * (end - start))

    def C(self, start, t_a, t_b):
        first_numerator = self.K ** 2 * torch.exp(-self.A * (t_a + t_b))
        first_denominator = 2 * (self.A - self.B)
        first_term = first_numerator / first_denominator
        second_term = torch.exp(2 * (self.A - self.B) * torch.minimum(t_a, t_b)) - torch.exp(2 * (self.A - self.B) * start)
        return first_term * second_term
    
class PeriodicVolatilitySDE(SDE):
    def __init__(self, alpha, k, eps, score_network):
        super().__init__(A=0, score_network=score_network)
        self.alpha = alpha
        self.k = k
        self.eps = eps
        return

    def sigma(self, t):
        return self.alpha / 2 * (1 - torch.cos(2 * math.pi * self.k * t)) + self.eps
    
    def phi(self, start, end):
        return torch.ones_like(start)

    def C(self, start, t_a, t_b):
        def integrand(s):
            first_term = (3 * self.alpha ** 2 / 8 + self.alpha * self.eps + self.eps ** 2) * s
            second_term = self.alpha * (self.alpha + 2 * self.eps) / (4 * math.pi * self.k) * torch.sin(2 * math.pi * self.k * s)
            third_term = self.alpha ** 2 / (32 * math.pi * self.k) * torch.sin(4 * math.pi * self.k * s)
            return first_term - second_term + third_term
        
        upper = torch.minimum(t_a, t_b)
        lower = start
        return integrand(upper) - integrand(lower)

class CosineDecayingVolatilitySDE(PeriodicVolatilitySDE):
    def __init__(self, alpha, eps, score_network):
        super().__init__(alpha=alpha, k=0.5, eps=eps, score_network=score_network)
        return
    def sigma(self, t):
        return super().sigma(t - 1)
    def C(self, start, t_a, t_b):
        return super().C(start=start-1, t_a=t_a-1, t_b=t_b-1)


class UniformVolatilitySDE(SDE):
    def __init__(self, A, K, score_network):
        super().__init__(A=A, score_network=score_network)
        self.K = K
        return

    def sigma(self, t):
        return torch.full_like(t, self.K)

    def phi(self, start, end):
        return torch.exp(-self.A * (end - start))

    def C(self, start, t_a, t_b):
        upper = torch.minimum(t_a, t_b)
        if self.A == 0:
            return (self.K ** 2) * (upper - start)

        numerator = (self.K ** 2) * torch.exp(-self.A * (t_a + t_b))
        denominator = 2 * self.A
        window = torch.exp(2 * self.A * upper) - torch.exp(2 * self.A * start)
        return numerator * window / denominator
