import torch
from torch.distributions.kl import kl_divergence
import numpy as np
from common.utils import set_flat_params_to

class OnTheFly(object):
    def __init__(self,
                 old_actor,
                 actor_critic,
                 omega,
                 max_kl,
                 damping):
        
        self.old_actor = old_actor
        self.actor_critic = actor_critic
        self.set_alias(actor_critic)

        self.omega = omega
        self.max_kl = max_kl
        self.damping = damping

    def set_alias(self, actor_critic):
        self.actor = actor_critic.actor
        self.critic = actor_critic.critic
        
    def linesearch(self,
                   params,
                   fullstep,
                   line_search_max_iter=10):

        actor_loss = self.get_loss(True).data

        for i, step_fraction in enumerate(.8**np.arange(line_search_max_iter)):
            direction = step_fraction * fullstep
            new_params = params + direction
            set_flat_params_to(self.actor, new_params)
            new_actor_loss = self.get_loss(True).data

            if self.get_kl(True) < self.max_kl and actor_loss > new_actor_loss:
                return True, direction

        return False, torch.zeros_like(params)

    def get_flat_grad(self, loss, **kwargs):
        grads = torch.autograd.grad(loss, self.actor.parameters(), **kwargs)
        return torch.cat([grad.view(-1) for grad in grads])
    
    def get_loss(self, volatile=False):
        if volatile:
            with torch.no_grad():
                _, log_probs, _ = self.actor_critic.evaluate_actions(self.states, self.actions)
        else:
            _, log_probs, _ = self.actor_critic.evaluate_actions(self.states, self.actions)

        ratio = torch.exp(log_probs - self.old_log_probs)
        action_loss = -(ratio * self.advantages).mean()
            
        return action_loss

    def get_kl(self, volatile=False):
        if volatile:
            with torch.no_grad():
                curr_dists = self.actor_critic.get_dist(self.states)
        else:
            curr_dists = self.actor_critic.get_dist(self.states)
            
        return kl_divergence(self.old_dists, curr_dists).mean()
    
    def product(self, x, kl_grad, retain_graph=False):
        kl_grad_dot_x = (kl_grad * x).sum()
        flat_grad_grad = self.get_flat_grad(kl_grad_dot_x, retain_graph=retain_graph)
        return flat_grad_grad + x * self.damping

    def conjugate_gradient_solver(self, b, kl_grad, steps, residual_tol=1e-10):
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        for i in range(steps):
            z = self.product(p, kl_grad, retain_graph=True)
            alpha = rdotr / p.dot(z)
            x += alpha * p
            r -= alpha * z
            new_rdotr = r.dot(r)
            if new_rdotr < residual_tol:
                break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
        return x

    def step(self, states, actions, advantages, old_log_probs):

        # store inputs
        self.states    = states
        self.actions   = actions
        self.advantages = advantages
        self.old_log_probs = old_log_probs

        # get old dists for multiple calls
        with torch.no_grad():
            self.old_dists = self.old_actor(states)
        
        action_loss = self.get_loss()

        # policy gradient
        g = self.get_flat_grad(action_loss)
        
        # KL gradient
        kl = self.get_kl()
        kl_grad = self.get_flat_grad(kl, create_graph=True)

        # search direction
        newton_direction = self.conjugate_gradient_solver(-g, kl_grad, 20)

        # quadratic form
        quadratic = (newton_direction * self.product(newton_direction, kl_grad)).sum(0, keepdim=True)
        assert quadratic > 0, "quadratic form must be positive"
        
        fullstep = torch.sqrt((2 * self.max_kl) / quadratic) * newton_direction
        
        params = torch.cat([param.data.view(-1) for param in self.actor.parameters()])
        success, direction = self.linesearch(params, fullstep)
        new_params = params + self.omega * direction
        
        if success:
            set_flat_params_to(self.actor, params)
        
        return new_params
