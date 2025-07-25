import torch
import torch.nn as nn
import torch.nn.functional as F

class GradientCollector:
    def __init__(self):
        self.grads = []
        self.hooks = []
    
    def collect_grad(self, grad):
        self.grads.append(grad.detach())
    
    def register_hooks(self, parameters):
        self.grads = []
        self.hooks = []
        for p in parameters:
            if p.requires_grad:
                hook = p.register_hook(self.collect_grad)
                self.hooks.append(hook)
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_grad_norm(self):
        if not self.grads:
            return torch.tensor(0.0)
        grads = torch.stack([g.norm() for g in self.grads])
        return grads.mean()

class MetaKDOptimizer(nn.Module):
    def __init__(self, num_losses=4, hidden_dim=64, lr=1e-4, activation='sigmoid'):
        """
        Meta Knowledge Distillation Optimizer
        Args:
            num_losses: Number of KD losses to optimize (default: 4 for kd_feat, kd_pix, kd_pix_local, kd_feat_intermediate)
            hidden_dim: Hidden dimension for weight prediction network
            lr: Learning rate for meta optimizer
            activation: 'sigmoid' or 'softmax' for output activation
        """
        super().__init__()
        self.num_losses = num_losses
        self.activation = activation
        
        # Input 12 dimensi: current, ema, relative improvement untuk 4 loss
        self.layer1 = nn.Linear(12, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, num_losses)
        
        # Meta optimizer
        self.meta_optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        # Moving averages for loss statistics
        self.register_buffer('loss_mean', torch.zeros(num_losses))
        self.register_buffer('loss_std', torch.ones(num_losses))
        self.momentum = 0.9
        
    def forward(self, loss_info):
        """Forward pass with explicit tensor operations"""
        # Ensure input is detached and on correct device
        x = loss_info.detach()
        
        # Layer 1
        x1 = self.layer1(x)
        x1 = F.relu(x1)
        
        # Layer 2
        x2 = self.layer2(x1)
        x2 = F.relu(x2)
        
        # Layer 3
        x3 = self.layer3(x2)
        
        # Pilih aktivasi output sesuai self.activation
        if self.activation == 'softmax':
            out = F.softmax(x3, dim=0)
        else:
            out = torch.sigmoid(x3)
        
        return out
    
    def update_statistics(self, losses):
        """Update moving averages of loss statistics"""
        with torch.no_grad():
            # Convert losses to tensor without modifying computation graph
            loss_values = torch.stack([l.detach() for l in losses])
            
            # Update mean
            new_mean = (1 - self.momentum) * loss_values + self.momentum * self.loss_mean
            self.loss_mean = new_mean
            
            # Update std
            diff = loss_values - self.loss_mean
            var = torch.mean(diff * diff, dim=0)
            new_std = torch.sqrt(var + 1e-6)
            self.loss_std = (1 - self.momentum) * new_std + self.momentum * self.loss_std
    
    def normalize_losses(self, losses):
        """Normalize losses using moving statistics"""
        normalized = []
        for loss, mean, std in zip(losses, self.loss_mean, self.loss_std):
            # Create new tensor for normalized loss
            norm_loss = (loss - mean.detach()) / (std.detach() + 1e-6)
            normalized.append(norm_loss)
        return normalized
    
    def get_weights(self, losses, grad_norms=None):
        """Get loss weights based on current losses and their gradients"""
        # Update statistics with current losses
        self.update_statistics(losses)
        
        # Normalize losses
        norm_losses = self.normalize_losses(losses)
        
        # Handle gradient norms
        if grad_norms is None:
            grad_norms = [torch.zeros(1, device=losses[0].device) for _ in losses]
        
        # Ensure all tensors are properly shaped
        loss_values = []
        grad_values = []
        
        for loss, grad in zip(norm_losses, grad_norms):
            # Convert loss to scalar if it's not already
            if loss.dim() > 0:
                loss = loss.mean()
            loss_values.append(loss)
            
            # Convert gradient norm to scalar if it's not already
            if grad.dim() > 0:
                grad = grad.mean()
            grad_values.append(grad)
        
        # Stack into tensors
        loss_tensor = torch.stack(loss_values)  # [num_losses]
        grad_tensor = torch.stack(grad_values)  # [num_losses]
        
        # Concatenate along feature dimension
        network_input = torch.cat([loss_tensor, grad_tensor])  # [num_losses * 2]
        
        # Get raw weights
        raw_weights = self(network_input)
        
        # Normalize weights to sum to 1
        weights_sum = raw_weights.sum() + 1e-6
        normalized_weights = raw_weights / weights_sum
        
        return normalized_weights
    
    def meta_step(self, student_model, teacher_model, batch, compute_losses_fn):
        """Perform meta optimization step"""
        # Create gradient collector
        collector = GradientCollector()
        
        # Get trainable parameters
        trainable_params = [p for p in student_model.parameters() if p.requires_grad]
        
        # Register hooks
        collector.register_hooks(trainable_params)
        
        try:
            # Compute losses
            losses = compute_losses_fn(student_model, teacher_model, batch)
            
            # Get gradient norms for each loss
            grad_norms = []
            for loss in losses:
                if loss.requires_grad:
                    # Clear previous gradients
                    student_model.zero_grad()
                    collector.grads = []
                    
                    # Backward pass
                    loss.backward(retain_graph=True)
                    
                    # Get gradient norm
                    grad_norm = collector.get_grad_norm()
                    grad_norms.append(grad_norm)
                else:
                    grad_norms.append(torch.tensor(0.0, device=loss.device))
            
            # Get weights based on current losses and gradients
            weights = self.get_weights([l.detach() for l in losses], grad_norms)
            
            # Compute meta loss
            meta_loss = -sum(w * l.detach() for w, l in zip(weights, losses)) + 0.1 * torch.sum(weights ** 2)
            
            # Update meta parameters
            self.meta_optimizer.zero_grad()
            meta_loss.backward()
            self.meta_optimizer.step()
            
            # Return normalized weights
            with torch.no_grad():
                weights_sum = weights.sum() + 1e-6
                normalized_weights = weights.detach() / weights_sum
            
            return normalized_weights
            
        finally:
            # Always remove hooks
            collector.remove_hooks()
            student_model.zero_grad() 