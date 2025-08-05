import torch
import torch.nn.functional as F

from models.ppc.model_search import Network
from config import PPCSearchConfig

class Batch:
  def __init__(self, input_train: torch.Tensor, target_train: torch.Tensor, input_search: torch.Tensor, target_search: torch.Tensor):
    self.input_train = input_train
    self.target_train = target_train
    self.input_search = input_search
    self.target_search = target_search
  
  def train(self) -> tuple[torch.Tensor, torch.Tensor]:
    return self.input_train, self.target_train
  
  def search(self) -> tuple[torch.Tensor, torch.Tensor]:
    return self.input_search, self.target_search

class FuncBo(object):
  """
    This class handles the training of the network during search stage.
    """

  def __init__(self, model: Network, args: PPCSearchConfig):
    self.model = model
    self.dual_model = self.model.new()
    self.dual_model.alphas_normal = self.model.alphas_normal
    self.dual_model.alphas_reduce = self.model.alphas_reduce
    self.alpha_optimizer = torch.optim.Adam(self.model.arch_parameters(),
                                        lr=args.arch_learning_rate,
                                        betas=(0.5, 0.999),
                                        weight_decay=args.arch_weight_decay)
    self.inner_optimizater = torch.optim.Adam(self.model.parameters(),
                                args.learning_rate,
                                weight_decay=args.weight_decay)
    self.dual_optimizater = torch.optim.Adam(self.dual_model.parameters(),
                                args.learning_rate,
                                weight_decay=args.weight_decay)


  def step(
      self,
      input_train: torch.Tensor,
      target_train: torch.Tensor,
      input_search: torch.Tensor,
      target_search: torch.Tensor,
  ):
    """
        If unrolled, make gradient step on alpha.
        Otherwise, make gradient step on ws.
        """
    batch = Batch(input_train, target_train, input_search, target_search)
    self.dual_model.train()
    self.inner_optimization(batch)
    self.dual_optimization(batch)
    self.total_optimization(batch)
  
  def inner_optimization(self, batch: Batch):
    """ Update model parameters based on input_train"""
    input, target  = batch.train()
    self.inner_optimizater.zero_grad()
    logits = self.model(input)
    loss = F.cross_entropy(logits, target)
    loss.backward()
    self.inner_optimizater.step()
  
  def dual_optimization(self, batch: Batch):
    """ Compute dual loss based on input_train, input_search and self.model"""
    self.dual_optimizater.zero_grad()  
    loss = self.dual_loss(batch)
    loss.backward()
    self.dual_optimizater.step()

  def dual_loss(self, batch: Batch):
    x_in, y_in = batch.train()
    v_in = self.model(x_in)
    a_in = self.dual_model(x_in)            # vector  a  (m_in, d_v)  <-- *NO* detach

    # ∂_v l  for all samples in one call
    loss_in_vec = F.cross_entropy(v_in, y_in, reduction='none')    # (m_in,)
    grad_v_in = torch.autograd.grad(loss_in_vec.sum(),
                                    v_in,
                                    create_graph=True)[0]         # (m_in,d_v)

    # Hessian-vector product   H a   (still batched)
    dot_in = (grad_v_in * a_in).sum(dim=-1)                        # (m_in,)
    # H a  : derivative of dot w.r.t. v
    Hv_in = torch.autograd.grad(dot_in.sum(), v_in,
                                create_graph=True)[0]              # (m_in,d_v)

    quad_in = (Hv_in * a_in).sum(dim=-1)                           # aᵀHa  (m_in,)
    term_H  = 0.5 * quad_in.mean()                                 # (½)(1/|B_in|) Σ

    # --------------------- 2. OUT-BATCH (gradient term) ------------------
    x_out, y_out = batch.search()
    v_out = self.model(x_out)                                         # (m_out,d_v)
    a_out = self.dual_model(x_out)                                         # (m_out,d_v)

    loss_out_vec = F.cross_entropy(v_out, y_out, reduction='none') # (m_out,)
    grad_v_out = torch.autograd.grad(loss_out_vec.sum(),
                                     v_out,
                                     create_graph=True)[0]         # (m_out,d_v)

    dot_out = (grad_v_out * a_out).sum(dim=-1)                     # aᵀ∇_v l
    term_G  = dot_out.mean()                                       # (1/|B_out|) Σ

    # --------------------- 3. FINAL LOSS -------------------------------
    return term_H + term_G
  
  def total_optimization(self, batch: Batch):
    _, _, g_total = self.total_grad(batch)
    self.alpha_optimizer.zero_grad()
    with torch.no_grad():
      self.model.alphas_normal.grad = g_total[0].clone()
      self.model.alphas_reduce.grad = g_total[1].clone()
    self.alpha_optimizer.step()
  
  def grad_alphas(self, loss: torch.Tensor):
    grad_normal = torch.autograd.grad(loss, self.model.alphas_normal, retain_graph=True)
    grad_reduce = torch.autograd.grad(loss, self.model.alphas_reduce, retain_graph=True)
    return (grad_normal, grad_reduce)
  
  def total_grad(self, batch: Batch):
    """
    B_out, B_in are batches (x, y) where
      x : (batch, d_in)  and  y : (batch,)
    Returns:   g_exp, g_imp, g_total   (lists of tensors, one per w-param)
    """
    input_train, target_train = batch.train()
    input_search, target_search = batch.search()
    # ------------------ g_exp : ∂_w L ------------------
    alpha_params = torch.cat([self.model.alphas_normal.view(-1), self.model.alphas_reduce.view(-1)])
    logits_out = self.model(input_search)
    loss_out   = F.cross_entropy(logits_out, target_search)
    g_exp = self.grad_alphas(loss_out) #torch.autograd.grad(loss_out, alpha_params, retain_graph=True)

    # ------------------ g_imp : Hessian-vector ------------ 
    # running sum of per-sample contributions
    v = self.model(input_train)             # logits from  hat h_w
    a = self.dual_model(input_train).detach()    # vector from  hat a_w  (stop grad!)

    # ----- ∂_v l  for every sample, but in one call -----------------------
    loss_vec = F.cross_entropy(v, target_train, reduction='none')      # (m,)
    grad_v   = torch.autograd.grad(loss_vec.sum(),             # scalar
                                   v,
                                   create_graph=True)[0]       # (m, d_v)

    # ----- dot product ⟨∂_v l , a⟩  per sample ----------------------------
    dot_per_sample = (grad_v * a).sum(dim=-1)                  # (m,)
    bar_s = dot_per_sample.mean()                              # scalar

    # ----- single backward gives (1/m) Σ_i H_i a_i ------------------------
    g_imp = self.grad_alphas(bar_s) # torch.autograd.grad(bar_s, alpha_params)
    
    # ------------------ total gradient ------------------
    assert len(g_imp[0]) == 1
    assert len(g_imp[1]) == 1
    assert len(g_exp[0]) == 1
    assert len(g_exp[1]) == 1
    g_total = [g_e + g_i for ga_e, ga_i in zip(g_exp, g_imp) for g_e, g_i in zip(ga_e, ga_i)]
    return g_exp, g_imp, g_total 