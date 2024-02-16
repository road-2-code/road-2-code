import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torchtune import KDRecipe

class ForwardKLLoss(torch.nn.Module):
  def __init__(self, ignore_index: int = -100):
    super().__init__()
    self.ignore_index = ignore_index

  def forward(self, student_logits, teacher_logits, labels) -> torch.Tensor:
    # Implementation from https://github.com/jongwooko/distillm
    # Computes the softmax of the teacher logits
    teacher_prob = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    # Computes the student log softmax probabilities
    student_logprob = F.log_softmax(student_logits, dim=-1, dtype=torch.float32)
    # Computes the forward KL divergence
    prod_probs = teacher_prob * student_logprob
    # Compute the sum
    x = torch.sum(prod_probs, dim=-1).view(-1)
    # We don't want to include the ignore labels in the average
    mask = (labels != self.ignore_index).int()
    # Loss is averaged over non-ignored targets
    return -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)