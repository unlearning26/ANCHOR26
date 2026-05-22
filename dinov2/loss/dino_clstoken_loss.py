# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


try:
    from xformers.ops import cross_entropy as _cross_entropy

    _XFORMERS_CROSS_ENTROPY_AVAILABLE = True

    def cross_entropy_distribution(student_logits, teacher_probs, temp, student_log_probs=None):
        student_logits = student_logits.float()
        teacher_probs = teacher_probs.float()
        if student_logits.ndim == 2:
            return _cross_entropy(student_logits.unsqueeze(0), teacher_probs.unsqueeze(0), temp, bw_inplace=True).squeeze(0)
        if student_logits.ndim == 3:
            return _cross_entropy(student_logits, teacher_probs, temp, bw_inplace=True)
        raise ValueError(f"Unsupported logits ndim: {student_logits.ndim}")

except ImportError:
    _XFORMERS_CROSS_ENTROPY_AVAILABLE = False

    def cross_entropy_distribution(student_logits, teacher_probs, temp, student_log_probs=None):
        if student_log_probs is None:
            student_log_probs = F.log_softmax(student_logits / temp, dim=-1)
        return torch.sum(teacher_probs * student_log_probs, dim=-1)


class DINOLoss(nn.Module):
    def __init__(
        self,
        out_dim,
        student_temp=0.1,
        center_momentum=0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.updated = True
        self.reduce_handle = None
        self.len_teacher_output = None
        self.async_batch_center = None

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_output, teacher_temp):
        self.apply_center_update()
        # teacher centering and sharpening
        return F.softmax((teacher_output - self.center) / teacher_temp, dim=-1)

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp, n_iterations=3):
        teacher_output = teacher_output.float()
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        # ====================================================================
        # NUMERICAL STABILITY FIX: LogSumExp trick to prevent exp() overflow
        # Problem: exp(logit / temp) overflows when logit/temp > 88.7 (FP32)
        # Solution: Subtract max before exp (standard softmax stabilization)
        # Mathematical guarantee: after x = x - max(x), all exp(x) <= 1.0
        # This is how PyTorch F.softmax is implemented internally.
        # To disable: comment the line "scaled_output = scaled_output - ..."
        # ====================================================================
        scaled_output = teacher_output / teacher_temp
        scaled_output = scaled_output - scaled_output.max(dim=-1, keepdim=True)[0]
        Q = torch.exp(scaled_output).t()  # Q is K-by-B for consistency with notations from our paper
        # ====================================================================
        
        B = Q.shape[1] * world_size  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if dist.is_initialized():
            dist.all_reduce(sum_Q)
        Q /= sum_Q

        for it in range(n_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the columns must sum to 1 so that Q is an assignment
        return Q.t()

    def forward(self, student_output_list, teacher_out_softmaxed_centered_list):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        total_loss = 0
        if _XFORMERS_CROSS_ENTROPY_AVAILABLE:
            for s in student_output_list:
                for t in teacher_out_softmaxed_centered_list:
                    loss = cross_entropy_distribution(s, t, self.student_temp)
                    total_loss -= loss.mean()
        else:
            for s in student_output_list:
                lsm = F.log_softmax(s / self.student_temp, dim=-1)
                for t in teacher_out_softmaxed_centered_list:
                    loss = cross_entropy_distribution(s, t, self.student_temp, student_log_probs=lsm)
                    total_loss -= loss.mean()
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        self.reduce_center_update(teacher_output)

    @torch.no_grad()
    def reduce_center_update(self, teacher_output):
        self.updated = False
        self.len_teacher_output = len(teacher_output)
        self.async_batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if dist.is_initialized():
            self.reduce_handle = dist.all_reduce(self.async_batch_center, async_op=True)

    @torch.no_grad()
    def apply_center_update(self):
        if self.updated is False:
            world_size = dist.get_world_size() if dist.is_initialized() else 1

            if self.reduce_handle is not None:
                self.reduce_handle.wait()
            _t = self.async_batch_center / (self.len_teacher_output * world_size)

            self.center = self.center * self.center_momentum + _t * (1 - self.center_momentum)

            self.updated = True
            
            # DEFENSIVE GUARD: Ensure center buffer never requires gradients
            assert not self.center.requires_grad, "Center buffer must not require gradients"
