# This code is adapted from the original DINOv2 repository: https://github.com/facebookresearch/dinov2
# This code is licensed under the CC BY-NC-ND 4.0 license
# found in the LICENSE file in the root directory of this source tree.

from functools import partial
import logging
import math

import torch
from torch import nn

from dinov2.loss import DINOLoss, iBOTPatchLoss, KoLeoLoss
from dinov2.models import build_model_from_cfg
from dinov2.layers import DINOHead
from dinov2.utils.utils import has_batchnorms
from dinov2.utils.param_groups import get_params_groups_with_decay, fuse_params_groups
from dinov2.fsdp import get_fsdp_wrapper, ShardedGradScaler, get_fsdp_modules, reshard_fsdp_model

from dinov2.models.vision_transformer import BlockChunk


logger = logging.getLogger("dinov2")


def interpolate_pos_encoding(state_dict, curr_img_size, patch_size):
    prev_pos_embed = state_dict["backbone.pos_embed"]
    prev_dtype = prev_pos_embed.dtype
    prev_npatch = prev_pos_embed.shape[1] - 1
    curr_npatch = (curr_img_size // patch_size) ** 3
    if prev_npatch == curr_npatch:
        return

    prev_pos_embed = prev_pos_embed.float()
    class_pos_embed = prev_pos_embed[:, 0]
    patch_pos_embed = prev_pos_embed[:, 1:]
    feat_dim = patch_pos_embed.shape[-1]

    size0 = curr_img_size // patch_size
    size0 = size0 + 0.1  # avoid floating point error in the interpolation

    cbrt_prev_npatch = round(math.pow(prev_npatch, (1/3)))
    scale_fact = float(size0) / cbrt_prev_npatch

    curr_pos_embed = nn.functional.interpolate(
        patch_pos_embed.reshape(
            1, cbrt_prev_npatch, cbrt_prev_npatch, cbrt_prev_npatch, feat_dim
        ).permute(0, 4, 1, 2, 3),
        scale_factor=(scale_fact, scale_fact, scale_fact),
        mode="trilinear",
    )

    assert int(size0) == curr_pos_embed.shape[-1] == curr_pos_embed.shape[-2] == curr_pos_embed.shape[-3]
    curr_pos_embed = curr_pos_embed.permute(0, 2, 3, 4, 1).reshape(1, -1, feat_dim)
    curr_pos_embed = torch.cat((class_pos_embed.unsqueeze(0), curr_pos_embed), dim=1).to(prev_dtype)
    print(curr_pos_embed.shape)
    state_dict["backbone.pos_embed"] = curr_pos_embed


class SSLMetaArch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.fp16_scaler = ShardedGradScaler() if cfg.compute_precision.grad_scaler else None

        student_model_dict = dict()
        teacher_model_dict = dict()

        student_backbone, teacher_backbone, embed_dim = build_model_from_cfg(cfg)
        student_model_dict["backbone"] = student_backbone
        teacher_model_dict["backbone"] = teacher_backbone
        logger.info(f"OPTIONS -- architecture : embed_dim: {embed_dim}")

        if cfg.student.pretrained_weights:
            chkpt = torch.load(cfg.student.pretrained_weights)
            logger.info(f"OPTIONS -- pretrained weights: loading from {cfg.student.pretrained_weights}")
            student_backbone.load_state_dict(chkpt["model"], strict=False)

        self.embed_dim = embed_dim
        self.dino_out_dim = cfg.dino.head_n_prototypes

        self.do_dino = cfg.dino.loss_weight > 0
        self.do_koleo = cfg.dino.koleo_loss_weight > 0
        self.do_ibot = cfg.ibot.loss_weight > 0
        self.ibot_separate_head = cfg.ibot.separate_head

        logger.info("OPTIONS -- DINO")
        if self.do_dino:
            logger.info(f"OPTIONS -- DINO -- loss_weight: {cfg.dino.loss_weight}")
            logger.info(f"OPTIONS -- DINO -- head_n_prototypes: {cfg.dino.head_n_prototypes}")
            logger.info(f"OPTIONS -- DINO -- head_bottleneck_dim: {cfg.dino.head_bottleneck_dim}")
            logger.info(f"OPTIONS -- DINO -- head_hidden_dim: {cfg.dino.head_hidden_dim}")
            self.dino_loss_weight = cfg.dino.loss_weight
            dino_head = partial(
                DINOHead,
                in_dim=embed_dim,
                out_dim=cfg.dino.head_n_prototypes,
                hidden_dim=cfg.dino.head_hidden_dim,
                bottleneck_dim=cfg.dino.head_bottleneck_dim,
                nlayers=cfg.dino.head_nlayers,
            )
            self.dino_loss = DINOLoss(self.dino_out_dim)
            if self.do_koleo:
                logger.info("OPTIONS -- DINO -- applying KOLEO regularization")
                self.koleo_loss = KoLeoLoss()

        else:
            logger.info("OPTIONS -- DINO -- not using DINO")

        if self.do_dino or self.do_ibot:
            student_model_dict["dino_head"] = dino_head()
            teacher_model_dict["dino_head"] = dino_head()

        logger.info("OPTIONS -- IBOT")
        logger.info(f"OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}")
        logger.info(f"OPTIONS -- IBOT masking -- ibot_mask_ratio_tuple: {cfg.ibot.mask_ratio_min_max}")
        logger.info(f"OPTIONS -- IBOT masking -- ibot_mask_sample_probability: {cfg.ibot.mask_sample_probability}")
        if self.do_ibot:
            self.ibot_loss_weight = cfg.ibot.loss_weight
            assert max(cfg.ibot.mask_ratio_min_max) > 0, "please provide a positive mask ratio tuple for ibot"
            assert cfg.ibot.mask_sample_probability > 0, "please provide a positive mask probability for ibot"
            self.ibot_out_dim = cfg.ibot.head_n_prototypes if self.ibot_separate_head else cfg.dino.head_n_prototypes
            self.ibot_patch_loss = iBOTPatchLoss(self.ibot_out_dim)
            if self.ibot_separate_head:
                logger.info(f"OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}")
                logger.info(f"OPTIONS -- IBOT -- head_n_prototypes: {cfg.ibot.head_n_prototypes}")
                logger.info(f"OPTIONS -- IBOT -- head_bottleneck_dim: {cfg.ibot.head_bottleneck_dim}")
                logger.info(f"OPTIONS -- IBOT -- head_hidden_dim: {cfg.ibot.head_hidden_dim}")
                ibot_head = partial(
                    DINOHead,
                    in_dim=embed_dim,
                    out_dim=cfg.ibot.head_n_prototypes,
                    hidden_dim=cfg.ibot.head_hidden_dim,
                    bottleneck_dim=cfg.ibot.head_bottleneck_dim,
                    nlayers=cfg.ibot.head_nlayers,
                )
                student_model_dict["ibot_head"] = ibot_head()
                teacher_model_dict["ibot_head"] = ibot_head()
            else:
                logger.info("OPTIONS -- IBOT -- head shared with DINO")

        self.need_to_synchronize_fsdp_streams = True

        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)

        # allow restarting from a checkpoint (adapting resolution experiments)
        if cfg.student.full_pretrained_weights:
            chkpt = torch.load(cfg.student.full_pretrained_weights)
            logger.info(f"OPTIONS -- full pretrained weights: loading from {cfg.student.full_pretrained_weights}")
            interpolate_pos_encoding(chkpt["teacher"], cfg.crops.global_crops_size, cfg.student.patch_size)
            msg = self.student.load_state_dict(chkpt["teacher"], strict=False)
            logger.info("Pretrained weights loaded with msg: {}".format(msg))

        # there is no backpropagation through the teacher, so no need for gradients
        for p in self.teacher.parameters():
            p.requires_grad = False
        logger.info(f"Student and Teacher are built: they are both {cfg.student.arch} network.")

    def forward(self, inputs):
        raise NotImplementedError

    def backprop_loss(self, loss):
        if self.fp16_scaler is not None:
            self.fp16_scaler.scale(loss).backward()
        else:
            loss.backward()

    def forward_backward(self, images, teacher_temp):
        n_global_crops = 2
        assert n_global_crops == 2
        n_local_crops = self.cfg.crops.local_crops_number

        global_crops = images["collated_global_crops"].cuda(non_blocking=True)
        local_crops = images["collated_local_crops"].cuda(non_blocking=True)

        masks = images["collated_masks"].cuda(non_blocking=True)
        mask_indices_list = images["mask_indices_list"].cuda(non_blocking=True)
        n_masked_patches_tensor = images["n_masked_patches"].cuda(non_blocking=True)
        n_masked_patches = mask_indices_list.shape[0]
        upperbound = images["upperbound"]
        masks_weight = images["masks_weight"].cuda(non_blocking=True)

        n_local_crops_loss_terms = max(n_local_crops * n_global_crops, 1)
        n_global_crops_loss_terms = (n_global_crops - 1) * n_global_crops

        do_dino = self.do_dino
        do_ibot = self.do_ibot

        # loss scales
        ibot_loss_scale = 1.0 / n_global_crops

        # teacher output
        @torch.no_grad()
        def get_teacher_output():
            x, n_global_crops_teacher = global_crops, n_global_crops
            teacher_backbone_output_dict = self.teacher.backbone(x, is_training=True)
            teacher_cls_tokens = teacher_backbone_output_dict["x_norm_clstoken"]
            # CRITICAL: Swap order for cross-view distillation
            # [crop0_samples, crop1_samples] → [crop1_samples, crop0_samples]
            teacher_cls_tokens = teacher_cls_tokens.chunk(n_global_crops_teacher)   # [(96, 1024), (96, 1024)]
            # watch out: these are chunked and cat'd in reverse so A is matched to B in the global crops dino loss
            teacher_cls_tokens = torch.cat((teacher_cls_tokens[1], teacher_cls_tokens[0]))  # (192, 1024)
            ibot_teacher_patch_tokens = teacher_backbone_output_dict["x_norm_patchtokens"]  # (192, 216, 1024)
            _dim = ibot_teacher_patch_tokens.shape[-1]
            n_cls_tokens = teacher_cls_tokens.shape[0]

            if do_ibot and not self.ibot_separate_head:
                buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(upperbound + n_cls_tokens, _dim)
                buffer_tensor_teacher[:n_cls_tokens].copy_(teacher_cls_tokens)
                torch.index_select(
                    ibot_teacher_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                    out=buffer_tensor_teacher[n_cls_tokens : n_cls_tokens + n_masked_patches],
                )
                tokens_after_head = self.teacher.dino_head(buffer_tensor_teacher)
                teacher_cls_tokens_after_head = tokens_after_head[:n_cls_tokens]
                masked_teacher_patch_tokens_after_head = tokens_after_head[
                    n_cls_tokens : n_cls_tokens + n_masked_patches
                ]
            elif do_ibot and self.ibot_separate_head:
                buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(upperbound, _dim)
                torch.index_select(
                    ibot_teacher_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                    out=buffer_tensor_teacher[:n_masked_patches],
                )
                teacher_cls_tokens_after_head = self.teacher.dino_head(teacher_cls_tokens)  # (192, 65536)
                masked_teacher_patch_tokens_after_head = self.teacher.ibot_head(buffer_tensor_teacher)[
                    :n_masked_patches
                ]
            else:
                teacher_cls_tokens_after_head = self.teacher.dino_head(teacher_cls_tokens)
                masked_teacher_ibot_softmaxed_centered = None

            # DEFENSIVE GUARD: Ensure intermediate teacher tensors are detached
            # (redundant with @torch.no_grad but explicit for safety)
            teacher_cls_tokens_after_head = teacher_cls_tokens_after_head.detach()
            if do_ibot and masked_teacher_patch_tokens_after_head is not None:
                masked_teacher_patch_tokens_after_head = masked_teacher_patch_tokens_after_head.detach()

            if self.cfg.train.centering == "centering":
                teacher_dino_softmaxed_centered_list = self.dino_loss.softmax_center_teacher(
                    teacher_cls_tokens_after_head, teacher_temp=teacher_temp
                ).view(n_global_crops_teacher, -1, *teacher_cls_tokens_after_head.shape[1:])
                self.dino_loss.update_center(teacher_cls_tokens_after_head)
                if do_ibot:
                    masked_teacher_patch_tokens_after_head = masked_teacher_patch_tokens_after_head.unsqueeze(0)
                    masked_teacher_ibot_softmaxed_centered = self.ibot_patch_loss.softmax_center_teacher(
                        masked_teacher_patch_tokens_after_head[:, :n_masked_patches], teacher_temp=teacher_temp
                    )
                    masked_teacher_ibot_softmaxed_centered = masked_teacher_ibot_softmaxed_centered.squeeze(0)
                    self.ibot_patch_loss.update_center(masked_teacher_patch_tokens_after_head[:n_masked_patches])

            elif self.cfg.train.centering == "sinkhorn_knopp":
                teacher_dino_softmaxed_centered_list = self.dino_loss.sinkhorn_knopp_teacher(
                    teacher_cls_tokens_after_head, teacher_temp=teacher_temp
                ).view(n_global_crops_teacher, -1, *teacher_cls_tokens_after_head.shape[1:])

                if do_ibot:
                    masked_teacher_ibot_softmaxed_centered = self.ibot_patch_loss.sinkhorn_knopp_teacher(
                        masked_teacher_patch_tokens_after_head,
                        teacher_temp=teacher_temp,
                        n_masked_patches_tensor=n_masked_patches_tensor,
                    )

            else:
                raise NotImplementedError

            return teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered

        teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered = get_teacher_output()
        
        # DEFENSIVE GUARDS: Validate teacher outputs
        # Guard 1: Teacher outputs must not require gradients (detach is implicit via @torch.no_grad)
        assert not teacher_dino_softmaxed_centered_list.requires_grad, (
            "Teacher DINO outputs must not require gradients"
        )
        if masked_teacher_ibot_softmaxed_centered is not None:
            assert not masked_teacher_ibot_softmaxed_centered.requires_grad, (
                "Teacher iBOT outputs must not require gradients"
            )
        
        # Guard 2: Teacher DINO shape validation [n_global_crops, batch_per_crop, out_dim]
        batch_per_crop = global_crops.shape[0] // n_global_crops
        assert teacher_dino_softmaxed_centered_list.shape[0] == n_global_crops, (
            f"Teacher DINO crop dim mismatch: {teacher_dino_softmaxed_centered_list.shape[0]} vs {n_global_crops}"
        )
        assert teacher_dino_softmaxed_centered_list.shape[1] == batch_per_crop, (
            f"Teacher DINO batch dim mismatch: {teacher_dino_softmaxed_centered_list.shape[1]} vs {batch_per_crop} "
            f"(possible B/V reshape mixing)"
        )
        
        # Guard 2b: Entropy check to detect mode collapse
        # High entropy (near-uniform) may indicate collapse or early training
        # Only warn, don't crash - early iterations naturally have high entropy
        with torch.no_grad():
            probs = teacher_dino_softmaxed_centered_list.flatten(0, 1)  # [B*V, K]
            entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean()
            max_entropy = torch.tensor(probs.shape[-1], dtype=probs.dtype, device=probs.device).log()
            entropy_ratio = entropy / max_entropy
            # Warn if entropy > 95% of max (near-uniform = possible collapse)
            if entropy_ratio > 0.95:
                logger.warning(
                    f"[CollapseGuard] Teacher entropy high: {entropy_ratio:.3f} of max"
                )
        
        # Guard 3: Check for NaN/Inf in teacher outputs
        assert torch.isfinite(teacher_dino_softmaxed_centered_list).all(), (
            "NaN/Inf detected in teacher DINO outputs"
        )
        if masked_teacher_ibot_softmaxed_centered is not None and masked_teacher_ibot_softmaxed_centered.numel() > 0:
            assert torch.isfinite(masked_teacher_ibot_softmaxed_centered).all(), (
                "NaN/Inf detected in teacher iBOT outputs"
            )
        
        reshard_fsdp_model(self.teacher)

        loss_dict = {}

        loss_accumulator = 0  # for backprop
        student_global_backbone_output_dict, student_local_backbone_output_dict = self.student.backbone(
            [global_crops, local_crops], masks=[masks, None], is_training=True
        )

        inputs_for_student_head_list = []

        # 1a: local crops cls tokens
        student_local_cls_tokens = student_local_backbone_output_dict["x_norm_clstoken"]
        inputs_for_student_head_list.append(student_local_cls_tokens.unsqueeze(0))

        # 1b: global crops cls tokens
        student_global_cls_tokens = student_global_backbone_output_dict["x_norm_clstoken"]
        inputs_for_student_head_list.append(student_global_cls_tokens.unsqueeze(0))

        # 1c: global crops patch tokens
        if do_ibot:
            _dim = student_global_backbone_output_dict["x_norm_clstoken"].shape[-1]
            ibot_student_patch_tokens = student_global_backbone_output_dict["x_norm_patchtokens"]
            buffer_tensor_patch_tokens = ibot_student_patch_tokens.new_zeros(upperbound, _dim)
            buffer_tensor_patch_tokens[:n_masked_patches].copy_(
                torch.index_select(ibot_student_patch_tokens.flatten(0, 1), dim=0, index=mask_indices_list)
            )
            if not self.ibot_separate_head:
                inputs_for_student_head_list.append(buffer_tensor_patch_tokens.unsqueeze(0))
            else:
                student_global_masked_patch_tokens_after_head = self.student.ibot_head(buffer_tensor_patch_tokens)[
                    :n_masked_patches
                ]

        # 2: run student head on batched inputs
        # =====================================================================
        # DINOHead is a pointwise MLP (no cross-token attention), so we can
        # efficiently batch all inputs by concatenating along token dimension,
        # running a single forward pass, then splitting by original sizes.
        # This replaces xFormers BlockDiagonalMask without any efficiency loss.
        # =====================================================================
        
        # Record sizes for splitting (each input has shape [1, n_tokens, dim])
        sizes = [inp.shape[1] for inp in inputs_for_student_head_list]
        
        # Concatenate all tokens: [1, n1, dim], [1, n2, dim], ... → [n1+n2+..., dim]
        cat_inputs = torch.cat([inp.squeeze(0) for inp in inputs_for_student_head_list], dim=0)
        
        # Single forward pass through MLP head
        cat_outputs = self.student.dino_head(cat_inputs)
        
        # Split back to original sizes and restore batch dim
        outputs_list = [out.unsqueeze(0) for out in cat_outputs.split(sizes, dim=0)]

        # 3a: local crops cls tokens
        student_local_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)

        # 3b: global crops cls tokens
        student_global_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)

        # 3c: global crops patch tokens
        if do_ibot and not self.ibot_separate_head:
            student_global_masked_patch_tokens_after_head = outputs_list.pop(0).squeeze(0)[:n_masked_patches]

        if n_local_crops > 0:
            dino_local_crops_loss = self.dino_loss(
                student_output_list=student_local_cls_tokens_after_head.chunk(n_local_crops),
                teacher_out_softmaxed_centered_list=teacher_dino_softmaxed_centered_list,
            ) / (n_global_crops_loss_terms + n_local_crops_loss_terms)

            # store for display
            loss_dict["dino_local_crops_loss"] = dino_local_crops_loss

            # accumulate loss
            loss_accumulator += self.dino_loss_weight * dino_local_crops_loss

        # process global crops
        loss_scales = 2  # this is here since we process global crops together

        if do_dino:
            # compute loss
            dino_global_crops_loss = (
                self.dino_loss(
                    student_output_list=[student_global_cls_tokens_after_head],
                    teacher_out_softmaxed_centered_list=[
                        teacher_dino_softmaxed_centered_list.flatten(0, 1)
                    ],  # these were chunked and stacked in reverse so A is matched to B
                )
                * loss_scales
                / (n_global_crops_loss_terms + n_local_crops_loss_terms)
            )

            loss_dict["dino_global_crops_loss"] = dino_global_crops_loss

            # accumulate loss
            loss_accumulator += self.dino_loss_weight * dino_global_crops_loss

            student_cls_tokens = student_global_cls_tokens

            if self.do_koleo:
                koleo_loss = self.cfg.dino.koleo_loss_weight * sum(
                    self.koleo_loss(p) for p in student_cls_tokens.chunk(2)
                )  # we don't apply koleo loss between cls tokens of a same image
                loss_accumulator += koleo_loss
                loss_dict["koleo_loss"] = (
                    koleo_loss / loss_scales
                )  # this is to display the same losses as before but we can remove eventually

        if do_ibot:
            # compute loss
            ibot_patch_loss = (
                self.ibot_patch_loss.forward_masked(
                    student_global_masked_patch_tokens_after_head,
                    masked_teacher_ibot_softmaxed_centered,
                    student_masks_flat=masks,
                    n_masked_patches=n_masked_patches,
                    masks_weight=masks_weight,
                )
                * loss_scales
                * ibot_loss_scale
            )

            # store for display
            loss_dict["ibot_loss"] = ibot_patch_loss / 2

            # accumulate loss
            loss_accumulator += self.ibot_loss_weight * ibot_patch_loss

        self.backprop_loss(loss_accumulator)

        self.fsdp_synchronize_streams()

        return loss_dict

    def fsdp_synchronize_streams(self):
        """Synchronize CUDA streams across FSDP-wrapped student/teacher models.
        
        In PyTorch 2.x FSDP, stream management is handled internally and the
        `_streams` attribute no longer exists on FSDP modules. The original
        DINOv2 code relied on this internal attribute to share streams across
        models for memory efficiency.
        
        For PyTorch 2.x compatibility, we simply call torch.cuda.synchronize()
        which ensures all CUDA operations complete before proceeding. This is
        sufficient for correctness in the teacher-student setup.
        
        Note: This may have a minor performance impact compared to the original
        stream-sharing approach, but ensures compatibility with PyTorch 2.x.
        """
        if self.need_to_synchronize_fsdp_streams:
            torch.cuda.synchronize()
            # PyTorch 2.x FSDP no longer exposes _streams attribute.
            # Stream synchronization is handled internally by FSDP.
            self.need_to_synchronize_fsdp_streams = False

    def update_teacher(self, m):
        """Update teacher parameters using EMA (Exponential Moving Average).
        
        In PyTorch 2.x FSDP, the internal `.params` attribute on FSDP modules
        no longer exists. Instead, we use the standard `.parameters()` method
        which returns the module parameters properly through FSDP's interface.
        
        The EMA update formula is:
            teacher_param = m * teacher_param + (1 - m) * student_param
        
        Args:
            m: Momentum coefficient for EMA (typically ~0.99)
        """
        student_param_list = []
        teacher_param_list = []
        with torch.no_grad():
            for k in self.student.keys():
                # In PyTorch 2.x FSDP, use .parameters() instead of .params
                # We iterate over FSDP modules and collect their parameters
                for ms, mt in zip(get_fsdp_modules(self.student[k]), get_fsdp_modules(self.teacher[k])):
                    student_param_list.extend(list(ms.parameters()))
                    teacher_param_list.extend(list(mt.parameters()))
            torch._foreach_mul_(teacher_param_list, m)
            torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)

    def train(self):
        super().train()
        self.teacher.eval()

    def get_maybe_fused_params_for_submodel(self, m):
        params_groups = get_params_groups_with_decay(
            model=m,
            lr_decay_rate=self.cfg.optim.layerwise_decay,
            patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult,
        )
        fused_params_groups = fuse_params_groups(params_groups)
        logger.info("fusing param groups")

        for g in fused_params_groups:
            g["foreach"] = True
        return fused_params_groups

    def get_params_groups(self):
        all_params_groups = []
        for m in self.student.values():
            all_params_groups += self.get_maybe_fused_params_for_submodel(m)
        return all_params_groups

    def prepare_for_distributed_training(self):
        logger.info("DISTRIBUTED FSDP -- preparing model for distributed training")
        if has_batchnorms(self.student):
            raise NotImplementedError
        # below will synchronize all student subnetworks across gpus:
        for k, v in self.student.items():
            self.teacher[k].load_state_dict(self.student[k].state_dict())
            student_model_cfg = self.cfg.compute_precision.student[k]
            self.student[k] = get_fsdp_wrapper(student_model_cfg, modules_to_wrap={BlockChunk})(self.student[k])
            teacher_model_cfg = self.cfg.compute_precision.teacher[k]
            self.teacher[k] = get_fsdp_wrapper(teacher_model_cfg, modules_to_wrap={BlockChunk})(self.teacher[k])
