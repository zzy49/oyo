"""Patch ManyDepth options.py and trainer.py to add depth supervision."""
import os

# ── Patch options.py ──
opt_path = r'e:\data1\manydepth-master\manydepth\options.py'
with open(opt_path, 'r', encoding='utf-8') as f:
    content = f.read()

old = ('self.parser.add_argument("--disparity_smoothness",\n'
       '                                 type=float,\n'
       '                                 help="disparity smoothness weight",\n'
       '                                 default=1e-3)')
new = (old + '\n        self.parser.add_argument("--depth_supervision_weight",\n'
       '                                 type=float,\n'
       '                                 help="weight for supervised depth L1 loss (0 to disable)",\n'
       '                                 default=0.1)')
content = content.replace(old, new)
with open(opt_path, 'w', encoding='utf-8') as f:
    f.write(content)
print('options.py patched')

# ── Patch trainer.py: add depth supervision loss in compute_losses ──
trainer_path = r'e:\data1\manydepth-master\manydepth\trainer.py'
with open(trainer_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Insert depth supervision before "total_loss /= self.num_scales"
insert_marker = '        total_loss /= self.num_scales\n        losses["loss"] = total_loss\n\n        return losses'
depth_sup_code = '''
        # --- GT depth supervision L1 loss (only for single-frame mono output) ---
        if not is_multi and hasattr(self.opt, "depth_supervision_weight") and self.opt.depth_supervision_weight > 0:
            if "depth_gt" in inputs:
                depth_gt = inputs["depth_gt"]
                depth_pred = outputs[("depth", 0, 0)]
                valid_mask = depth_gt > 1.0
                if valid_mask.sum() > 100:
                    med_gt = depth_gt[valid_mask].median()
                    med_pred = depth_pred[valid_mask].median().detach()
                    scale = med_gt / (med_pred + 1e-6)
                    depth_pred_scaled = depth_pred * scale
                    depth_loss = F.l1_loss(
                        depth_pred_scaled[valid_mask], depth_gt[valid_mask])
                    total_loss += self.opt.depth_supervision_weight * depth_loss
                    losses["depth_sup_loss"] = depth_loss

'''
new_insert = depth_sup_code + '        total_loss /= self.num_scales\n        losses["loss"] = total_loss\n\n        return losses'
content = content.replace(insert_marker, new_insert)
with open(trainer_path, 'w', encoding='utf-8') as f:
    f.write(content)
print('trainer.py patched')
