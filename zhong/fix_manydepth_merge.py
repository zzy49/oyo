"""Fix ManyDepth trainer merge for depth_sup_loss key."""
trainer_path = r'e:\data1\manydepth-master\manydepth\trainer.py'
with open(trainer_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the merge to handle missing keys gracefully
old_merge = """        # update losses with single frame losses
        if self.train_teacher_and_pose:
            for key, val in mono_losses.items():
                losses[key] += val"""

new_merge = """        # update losses with single frame losses
        if self.train_teacher_and_pose:
            for key, val in mono_losses.items():
                if key in losses:
                    losses[key] += val
                else:
                    losses[key] = val"""

content = content.replace(old_merge, new_merge)
with open(trainer_path, 'w', encoding='utf-8') as f:
    f.write(content)
print('trainer.py merge fixed')
