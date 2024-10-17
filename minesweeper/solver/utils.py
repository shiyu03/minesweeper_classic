
def update_checkpoint(checkpoint, *, mapping_model=None, mapping_optimizer_old=None, mapping_optimizer_new=None, checkpoint_file_old_model=None):
    if mapping_model is not None:
        model_state_dict = checkpoint['model_state_dict']
        for old_param, new_param in mapping_model.items():
            for suffix in ['.weight', '.bias']:
                old_param_suffix = old_param + suffix
                new_param_suffix = new_param + suffix
                model_state_dict[new_param_suffix] = model_state_dict.pop(old_param_suffix)
        checkpoint['model_state_dict'] = model_state_dict

    # 如果只训练新添加的层，optimizer的param_groups就只对应新参数，旧参数就不需要了。
    if mapping_optimizer_old is not None and mapping_optimizer_new is not None:
        old_optimizer_state_dict = checkpoint_file_old_model['optimizer_state_dict']
        new_partial_optimizer_state_dict = checkpoint['optimizer_state_dict']

        new_partial_optimizer_state_dict['param_groups'][0]['params'] = list(range(32))

        # 先把新中间层对应的0~15映射到整个模型的14~29
        for new_partial_param, new_full_param in mapping_optimizer_new.items():
            new_partial_optimizer_state_dict['state'][new_full_param] = new_partial_optimizer_state_dict['state'].pop(new_partial_param)
        # 再把旧的头尾层0~15映射到整个模型0~13, 30~31
        for old_param, new_full_param in mapping_optimizer_old.items():
            new_partial_optimizer_state_dict['state'][new_full_param] = old_optimizer_state_dict.pop(old_param)
        checkpoint['optimizer_state_dict'] = new_partial_optimizer_state_dict
    return checkpoint
