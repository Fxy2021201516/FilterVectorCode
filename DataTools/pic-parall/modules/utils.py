def build_paths_for_exp(config, dataset_name, exp_params):
    """
    Constructs the results directory path for a given experiment based on config templates.
    """
    global_settings = config['global_settings']
    dataset_conf = config['dataset_configurations'][dataset_name]
    templates = dataset_conf['structure_templates']
    build_params = dataset_conf.get('build_params', {})
    
    format_params = {**build_params, **exp_params, 'dataset': dataset_name, **global_settings}
    
    if 'query_dir_name' not in format_params:
        raise ValueError("实验参数中缺少 'query_dir_name'")
        
    format_params['safe_query_name'] = format_params['query_dir_name'].replace('/', '_')
    
    # Dynamically format handles
    ung_index_handle = templates['ung_index_handle'].format(**format_params)
    ung_gt_handle = templates['ung_gt_handle'].format(**format_params)
    search_params_handle = templates['search_params_handle'].format(**format_params)
    
    format_params.update({
      'ung_index_handle': ung_index_handle,
      'ung_gt_handle': ung_gt_handle,
      'search_params_handle': search_params_handle
    })
    
    return templates['result_dir_template'].format(**format_params)