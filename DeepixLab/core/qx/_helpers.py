
def q_init(kwarg_name, cls_, cls_check_=None, **kwargs):
    if cls_check_ is None:
        cls_check_ = cls_
        
    val = kwargs.get(kwarg_name, None)
    if val is None and cls_ is not None:
        val = cls_()
    
    # if not isinstance(val, cls_check_):
    #     raise ValueError(f'{kwarg_name} must be an instance of {cls_check_}')
    return val


"""
qx.Hfmt.span_shortcut_keycomb



"""