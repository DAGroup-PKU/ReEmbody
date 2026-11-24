"""Point JSON repair at the failing field, without modifying rejected data."""


def _matches_type(instance, kind):
    kinds=kind if isinstance(kind,list) else [kind]
    actual=('null' if instance is None else 'boolean' if type(instance) is bool else
            'object' if isinstance(instance,dict) else 'array' if isinstance(instance,(list,tuple)) else
            'string' if isinstance(instance,str) else 'integer' if type(instance) is int else
            'number' if type(instance) is float else 'unknown')
    return actual in kinds or (actual=='integer' and 'number' in kinds)


def _leaves(error):
    if error.validator not in ('oneOf','anyOf') or not error.context:
        return [error]
    branches=error.validator_value;indices=list(range(len(branches)))
    if isinstance(error.instance,dict):
        # A declared discriminator (e.g. schema="rssr.scene.v2") selects
        # which failed alternative to explain. It never makes it valid.
        matched=[]
        for i,branch in enumerate(branches):
            constants={k:v['const'] for k,v in branch.get('properties',{}).items()
                       if isinstance(v,dict) and 'const' in v and k in error.instance}
            if constants and all(error.instance[k]==v for k,v in constants.items()):matched.append(i)
        if matched:indices=matched
    typed=[i for i in indices if 'type' not in branches[i] or
           _matches_type(error.instance,branches[i]['type'])]
    if typed:indices=typed
    children=[e for e in error.context if e.schema_path and e.schema_path[0] in indices]
    return [leaf for child in children for leaf in _leaves(child)] or [error]


def explain_errors(errors, maximum=3):
    leaves=[]
    for error in errors:
        for leaf in _leaves(error):
            path='/'.join(map(str,leaf.absolute_path)) or '<root>'
            message=leaf.message
            if len(message)>260:message=message[:257]+'...'
            text=path+': '+message
            if text not in leaves:leaves.append(text)
    return '; '.join(leaves[:maximum])[:1200]
