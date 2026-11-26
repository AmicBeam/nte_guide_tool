"""Resolve ordinary rule functions and synchronous built-in methods directly.

Argument binding, frame allocation, generator suspension, method allocation and
return handling retain their canonical bodies. This is explicit tag routing,
not speculative inlining of effects or replacement of their rule semantics.
"""
from .cuda_vm import _VM_SOURCE
from .native_array_codegen import _matching_brace_end


_METHODS = ('DICT_GET', 'DICT_KEYS', 'DICT_VALUES', 'DICT_ITEMS',
            'DICT_SETDEFAULT', 'DICT_POP', 'DICT_COPY', 'LIST_APPEND',
            'LIST_POP', 'LIST_INSERT', 'LIST_REMOVE', 'LIST_REVERSE', 'LIST_COPY')


def integrate_native_calls(source):
    marker = '__device__ inline void call(Context* c,int callable,const int* args,int nargs,int caller,int kw_names=MISSING,const int* kw_values=0,int nkw=0){'
    if source.count(marker) != 1:
        raise ValueError('Canonical native call definition boundary changed')
    begin = _VM_SOURCE.index(marker)
    end = _matching_brace_end(_VM_SOURCE, _VM_SOURCE.index('{', begin))
    canonical = _VM_SOURCE[begin:end]
    start = canonical.index(' int fid=-1,globals=MISSING,closure=MISSING,defaults=MISSING,kwdefaults=MISSING;')
    tail = canonical[start:-1]
    if source.count(tail) != 1:
        raise ValueError('Canonical native function binding body changed')
    # The original function path is reached only after all library handlers
    # reject these two tags. Reuse its exact binding/generator body.
    helper = '''__device__ inline int native_fast_call(Context* c,int callable,const int* args,int nargs,int caller,
 int kw_names,const int* kw_values,int nkw){
 const int* n=node(c,callable);if(!n)return 0;
 if(n[0]==T_FUNCTION||n[0]==T_DYNFN){
'''
    # The outer helper reports whether it consumed the call; original void
    # exits report consumption, including argument/frame failures.
    helper += tail.replace('return;', 'return 1;') + ' return 1;\n }\n'
    helper += ' if(n[0]!=34)return 0;\n switch(n[3]){\n'
    helper += ''.join(' case METHOD_' + name + ':\n' for name in _METHODS)
    helper += ''' {
  if(nkw){c->error=BAD_CALL;return 1;}
  int result=MISSING;
  if(!native_builtin_call(c,callable,args,nargs,&result)){c->error=BAD_CALL;return 1;}
  if(!c->error&&!c->h.error)push(c,frame(c,caller),result);
  return 1;
 }
 default:return 0;
 }
}
'''
    hook = ' if(n[0]==13){\n'
    call = (' if(native_fast_call(c,callable,args,nargs,caller,kw_names,kw_values,nkw))return;\n')
    # Only patch the call function. Other constructors and library handlers
    # may use exactly the same class tag test.
    offset = source.index(marker)
    end = _matching_brace_end(source, source.index('{', offset))
    body = source[offset:end]
    if body.count(hook) != 1:
        raise ValueError('Canonical native call routing boundary changed')
    body = body.replace(hook, call + hook, 1)
    return source[:offset] + helper + body + source[end:]
