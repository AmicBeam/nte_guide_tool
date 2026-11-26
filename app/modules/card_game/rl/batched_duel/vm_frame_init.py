"""Initialize only addressable frame slots; operand stacks are length guarded."""

HOOK = 'int* f=frame(c,id);for(int i=0;i<STRIDE;i++)f[i]=MISSING;'
REPLACEMENT = '''int* f=frame(c,id);
 for(int i=0;i<HEADER;i++)f[i]=MISSING;
 for(int i=0;i<fn[5];i++)f[LBASE+i]=MISSING;
 f[11]=0; // No cells are live until their explicit construction below.
'''


def integrate_frame_init(source):
    if type(source) is not str:
        raise TypeError('CUDA source text required')
    if source.count(HOOK) != 1:
        raise ValueError('Expected unique frozen frame initialization hook')
    return source.replace(HOOK, REPLACEMENT, 1)
