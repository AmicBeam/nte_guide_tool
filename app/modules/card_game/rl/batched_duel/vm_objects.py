"""Frozen namespace and descriptor lookup for the CUDA rule interpreter.

Compiles class MRO from frozen metadata; executes attribute lookup on device.
Ordinary object constructors use captured __init__; native descriptors,
native-base constructors and arbitrary metaclasses are not implemented.
This component is not a full-rule executor.
"""
OBJECT_OPS=frozenset(('LOAD_ATTR','LOAD_METHOD','CALL_METHOD','STORE_ATTR','IMPORT_NAME','IMPORT_FROM'))


def emit_object_source(image):
    def node(ref):
        if ref>=0:raise ValueError('Immutable class metadata required')
        return image.nodes[-1-ref]
    def seq(ref):
        n=node(ref)
        if n[0]==0:return []
        if n[0] not in (5,6):raise ValueError('Class base sequence required')
        return [int(x) for x in image.edges[n[1]:n[1]+n[2],0]]
    cache={};visiting=set()
    def mro(key):
        if key in cache:return cache[key]
        if key in visiting:raise ValueError('Cyclic frozen class hierarchy')
        visiting.add(key)
        tag,index=key
        bases=[(int(node(r)[0]),int(node(r)[3])) for r in seq(int(image.classes[index,0]))] if tag==13 else []
        pending=[mro(b)[:] for b in bases]+[bases[:]];out=[key]
        while any(pending):
            pending=[x for x in pending if x]
            candidate=next((x[0] for x in pending if not any(x[0] in y[1:] for y in pending)),None)
            if candidate is None:raise ValueError('Inconsistent frozen MRO')
            out.append(candidate)
            for x in pending:
                if x and x[0]==candidate:x.pop(0)
        visiting.remove(key);cache[key]=out;return out
    orders=[[index for tag,index in mro((13,i)) if tag==13] for i in range(len(image.classes))]
    refs={int(n[3]):-1-i for i,n in enumerate(image.nodes) if n[0]==13}
    def entity_ref(name):
        found=[i for i,k in enumerate(image.class_ids) if ':entities:'+name+':' in k]
        return refs[found[0]] if len(found)==1 else -2147483648
    constructible=[all(tag==13 or image.intrinsic_names[index]=='object' for tag,index in mro((13,i))) for i in range(len(image.classes))]
    constructors=''.join('case '+str(i)+':return '+str(int(ok))+';' for i,ok in enumerate(constructible))
    switches=''.join('case '+str(i)+':{const int order[]={'+','.join(map(str,order))+'}; for(int j=0;j<'+str(len(order))+';j++){int out=lookup_own(c,order[j],name,category);if(out!=MISSING)return out;}break;}\n' for i,order in enumerate(orders))
    return '__device__ inline int class_constructible(int cid){switch(cid){'+constructors+'default:return 0;}}\n'+r'''
__device__ inline int vm_text_equals(Context* c,int ref,const char* text){
 const int* n=node(c,ref);if(!n||n[0]!=TAG_STR)return 0;NodeView v;if(!get_node(&c->h,ref,&v))return 0;
 const unsigned char* p=get_payload_ptr(&c->h,&v);int size=0;while(text[size])size++;
 if(size!=v.count)return 0;for(int i=0;i<size;i++)if(p[i]!=(unsigned char)text[i])return 0;return 1;
}
__device__ inline int dict_text_get(Context* c,int mapping,const char* key){
 const int* n=node(c,mapping);if(!n||n[0]!=TAG_DICT){c->error=BAD_PROGRAM;return MISSING;}
 for(int i=0;i<n[2];i++)if(vm_text_equals(c,edge(c,mapping,i),key))return edge(c,mapping,i,1);return MISSING;
}
__device__ inline int lookup_own(Context* c,int cid,int name,int* category){
 if(cid<0||cid>=c->class_count){c->error=BAD_PROGRAM;return MISSING;}
 const int* cls=c->classes+cid*6;
 // Data descriptors precede methods and frozen attributes.
 const int columns[5]={5,2,3,4,1};
 for(int j=0;j<5;j++){int col=columns[j],found=dict_get(&c->h,cls[col],name,MISSING);if(found!=MISSING){*category=col;return found;}}
 return MISSING;
}
__device__ inline int lookup_class(Context* c,int cid,int name,int* category){
 switch(cid){
''' + switches + r'''
 default:c->error=BAD_PROGRAM;
 }return MISSING;
}
__device__ inline int bind_method(Context* c,int callable,int owner){
 int refs[2]={callable,owner};int bound=make_tuple(&c->h,refs,2);int* n=mutable_node(c,bound);if(n)n[0]=18;return bound;
}
__device__ inline int class_ref(Context* c,int owner){
 int tag=kind(c,owner);if(tag==13)return owner;if(tag==29||tag==27)return edge(c,owner,0);
 if(tag==8)return '''+str(entity_ref('PlayerEntity'))+r''';
 if(tag==9)return '''+str(entity_ref('CharacterEntity'))+r''';
 if(tag==10)return '''+str(entity_ref('CardEntity'))+r''';
 return MISSING;
}
__device__ inline int object_attribute(Context* c,int owner,int name,int* getter){
 *getter=0;int tag=kind(c,owner);
 if(tag==14){int id=node(c,owner)[3];if(id<0||id>=c->module_count){c->error=BAD_PROGRAM;return MISSING;}
  return dict_get(&c->h,c->modules[id],name,MISSING);}
 int cls=class_ref(c,owner);if(cls==MISSING)return MISSING;
 if(kind(c,cls)!=13){c->error=BAD_PROGRAM;return MISSING;}
 int category=-1,found=lookup_class(c,node(c,cls)[3],name,&category);
 if(category==5 && tag!=13){
  int fn=dict_text_get(c,found,"fget");if(fn==MISSING||kind(c,fn)==TAG_NONE){c->error=UNSUPPORTED_TYPE;return MISSING;}
  *getter=1;return bind_method(c,fn,owner);
 }
 if(tag==29){int attrs=edge(c,owner,1),value=dict_get(&c->h,attrs,name,MISSING);if(value!=MISSING)return value;}
 if(found==MISSING)return MISSING;
 if(category==2 && tag!=13)return bind_method(c,found,owner);
 if(category==4)return bind_method(c,found,cls);
 return found;
}
__device__ inline void object_store(Context* c,int owner,int name,int value){
 if(kind(c,owner)!=29){c->error=UNSUPPORTED_TYPE;return;}
 int cls=class_ref(c,owner),category=-1;
 lookup_class(c,node(c,cls)[3],name,&category);
 if(category==5){c->error=UNSUPPORTED_TYPE;return;}
 int attrs=edge(c,owner,1);dict_set(&c->h,attrs,name,value);
}
'''
