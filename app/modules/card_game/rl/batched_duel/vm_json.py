"""Device json.dumps for recovery own-deck canonical ordering.

Produces CPython json.dumps(..., sort_keys=True) for the tuple/list/dict/
numeric/string metadata used by recovery_sampler. Dumps produces ordinary
string nodes and allocates no private heap tags. Standard-library objects
use tags 70-79; NumPy objects use tags 60-69.

Hooks in namespace duel_rule_vm:

- json_call(c, callee, args, nargs, kw_names, kw_values, nkw, out)
  Handles T_NATIVE json.dumps. Supported kwargs: sort_keys, separators,
  ensure_ascii. Unknown kwargs, cycles, nonfinite floats, non-string dict
  keys, unsupported types and capacity failures are explicit.
  Uses integer_string/float_string; never calls Python json.dumps.

Primary: call json_call before T_NATIVE nkw rejection, next to numpy_call.
"""
from __future__ import annotations

from typing import Sequence

SUPPORTED_JSON_NAMES: tuple[str, ...] = ("json.dumps",)

JSON_INTEGRATION = (
    "from .vm_json import emit_json_source, SUPPORTED_JSON_NAMES; "
    "SUPPORTED_INTRINSICS |= frozenset(SUPPORTED_JSON_NAMES); "
    "insert emit_json_source(image.intrinsic_names) after numpy source; "
    "if(json_call(...)) before T_NATIVE nkw rejection. No private heap tags."
)


def emit_json_source(intrinsic_names: tuple[str, ...] | Sequence[str] = ()) -> str:
    names = tuple(intrinsic_names)
    native = names.index("json.dumps") if "json.dumps" in names else -8001
    src = r'''
const int NATIVE_JSON_DUMPS=@J@;
const int JSON_MAX_DEPTH=32,JSON_MAX_KEYS=64;
#ifndef T_NATIVE
#define T_NATIVE 15
#endif
__device__ inline int integer_string(Context* c,int ref,int plus);
__device__ inline int float_string(Context* c,int ref);
__device__ inline int numeric_int(Context* c,int ref,long long* value);
__device__ inline int vm_text_equals(Context* c,int ref,const char* text);
__device__ inline int json_fail_type(Context* c){c->error=UNSUPPORTED_TYPE;return 0;}
__device__ inline int json_fail_call(Context* c){c->error=BAD_CALL;return 0;}
__device__ inline int json_append(Context* c,int* off,int* n,const char* s,int len){
 if(len<0||*off<0||*n<0||(long long)(*off)+*n+len>c->h.max_payload){c->h.error=ERROR_PAYLOAD_CAPACITY;return 0;}
 for(int i=0;i<len;i++)c->h.payload[(*off)+(*n)+i]=(unsigned char)s[i];*n+=len;
 if(*off+*n>c->h.counts[2])c->h.counts[2]=*off+*n;return 1;
}
__device__ inline int json_append_ch(Context* c,int* off,int* n,char ch){char s[1];s[0]=ch;return json_append(c,off,n,s,1);}
__device__ inline int json_hex(int v){v&=15;return v<10?'0'+v:'a'+(v-10);}
__device__ inline int json_u16(Context* c,int* off,int* n,unsigned int cp){
 char b[6];b[0]='\\';b[1]='u';b[2]=(char)json_hex(cp>>12);b[3]=(char)json_hex(cp>>8);b[4]=(char)json_hex(cp>>4);b[5]=(char)json_hex(cp);
 return json_append(c,off,n,b,6);
}
__device__ inline int json_seen(const int* stack,int depth,int ref){for(int i=0;i<depth;i++)if(stack[i]==ref)return 1;return 0;}
__device__ inline int json_kw_only(Context* c,int names,int nkw){
 if(!nkw)return 1;if(names==MISSING||length(c,names)!=nkw)return json_fail_call(c);
 for(int i=0;i<nkw;i++){int nm=edge(c,names,i);int ok=vm_text_equals(c,nm,"sort_keys")||vm_text_equals(c,nm,"separators")||vm_text_equals(c,nm,"ensure_ascii");if(!ok)return json_fail_call(c);}return 1;
}
__device__ inline int json_kw_get(Context* c,int names,const int* vals,int nkw,const char* key){if(!nkw||names==MISSING||!vals)return MISSING;for(int i=0;i<nkw;i++)if(vm_text_equals(c,edge(c,names,i),key))return vals[i];return MISSING;}
__device__ inline int json_truth(Context* c,int ref,int* out){int t=kind(c,ref);if(t==TAG_BOOL){*out=node(c,ref)[3]?1:0;return 1;}if(t==TAG_INT){long long v=0;if(numeric_int(c,ref,&v)<0)return 0;*out=v!=0;return 1;}if(t==TAG_NONE){*out=0;return 1;}c->error=BAD_CALL;return 0;}
__device__ inline int json_read_seps(Context* c,int ref,int* item,int* key){
 if(kind(c,ref)!=TAG_TUPLE&&kind(c,ref)!=TAG_LIST)return json_fail_call(c);if(length(c,ref)!=2)return json_fail_call(c);
 int a=edge(c,ref,0),b=edge(c,ref,1);if(kind(c,a)!=TAG_STR||kind(c,b)!=TAG_STR)return json_fail_call(c);*item=a;*key=b;return 1;
}
__device__ inline int json_append_str_payload(Context* c,int* off,int* n,int ref){
 const int* nd=node(c,ref);if(!nd||nd[0]!=TAG_STR)return json_fail_type(c);
 NodeView v;if(!get_node(&c->h,ref,&v))return 0;const unsigned char* p=get_payload_ptr(&c->h,&v);
 if(v.count&&!p){c->error=BAD_PROGRAM;return 0;}return json_append(c,off,n,(const char*)p,v.count);
}
__device__ inline int json_escape(Context* c,int* off,int* n,int ref,int ensure_ascii){
 const int* nd=node(c,ref);if(!nd||nd[0]!=TAG_STR)return json_fail_type(c);
 NodeView v;if(!get_node(&c->h,ref,&v))return 0;const unsigned char* p=get_payload_ptr(&c->h,&v);
 if(v.count&&!p){c->error=BAD_PROGRAM;return 0;}
 if(!json_append_ch(c,off,n,'"'))return 0;
 int i=0;while(i<v.count){
  unsigned char b=p[i];unsigned int cp=0;int w=1;
  if(b<0x80){cp=b;w=1;}
  else if((b&0xE0)==0xC0&&i+1<v.count){cp=((b&0x1F)<<6)|(p[i+1]&0x3F);w=2;if(cp<0x80){c->h.error=ERROR_INVALID_UTF8;return 0;}}
  else if((b&0xF0)==0xE0&&i+2<v.count){cp=((b&0x0F)<<12)|((p[i+1]&0x3F)<<6)|(p[i+2]&0x3F);w=3;if(cp<0x800){c->h.error=ERROR_INVALID_UTF8;return 0;}}
  else if((b&0xF8)==0xF0&&i+3<v.count){cp=((b&0x07)<<18)|((p[i+1]&0x3F)<<12)|((p[i+2]&0x3F)<<6)|(p[i+3]&0x3F);w=4;if(cp<0x10000||cp>0x10FFFF){c->h.error=ERROR_INVALID_UTF8;return 0;}}
  else{c->h.error=ERROR_INVALID_UTF8;return 0;}
  i+=w;
  if(cp=='"'||cp=='\\'){if(!json_append_ch(c,off,n,'\\' )||!json_append_ch(c,off,n,(char)cp))return 0;}
  else if(cp=='\b'){if(!json_append(c,off,n,"\\b",2))return 0;}
  else if(cp=='\f'){if(!json_append(c,off,n,"\\f",2))return 0;}
  else if(cp=='\n'){if(!json_append(c,off,n,"\\n",2))return 0;}
  else if(cp=='\r'){if(!json_append(c,off,n,"\\r",2))return 0;}
  else if(cp=='\t'){if(!json_append(c,off,n,"\\t",2))return 0;}
  else if(cp<0x20){if(!json_u16(c,off,n,cp))return 0;}
  else if(ensure_ascii&&cp>=0x80){
   if(cp<=0xFFFF){if(!json_u16(c,off,n,cp))return 0;}
   else{unsigned int u=cp-0x10000;if(!json_u16(c,off,n,0xD800+((u>>10)&0x3FF))||!json_u16(c,off,n,0xDC00+(u&0x3FF)))return 0;}
  }else{
   for(int k=0;k<w;k++)if(!json_append_ch(c,off,n,(char)p[i-w+k]))return 0;
  }
 }
 return json_append_ch(c,off,n,'"');
}
__device__ inline int json_cmp_keys(Context* c,int a,int b){
 NodeView va,vb;if(!get_node(&c->h,a,&va)||!get_node(&c->h,b,&vb)||va.tag!=TAG_STR||vb.tag!=TAG_STR)return 0;
 const unsigned char* pa=get_payload_ptr(&c->h,&va);const unsigned char* pb=get_payload_ptr(&c->h,&vb);
 int n=va.count<vb.count?va.count:vb.count;int cmp=vm_memcmp(pa,pb,n);if(cmp)return cmp;return va.count-vb.count;
}
__device__ inline int json_encode(Context* c,int ref,int* off,int* n,int* stack,int depth,int sort_keys,int item_sep,int key_sep,int ensure_ascii);
__device__ inline int json_encode_seq(Context* c,int ref,int* off,int* n,int* stack,int depth,int sort_keys,int item_sep,int key_sep,int ensure_ascii){
 if(!json_append_ch(c,off,n,'['))return 0;int count=length(c,ref);
 for(int i=0;i<count;i++){if(i&&!json_append_str_payload(c,off,n,item_sep))return 0;if(!json_encode(c,edge(c,ref,i),off,n,stack,depth,sort_keys,item_sep,key_sep,ensure_ascii))return 0;}
 return json_append_ch(c,off,n,']');
}
__device__ inline int json_encode_dict(Context* c,int ref,int* off,int* n,int* stack,int depth,int sort_keys,int item_sep,int key_sep,int ensure_ascii){
 int count=length(c,ref);if(count>JSON_MAX_KEYS)return json_fail_type(c);
 int keys[64];for(int i=0;i<count;i++){int k=edge(c,ref,i);if(kind(c,k)!=TAG_STR)return json_fail_type(c);keys[i]=k;}
 if(sort_keys){for(int i=1;i<count;i++){int x=keys[i],j=i;while(j>0&&json_cmp_keys(c,keys[j-1],x)>0){keys[j]=keys[j-1];j--;}keys[j]=x;}}
 if(!json_append_ch(c,off,n,'{'))return 0;
 for(int i=0;i<count;i++){if(i&&!json_append_str_payload(c,off,n,item_sep))return 0;
  if(!json_escape(c,off,n,keys[i],ensure_ascii)||!json_append_str_payload(c,off,n,key_sep))return 0;
  int val=dict_get(&c->h,ref,keys[i],MISSING);if(val==MISSING)return json_fail_type(c);
  if(!json_encode(c,val,off,n,stack,depth,sort_keys,item_sep,key_sep,ensure_ascii))return 0;}
 return json_append_ch(c,off,n,'}');
}
__device__ inline int json_encode(Context* c,int ref,int* off,int* n,int* stack,int depth,int sort_keys,int item_sep,int key_sep,int ensure_ascii){
 if(c->error||c->h.error)return 0;int t=kind(c,ref);
 if(t==TAG_NONE)return json_append(c,off,n,"null",4);
 if(t==TAG_BOOL)return json_append(c,off,n,node(c,ref)[3]?"true":"false",node(c,ref)[3]?4:5);
 if(t==TAG_INT){int s=integer_string(c,ref,0);if(s==MISSING)return 0;return json_append_str_payload(c,off,n,s);}
 if(t==TAG_FLOAT){int s=float_string(c,ref);if(s==MISSING)return 0;return json_append_str_payload(c,off,n,s);}
 if(t==TAG_STR)return json_escape(c,off,n,ref,ensure_ascii);
 if(t==TAG_LIST||t==TAG_TUPLE||t==TAG_DICT){
  if(depth>=JSON_MAX_DEPTH)return json_fail_type(c);if(json_seen(stack,depth,ref))return json_fail_type(c);
  stack[depth]=ref;if(t==TAG_DICT)return json_encode_dict(c,ref,off,n,stack,depth+1,sort_keys,item_sep,key_sep,ensure_ascii);
  return json_encode_seq(c,ref,off,n,stack,depth+1,sort_keys,item_sep,key_sep,ensure_ascii);
 }
 return json_fail_type(c);
}
__device__ inline int json_dumps(Context* c,int obj,int sort_keys,int item_sep,int key_sep,int ensure_ascii,int* out){
 int off=c->h.counts[2],n=0,stack[32];
 if(!json_encode(c,obj,&off,&n,stack,0,sort_keys,item_sep,key_sep,ensure_ascii))return 0;
 if(c->h.counts[0]>=c->h.max_nodes){c->h.error=ERROR_NODE_CAPACITY;return 0;}
 int ref=c->h.counts[0]++;int* nd=c->h.nodes+ref*4;nd[0]=TAG_STR;nd[1]=off;nd[2]=n;nd[3]=0;c->h.counts[2]=off+n;*out=ref;return 1;
}
__device__ inline int json_call(Context* c,int callee,const int* args,int nargs,int kn,const int* kv,int nkw,int* out){
 if(kind(c,callee)!=T_NATIVE||node(c,callee)[3]!=NATIVE_JSON_DUMPS)return 0;*out=MISSING;
 if(nargs!=1||!json_kw_only(c,kn,nkw)){if(!c->error)c->error=BAD_CALL;return 1;}
 int sort_ref=json_kw_get(c,kn,kv,nkw,"sort_keys"),sep_ref=json_kw_get(c,kn,kv,nkw,"separators"),ascii_ref=json_kw_get(c,kn,kv,nkw,"ensure_ascii");
 int sort_keys=0;if(sort_ref!=MISSING&&!json_truth(c,sort_ref,&sort_keys))return 1;
 int ensure_ascii=1;if(ascii_ref!=MISSING&&!json_truth(c,ascii_ref,&ensure_ascii))return 1;
 int item_sep=MISSING,key_sep=MISSING;
 if(sep_ref!=MISSING){if(!json_read_seps(c,sep_ref,&item_sep,&key_sep))return 1;}
 else{item_sep=make_utf8(&c->h,", ",2);key_sep=make_utf8(&c->h,": ",2);if(item_sep==MISSING||key_sep==MISSING)return 1;}
 if(!json_dumps(c,args[0],sort_keys,item_sep,key_sep,ensure_ascii,out)){if(!c->error&&!c->h.error)c->error=UNSUPPORTED_TYPE;return 1;}
 return 1;
}
'''
    return src.replace("@J@", str(native))
