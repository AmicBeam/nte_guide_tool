"""Device scalar strings and f-string operations for frozen rule programs.

Uses pinned Ryu for exact shortest finite binary64 digits and Python's display
thresholds. Supports plain scalar formatting and integer d/+d; generic object
repr, encoding conversions and other format specifications fail explicitly.
"""
STRING_OPS=frozenset(('FORMAT_VALUE','BUILD_STRING'))
STRING_INTRINSICS=('str','enum.Enum.__str__')


def emit_string_source(intrinsic_names):
    string_id=intrinsic_names.index('str') if 'str' in intrinsic_names else -5001
    enum_id=intrinsic_names.index('enum.Enum.__str__') if 'enum.Enum.__str__' in intrinsic_names else -5002
    source=r'''
const int NATIVE_STR=@STR@,NATIVE_ENUM_STR=@ENUM@;
__device__ inline int integer_string(Context* c,int ref,int plus){
 if(kind(c,ref)==TAG_BOOL){char chars[2]={'+' ,node(c,ref)[3]?'1':'0'};return make_utf8(&c->h,chars+(plus?0:1),plus?2:1);}
 NodeView n;if(!get_node(&c->h,ref,&n)||n.tag!=TAG_INT||n.count<1){c->error=BAD_CALL;return MISSING;}
 if(n.count>256){c->error=UNSUPPORTED_TYPE;return MISSING;}
 const unsigned char* p=get_payload_ptr(&c->h,&n);unsigned char work[256];int negative=(p[n.count-1]&128)!=0,carry=negative?1:0;
 for(int i=0;i<n.count;i++){int value=negative?255-p[i]+carry:p[i];work[i]=(unsigned char)(value&255);if(negative)carry=value>>8;}
 int width=n.count;while(width&&work[width-1]==0)width--;
 char reverse[624],result[624];int count=0;
 do{
  int remainder=0;
  for(int i=width-1;i>=0;i--){int value=remainder*256+work[i];work[i]=(unsigned char)(value/10);remainder=value%10;}
  reverse[count++]=(char)('0'+remainder);while(width&&work[width-1]==0)width--;
 }while(width&&count<623);
 if(width){c->error=UNSUPPORTED_TYPE;return MISSING;}
 int out=0;if(negative)result[out++]='-';else if(plus)result[out++]='+';
 for(int i=count-1;i>=0;i--)result[out++]=reverse[i];return make_utf8(&c->h,result,out);
}
__device__ inline int float_string(Context* c,int ref){
 double value=0;if(read_float64(&c->h,ref,&value)<0)return MISSING;
 unsigned int row=duel_ryu::row_index();if(row>=duel_ryu::MAX_ROWS){c->error=UNSUPPORTED_TYPE;return MISSING;}
 duel_ryu::error_flags[row]=0;char raw[32];int size=duel_ryu::d2s_buffered_n(value,raw);
 if(duel_ryu::error_flags[row]||size<3||size>31){c->error=BAD_PROGRAM;return MISSING;}
 int cursor=0,negative=raw[0]=='-';if(negative)cursor++;
 char digits[18];int count=0;
 while(cursor<size&&raw[cursor]!='E'){
  char ch=raw[cursor++];if(ch=='.')continue;
  if(ch<'0'||ch>'9'||count>=17){c->error=BAD_PROGRAM;return MISSING;}digits[count++]=ch;
 }
 if(!count||cursor>=size){c->error=BAD_PROGRAM;return MISSING;}cursor++;
 int sign=1;if(cursor<size&&(raw[cursor]=='-'||raw[cursor]=='+')){sign=raw[cursor]=='-'?-1:1;cursor++;}
 int exponent=0;
 for(;cursor<size;cursor++){char ch=raw[cursor];if(ch<'0'||ch>'9'){c->error=BAD_PROGRAM;return MISSING;}if(exponent>1000){c->error=BAD_PROGRAM;return MISSING;}exponent=exponent*10+ch-'0';}
 exponent*=sign;if(exponent< -400||exponent>400){c->error=BAD_PROGRAM;return MISSING;}char result[48];int out=0;if(negative)result[out++]='-';
 if(exponent>= -4&&exponent<16){
  int point=exponent+1;
  if(point<=0){result[out++]='0';result[out++]='.';for(int i=0;i< -point;i++)result[out++]='0';for(int i=0;i<count;i++)result[out++]=digits[i];}
  else if(point>=count){for(int i=0;i<count;i++)result[out++]=digits[i];for(int i=count;i<point;i++)result[out++]='0';result[out++]='.';result[out++]='0';}
  else{for(int i=0;i<count;i++){if(i==point)result[out++]='.';result[out++]=digits[i];}}
 }else{
  result[out++]=digits[0];if(count>1){result[out++]='.';for(int i=1;i<count;i++)result[out++]=digits[i];}
  result[out++]='e';result[out++]=exponent<0?'-':'+';int absolute=exponent<0?-exponent:exponent;
  if(absolute>=100)result[out++]=(char)('0'+absolute/100);
  result[out++]=(char)('0'+absolute/10%10);result[out++]=(char)('0'+absolute%10);
 }
 return make_utf8(&c->h,result,out);
}
__device__ inline int scalar_string(Context* c,int ref){
 int type=kind(c,ref);
 if(type==TAG_STR)return ref;
 if(type==TAG_NONE)return make_utf8(&c->h,"None",4);
 if(type==TAG_BOOL)return node(c,ref)[3]?make_utf8(&c->h,"True",4):make_utf8(&c->h,"False",5);
 if(type==TAG_INT)return integer_string(c,ref,0);
 if(type==TAG_FLOAT)return float_string(c,ref);
 if(type==27)return edge(c,ref,3);
 if(type==28)return decimal_string(c,ref);
 c->error=UNSUPPORTED_TYPE;return MISSING;
}
__device__ inline int string_call(Context* c,int callable,const int* args,int nargs,int* out){
 if(kind(c,callable)!=T_NATIVE)return 0;int id=node(c,callable)[3];
 if(id==NATIVE_STR){if(nargs==0)*out=make_utf8(&c->h,"",0);else if(nargs==1)*out=scalar_string(c,args[0]);else c->error=BAD_CALL;return 1;}
 if(id==NATIVE_ENUM_STR){if(nargs!=1||kind(c,args[0])!=27)c->error=UNSUPPORTED_TYPE;else *out=edge(c,args[0],3);return 1;}
 return 0;
}
__device__ inline int handle_strings(Context* c,int* f,int op,int arg){
 if(op==OP_BUILD_STRING){
  if(arg<0||arg>f[2]){c->error=BAD_PROGRAM;return 1;}int ref=concat_strings(c,f+SBASE+f[2]-arg,arg);f[2]-=arg;push(c,f,ref);return 1;
 }
 if(op!=OP_FORMAT_VALUE)return 0;
 if(arg&~7){c->error=UNSUPPORTED_OP;return 1;}
 int spec=arg&4?pop(c,f):MISSING,value=pop(c,f),conversion=arg&3;
 if(conversion>=2){c->error=UNSUPPORTED_TYPE;return 1;}
 if(conversion==1)value=scalar_string(c,value);
 if(c->error||c->h.error)return 1;
 if(spec==MISSING||(kind(c,spec)==TAG_STR&&node(c,spec)[2]==0)){
  if(kind(c,value)==27&&conversion==0){c->error=UNSUPPORTED_TYPE;return 1;}
  push(c,f,scalar_string(c,value));return 1;
 }
 if(kind(c,spec)!=TAG_STR){c->error=BAD_CALL;return 1;}
 if(vm_text_equals(c,spec,"d")||vm_text_equals(c,spec,"+d"))push(c,f,integer_string(c,value,vm_text_equals(c,spec,"+d")));
 else c->error=UNSUPPORTED_TYPE;
 return 1;
}
'''
    return source.replace('@STR@',str(string_id)).replace('@ENUM@',str(enum_id))
