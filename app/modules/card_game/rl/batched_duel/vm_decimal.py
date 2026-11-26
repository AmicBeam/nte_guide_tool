"""Exact, bounded Decimal primitives for the CUDA rule VM.

Decimal nodes retain image ABI tag 28 and one canonical string edge. Finite
values have a signed int64 coefficient and an exponent in [-36, 36]. Parsing,
quantize(ROUND_HALF_UP), integer conversion and comparisons use integer math.
The default Decimal context's precision is sufficient for this coefficient
range; alternate contexts and rounding modes are explicitly unsupported.

Float construction is supported only when its *exact binary value* has a
Decimal representation inside these bounds. Float output uses one correctly
rounded binary64 division with exactly representable integer operands, each
at most 2**53; wider conversions fail rather than approximate the rounding.
NaN, infinity, Unicode digits, underscores and contexts are unsupported.
This emitter neither initializes CUDA nor changes the formal rule engine.
"""

SUPPORTED_DECIMAL_NAMES = ('decimal.Decimal',)


def emit_decimal_source(intrinsic_names=()) -> str:
    names = tuple(intrinsic_names)
    decimal_id = names.index('decimal.Decimal') if 'decimal.Decimal' in names else -6001
    return _SOURCE.replace('@DECIMAL_ID@', str(decimal_id))


_SOURCE = r'''
const int NATIVE_DECIMAL=@DECIMAL_ID@;
const int T_DECIMAL=28,T_DECIMAL_BOUND=49;
__device__ inline int numeric_int(Context* c,int ref,long long* value);
__device__ inline int integer_string(Context* c,int ref,int plus);
__device__ inline int vm_text_equals(Context* c,int ref,const char* text);

struct VmDecimal { long long coefficient; int exponent; int negative; };
__device__ inline unsigned long long decimal_magnitude(long long value){
 return value<0?0ULL-(unsigned long long)value:(unsigned long long)value;
}
__device__ inline int decimal_fail(Context* c){c->error=UNSUPPORTED_TYPE;return 0;}
__device__ inline int decimal_assign(Context* c,VmDecimal* out,unsigned long long value,int exponent,int negative){
 unsigned long long limit=negative?9223372036854775808ULL:9223372036854775807ULL;
 if(value>limit||exponent< -36||exponent>36)return decimal_fail(c);
 out->coefficient=negative?(long long)(0ULL-value):(long long)value;
 out->exponent=exponent;out->negative=negative;return 1;
}
__device__ inline unsigned long long decimal_pow10(int exponent){
 unsigned long long value=1;for(int i=0;i<exponent;i++)value*=10ULL;return value;
}
__device__ inline int decimal_space(unsigned char ch){
 return ch==' '||ch=='\t'||ch=='\n'||ch=='\r'||ch=='\f'||ch=='\v';
}
__device__ inline int decimal_parse(Context* c,int text,VmDecimal* out){
 NodeView view;if(!get_node(&c->h,text,&view)){c->error=BAD_PROGRAM;return 0;}
 if(view.tag!=TAG_STR||view.count<1||view.count>256)return decimal_fail(c);
 const unsigned char* p=get_payload_ptr(&c->h,&view);if(!p){c->error=BAD_PROGRAM;return 0;}
 int start=0,end=view.count;while(start<end&&decimal_space(p[start]))start++;
 while(end>start&&decimal_space(p[end-1]))end--;
 int negative=0;if(start<end&&(p[start]=='+'||p[start]=='-'))negative=p[start++]=='-';
 unsigned long long magnitude=0,limit=negative?9223372036854775808ULL:9223372036854775807ULL;
 int point=0,fraction=0,digits=0;
 while(start<end&&p[start]!='e'&&p[start]!='E'){
  unsigned char ch=p[start++];if(ch=='.'&&!point){point=1;continue;}
  if(ch<'0'||ch>'9')return decimal_fail(c);
  unsigned int digit=(unsigned int)(ch-'0');
  if(magnitude>(limit-digit)/10ULL)return decimal_fail(c);
  magnitude=magnitude*10ULL+digit;digits++;if(point)fraction++;
 }
 if(!digits)return decimal_fail(c);
 int exponent=0,sign=1;
 if(start<end){
  start++;if(start<end&&(p[start]=='+'||p[start]=='-'))sign=p[start++]=='-'?-1:1;
  int exponent_digits=0;
  while(start<end){unsigned char ch=p[start++];if(ch<'0'||ch>'9'||exponent>1000)return decimal_fail(c);
   exponent=exponent*10+ch-'0';exponent_digits++;
  }
  if(!exponent_digits)return decimal_fail(c);
 }
 return decimal_assign(c,out,magnitude,sign*exponent-fraction,negative);
}
__device__ inline int decimal_read(Context* c,int ref,VmDecimal* out){
 const int* n=node(c,ref);if(!n)return 0;
 if(n[0]!=T_DECIMAL||n[2]!=1){c->error=BAD_PROGRAM;return 0;}
 return decimal_parse(c,edge(c,ref,0),out);
}
// Decode IEEE binary64 exactly; never use its rounded display string.
__device__ inline int decimal_from_float(Context* c,int ref,VmDecimal* out){
 double value=0;if(read_float64(&c->h,ref,&value)<0)return 0;
 union Bits {double d;unsigned long long u;} bits;bits.d=value;
 int negative=(int)(bits.u>>63),biased=(int)((bits.u>>52)&2047ULL);
 unsigned long long coefficient=bits.u&0xFFFFFFFFFFFFFULL;
 if(biased==2047)return decimal_fail(c);
 if(!biased&&!coefficient)return decimal_assign(c,out,0,0,negative);
 int exponent=biased?biased-1075:-1074;if(biased)coefficient|=0x10000000000000ULL;
 while(exponent<0&&(coefficient&1ULL)==0ULL){coefficient>>=1;exponent++;}
 unsigned long long limit=negative?9223372036854775808ULL:9223372036854775807ULL;
 if(exponent>=0){
  if(exponent>63||coefficient>(limit>>exponent))return decimal_fail(c);
  return decimal_assign(c,out,coefficient<<exponent,0,negative);
 }
 if(exponent< -36)return decimal_fail(c);
 for(int i=0;i< -exponent;i++){if(coefficient>limit/5ULL)return decimal_fail(c);coefficient*=5ULL;}
 return decimal_assign(c,out,coefficient,exponent,negative);
}
__device__ inline int decimal_from_number(Context* c,int ref,VmDecimal* out){
 int type=kind(c,ref);if(type==T_DECIMAL)return decimal_read(c,ref,out);
 if(type==TAG_FLOAT)return decimal_from_float(c,ref,out);
 if(type==TAG_INT||type==TAG_BOOL){long long value=0;if(numeric_int(c,ref,&value)<0)return 0;
  out->coefficient=value;out->exponent=0;out->negative=value<0;return 1;}
 return decimal_fail(c);
}
__device__ inline int decimal_digits(unsigned long long value,char* digits){
 char reverse[20];int count=0;do{reverse[count++]=(char)('0'+value%10ULL);value/=10ULL;}while(value);
 for(int i=0;i<count;i++)digits[i]=reverse[count-i-1];return count;
}
__device__ inline int decimal_format(Context* c,const VmDecimal* value){
 char digits[20],result[96];int count=decimal_digits(decimal_magnitude(value->coefficient),digits);
 int adjusted=value->exponent+count-1,out=0;if(value->negative)result[out++]='-';
 // Python Decimal.__str__: fixed iff exponent <= 0 and adjusted >= -6.
 if(value->exponent<=0&&adjusted>= -6){
  int point=count+value->exponent;
  if(point<=0){result[out++]='0';result[out++]='.';for(int i=0;i< -point;i++)result[out++]='0';
   for(int i=0;i<count;i++)result[out++]=digits[i];}
  else{for(int i=0;i<count;i++){if(i==point)result[out++]='.';result[out++]=digits[i];}}
 }else{
  result[out++]=digits[0];if(count>1){result[out++]='.';for(int i=1;i<count;i++)result[out++]=digits[i];}
  result[out++]='E';result[out++]=adjusted<0?'-':'+';
  char exponent_digits[20];int n=decimal_digits((unsigned long long)(adjusted<0?-adjusted:adjusted),exponent_digits);
  for(int i=0;i<n;i++)result[out++]=exponent_digits[i];
 }
 return make_utf8(&c->h,result,out);
}
__device__ inline int decimal_make(Context* c,const VmDecimal* value){
 int text=decimal_format(c,value);if(c->error||c->h.error)return MISSING;
 int ref=make_tuple(&c->h,&text,1);if(c->h.error)return MISSING;
 int* n=mutable_node(c,ref);if(!n)return MISSING;n[0]=T_DECIMAL;return ref;
}
__device__ inline int decimal_string(Context* c,int ref){
 // Validate bounded finite input, including frozen Decimal constants.
 VmDecimal value;if(!decimal_read(c,ref,&value))return MISSING;
 return decimal_format(c,&value);
}
__device__ inline int decimal_quantize(Context* c,const VmDecimal* value,int target,VmDecimal* out){
 if(target< -36||target>36)return decimal_fail(c);
 unsigned long long magnitude=decimal_magnitude(value->coefficient);
 unsigned long long limit=value->negative?9223372036854775808ULL:9223372036854775807ULL;
 int shift=value->exponent-target;
 if(shift>0){
  if(magnitude){if(shift>18)return decimal_fail(c);unsigned long long factor=decimal_pow10(shift);
   if(magnitude>limit/factor)return decimal_fail(c);magnitude*=factor;}
 }else if(shift<0){
  int places=-shift;
  if(places>19)magnitude=0;
  else{unsigned long long factor=decimal_pow10(places),remainder=magnitude%factor;
   magnitude/=factor;if(remainder>=factor/2ULL)magnitude++;}
 }
 return decimal_assign(c,out,magnitude,target,value->negative);
}
__device__ inline int decimal_integer(Context* c,const VmDecimal* value,long long* out){
 unsigned long long magnitude=decimal_magnitude(value->coefficient);
 unsigned long long limit=value->negative?9223372036854775808ULL:9223372036854775807ULL;
 if(value->exponent>0&&magnitude){
  if(value->exponent>18)return decimal_fail(c);unsigned long long factor=decimal_pow10(value->exponent);
  if(magnitude>limit/factor)return decimal_fail(c);magnitude*=factor;
 }else if(value->exponent<0)magnitude=value->exponent< -19?0:magnitude/decimal_pow10(-value->exponent);
 *out=value->negative?(long long)(0ULL-magnitude):(long long)magnitude;return 1;
}
__device__ inline int decimal_float(Context* c,const VmDecimal* value,double* out){
 unsigned long long numerator=decimal_magnitude(value->coefficient);int exponent=value->exponent;
 if(!numerator){*out=value->negative?-0.0:0.0;return 1;}
 // Cancel decimal factors before requiring exactly represented operands.
 while(exponent<0&&numerator%10ULL==0ULL){numerator/=10ULL;exponent++;}
 if(numerator>9007199254740992ULL)return decimal_fail(c);
 if(exponent>0){
  if(exponent>15)return decimal_fail(c);unsigned long long factor=decimal_pow10(exponent);
  if(numerator>9007199254740992ULL/factor)return decimal_fail(c);numerator*=factor;
  *out=(double)numerator;
 }else{
  if(exponent< -15)return decimal_fail(c);unsigned long long denominator=decimal_pow10(-exponent);
  *out=__ddiv_rn((double)numerator,(double)denominator);
 }
 if(value->negative)*out=-*out;return 1;
}
__device__ inline int decimal_numeric(Context* c,int ref,int as_float,int* out){
 if(kind(c,ref)!=T_DECIMAL)return 0;VmDecimal value;
 if(!decimal_read(c,ref,&value)){*out=MISSING;return 1;}
 if(as_float){double converted=0;*out=decimal_float(c,&value,&converted)?make_float64(&c->h,converted):MISSING;}
 else{long long converted=0;*out=decimal_integer(c,&value,&converted)?make_int64(&c->h,converted):MISSING;}
 return 1;
}
__device__ inline int decimal_truth(Context* c,int ref){
 VmDecimal value;return decimal_read(c,ref,&value)&&value.coefficient!=0;
}
__device__ inline int decimal_cmp_values(const VmDecimal* a,const VmDecimal* b){
 unsigned long long aa=decimal_magnitude(a->coefficient),bb=decimal_magnitude(b->coefficient);
 if(!aa&&!bb)return 0;if(!aa)return b->negative?1:-1;if(!bb)return a->negative?-1:1;
 if(a->negative!=b->negative)return a->negative?-1:1;
 char ad[20],bd[20];int an=decimal_digits(aa,ad),bn=decimal_digits(bb,bd);
 int ax=a->exponent+an,bx=b->exponent+bn,cmp=ax<bx?-1:ax>bx?1:0;
 if(!cmp)for(int i=0;i<an||i<bn;i++){
  char ac=i<an?ad[i]:'0',bc=i<bn?bd[i]:'0';if(ac!=bc){cmp=ac<bc?-1:1;break;}
 }
 return a->negative?-cmp:cmp;
}
__device__ inline int decimal_comparison_values(Context* c,int a,int b,VmDecimal* av,VmDecimal* bv){
 return decimal_from_number(c,a,av)&&decimal_from_number(c,b,bv);
}
__device__ inline int decimal_equal(Context* c,int a,int b,int* out){
 int ta=kind(c,a),tb=kind(c,b);if(ta!=T_DECIMAL&&tb!=T_DECIMAL)return 0;
 *out=0;
 // Equality against ordinary nonnumeric types is False, matching Decimal.
 if((ta!=T_DECIMAL&&ta!=TAG_INT&&ta!=TAG_BOOL&&ta!=TAG_FLOAT)||
    (tb!=T_DECIMAL&&tb!=TAG_INT&&tb!=TAG_BOOL&&tb!=TAG_FLOAT))return 1;
 VmDecimal av,bv;if(decimal_comparison_values(c,a,b,&av,&bv))*out=decimal_cmp_values(&av,&bv)==0;
 return 1;
}
__device__ inline int decimal_compare(Context* c,int a,int b,int mode,int* out){
 if(kind(c,a)!=T_DECIMAL&&kind(c,b)!=T_DECIMAL)return 0;*out=0;
 VmDecimal av,bv;if(!decimal_comparison_values(c,a,b,&av,&bv))return 1;
 int cmp=decimal_cmp_values(&av,&bv);
 if(mode==0)*out=cmp<0;else if(mode==1)*out=cmp<=0;else if(mode==2)*out=cmp==0;
 else if(mode==3)*out=cmp!=0;else if(mode==4)*out=cmp>0;else if(mode==5)*out=cmp>=0;
 else c->error=UNSUPPORTED_OP;return 1;
}
__device__ inline int decimal_attribute(Context* c,int owner,int name){
 if(kind(c,owner)!=T_DECIMAL)return MISSING;
 if(!vm_text_equals(c,name,"quantize")){c->error=UNSUPPORTED_TYPE;return MISSING;}
 int ref=make_tuple(&c->h,&owner,1);if(c->h.error)return MISSING;
 int* n=mutable_node(c,ref);if(!n)return MISSING;n[0]=T_DECIMAL_BOUND;return ref;
}
__device__ inline int decimal_call(Context* c,int callee,const int* args,int nargs,int kw_names,const int* kw_values,int nkw,int* out){
 int type=kind(c,callee);if(type!=T_DECIMAL_BOUND&&(type!=T_NATIVE||node(c,callee)[3]!=NATIVE_DECIMAL))return 0;
 *out=MISSING;
 if(type==T_NATIVE){
  if(nargs>1||nkw){c->error=BAD_CALL;return 1;}
  VmDecimal value;
  if(!nargs){decimal_assign(c,&value,0,0,0);*out=decimal_make(c,&value);return 1;}
  if(kind(c,args[0])==T_DECIMAL){if(decimal_read(c,args[0],&value))*out=args[0];return 1;}
  int ok=kind(c,args[0])==TAG_STR?decimal_parse(c,args[0],&value):decimal_from_number(c,args[0],&value);
  if(ok)*out=decimal_make(c,&value);return 1;
 }
 // Explicit ROUND_HALF_UP is required: the default context uses HALF_EVEN.
 if(nargs<1||nargs>2||nkw<0||nkw>1){c->error=BAD_CALL;return 1;}
 int rounding=nargs==2?args[1]:MISSING;
 if(nkw){
  if(nargs==2||kw_names==MISSING||!kw_values||length(c,kw_names)!=1||
     !vm_text_equals(c,edge(c,kw_names,0),"rounding")){c->error=BAD_CALL;return 1;}
  rounding=kw_values[0];
 }
 if(rounding==MISSING||!vm_text_equals(c,rounding,"ROUND_HALF_UP")){c->error=UNSUPPORTED_TYPE;return 1;}
 if(kind(c,args[0])!=T_DECIMAL){c->error=BAD_CALL;return 1;}
 VmDecimal value,quantum,result;
 if(decimal_read(c,edge(c,callee,0),&value)&&decimal_read(c,args[0],&quantum)&&
    decimal_quantize(c,&value,quantum.exponent,&result))*out=decimal_make(c,&result);
 return 1;
}
'''
