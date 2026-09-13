"""Experimental handwritten combat kernel from the 2026-09-14 probe.

Not a business-rule source. Used only to check that IR-generated CUDA matches
this slice on the same layout, inputs, and launch shape.
"""

HANDWRITTEN_FN = 'combat_batch'
HANDWRITTEN_SOURCE = r'''
extern "C" __global__ void combat_batch(const int* input, int* output, int n, int steps) {
 int row=blockIdx.x*blockDim.x+threadIdx.x;
 if(row>=n)return;
 int x[19];
 #pragma unroll
 for(int j=0;j<19;j++)x[j]=input[row*19+j];
 for(int k=0;k<steps;k++){
  int b=k%2;
  int ah=b?3:0, sh=b?4:1, at=b?5:2, dh=b?0:3, ds=b?1:4, da=b?2:5;
  int ph=b?6:8, ps=b?7:9, af=b?11:10, df=b?10:11, ad=b?13:12, dd=b?12:13;
  if(x[14] || x[ah]<=0 || !x[af])continue;
  bool front=x[df]!=0;
  int amount=max(0,x[at]+x[18]);
  int counter=front && !x[16]?x[da]:0;
  int th=front?x[dh]:x[ph],ts=front?x[ds]:x[ps];
  int absorbed=min(ts,amount), nh=max(0,th-amount+absorbed),ns=ts-absorbed;
  int sa=min(x[sh],counter),selfhp=max(0,x[ah]-counter+sa),selfsh=x[sh]-sa;
  int overflow=front && x[17]?max(0,amount-ts-th):0;
  int pa=min(x[ps],overflow);
  x[ph]=front?max(0,x[ph]-overflow+pa):nh;
  x[ps]=front?x[ps]-pa:ns;
  x[ah]=selfhp;x[sh]=selfsh;
  if(front){x[dh]=nh;x[ds]=ns;}
  if(!selfhp){x[sh]=0;x[af]=0;x[ad]=3;}
  if(front && !nh){x[ds]=0;x[df]=0;x[dd]=3;}
  bool adead=x[6]<=0,bdead=x[8]<=0;
  if(adead||bdead){x[14]=1;x[15]=adead?(bdead?2:1):0;}
 }
 #pragma unroll
 for(int j=0;j<19;j++)output[row*19+j]=x[j];
}
'''
