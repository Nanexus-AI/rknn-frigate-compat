/*
 * Project-maintained SHA-256 implementation for rknn-frigate-compat.
 * Implements SHA-256 per FIPS PUB 180-4. Maintained as project source; not
 * presented as code copied from a named third-party library.
 */
#ifndef RKNN_PROBE_SHA256_H
#define RKNN_PROBE_SHA256_H

#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef struct {
  uint32_t state[8];
  uint64_t bits;
  unsigned char block[64];
  size_t used;
} rfc_sha256_ctx;

static uint32_t rfc_rotr(uint32_t value, unsigned shift) {
  return (value >> shift) | (value << (32U - shift));
}

static void rfc_sha256_transform(rfc_sha256_ctx *ctx, const unsigned char block[64]) {
  static const uint32_t constants[64] = {
    0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,
    0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,
    0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,
    0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,
    0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,
    0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,
    0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,
    0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U
  };
  uint32_t words[64], a,b,c,d,e,f,g,h;
  unsigned i;
  for (i=0;i<16;i++) words[i]=((uint32_t)block[i*4]<<24)|((uint32_t)block[i*4+1]<<16)|((uint32_t)block[i*4+2]<<8)|block[i*4+3];
  for (i=16;i<64;i++) {
    uint32_t s0=rfc_rotr(words[i-15],7)^rfc_rotr(words[i-15],18)^(words[i-15]>>3);
    uint32_t s1=rfc_rotr(words[i-2],17)^rfc_rotr(words[i-2],19)^(words[i-2]>>10);
    words[i]=words[i-16]+s0+words[i-7]+s1;
  }
  a=ctx->state[0]; b=ctx->state[1]; c=ctx->state[2]; d=ctx->state[3];
  e=ctx->state[4]; f=ctx->state[5]; g=ctx->state[6]; h=ctx->state[7];
  for (i=0;i<64;i++) {
    uint32_t s1=rfc_rotr(e,6)^rfc_rotr(e,11)^rfc_rotr(e,25);
    uint32_t choice=(e&f)^((~e)&g);
    uint32_t temp1=h+s1+choice+constants[i]+words[i];
    uint32_t s0=rfc_rotr(a,2)^rfc_rotr(a,13)^rfc_rotr(a,22);
    uint32_t majority=(a&b)^(a&c)^(b&c);
    uint32_t temp2=s0+majority;
    h=g; g=f; f=e; e=d+temp1; d=c; c=b; b=a; a=temp1+temp2;
  }
  ctx->state[0]+=a; ctx->state[1]+=b; ctx->state[2]+=c; ctx->state[3]+=d;
  ctx->state[4]+=e; ctx->state[5]+=f; ctx->state[6]+=g; ctx->state[7]+=h;
}

static void rfc_sha256_init(rfc_sha256_ctx *ctx) {
  static const uint32_t initial[8]={0x6a09e667U,0xbb67ae85U,0x3c6ef372U,0xa54ff53aU,0x510e527fU,0x9b05688cU,0x1f83d9abU,0x5be0cd19U};
  memcpy(ctx->state,initial,sizeof(initial)); ctx->bits=0; ctx->used=0;
}

static void rfc_sha256_update(rfc_sha256_ctx *ctx,const unsigned char *data,size_t length) {
  ctx->bits+=(uint64_t)length*8U;
  while(length>0){size_t take=64-ctx->used;if(take>length)take=length;memcpy(ctx->block+ctx->used,data,take);ctx->used+=take;data+=take;length-=take;if(ctx->used==64){rfc_sha256_transform(ctx,ctx->block);ctx->used=0;}}
}

static void rfc_sha256_final(rfc_sha256_ctx *ctx,unsigned char output[32]) {
  uint64_t bits=ctx->bits; unsigned i;
  ctx->block[ctx->used++]=0x80;
  if(ctx->used>56){while(ctx->used<64)ctx->block[ctx->used++]=0;rfc_sha256_transform(ctx,ctx->block);ctx->used=0;}
  while(ctx->used<56)ctx->block[ctx->used++]=0;
  for(i=0;i<8;i++)ctx->block[63-i]=(unsigned char)(bits>>(i*8));
  rfc_sha256_transform(ctx,ctx->block);
  for(i=0;i<8;i++){output[i*4]=(unsigned char)(ctx->state[i]>>24);output[i*4+1]=(unsigned char)(ctx->state[i]>>16);output[i*4+2]=(unsigned char)(ctx->state[i]>>8);output[i*4+3]=(unsigned char)ctx->state[i];}
}

static int rfc_sha256_file(const char *path,char hex[65]) {
  unsigned char buffer[32768],output[32]; size_t count; unsigned i;
  FILE *stream=fopen(path,"rb"); rfc_sha256_ctx ctx;
  if(stream==NULL)return -1;
  rfc_sha256_init(&ctx);
  while((count=fread(buffer,1,sizeof(buffer),stream))>0)rfc_sha256_update(&ctx,buffer,count);
  if(ferror(stream)||fclose(stream)!=0)return -1;
  rfc_sha256_final(&ctx,output);
  for(i=0;i<32;i++)sprintf(hex+i*2,"%02x",output[i]);
  hex[64]='\0'; return 0;
}

#endif
