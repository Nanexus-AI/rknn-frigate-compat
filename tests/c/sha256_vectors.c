#include <stdio.h>
#include <string.h>
#include "../../tools/rknn_probe_sha256.h"

int main(void) {
  rfc_sha256_ctx context; unsigned char digest[32]; char hex[65]; unsigned i;
  (void)rfc_sha256_file;
  rfc_sha256_init(&context);
  rfc_sha256_final(&context,digest);
  for(i=0;i<32;i++)sprintf(hex+i*2,"%02x",digest[i]);
  hex[64]='\0';
  if(strcmp(hex,"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")!=0)return 1;
  rfc_sha256_init(&context);
  rfc_sha256_update(&context,(const unsigned char *)"abc",3);
  rfc_sha256_final(&context,digest);
  for(i=0;i<32;i++)sprintf(hex+i*2,"%02x",digest[i]);
  hex[64]='\0';
  return strcmp(hex,"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")!=0;
}
