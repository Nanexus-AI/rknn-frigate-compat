#define _GNU_SOURCE
/* Read-only RKNN public metadata probe. It never calls rknn_run. */
#include <dlfcn.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef RKNN_FRIGATE_COMPAT_HOST_SHIM
#include "rknn_host_api.h"
#else
#include <rknn_api.h>
#endif
#include "rknn_probe_sha256.h"

typedef struct {
  char path[PATH_MAX];
  char sha256[65];
  int established;
} runtime_identity;

static void json_string(const char *value) {
  const unsigned char *p=(const unsigned char *)value; putchar('"');
  while(*p){switch(*p){case '"':fputs("\\\"",stdout);break;case '\\':fputs("\\\\",stdout);break;case '\n':fputs("\\n",stdout);break;case '\r':fputs("\\r",stdout);break;case '\t':fputs("\\t",stdout);break;default:if(*p<0x20)printf("\\u%04x",(unsigned)*p);else putchar((int)*p);}p++;} putchar('"');
}

static void json_bounded(const char *value,size_t capacity) {
  size_t i; putchar('"');
  for(i=0;i<capacity&&value[i]!='\0';i++){
    unsigned char byte=(unsigned char)value[i];
    switch(byte){case '"':fputs("\\\"",stdout);break;case '\\':fputs("\\\\",stdout);break;case '\n':fputs("\\n",stdout);break;case '\r':fputs("\\r",stdout);break;case '\t':fputs("\\t",stdout);break;default:if(byte<0x20)printf("\\u%04x",(unsigned)byte);else putchar((int)byte);}
  }
  putchar('"');
}

static int identify_runtime(runtime_identity *identity) {
  Dl_info info; char *resolved;
  memset(identity,0,sizeof(*identity)); memset(&info,0,sizeof(info));
  if(dladdr((const void *)rknn_init,&info)==0||info.dli_fname==NULL)return -1;
  resolved=realpath(info.dli_fname,identity->path); if(resolved==NULL)return -1;
  if(rfc_sha256_file(identity->path,identity->sha256)!=0)return -1;
  identity->established=1; return 0;
}

static void print_runtime(const runtime_identity *identity,const rknn_sdk_version *version,int have_version) {
  fputs("\"runtime\":{\"loaded_library_path\":",stdout);json_string(identity->path);
  fputs(",\"loaded_library_sha256\":",stdout);json_string(identity->sha256);
  if(have_version){fputs(",\"api_version\":",stdout);json_bounded(version->api_version,sizeof(version->api_version));fputs(",\"driver_version\":",stdout);json_bounded(version->drv_version,sizeof(version->drv_version));}
  putchar('}');
}

static void print_error(const char *stage,const char *code,const char *message,int native_code,int have_native,const runtime_identity *identity,const rknn_sdk_version *version,int have_version,int tensor_index) {
  fputs("{\"protocol_version\":1,\"status\":\"error\",\"error\":{\"stage\":",stdout);json_string(stage);
  fputs(",\"code\":",stdout);json_string(code);fputs(",\"message\":",stdout);json_string(message);
  if(have_native)printf(",\"native_code\":%d",native_code);else fputs(",\"native_code\":null",stdout);
  if(tensor_index>=0)printf(",\"tensor_index\":%d",tensor_index);
  putchar('}');
  if(identity!=NULL&&identity->established){putchar(',');print_runtime(identity,version,have_version);}
  fputs("}\n",stdout);
}

static void print_tensor(const rknn_tensor_attr *attr) {
  uint32_t i; fputs("{\"index\":",stdout);printf("%" PRIu32,attr->index);fputs(",\"name\":",stdout);json_bounded(attr->name,sizeof(attr->name));
  printf(",\"n_dims\":%" PRIu32 ",\"dims\":[",attr->n_dims);
  for(i=0;i<attr->n_dims&&i<RKNN_MAX_DIMS;i++){if(i)putchar(',');printf("%" PRIu32,attr->dims[i]);}
  fputs("],\"format\":",stdout);json_string(get_format_string(attr->fmt));fputs(",\"type\":",stdout);json_string(get_type_string(attr->type));
  fputs(",\"quantization_type\":",stdout);json_string(get_qnt_type_string(attr->qnt_type));
  printf(",\"zero_point\":%" PRId32 ",\"scale\":%.9g,\"fractional_length\":%d",attr->zp,(double)attr->scale,(int)attr->fl);
  printf(",\"element_count\":%" PRIu32 ",\"byte_size\":%" PRIu32 ",\"width_stride\":%" PRIu32 ",\"byte_size_with_stride\":%" PRIu32 "}",attr->n_elems,attr->size,attr->w_stride,attr->size_with_stride);
}

static int valid_tensor(const rknn_tensor_attr *attr) {
  uint32_t i;
  if(attr->n_dims==0||attr->n_dims>RKNN_MAX_DIMS||!isfinite((double)attr->scale))return 0;
  for(i=0;i<attr->n_dims;i++)if(attr->dims[i]==0)return 0;
  return get_format_string(attr->fmt)!=NULL&&get_type_string(attr->type)!=NULL&&get_qnt_type_string(attr->qnt_type)!=NULL;
}

static void print_success(const char *model_path,const runtime_identity *identity,const rknn_sdk_version *version,const rknn_input_output_num *count,const rknn_tensor_attr *inputs,const rknn_tensor_attr *outputs) {
  uint32_t i; fputs("{\"protocol_version\":1,\"status\":\"ok\",",stdout);print_runtime(identity,version,1);
  fputs(",\"model\":{\"model_path\":",stdout);json_string(model_path);printf(",\"input_count\":%" PRIu32 ",\"output_count\":%" PRIu32 ",\"inputs\":[",count->n_input,count->n_output);
  for(i=0;i<count->n_input;i++){if(i)putchar(',');print_tensor(&inputs[i]);}fputs("],\"outputs\":[",stdout);
  for(i=0;i<count->n_output;i++){if(i)putchar(',');print_tensor(&outputs[i]);}fputs("]}}\n",stdout);
}

int main(int argc,char **argv) {
  runtime_identity identity; rknn_context context=0; rknn_sdk_version version; rknn_input_output_num count;
  rknn_tensor_attr *inputs=NULL,*outputs=NULL; const char *stage=NULL,*code=NULL,*message=NULL; int native=0,have_native=0,tensor_index=-1,have_version=0,result; uint32_t i; FILE *model;
  if(argc!=2){print_error("startup","INVALID_ARGUMENTS","expected exactly one model path",0,0,NULL,NULL,0,-1);return 1;}
  if(identify_runtime(&identity)!=0){print_error("runtime_identity","IDENTITY_UNAVAILABLE","cannot identify loaded librknnrt.so",0,0,NULL,NULL,0,-1);return 1;}
  model=fopen(argv[1],"rb");if(model==NULL){print_error("model_open","MODEL_OPEN_FAILED","cannot open model read-only",0,0,&identity,NULL,0,-1);return 1;}fclose(model);
  result=rknn_init(&context,argv[1],0,0,NULL);if(result!=RKNN_SUCC){print_error("rknn_init","RKNN_INIT_FAILED","rknn_init failed",result,1,&identity,NULL,0,-1);return 1;}
  memset(&version,0,sizeof(version));result=rknn_query(context,RKNN_QUERY_SDK_VERSION,&version,sizeof(version));
  if(result!=RKNN_SUCC){stage="query_sdk_version";code="QUERY_SDK_VERSION_FAILED";message="SDK version query failed";native=result;have_native=1;goto cleanup;}have_version=1;
  memset(&count,0,sizeof(count));result=rknn_query(context,RKNN_QUERY_IN_OUT_NUM,&count,sizeof(count));
  if(result!=RKNN_SUCC){stage="query_io_count";code="QUERY_IO_COUNT_FAILED";message="input/output count query failed";native=result;have_native=1;goto cleanup;}
  inputs=calloc(count.n_input?count.n_input:1,sizeof(*inputs));outputs=calloc(count.n_output?count.n_output:1,sizeof(*outputs));
  if(inputs==NULL||outputs==NULL){stage="serialization";code="ALLOCATION_FAILED";message="metadata allocation failed";goto cleanup;}
  for(i=0;i<count.n_input;i++){inputs[i].index=i;result=rknn_query(context,RKNN_QUERY_INPUT_ATTR,&inputs[i],sizeof(inputs[i]));if(result!=RKNN_SUCC){stage="query_input_attr";code="QUERY_INPUT_ATTR_FAILED";message="input attribute query failed";native=result;have_native=1;tensor_index=(int)i;goto cleanup;}}
  for(i=0;i<count.n_output;i++){outputs[i].index=i;result=rknn_query(context,RKNN_QUERY_OUTPUT_ATTR,&outputs[i],sizeof(outputs[i]));if(result!=RKNN_SUCC){stage="query_output_attr";code="QUERY_OUTPUT_ATTR_FAILED";message="output attribute query failed";native=result;have_native=1;tensor_index=(int)i;goto cleanup;}}
  if(version.api_version[0]=='\0'){stage="serialization";code="INVALID_METADATA";message="Runtime returned an empty API version";goto cleanup;}
  for(i=0;i<count.n_input;i++)if(!valid_tensor(&inputs[i])){stage="serialization";code="INVALID_METADATA";message="Runtime returned invalid input metadata";tensor_index=-1;goto cleanup;}
  for(i=0;i<count.n_output;i++)if(!valid_tensor(&outputs[i])){stage="serialization";code="INVALID_METADATA";message="Runtime returned invalid output metadata";tensor_index=-1;goto cleanup;}
cleanup:
  result=rknn_destroy(context);
  if(stage==NULL&&result!=RKNN_SUCC){stage="destroy";code="RKNN_DESTROY_FAILED";message="rknn_destroy failed";native=result;have_native=1;}
  if(stage!=NULL)print_error(stage,code,message,native,have_native,&identity,&version,have_version,tensor_index);else print_success(argv[1],&identity,&version,&count,inputs,outputs);
  free(inputs);free(outputs);
  if(ferror(stdout))return 1;
  return stage==NULL?0:1;
}
