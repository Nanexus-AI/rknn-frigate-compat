#include <stdlib.h>
#include <string.h>
#include "rknn_host_api.h"

static int failing(const char *name) {
  const char *value=getenv("RKNN_STUB_FAIL");
  return value!=NULL&&strstr(value,name)!=NULL;
}

int rknn_init(rknn_context *context,const char *path,uint32_t flags,uint32_t extend,void *config) {
  (void)path;(void)flags;(void)extend;(void)config;
  if(failing("rknn_init"))return -10;
  *context=1;
  return RKNN_SUCC;
}

int rknn_query(rknn_context context,rknn_query_cmd command,void *data,uint32_t size) {
  (void)context;(void)size;
  if(command==RKNN_QUERY_SDK_VERSION){rknn_sdk_version *value=data;if(failing("query_sdk_version"))return -11;strcpy(value->api_version,"stub-2.3.2");strcpy(value->drv_version,"stub-driver");return 0;}
  if(command==RKNN_QUERY_IN_OUT_NUM){rknn_input_output_num *value=data;if(failing("query_io_count"))return -12;value->n_input=1;value->n_output=2;return 0;}
  if(command==RKNN_QUERY_INPUT_ATTR&&failing("query_input_attr"))return -13;
  if(command==RKNN_QUERY_OUTPUT_ATTR&&failing("query_output_attr"))return -14;
  {
    rknn_tensor_attr *value=data;uint32_t index=value->index;memset(value,0,sizeof(*value));value->index=index;value->n_dims=4;value->fmt=command==RKNN_QUERY_INPUT_ATTR?1:2;value->type=1;value->qnt_type=1;value->scale=0.1f;
    if(command==RKNN_QUERY_INPUT_ATTR){strcpy(value->name,"input");value->dims[0]=1;value->dims[1]=640;value->dims[2]=640;value->dims[3]=3;value->n_elems=1228800;value->w_stride=640;}
    else{strcpy(value->name,index?"output1":"output0");value->dims[0]=1;value->dims[1]=255;value->dims[2]=index?40:80;value->dims[3]=index?40:80;value->n_elems=index?408000:1632000;value->w_stride=value->dims[3];}
    value->size=value->n_elems;value->size_with_stride=value->size;return 0;
  }
}

int rknn_destroy(rknn_context context){(void)context;return failing("destroy")?-15:0;}
const char *get_format_string(int value){return value==1?"NHWC":"NCHW";}
const char *get_type_string(int value){(void)value;return "INT8";}
const char *get_qnt_type_string(int value){(void)value;return "AFFINE";}
