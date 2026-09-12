/*
 * Project-authored Host-test RKNN API shim for rknn-frigate-compat.
 *
 * This header exists only so Host-side compilation and tests can build without
 * a Rockchip RKNN Runtime SDK. It is NOT Rockchip's rknn_api.h and must not be
 * treated as a vendor SDK substitute.
 *
 * Real Target builds must use the user-supplied Rockchip rknn_api.h (via
 * tools/build_rknn_metadata_probe.py --include-dir) and genuine librknnrt.
 */
#ifndef RKNN_FRIGATE_COMPAT_HOST_API_H
#define RKNN_FRIGATE_COMPAT_HOST_API_H

#include <stddef.h>
#include <stdint.h>

#define RKNN_MAX_DIMS 16
#define RKNN_SUCC 0
#define RKNN_QUERY_SDK_VERSION 1
#define RKNN_QUERY_IN_OUT_NUM 2
#define RKNN_QUERY_INPUT_ATTR 3
#define RKNN_QUERY_OUTPUT_ATTR 4

typedef uint64_t rknn_context;
typedef int rknn_query_cmd;
typedef struct { char api_version[256]; char drv_version[256]; } rknn_sdk_version;
typedef struct { uint32_t n_input; uint32_t n_output; } rknn_input_output_num;
typedef struct {
  uint32_t index; char name[256]; uint32_t n_dims; uint32_t dims[RKNN_MAX_DIMS];
  int fmt; int type; int qnt_type; int32_t zp; float scale; int8_t fl;
  uint32_t n_elems; uint32_t size; uint32_t w_stride; uint32_t size_with_stride;
} rknn_tensor_attr;

int rknn_init(rknn_context *, const char *, uint32_t, uint32_t, void *);
int rknn_query(rknn_context, rknn_query_cmd, void *, uint32_t);
int rknn_destroy(rknn_context);
const char *get_format_string(int);
const char *get_type_string(int);
const char *get_qnt_type_string(int);

#endif
