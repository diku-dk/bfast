import sys
import numpy as np
import ctypes as ct
# Stub code for OpenCL setup.

import pyopencl as cl
import numpy as np
import sys

if cl.version.VERSION < (2015,2):
    raise Exception('Futhark requires at least PyOpenCL version 2015.2.  Installed version is %s.' %
                    cl.version.VERSION_TEXT)

def parse_preferred_device(s):
    pref_num = 0
    if len(s) > 1 and s[0] == '#':
        i = 1
        while i < len(s):
            if not s[i].isdigit():
                break
            else:
                pref_num = pref_num * 10 + int(s[i])
            i += 1
        while i < len(s) and s[i].isspace():
            i += 1
        return (s[i:], pref_num)
    else:
        return (s, 0)

def get_prefered_context(interactive=False, platform_pref=None, device_pref=None):
    if device_pref != None:
        (device_pref, device_num) = parse_preferred_device(device_pref)
    else:
        device_num = 0

    if interactive:
        return cl.create_some_context(interactive=True)

    def blacklisted(p, d):
        return platform_pref == None and device_pref == None and \
            p.name == "Apple" and d.name.find("Intel(R) Core(TM)") >= 0
    def platform_ok(p):
        return not platform_pref or p.name.find(platform_pref) >= 0
    def device_ok(d):
        return not device_pref or d.name.find(device_pref) >= 0

    device_matches = 0

    for p in cl.get_platforms():
        if not platform_ok(p):
            continue
        for d in p.get_devices():
            if blacklisted(p,d) or not device_ok(d):
                continue
            if device_matches == device_num:
                return cl.Context(devices=[d])
            else:
                device_matches += 1
    raise Exception('No OpenCL platform and device matching constraints found.')

def size_assignment(s):
    name, value = s.split('=')
    return (name, int(value))

def check_types(self, required_types):
    if 'f64' in required_types:
        if self.device.get_info(cl.device_info.PREFERRED_VECTOR_WIDTH_DOUBLE) == 0:
            raise Exception('Program uses double-precision floats, but this is not supported on chosen device: %s' % self.device.name)

def apply_size_heuristics(self, size_heuristics, sizes):
    for (platform_name, device_type, size, value) in size_heuristics:
        if sizes[size] == None \
           and self.platform.name.find(platform_name) >= 0 \
           and self.device.type == device_type:
               if type(value) == str:
                   sizes[size] = self.device.get_info(getattr(cl.device_info,value))
               else:
                   sizes[size] = value
    return sizes

def initialise_opencl_object(self,
                             program_src='',
                             command_queue=None,
                             interactive=False,
                             platform_pref=None,
                             device_pref=None,
                             default_group_size=None,
                             default_num_groups=None,
                             default_tile_size=None,
                             default_threshold=None,
                             size_heuristics=[],
                             required_types=[],
                             all_sizes={},
                             user_sizes={}):
    if command_queue is None:
        self.ctx = get_prefered_context(interactive, platform_pref, device_pref)
        self.queue = cl.CommandQueue(self.ctx)
    else:
        self.ctx = command_queue.context
        self.queue = command_queue
    self.device = self.queue.device
    self.platform = self.device.platform
    self.pool = cl.tools.MemoryPool(cl.tools.ImmediateAllocator(self.queue))
    device_type = self.device.type

    check_types(self, required_types)

    max_group_size = int(self.device.max_work_group_size)
    max_tile_size = int(np.sqrt(self.device.max_work_group_size))

    self.max_group_size = max_group_size
    self.max_tile_size = max_tile_size
    self.max_threshold = 0
    self.max_num_groups = 0
    self.max_local_memory = int(self.device.local_mem_size)
    self.free_list = {}

    if 'default_group_size' in sizes:
        default_group_size = sizes['default_group_size']
        del sizes['default_group_size']

    if 'default_num_groups' in sizes:
        default_num_groups = sizes['default_num_groups']
        del sizes['default_num_groups']

    if 'default_tile_size' in sizes:
        default_tile_size = sizes['default_tile_size']
        del sizes['default_tile_size']

    if 'default_threshold' in sizes:
        default_threshold = sizes['default_threshold']
        del sizes['default_threshold']

    default_group_size_set = default_group_size != None
    default_tile_size_set = default_tile_size != None
    default_sizes = apply_size_heuristics(self, size_heuristics,
                                          {'group_size': default_group_size,
                                           'tile_size': default_tile_size,
                                           'num_groups': default_num_groups,
                                           'lockstep_width': None,
                                           'threshold': default_threshold})
    default_group_size = default_sizes['group_size']
    default_num_groups = default_sizes['num_groups']
    default_threshold = default_sizes['threshold']
    default_tile_size = default_sizes['tile_size']
    lockstep_width = default_sizes['lockstep_width']

    if default_group_size > max_group_size:
        if default_group_size_set:
            sys.stderr.write('Note: Device limits group size to {} (down from {})\n'.
                             format(max_tile_size, default_group_size))
        default_group_size = max_group_size

    if default_tile_size > max_tile_size:
        if default_tile_size_set:
            sys.stderr.write('Note: Device limits tile size to {} (down from {})\n'.
                             format(max_tile_size, default_tile_size))
        default_tile_size = max_tile_size

    for (k,v) in user_sizes.items():
        if k in all_sizes:
            all_sizes[k]['value'] = v
        else:
            raise Exception('Unknown size: {}\nKnown sizes: {}'.format(k, ' '.join(all_sizes.keys())))

    self.sizes = {}
    for (k,v) in all_sizes.items():
        if v['class'] == 'group_size':
            max_value = max_group_size
            default_value = default_group_size
        elif v['class'] == 'num_groups':
            max_value = max_group_size # Intentional!
            default_value = default_num_groups
        elif v['class'] == 'tile_size':
            max_value = max_tile_size
            default_value = default_tile_size
        elif v['class'].startswith('threshold'):
            max_value = None
            default_value = default_threshold
        else:
            raise Exception('Unknown size class for size \'{}\': {}'.format(k, v['class']))
        if v['value'] == None:
            self.sizes[k] = default_value
        elif max_value != None and v['value'] > max_value:
            sys.stderr.write('Note: Device limits {} to {} (down from {}\n'.
                             format(k, max_value, v['value']))
            self.sizes[k] = max_value
        else:
            self.sizes[k] = v['value']

    # XXX: we perform only a subset of z-encoding here.  Really, the
    # compiler should provide us with the variables to which
    # parameters are mapped.
    if (len(program_src) >= 0):
        return cl.Program(self.ctx, program_src).build(
            ["-DLOCKSTEP_WIDTH={}".format(lockstep_width)]
            + ["-D{}={}".format(s.replace('z', 'zz').replace('.', 'zi'),v) for (s,v) in self.sizes.items()])

def opencl_alloc(self, min_size, tag):
    min_size = 1 if min_size == 0 else min_size
    assert min_size > 0
    return self.pool.allocate(min_size)

def opencl_free_all(self):
    self.pool.free_held()
import pyopencl.array
import time
import argparse
sizes = {}
synchronous = False
preferred_platform = None
preferred_device = None
default_threshold = None
default_group_size = None
default_num_groups = None
default_tile_size = None
fut_opencl_src = """#ifdef cl_clang_storage_class_specifiers
#pragma OPENCL EXTENSION cl_clang_storage_class_specifiers : enable
#endif
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
__kernel void dummy_kernel(__global unsigned char *dummy, int n)
{
    const int thread_gid = get_global_id(0);
    
    if (thread_gid >= n)
        return;
}
typedef char int8_t;
typedef short int16_t;
typedef int int32_t;
typedef long int64_t;
typedef uchar uint8_t;
typedef ushort uint16_t;
typedef uint uint32_t;
typedef ulong uint64_t;
#define ALIGNED_LOCAL_MEMORY(m,size) __local unsigned char m[size] __attribute__ ((align))
#ifdef cl_nv_pragma_unroll
static inline void mem_fence_global()
{
    asm("membar.gl;");
}
#else
static inline void mem_fence_global()
{
    mem_fence(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}
#endif
static inline void mem_fence_local()
{
    mem_fence(CLK_LOCAL_MEM_FENCE);
}
static inline int8_t add8(int8_t x, int8_t y)
{
    return x + y;
}
static inline int16_t add16(int16_t x, int16_t y)
{
    return x + y;
}
static inline int32_t add32(int32_t x, int32_t y)
{
    return x + y;
}
static inline int64_t add64(int64_t x, int64_t y)
{
    return x + y;
}
static inline int8_t sub8(int8_t x, int8_t y)
{
    return x - y;
}
static inline int16_t sub16(int16_t x, int16_t y)
{
    return x - y;
}
static inline int32_t sub32(int32_t x, int32_t y)
{
    return x - y;
}
static inline int64_t sub64(int64_t x, int64_t y)
{
    return x - y;
}
static inline int8_t mul8(int8_t x, int8_t y)
{
    return x * y;
}
static inline int16_t mul16(int16_t x, int16_t y)
{
    return x * y;
}
static inline int32_t mul32(int32_t x, int32_t y)
{
    return x * y;
}
static inline int64_t mul64(int64_t x, int64_t y)
{
    return x * y;
}
static inline uint8_t udiv8(uint8_t x, uint8_t y)
{
    return x / y;
}
static inline uint16_t udiv16(uint16_t x, uint16_t y)
{
    return x / y;
}
static inline uint32_t udiv32(uint32_t x, uint32_t y)
{
    return x / y;
}
static inline uint64_t udiv64(uint64_t x, uint64_t y)
{
    return x / y;
}
static inline uint8_t umod8(uint8_t x, uint8_t y)
{
    return x % y;
}
static inline uint16_t umod16(uint16_t x, uint16_t y)
{
    return x % y;
}
static inline uint32_t umod32(uint32_t x, uint32_t y)
{
    return x % y;
}
static inline uint64_t umod64(uint64_t x, uint64_t y)
{
    return x % y;
}
static inline int8_t sdiv8(int8_t x, int8_t y)
{
    int8_t q = x / y;
    int8_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int16_t sdiv16(int16_t x, int16_t y)
{
    int16_t q = x / y;
    int16_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int32_t sdiv32(int32_t x, int32_t y)
{
    int32_t q = x / y;
    int32_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int64_t sdiv64(int64_t x, int64_t y)
{
    int64_t q = x / y;
    int64_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int8_t smod8(int8_t x, int8_t y)
{
    int8_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int16_t smod16(int16_t x, int16_t y)
{
    int16_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int32_t smod32(int32_t x, int32_t y)
{
    int32_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int64_t smod64(int64_t x, int64_t y)
{
    int64_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int8_t squot8(int8_t x, int8_t y)
{
    return x / y;
}
static inline int16_t squot16(int16_t x, int16_t y)
{
    return x / y;
}
static inline int32_t squot32(int32_t x, int32_t y)
{
    return x / y;
}
static inline int64_t squot64(int64_t x, int64_t y)
{
    return x / y;
}
static inline int8_t srem8(int8_t x, int8_t y)
{
    return x % y;
}
static inline int16_t srem16(int16_t x, int16_t y)
{
    return x % y;
}
static inline int32_t srem32(int32_t x, int32_t y)
{
    return x % y;
}
static inline int64_t srem64(int64_t x, int64_t y)
{
    return x % y;
}
static inline int8_t smin8(int8_t x, int8_t y)
{
    return x < y ? x : y;
}
static inline int16_t smin16(int16_t x, int16_t y)
{
    return x < y ? x : y;
}
static inline int32_t smin32(int32_t x, int32_t y)
{
    return x < y ? x : y;
}
static inline int64_t smin64(int64_t x, int64_t y)
{
    return x < y ? x : y;
}
static inline uint8_t umin8(uint8_t x, uint8_t y)
{
    return x < y ? x : y;
}
static inline uint16_t umin16(uint16_t x, uint16_t y)
{
    return x < y ? x : y;
}
static inline uint32_t umin32(uint32_t x, uint32_t y)
{
    return x < y ? x : y;
}
static inline uint64_t umin64(uint64_t x, uint64_t y)
{
    return x < y ? x : y;
}
static inline int8_t smax8(int8_t x, int8_t y)
{
    return x < y ? y : x;
}
static inline int16_t smax16(int16_t x, int16_t y)
{
    return x < y ? y : x;
}
static inline int32_t smax32(int32_t x, int32_t y)
{
    return x < y ? y : x;
}
static inline int64_t smax64(int64_t x, int64_t y)
{
    return x < y ? y : x;
}
static inline uint8_t umax8(uint8_t x, uint8_t y)
{
    return x < y ? y : x;
}
static inline uint16_t umax16(uint16_t x, uint16_t y)
{
    return x < y ? y : x;
}
static inline uint32_t umax32(uint32_t x, uint32_t y)
{
    return x < y ? y : x;
}
static inline uint64_t umax64(uint64_t x, uint64_t y)
{
    return x < y ? y : x;
}
static inline uint8_t shl8(uint8_t x, uint8_t y)
{
    return x << y;
}
static inline uint16_t shl16(uint16_t x, uint16_t y)
{
    return x << y;
}
static inline uint32_t shl32(uint32_t x, uint32_t y)
{
    return x << y;
}
static inline uint64_t shl64(uint64_t x, uint64_t y)
{
    return x << y;
}
static inline uint8_t lshr8(uint8_t x, uint8_t y)
{
    return x >> y;
}
static inline uint16_t lshr16(uint16_t x, uint16_t y)
{
    return x >> y;
}
static inline uint32_t lshr32(uint32_t x, uint32_t y)
{
    return x >> y;
}
static inline uint64_t lshr64(uint64_t x, uint64_t y)
{
    return x >> y;
}
static inline int8_t ashr8(int8_t x, int8_t y)
{
    return x >> y;
}
static inline int16_t ashr16(int16_t x, int16_t y)
{
    return x >> y;
}
static inline int32_t ashr32(int32_t x, int32_t y)
{
    return x >> y;
}
static inline int64_t ashr64(int64_t x, int64_t y)
{
    return x >> y;
}
static inline uint8_t and8(uint8_t x, uint8_t y)
{
    return x & y;
}
static inline uint16_t and16(uint16_t x, uint16_t y)
{
    return x & y;
}
static inline uint32_t and32(uint32_t x, uint32_t y)
{
    return x & y;
}
static inline uint64_t and64(uint64_t x, uint64_t y)
{
    return x & y;
}
static inline uint8_t or8(uint8_t x, uint8_t y)
{
    return x | y;
}
static inline uint16_t or16(uint16_t x, uint16_t y)
{
    return x | y;
}
static inline uint32_t or32(uint32_t x, uint32_t y)
{
    return x | y;
}
static inline uint64_t or64(uint64_t x, uint64_t y)
{
    return x | y;
}
static inline uint8_t xor8(uint8_t x, uint8_t y)
{
    return x ^ y;
}
static inline uint16_t xor16(uint16_t x, uint16_t y)
{
    return x ^ y;
}
static inline uint32_t xor32(uint32_t x, uint32_t y)
{
    return x ^ y;
}
static inline uint64_t xor64(uint64_t x, uint64_t y)
{
    return x ^ y;
}
static inline char ult8(uint8_t x, uint8_t y)
{
    return x < y;
}
static inline char ult16(uint16_t x, uint16_t y)
{
    return x < y;
}
static inline char ult32(uint32_t x, uint32_t y)
{
    return x < y;
}
static inline char ult64(uint64_t x, uint64_t y)
{
    return x < y;
}
static inline char ule8(uint8_t x, uint8_t y)
{
    return x <= y;
}
static inline char ule16(uint16_t x, uint16_t y)
{
    return x <= y;
}
static inline char ule32(uint32_t x, uint32_t y)
{
    return x <= y;
}
static inline char ule64(uint64_t x, uint64_t y)
{
    return x <= y;
}
static inline char slt8(int8_t x, int8_t y)
{
    return x < y;
}
static inline char slt16(int16_t x, int16_t y)
{
    return x < y;
}
static inline char slt32(int32_t x, int32_t y)
{
    return x < y;
}
static inline char slt64(int64_t x, int64_t y)
{
    return x < y;
}
static inline char sle8(int8_t x, int8_t y)
{
    return x <= y;
}
static inline char sle16(int16_t x, int16_t y)
{
    return x <= y;
}
static inline char sle32(int32_t x, int32_t y)
{
    return x <= y;
}
static inline char sle64(int64_t x, int64_t y)
{
    return x <= y;
}
static inline int8_t pow8(int8_t x, int8_t y)
{
    int8_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int16_t pow16(int16_t x, int16_t y)
{
    int16_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int32_t pow32(int32_t x, int32_t y)
{
    int32_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int64_t pow64(int64_t x, int64_t y)
{
    int64_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline bool itob_i8_bool(int8_t x)
{
    return x;
}
static inline bool itob_i16_bool(int16_t x)
{
    return x;
}
static inline bool itob_i32_bool(int32_t x)
{
    return x;
}
static inline bool itob_i64_bool(int64_t x)
{
    return x;
}
static inline int8_t btoi_bool_i8(bool x)
{
    return x;
}
static inline int16_t btoi_bool_i16(bool x)
{
    return x;
}
static inline int32_t btoi_bool_i32(bool x)
{
    return x;
}
static inline int64_t btoi_bool_i64(bool x)
{
    return x;
}
#define sext_i8_i8(x) ((int8_t) (int8_t) x)
#define sext_i8_i16(x) ((int16_t) (int8_t) x)
#define sext_i8_i32(x) ((int32_t) (int8_t) x)
#define sext_i8_i64(x) ((int64_t) (int8_t) x)
#define sext_i16_i8(x) ((int8_t) (int16_t) x)
#define sext_i16_i16(x) ((int16_t) (int16_t) x)
#define sext_i16_i32(x) ((int32_t) (int16_t) x)
#define sext_i16_i64(x) ((int64_t) (int16_t) x)
#define sext_i32_i8(x) ((int8_t) (int32_t) x)
#define sext_i32_i16(x) ((int16_t) (int32_t) x)
#define sext_i32_i32(x) ((int32_t) (int32_t) x)
#define sext_i32_i64(x) ((int64_t) (int32_t) x)
#define sext_i64_i8(x) ((int8_t) (int64_t) x)
#define sext_i64_i16(x) ((int16_t) (int64_t) x)
#define sext_i64_i32(x) ((int32_t) (int64_t) x)
#define sext_i64_i64(x) ((int64_t) (int64_t) x)
#define zext_i8_i8(x) ((uint8_t) (uint8_t) x)
#define zext_i8_i16(x) ((uint16_t) (uint8_t) x)
#define zext_i8_i32(x) ((uint32_t) (uint8_t) x)
#define zext_i8_i64(x) ((uint64_t) (uint8_t) x)
#define zext_i16_i8(x) ((uint8_t) (uint16_t) x)
#define zext_i16_i16(x) ((uint16_t) (uint16_t) x)
#define zext_i16_i32(x) ((uint32_t) (uint16_t) x)
#define zext_i16_i64(x) ((uint64_t) (uint16_t) x)
#define zext_i32_i8(x) ((uint8_t) (uint32_t) x)
#define zext_i32_i16(x) ((uint16_t) (uint32_t) x)
#define zext_i32_i32(x) ((uint32_t) (uint32_t) x)
#define zext_i32_i64(x) ((uint64_t) (uint32_t) x)
#define zext_i64_i8(x) ((uint8_t) (uint64_t) x)
#define zext_i64_i16(x) ((uint16_t) (uint64_t) x)
#define zext_i64_i32(x) ((uint32_t) (uint64_t) x)
#define zext_i64_i64(x) ((uint64_t) (uint64_t) x)
static inline float fdiv32(float x, float y)
{
    return x / y;
}
static inline float fadd32(float x, float y)
{
    return x + y;
}
static inline float fsub32(float x, float y)
{
    return x - y;
}
static inline float fmul32(float x, float y)
{
    return x * y;
}
static inline float fmin32(float x, float y)
{
    return x < y ? x : y;
}
static inline float fmax32(float x, float y)
{
    return x < y ? y : x;
}
static inline float fpow32(float x, float y)
{
    return pow(x, y);
}
static inline char cmplt32(float x, float y)
{
    return x < y;
}
static inline char cmple32(float x, float y)
{
    return x <= y;
}
static inline float sitofp_i8_f32(int8_t x)
{
    return x;
}
static inline float sitofp_i16_f32(int16_t x)
{
    return x;
}
static inline float sitofp_i32_f32(int32_t x)
{
    return x;
}
static inline float sitofp_i64_f32(int64_t x)
{
    return x;
}
static inline float uitofp_i8_f32(uint8_t x)
{
    return x;
}
static inline float uitofp_i16_f32(uint16_t x)
{
    return x;
}
static inline float uitofp_i32_f32(uint32_t x)
{
    return x;
}
static inline float uitofp_i64_f32(uint64_t x)
{
    return x;
}
static inline int8_t fptosi_f32_i8(float x)
{
    return x;
}
static inline int16_t fptosi_f32_i16(float x)
{
    return x;
}
static inline int32_t fptosi_f32_i32(float x)
{
    return x;
}
static inline int64_t fptosi_f32_i64(float x)
{
    return x;
}
static inline uint8_t fptoui_f32_i8(float x)
{
    return x;
}
static inline uint16_t fptoui_f32_i16(float x)
{
    return x;
}
static inline uint32_t fptoui_f32_i32(float x)
{
    return x;
}
static inline uint64_t fptoui_f32_i64(float x)
{
    return x;
}
static inline float futrts_log32(float x)
{
    return log(x);
}
static inline float futrts_log2_32(float x)
{
    return log2(x);
}
static inline float futrts_log10_32(float x)
{
    return log10(x);
}
static inline float futrts_sqrt32(float x)
{
    return sqrt(x);
}
static inline float futrts_exp32(float x)
{
    return exp(x);
}
static inline float futrts_cos32(float x)
{
    return cos(x);
}
static inline float futrts_sin32(float x)
{
    return sin(x);
}
static inline float futrts_tan32(float x)
{
    return tan(x);
}
static inline float futrts_acos32(float x)
{
    return acos(x);
}
static inline float futrts_asin32(float x)
{
    return asin(x);
}
static inline float futrts_atan32(float x)
{
    return atan(x);
}
static inline float futrts_atan2_32(float x, float y)
{
    return atan2(x, y);
}
static inline float futrts_round32(float x)
{
    return rint(x);
}
static inline char futrts_isnan32(float x)
{
    return isnan(x);
}
static inline char futrts_isinf32(float x)
{
    return isinf(x);
}
static inline int32_t futrts_to_bits32(float x)
{
    union {
        float f;
        int32_t t;
    } p;
    
    p.f = x;
    return p.t;
}
static inline float futrts_from_bits32(int32_t x)
{
    union {
        int32_t f;
        float t;
    } p;
    
    p.f = x;
    return p.t;
}
__kernel void copy_37930(int32_t sizze_30757, int32_t res_30780,
                         int32_t j_30912, int32_t j_m_i_30913, __global
                         unsigned char *mem_37330, __global
                         unsigned char *mem_37343)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t copy_gtid_37930;
    int32_t copy_ltid_37931;
    int32_t copy_gid_37932;
    
    copy_gtid_37930 = get_global_id(0);
    copy_ltid_37931 = get_local_id(0);
    copy_gid_37932 = get_group_id(0);
    if (slt32(copy_gtid_37930, sizze_30757 * (res_30780 * j_m_i_30913))) {
        *(__global float *) &mem_37343[(squot32(copy_gtid_37930, res_30780 *
                                                j_m_i_30913) * (j_m_i_30913 *
                                                                res_30780) +
                                        squot32(copy_gtid_37930 -
                                                squot32(copy_gtid_37930,
                                                        res_30780 *
                                                        j_m_i_30913) *
                                                (res_30780 * j_m_i_30913),
                                                j_m_i_30913) * j_m_i_30913 +
                                        (copy_gtid_37930 -
                                         squot32(copy_gtid_37930, res_30780 *
                                                 j_m_i_30913) * (res_30780 *
                                                                 j_m_i_30913) -
                                         squot32(copy_gtid_37930 -
                                                 squot32(copy_gtid_37930,
                                                         res_30780 *
                                                         j_m_i_30913) *
                                                 (res_30780 * j_m_i_30913),
                                                 j_m_i_30913) * j_m_i_30913)) *
                                       4] = *(__global
                                              float *) &mem_37330[(res_30780 +
                                                                   (squot32(copy_gtid_37930,
                                                                            res_30780 *
                                                                            j_m_i_30913) *
                                                                    (j_30912 *
                                                                     res_30780) +
                                                                    squot32(copy_gtid_37930 -
                                                                            squot32(copy_gtid_37930,
                                                                                    res_30780 *
                                                                                    j_m_i_30913) *
                                                                            (res_30780 *
                                                                             j_m_i_30913),
                                                                            j_m_i_30913) *
                                                                    j_30912 +
                                                                    (copy_gtid_37930 -
                                                                     squot32(copy_gtid_37930,
                                                                             res_30780 *
                                                                             j_m_i_30913) *
                                                                     (res_30780 *
                                                                      j_m_i_30913) -
                                                                     squot32(copy_gtid_37930 -
                                                                             squot32(copy_gtid_37930,
                                                                                     res_30780 *
                                                                                     j_m_i_30913) *
                                                                             (res_30780 *
                                                                              j_m_i_30913),
                                                                             j_m_i_30913) *
                                                                     j_m_i_30913))) *
                                                                  4];
    }
}
__kernel void copy_37996(int32_t sizze_30757, int32_t res_30780,
                         int32_t j_m_i_30913, __global
                         unsigned char *res_mem_37344, __global
                         unsigned char *mem_37402)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t copy_gtid_37996;
    int32_t copy_ltid_37997;
    int32_t copy_gid_37998;
    
    copy_gtid_37996 = get_global_id(0);
    copy_ltid_37997 = get_local_id(0);
    copy_gid_37998 = get_group_id(0);
    if (slt32(copy_gtid_37996, sizze_30757 * (res_30780 * j_m_i_30913))) {
        *(__global float *) &mem_37402[((copy_gtid_37996 -
                                         squot32(copy_gtid_37996, res_30780 *
                                                 j_m_i_30913) * (res_30780 *
                                                                 j_m_i_30913) -
                                         squot32(copy_gtid_37996 -
                                                 squot32(copy_gtid_37996,
                                                         res_30780 *
                                                         j_m_i_30913) *
                                                 (res_30780 * j_m_i_30913),
                                                 j_m_i_30913) * j_m_i_30913) *
                                        (sizze_30757 * res_30780) +
                                        squot32(copy_gtid_37996 -
                                                squot32(copy_gtid_37996,
                                                        res_30780 *
                                                        j_m_i_30913) *
                                                (res_30780 * j_m_i_30913),
                                                j_m_i_30913) * sizze_30757 +
                                        squot32(copy_gtid_37996, res_30780 *
                                                j_m_i_30913)) * 4] = *(__global
                                                                       float *) &res_mem_37344[(squot32(copy_gtid_37996,
                                                                                                        res_30780 *
                                                                                                        j_m_i_30913) *
                                                                                                (j_m_i_30913 *
                                                                                                 res_30780) +
                                                                                                squot32(copy_gtid_37996 -
                                                                                                        squot32(copy_gtid_37996,
                                                                                                                res_30780 *
                                                                                                                j_m_i_30913) *
                                                                                                        (res_30780 *
                                                                                                         j_m_i_30913),
                                                                                                        j_m_i_30913) *
                                                                                                j_m_i_30913 +
                                                                                                (copy_gtid_37996 -
                                                                                                 squot32(copy_gtid_37996,
                                                                                                         res_30780 *
                                                                                                         j_m_i_30913) *
                                                                                                 (res_30780 *
                                                                                                  j_m_i_30913) -
                                                                                                 squot32(copy_gtid_37996 -
                                                                                                         squot32(copy_gtid_37996,
                                                                                                                 res_30780 *
                                                                                                                 j_m_i_30913) *
                                                                                                         (res_30780 *
                                                                                                          j_m_i_30913),
                                                                                                         j_m_i_30913) *
                                                                                                 j_m_i_30913)) *
                                                                                               4];
    }
}
__kernel void copy_38192(int32_t sizze_30756, int32_t sizze_30757,
                         int32_t i_31033, __global unsigned char *mem_37577,
                         __global unsigned char *mem_37584)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t copy_gtid_38192;
    int32_t copy_ltid_38193;
    int32_t copy_gid_38194;
    
    copy_gtid_38192 = get_global_id(0);
    copy_ltid_38193 = get_local_id(0);
    copy_gid_38194 = get_group_id(0);
    if (slt32(copy_gtid_38192, sizze_30757)) {
        *(__global int32_t *) &mem_37584[copy_gtid_38192 * 4] = *(__global
                                                                  int32_t *) &mem_37577[(i_31033 +
                                                                                         copy_gtid_38192 *
                                                                                         sizze_30756) *
                                                                                        4];
    }
}
__kernel void map_31399(int32_t sizze_30742, int32_t sizze_30743,
                        int32_t sizze_30744, int16_t nan_value_30745, __global
                        unsigned char *images_mem_37200, __global
                        unsigned char *mem_37205)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_31399;
    int32_t local_tid_31400;
    int32_t group_sizze_37798;
    int32_t wave_sizze_37797;
    int32_t group_id_31401;
    
    global_tid_31399 = get_global_id(0);
    local_tid_31400 = get_local_id(0);
    group_sizze_37798 = get_local_size(0);
    wave_sizze_37797 = LOCKSTEP_WIDTH;
    group_id_31401 = get_group_id(0);
    
    int32_t gtid_31388;
    int32_t gtid_31389;
    int32_t gtid_31390;
    
    gtid_31388 = squot32(global_tid_31399, sizze_30743 * sizze_30744);
    gtid_31389 = squot32(global_tid_31399 - squot32(global_tid_31399,
                                                    sizze_30743 * sizze_30744) *
                         (sizze_30743 * sizze_30744), sizze_30744);
    gtid_31390 = global_tid_31399 - squot32(global_tid_31399, sizze_30743 *
                                            sizze_30744) * (sizze_30743 *
                                                            sizze_30744) -
        squot32(global_tid_31399 - squot32(global_tid_31399, sizze_30743 *
                                           sizze_30744) * (sizze_30743 *
                                                           sizze_30744),
                sizze_30744) * sizze_30744;
    
    int16_t x_31462;
    bool cond_31463;
    float res_31464;
    
    if ((slt32(gtid_31388, sizze_30742) && slt32(gtid_31389, sizze_30743)) &&
        slt32(gtid_31390, sizze_30744)) {
        x_31462 = *(__global int16_t *) &images_mem_37200[(gtid_31388 *
                                                           (sizze_30744 *
                                                            sizze_30743) +
                                                           gtid_31389 *
                                                           sizze_30744 +
                                                           gtid_31390) * 2];
        cond_31463 = x_31462 == nan_value_30745;
        if (cond_31463) {
            res_31464 = NAN;
        } else {
            float res_31465 = sitofp_i16_f32(x_31462);
            
            res_31464 = res_31465;
        }
    }
    if ((slt32(gtid_31388, sizze_30742) && slt32(gtid_31389, sizze_30743)) &&
        slt32(gtid_31390, sizze_30744)) {
        *(__global float *) &mem_37205[(gtid_31388 * (sizze_30744 *
                                                      sizze_30743) +
                                        gtid_31389 * sizze_30744 + gtid_31390) *
                                       4] = res_31464;
    }
}
__kernel void map_31576(int32_t sizze_30756, float freq_30762,
                        int32_t res_30780, __global
                        unsigned char *mappingindices_mem_37200, __global
                        unsigned char *mem_37205)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_31576;
    int32_t local_tid_31577;
    int32_t group_sizze_37804;
    int32_t wave_sizze_37803;
    int32_t group_id_31578;
    
    global_tid_31576 = get_global_id(0);
    local_tid_31577 = get_local_id(0);
    group_sizze_37804 = get_local_size(0);
    wave_sizze_37803 = LOCKSTEP_WIDTH;
    group_id_31578 = get_group_id(0);
    
    int32_t gtid_31567;
    int32_t gtid_31568;
    
    gtid_31567 = squot32(global_tid_31576, sizze_30756);
    gtid_31568 = global_tid_31576 - squot32(global_tid_31576, sizze_30756) *
        sizze_30756;
    
    bool index_primexp_36457;
    bool index_primexp_36456;
    int32_t cmpop_x_36454;
    bool index_primexp_36455;
    int32_t convop_x_36451;
    float binop_y_36452;
    float index_primexp_36453;
    int32_t x_31667;
    float res_31668;
    
    if (slt32(gtid_31567, res_30780) && slt32(gtid_31568, sizze_30756)) {
        index_primexp_36457 = gtid_31567 == 0;
        index_primexp_36456 = gtid_31567 == 1;
        cmpop_x_36454 = smod32(gtid_31567, 2);
        index_primexp_36455 = cmpop_x_36454 == 0;
        convop_x_36451 = sdiv32(gtid_31567, 2);
        binop_y_36452 = sitofp_i32_f32(convop_x_36451);
        index_primexp_36453 = 6.2831855F * binop_y_36452;
        x_31667 = *(__global int32_t *) &mappingindices_mem_37200[gtid_31568 *
                                                                  4];
        if (index_primexp_36457) {
            res_31668 = 1.0F;
        } else {
            float res_31669;
            
            if (index_primexp_36456) {
                float res_31670 = sitofp_i32_f32(x_31667);
                
                res_31669 = res_31670;
            } else {
                float res_31671;
                float x_31672;
                float res_31673;
                float res_31674;
                
                res_31671 = sitofp_i32_f32(x_31667);
                x_31672 = res_31671 * index_primexp_36453;
                res_31673 = x_31672 / freq_30762;
                if (index_primexp_36455) {
                    float res_31675;
                    
                    res_31675 = futrts_sin32(res_31673);
                    res_31674 = res_31675;
                } else {
                    float res_31676;
                    
                    res_31676 = futrts_cos32(res_31673);
                    res_31674 = res_31676;
                }
                res_31669 = res_31674;
            }
            res_31668 = res_31669;
        }
    }
    if (slt32(gtid_31567, res_30780) && slt32(gtid_31568, sizze_30756)) {
        *(__global float *) &mem_37205[(gtid_31567 * sizze_30756 + gtid_31568) *
                                       4] = res_31668;
    }
}
__kernel void map_31778(int32_t sizze_30756, float freq_30762,
                        int32_t res_30780, __global
                        unsigned char *mappingindices_mem_37200, __global
                        unsigned char *mem_37209)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_31778;
    int32_t local_tid_31779;
    int32_t group_sizze_37806;
    int32_t wave_sizze_37805;
    int32_t group_id_31780;
    
    global_tid_31778 = get_global_id(0);
    local_tid_31779 = get_local_id(0);
    group_sizze_37806 = get_local_size(0);
    wave_sizze_37805 = LOCKSTEP_WIDTH;
    group_id_31780 = get_group_id(0);
    
    int32_t gtid_31769;
    int32_t gtid_31770;
    
    gtid_31769 = squot32(global_tid_31778, sizze_30756);
    gtid_31770 = global_tid_31778 - squot32(global_tid_31778, sizze_30756) *
        sizze_30756;
    
    bool index_primexp_36465;
    int32_t binop_x_36462;
    int32_t cmpop_x_36463;
    bool index_primexp_36464;
    int32_t convop_x_36459;
    float binop_y_36460;
    float index_primexp_36461;
    int32_t x_31861;
    float res_31862;
    
    if (slt32(gtid_31769, res_30780) && slt32(gtid_31770, sizze_30756)) {
        index_primexp_36465 = gtid_31769 == 0;
        binop_x_36462 = 1 + gtid_31769;
        cmpop_x_36463 = smod32(binop_x_36462, 2);
        index_primexp_36464 = cmpop_x_36463 == 0;
        convop_x_36459 = sdiv32(binop_x_36462, 2);
        binop_y_36460 = sitofp_i32_f32(convop_x_36459);
        index_primexp_36461 = 6.2831855F * binop_y_36460;
        x_31861 = *(__global int32_t *) &mappingindices_mem_37200[gtid_31770 *
                                                                  4];
        if (index_primexp_36465) {
            res_31862 = 1.0F;
        } else {
            float res_31863;
            float x_31864;
            float res_31865;
            float res_31866;
            
            res_31863 = sitofp_i32_f32(x_31861);
            x_31864 = res_31863 * index_primexp_36461;
            res_31865 = x_31864 / freq_30762;
            if (index_primexp_36464) {
                float res_31867;
                
                res_31867 = futrts_sin32(res_31865);
                res_31866 = res_31867;
            } else {
                float res_31868;
                
                res_31868 = futrts_cos32(res_31865);
                res_31866 = res_31868;
            }
            res_31862 = res_31866;
        }
    }
    if (slt32(gtid_31769, res_30780) && slt32(gtid_31770, sizze_30756)) {
        *(__global float *) &mem_37209[(gtid_31769 * sizze_30756 + gtid_31770) *
                                       4] = res_31862;
    }
}
__kernel void map_31932(int32_t sizze_30756, int32_t res_30780, float res_30853,
                        __global unsigned char *mem_37214, __global
                        unsigned char *mem_37218)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_31932;
    int32_t local_tid_31933;
    int32_t group_sizze_37808;
    int32_t wave_sizze_37807;
    int32_t group_id_31934;
    
    global_tid_31932 = get_global_id(0);
    local_tid_31933 = get_local_id(0);
    group_sizze_37808 = get_local_size(0);
    wave_sizze_37807 = LOCKSTEP_WIDTH;
    group_id_31934 = get_group_id(0);
    
    int32_t gtid_31923;
    int32_t gtid_31924;
    
    gtid_31923 = squot32(global_tid_31932, res_30780);
    gtid_31924 = global_tid_31932 - squot32(global_tid_31932, res_30780) *
        res_30780;
    
    float x_31960;
    float res_31961;
    
    if (slt32(gtid_31923, sizze_30756) && slt32(gtid_31924, res_30780)) {
        x_31960 = *(__global float *) &mem_37214[(gtid_31923 * res_30780 +
                                                  gtid_31924) * 4];
        res_31961 = res_30853 + x_31960;
    }
    if (slt32(gtid_31923, sizze_30756) && slt32(gtid_31924, res_30780)) {
        *(__global float *) &mem_37218[(gtid_31923 * res_30780 + gtid_31924) *
                                       4] = res_31961;
    }
}
__kernel void map_31999(int32_t sizze_30756, int32_t sizze_30757,
                        int32_t n_30761, int32_t res_30780, __global
                        unsigned char *arg_mem_37210, __global
                        unsigned char *mem_37218, __global
                        unsigned char *mem_37222, __global
                        unsigned char *mem_37226, __global
                        unsigned char *mem_37237)
{
    const int32_t group_sizze_32028 = mainzigroup_sizze_31993;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_31999;
    int32_t local_tid_32000;
    int32_t group_sizze_37810;
    int32_t wave_sizze_37809;
    int32_t group_id_32001;
    
    global_tid_31999 = get_global_id(0);
    local_tid_32000 = get_local_id(0);
    group_sizze_37810 = get_local_size(0);
    wave_sizze_37809 = LOCKSTEP_WIDTH;
    group_id_32001 = get_group_id(0);
    
    int32_t gtid_31992;
    
    gtid_31992 = global_tid_31999;
    if (slt32(gtid_31992, sizze_30757)) {
        for (int32_t i_32038 = 0; i_32038 < res_30780; i_32038++) {
            for (int32_t i_32043 = 0; i_32043 < res_30780; i_32043++) {
                float res_32045;
                float redout_32046 = 0.0F;
                
                for (int32_t i_32047 = 0; i_32047 < n_30761; i_32047++) {
                    float x_32048;
                    float x_32049;
                    float x_32050;
                    float x_32051;
                    bool res_32052;
                    float y_32053;
                    float res_32054;
                    float res_32057;
                    
                    x_32048 = *(__global float *) &mem_37222[(i_32047 *
                                                              sizze_30757 +
                                                              gtid_31992) * 4];
                    x_32049 = *(__global float *) &arg_mem_37210[(i_32038 *
                                                                  sizze_30756 +
                                                                  i_32047) * 4];
                    x_32050 = *(__global float *) &mem_37218[(i_32047 *
                                                              res_30780 +
                                                              i_32043) * 4];
                    x_32051 = x_32049 * x_32050;
                    res_32052 = futrts_isnan32(x_32048);
                    if (res_32052) {
                        y_32053 = 0.0F;
                    } else {
                        y_32053 = 1.0F;
                    }
                    res_32054 = x_32051 * y_32053;
                    res_32057 = redout_32046 + res_32054;
                    
                    float redout_tmp_37813 = res_32057;
                    
                    redout_32046 = redout_tmp_37813;
                }
                res_32045 = redout_32046;
                *(__global float *) &mem_37226[(group_id_32001 *
                                                (group_sizze_32028 * res_30780 *
                                                 res_30780) + local_tid_32000 +
                                                i_32038 * (group_sizze_32028 *
                                                           res_30780) +
                                                i_32043 * group_sizze_32028) *
                                               4] = res_32045;
            }
        }
    }
    if (slt32(gtid_31992, sizze_30757)) {
        for (int32_t i_37814 = 0; i_37814 < res_30780; i_37814++) {
            for (int32_t i_37815 = 0; i_37815 < res_30780; i_37815++) {
                *(__global float *) &mem_37237[(sizze_30757 * res_30780 * 0 +
                                                gtid_31992 + (i_37814 *
                                                              (sizze_30757 *
                                                               res_30780) +
                                                              i_37815 *
                                                              sizze_30757)) *
                                               4] = *(__global
                                                      float *) &mem_37226[(group_id_32001 *
                                                                           (group_sizze_32028 *
                                                                            res_30780 *
                                                                            res_30780) +
                                                                           local_tid_32000 +
                                                                           (i_37814 *
                                                                            (group_sizze_32028 *
                                                                             res_30780) +
                                                                            i_37815 *
                                                                            group_sizze_32028)) *
                                                                          4];
            }
        }
    }
}
__kernel void map_32103(int32_t sizze_30757, int32_t sizze_30758,
                        int32_t n_30761, int32_t res_30780, __global
                        unsigned char *images_mem_37201, __global
                        unsigned char *mem_37214, __global
                        unsigned char *mem_37218, __global
                        unsigned char *mem_37258, __global
                        unsigned char *mem_37263)
{
    const int32_t group_sizze_32456 = mainzigroup_sizze_32097;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_32103;
    int32_t local_tid_32104;
    int32_t group_sizze_37825;
    int32_t wave_sizze_37824;
    int32_t group_id_32105;
    
    global_tid_32103 = get_global_id(0);
    local_tid_32104 = get_local_id(0);
    group_sizze_37825 = get_local_size(0);
    wave_sizze_37824 = LOCKSTEP_WIDTH;
    group_id_32105 = get_group_id(0);
    
    int32_t gtid_32094;
    int32_t gtid_32095;
    
    gtid_32094 = squot32(global_tid_32103, res_30780);
    gtid_32095 = global_tid_32103 - squot32(global_tid_32103, res_30780) *
        res_30780;
    if (slt32(gtid_32094, sizze_30757) && slt32(gtid_32095, res_30780)) {
        for (int32_t i_32475 = 0; i_32475 < res_30780; i_32475++) {
            float res_32477;
            float redout_32478 = 0.0F;
            
            for (int32_t i_32479 = 0; i_32479 < n_30761; i_32479++) {
                float x_32480;
                float x_32481;
                float x_32482;
                float x_32483;
                bool res_32484;
                float y_32485;
                float res_32486;
                float res_32489;
                
                x_32480 = *(__global float *) &images_mem_37201[(gtid_32094 *
                                                                 sizze_30758 +
                                                                 i_32479) * 4];
                x_32481 = *(__global float *) &mem_37214[(i_32479 * res_30780 +
                                                          gtid_32095) * 4];
                x_32482 = *(__global float *) &mem_37218[(i_32479 * res_30780 +
                                                          i_32475) * 4];
                x_32483 = x_32481 * x_32482;
                res_32484 = futrts_isnan32(x_32480);
                if (res_32484) {
                    y_32485 = 0.0F;
                } else {
                    y_32485 = 1.0F;
                }
                res_32486 = x_32483 * y_32485;
                res_32489 = redout_32478 + res_32486;
                
                float redout_tmp_37827 = res_32489;
                
                redout_32478 = redout_tmp_37827;
            }
            res_32477 = redout_32478;
            *(__global float *) &mem_37258[(group_id_32105 *
                                            (group_sizze_32456 * res_30780) +
                                            local_tid_32104 + i_32475 *
                                            group_sizze_32456) * 4] = res_32477;
        }
    }
    if (slt32(gtid_32094, sizze_30757) && slt32(gtid_32095, res_30780)) {
        for (int32_t i_37828 = 0; i_37828 < res_30780; i_37828++) {
            *(__global float *) &mem_37263[(res_30780 * sizze_30757 * 0 +
                                            gtid_32094 * res_30780 +
                                            gtid_32095 + i_37828 * (res_30780 *
                                                                    sizze_30757)) *
                                           4] = *(__global
                                                  float *) &mem_37258[(group_id_32105 *
                                                                       (group_sizze_32456 *
                                                                        res_30780) +
                                                                       local_tid_32104 +
                                                                       i_37828 *
                                                                       group_sizze_32456) *
                                                                      4];
        }
    }
}
__kernel void map_32210(__local volatile int64_t *mem_37288_backing_aligned_0,
                        int32_t sizze_30757, int32_t n_30761, int32_t res_30780,
                        int32_t gidzz_range_36508, int32_t tile_sizze_x_36512,
                        int32_t tiled_group_sizze_36514, __global
                        unsigned char *mem_37214, __global
                        unsigned char *mem_37218, __global
                        unsigned char *mem_37281, __global
                        unsigned char *mem_37285)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict mem_37288_backing_0 =
                          mem_37288_backing_aligned_0;
    int32_t global_tid_32210;
    int32_t local_tid_32211;
    int32_t group_sizze_37836;
    int32_t wave_sizze_37835;
    int32_t group_id_32212;
    
    global_tid_32210 = get_global_id(0);
    local_tid_32211 = get_local_id(0);
    group_sizze_37836 = get_local_size(0);
    wave_sizze_37835 = LOCKSTEP_WIDTH;
    group_id_32212 = get_group_id(0);
    
    int32_t gtid_32199;
    int32_t gtid_32200;
    int32_t gtid_32201;
    int32_t ltid_36515;
    int32_t ltid_36516;
    int32_t ltid_36517;
    
    gtid_32199 = squot32(srem32(global_tid_32210, tile_sizze_x_36512 *
                                tile_sizze_x_36512), tile_sizze_x_36512 *
                         tile_sizze_x_36512) + squot32(squot32(global_tid_32210,
                                                               tile_sizze_x_36512 *
                                                               tile_sizze_x_36512),
                                                       squot32(res_30780 +
                                                               tile_sizze_x_36512 -
                                                               1,
                                                               tile_sizze_x_36512) *
                                                       squot32(res_30780 +
                                                               tile_sizze_x_36512 -
                                                               1,
                                                               tile_sizze_x_36512));
    gtid_32200 = squot32(srem32(global_tid_32210, tile_sizze_x_36512 *
                                tile_sizze_x_36512) -
                         squot32(srem32(global_tid_32210, tile_sizze_x_36512 *
                                        tile_sizze_x_36512),
                                 tile_sizze_x_36512 * tile_sizze_x_36512) *
                         (tile_sizze_x_36512 * tile_sizze_x_36512),
                         tile_sizze_x_36512) + squot32(squot32(global_tid_32210,
                                                               tile_sizze_x_36512 *
                                                               tile_sizze_x_36512) -
                                                       squot32(squot32(global_tid_32210,
                                                                       tile_sizze_x_36512 *
                                                                       tile_sizze_x_36512),
                                                               squot32(res_30780 +
                                                                       tile_sizze_x_36512 -
                                                                       1,
                                                                       tile_sizze_x_36512) *
                                                               squot32(res_30780 +
                                                                       tile_sizze_x_36512 -
                                                                       1,
                                                                       tile_sizze_x_36512)) *
                                                       (squot32(res_30780 +
                                                                tile_sizze_x_36512 -
                                                                1,
                                                                tile_sizze_x_36512) *
                                                        squot32(res_30780 +
                                                                tile_sizze_x_36512 -
                                                                1,
                                                                tile_sizze_x_36512)),
                                                       squot32(res_30780 +
                                                               tile_sizze_x_36512 -
                                                               1,
                                                               tile_sizze_x_36512)) *
        tile_sizze_x_36512;
    gtid_32201 = srem32(global_tid_32210, tile_sizze_x_36512 *
                        tile_sizze_x_36512) - squot32(srem32(global_tid_32210,
                                                             tile_sizze_x_36512 *
                                                             tile_sizze_x_36512),
                                                      tile_sizze_x_36512 *
                                                      tile_sizze_x_36512) *
        (tile_sizze_x_36512 * tile_sizze_x_36512) -
        squot32(srem32(global_tid_32210, tile_sizze_x_36512 *
                       tile_sizze_x_36512) - squot32(srem32(global_tid_32210,
                                                            tile_sizze_x_36512 *
                                                            tile_sizze_x_36512),
                                                     tile_sizze_x_36512 *
                                                     tile_sizze_x_36512) *
                (tile_sizze_x_36512 * tile_sizze_x_36512), tile_sizze_x_36512) *
        tile_sizze_x_36512 + (squot32(global_tid_32210, tile_sizze_x_36512 *
                                      tile_sizze_x_36512) -
                              squot32(squot32(global_tid_32210,
                                              tile_sizze_x_36512 *
                                              tile_sizze_x_36512),
                                      squot32(res_30780 + tile_sizze_x_36512 -
                                              1, tile_sizze_x_36512) *
                                      squot32(res_30780 + tile_sizze_x_36512 -
                                              1, tile_sizze_x_36512)) *
                              (squot32(res_30780 + tile_sizze_x_36512 - 1,
                                       tile_sizze_x_36512) * squot32(res_30780 +
                                                                     tile_sizze_x_36512 -
                                                                     1,
                                                                     tile_sizze_x_36512)) -
                              squot32(squot32(global_tid_32210,
                                              tile_sizze_x_36512 *
                                              tile_sizze_x_36512) -
                                      squot32(squot32(global_tid_32210,
                                                      tile_sizze_x_36512 *
                                                      tile_sizze_x_36512),
                                              squot32(res_30780 +
                                                      tile_sizze_x_36512 - 1,
                                                      tile_sizze_x_36512) *
                                              squot32(res_30780 +
                                                      tile_sizze_x_36512 - 1,
                                                      tile_sizze_x_36512)) *
                                      (squot32(res_30780 + tile_sizze_x_36512 -
                                               1, tile_sizze_x_36512) *
                                       squot32(res_30780 + tile_sizze_x_36512 -
                                               1, tile_sizze_x_36512)),
                                      squot32(res_30780 + tile_sizze_x_36512 -
                                              1, tile_sizze_x_36512)) *
                              squot32(res_30780 + tile_sizze_x_36512 - 1,
                                      tile_sizze_x_36512)) * tile_sizze_x_36512;
    ltid_36515 = squot32(srem32(global_tid_32210, tile_sizze_x_36512 *
                                tile_sizze_x_36512), tile_sizze_x_36512 *
                         tile_sizze_x_36512);
    ltid_36516 = squot32(srem32(global_tid_32210, tile_sizze_x_36512 *
                                tile_sizze_x_36512) -
                         squot32(srem32(global_tid_32210, tile_sizze_x_36512 *
                                        tile_sizze_x_36512),
                                 tile_sizze_x_36512 * tile_sizze_x_36512) *
                         (tile_sizze_x_36512 * tile_sizze_x_36512),
                         tile_sizze_x_36512);
    ltid_36517 = srem32(global_tid_32210, tile_sizze_x_36512 *
                        tile_sizze_x_36512) - squot32(srem32(global_tid_32210,
                                                             tile_sizze_x_36512 *
                                                             tile_sizze_x_36512),
                                                      tile_sizze_x_36512 *
                                                      tile_sizze_x_36512) *
        (tile_sizze_x_36512 * tile_sizze_x_36512) -
        squot32(srem32(global_tid_32210, tile_sizze_x_36512 *
                       tile_sizze_x_36512) - squot32(srem32(global_tid_32210,
                                                            tile_sizze_x_36512 *
                                                            tile_sizze_x_36512),
                                                     tile_sizze_x_36512 *
                                                     tile_sizze_x_36512) *
                (tile_sizze_x_36512 * tile_sizze_x_36512), tile_sizze_x_36512) *
        tile_sizze_x_36512;
    
    int32_t mm_36505;
    int32_t m_36535;
    bool is_active_37121;
    bool is_active_37122;
    bool active_37124;
    
    if ((slt32(gtid_32199, gidzz_range_36508) && slt32(gtid_32200,
                                                       res_30780)) &&
        slt32(gtid_32201, res_30780)) {
        mm_36505 = 30 * gtid_32199;
        m_36535 = local_tid_32211 + mm_36505;
        is_active_37121 = slt32(local_tid_32211, 30);
        is_active_37122 = slt32(m_36535, sizze_30757);
        active_37124 = is_active_37121 && is_active_37122;
    }
    
    __local char *mem_37288;
    
    mem_37288 = (__local char *) mem_37288_backing_0;
    
    float res_36871;
    float res_36872;
    float res_36873;
    float res_36874;
    float res_36875;
    float res_36876;
    float res_36877;
    float res_36878;
    float res_36879;
    float res_36880;
    float res_36881;
    float res_36882;
    float res_36883;
    float res_36884;
    float res_36885;
    float res_36886;
    float res_36887;
    float res_36888;
    float res_36889;
    float res_36890;
    float res_36891;
    float res_36892;
    float res_36893;
    float res_36894;
    float res_36895;
    float res_36896;
    float res_36897;
    float res_36898;
    float res_36899;
    float res_36900;
    int32_t m_36906;
    int32_t m_36909;
    int32_t m_36912;
    int32_t m_36915;
    int32_t m_36918;
    int32_t m_36921;
    int32_t m_36924;
    int32_t m_36927;
    int32_t m_36930;
    int32_t m_36933;
    int32_t m_36936;
    int32_t m_36939;
    int32_t m_36942;
    int32_t m_36945;
    int32_t m_36948;
    int32_t m_36951;
    int32_t m_36954;
    int32_t m_36957;
    int32_t m_36960;
    int32_t m_36963;
    int32_t m_36966;
    int32_t m_36969;
    int32_t m_36972;
    int32_t m_36975;
    int32_t m_36978;
    int32_t m_36981;
    int32_t m_36984;
    int32_t m_36987;
    int32_t m_36990;
    
    if ((slt32(gtid_32199, gidzz_range_36508) && slt32(gtid_32200,
                                                       res_30780)) &&
        slt32(gtid_32201, res_30780)) {
        float acc_clone_36541;
        float acc_clone_36552;
        float acc_clone_36563;
        float acc_clone_36574;
        float acc_clone_36585;
        float acc_clone_36596;
        float acc_clone_36607;
        float acc_clone_36618;
        float acc_clone_36629;
        float acc_clone_36640;
        float acc_clone_36651;
        float acc_clone_36662;
        float acc_clone_36673;
        float acc_clone_36684;
        float acc_clone_36695;
        float acc_clone_36706;
        float acc_clone_36717;
        float acc_clone_36728;
        float acc_clone_36739;
        float acc_clone_36750;
        float acc_clone_36761;
        float acc_clone_36772;
        float acc_clone_36783;
        float acc_clone_36794;
        float acc_clone_36805;
        float acc_clone_36816;
        float acc_clone_36827;
        float acc_clone_36838;
        float acc_clone_36849;
        float acc_clone_36860;
        
        acc_clone_36541 = 0.0F;
        acc_clone_36552 = 0.0F;
        acc_clone_36563 = 0.0F;
        acc_clone_36574 = 0.0F;
        acc_clone_36585 = 0.0F;
        acc_clone_36596 = 0.0F;
        acc_clone_36607 = 0.0F;
        acc_clone_36618 = 0.0F;
        acc_clone_36629 = 0.0F;
        acc_clone_36640 = 0.0F;
        acc_clone_36651 = 0.0F;
        acc_clone_36662 = 0.0F;
        acc_clone_36673 = 0.0F;
        acc_clone_36684 = 0.0F;
        acc_clone_36695 = 0.0F;
        acc_clone_36706 = 0.0F;
        acc_clone_36717 = 0.0F;
        acc_clone_36728 = 0.0F;
        acc_clone_36739 = 0.0F;
        acc_clone_36750 = 0.0F;
        acc_clone_36761 = 0.0F;
        acc_clone_36772 = 0.0F;
        acc_clone_36783 = 0.0F;
        acc_clone_36794 = 0.0F;
        acc_clone_36805 = 0.0F;
        acc_clone_36816 = 0.0F;
        acc_clone_36827 = 0.0F;
        acc_clone_36838 = 0.0F;
        acc_clone_36849 = 0.0F;
        acc_clone_36860 = 0.0F;
        for (int32_t loop_ind_36870 = 0; loop_ind_36870 < n_30761;
             loop_ind_36870++) {
            int32_t i_32551;
            
            i_32551 = loop_ind_36870;
            barrier(CLK_LOCAL_MEM_FENCE);
            
            float x_32557;
            float x_32558;
            float x_32560;
            float x_32556;
            
            x_32557 = *(__global float *) &mem_37214[(i_32551 * res_30780 +
                                                      gtid_32200) * 4];
            x_32558 = *(__global float *) &mem_37218[(i_32551 * res_30780 +
                                                      gtid_32201) * 4];
            x_32560 = x_32557 * x_32558;
            if (active_37124) {
                float x_37125 = *(__global float *) &mem_37285[(i_32551 *
                                                                sizze_30757 +
                                                                m_36535) * 4];
                
                x_32556 = x_37125;
            } else {
                x_32556 = 0.0F;
            }
            for (int32_t comb_iter_37867 = 0; comb_iter_37867 < 1;
                 comb_iter_37867++) {
                int32_t cid_36539;
                int32_t flat_comb_id_37868 = comb_iter_37867 *
                        tiled_group_sizze_36514 + local_tid_32211;
                
                cid_36539 = flat_comb_id_37868;
                if (slt32(cid_36539, tiled_group_sizze_36514) &&
                    (slt32(local_tid_32211, 30) && slt32(m_36535,
                                                         sizze_30757))) {
                    *(__local float *) &mem_37288[cid_36539 * 4] = x_32556;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            float x_36546;
            bool res_36547;
            float y_36548;
            float res_36549;
            float res_36550;
            float x_36557;
            bool res_36558;
            float y_36559;
            float res_36560;
            float res_36561;
            float x_36568;
            bool res_36569;
            float y_36570;
            float res_36571;
            float res_36572;
            float x_36579;
            bool res_36580;
            float y_36581;
            float res_36582;
            float res_36583;
            float x_36590;
            bool res_36591;
            float y_36592;
            float res_36593;
            float res_36594;
            float x_36601;
            bool res_36602;
            float y_36603;
            float res_36604;
            float res_36605;
            float x_36612;
            bool res_36613;
            float y_36614;
            float res_36615;
            float res_36616;
            float x_36623;
            bool res_36624;
            float y_36625;
            float res_36626;
            float res_36627;
            float x_36634;
            bool res_36635;
            float y_36636;
            float res_36637;
            float res_36638;
            float x_36645;
            bool res_36646;
            float y_36647;
            float res_36648;
            float res_36649;
            float x_36656;
            bool res_36657;
            float y_36658;
            float res_36659;
            float res_36660;
            float x_36667;
            bool res_36668;
            float y_36669;
            float res_36670;
            float res_36671;
            float x_36678;
            bool res_36679;
            float y_36680;
            float res_36681;
            float res_36682;
            float x_36689;
            bool res_36690;
            float y_36691;
            float res_36692;
            float res_36693;
            float x_36700;
            bool res_36701;
            float y_36702;
            float res_36703;
            float res_36704;
            float x_36711;
            bool res_36712;
            float y_36713;
            float res_36714;
            float res_36715;
            float x_36722;
            bool res_36723;
            float y_36724;
            float res_36725;
            float res_36726;
            float x_36733;
            bool res_36734;
            float y_36735;
            float res_36736;
            float res_36737;
            float x_36744;
            bool res_36745;
            float y_36746;
            float res_36747;
            float res_36748;
            float x_36755;
            bool res_36756;
            float y_36757;
            float res_36758;
            float res_36759;
            float x_36766;
            bool res_36767;
            float y_36768;
            float res_36769;
            float res_36770;
            float x_36777;
            bool res_36778;
            float y_36779;
            float res_36780;
            float res_36781;
            float x_36788;
            bool res_36789;
            float y_36790;
            float res_36791;
            float res_36792;
            float x_36799;
            bool res_36800;
            float y_36801;
            float res_36802;
            float res_36803;
            float x_36810;
            bool res_36811;
            float y_36812;
            float res_36813;
            float res_36814;
            float x_36821;
            bool res_36822;
            float y_36823;
            float res_36824;
            float res_36825;
            float x_36832;
            bool res_36833;
            float y_36834;
            float res_36835;
            float res_36836;
            float x_36843;
            bool res_36844;
            float y_36845;
            float res_36846;
            float res_36847;
            float x_36854;
            bool res_36855;
            float y_36856;
            float res_36857;
            float res_36858;
            float x_36865;
            bool res_36866;
            float y_36867;
            float res_36868;
            float res_36869;
            
            x_36546 = *(__local float *) &mem_37288[0];
            res_36547 = futrts_isnan32(x_36546);
            if (res_36547) {
                y_36548 = 0.0F;
            } else {
                y_36548 = 1.0F;
            }
            res_36549 = x_32560 * y_36548;
            res_36550 = acc_clone_36541 + res_36549;
            x_36557 = *(__local float *) &mem_37288[4];
            res_36558 = futrts_isnan32(x_36557);
            if (res_36558) {
                y_36559 = 0.0F;
            } else {
                y_36559 = 1.0F;
            }
            res_36560 = x_32560 * y_36559;
            res_36561 = acc_clone_36552 + res_36560;
            x_36568 = *(__local float *) &mem_37288[8];
            res_36569 = futrts_isnan32(x_36568);
            if (res_36569) {
                y_36570 = 0.0F;
            } else {
                y_36570 = 1.0F;
            }
            res_36571 = x_32560 * y_36570;
            res_36572 = acc_clone_36563 + res_36571;
            x_36579 = *(__local float *) &mem_37288[12];
            res_36580 = futrts_isnan32(x_36579);
            if (res_36580) {
                y_36581 = 0.0F;
            } else {
                y_36581 = 1.0F;
            }
            res_36582 = x_32560 * y_36581;
            res_36583 = acc_clone_36574 + res_36582;
            x_36590 = *(__local float *) &mem_37288[16];
            res_36591 = futrts_isnan32(x_36590);
            if (res_36591) {
                y_36592 = 0.0F;
            } else {
                y_36592 = 1.0F;
            }
            res_36593 = x_32560 * y_36592;
            res_36594 = acc_clone_36585 + res_36593;
            x_36601 = *(__local float *) &mem_37288[20];
            res_36602 = futrts_isnan32(x_36601);
            if (res_36602) {
                y_36603 = 0.0F;
            } else {
                y_36603 = 1.0F;
            }
            res_36604 = x_32560 * y_36603;
            res_36605 = acc_clone_36596 + res_36604;
            x_36612 = *(__local float *) &mem_37288[24];
            res_36613 = futrts_isnan32(x_36612);
            if (res_36613) {
                y_36614 = 0.0F;
            } else {
                y_36614 = 1.0F;
            }
            res_36615 = x_32560 * y_36614;
            res_36616 = acc_clone_36607 + res_36615;
            x_36623 = *(__local float *) &mem_37288[28];
            res_36624 = futrts_isnan32(x_36623);
            if (res_36624) {
                y_36625 = 0.0F;
            } else {
                y_36625 = 1.0F;
            }
            res_36626 = x_32560 * y_36625;
            res_36627 = acc_clone_36618 + res_36626;
            x_36634 = *(__local float *) &mem_37288[32];
            res_36635 = futrts_isnan32(x_36634);
            if (res_36635) {
                y_36636 = 0.0F;
            } else {
                y_36636 = 1.0F;
            }
            res_36637 = x_32560 * y_36636;
            res_36638 = acc_clone_36629 + res_36637;
            x_36645 = *(__local float *) &mem_37288[36];
            res_36646 = futrts_isnan32(x_36645);
            if (res_36646) {
                y_36647 = 0.0F;
            } else {
                y_36647 = 1.0F;
            }
            res_36648 = x_32560 * y_36647;
            res_36649 = acc_clone_36640 + res_36648;
            x_36656 = *(__local float *) &mem_37288[40];
            res_36657 = futrts_isnan32(x_36656);
            if (res_36657) {
                y_36658 = 0.0F;
            } else {
                y_36658 = 1.0F;
            }
            res_36659 = x_32560 * y_36658;
            res_36660 = acc_clone_36651 + res_36659;
            x_36667 = *(__local float *) &mem_37288[44];
            res_36668 = futrts_isnan32(x_36667);
            if (res_36668) {
                y_36669 = 0.0F;
            } else {
                y_36669 = 1.0F;
            }
            res_36670 = x_32560 * y_36669;
            res_36671 = acc_clone_36662 + res_36670;
            x_36678 = *(__local float *) &mem_37288[48];
            res_36679 = futrts_isnan32(x_36678);
            if (res_36679) {
                y_36680 = 0.0F;
            } else {
                y_36680 = 1.0F;
            }
            res_36681 = x_32560 * y_36680;
            res_36682 = acc_clone_36673 + res_36681;
            x_36689 = *(__local float *) &mem_37288[52];
            res_36690 = futrts_isnan32(x_36689);
            if (res_36690) {
                y_36691 = 0.0F;
            } else {
                y_36691 = 1.0F;
            }
            res_36692 = x_32560 * y_36691;
            res_36693 = acc_clone_36684 + res_36692;
            x_36700 = *(__local float *) &mem_37288[56];
            res_36701 = futrts_isnan32(x_36700);
            if (res_36701) {
                y_36702 = 0.0F;
            } else {
                y_36702 = 1.0F;
            }
            res_36703 = x_32560 * y_36702;
            res_36704 = acc_clone_36695 + res_36703;
            x_36711 = *(__local float *) &mem_37288[60];
            res_36712 = futrts_isnan32(x_36711);
            if (res_36712) {
                y_36713 = 0.0F;
            } else {
                y_36713 = 1.0F;
            }
            res_36714 = x_32560 * y_36713;
            res_36715 = acc_clone_36706 + res_36714;
            x_36722 = *(__local float *) &mem_37288[64];
            res_36723 = futrts_isnan32(x_36722);
            if (res_36723) {
                y_36724 = 0.0F;
            } else {
                y_36724 = 1.0F;
            }
            res_36725 = x_32560 * y_36724;
            res_36726 = acc_clone_36717 + res_36725;
            x_36733 = *(__local float *) &mem_37288[68];
            res_36734 = futrts_isnan32(x_36733);
            if (res_36734) {
                y_36735 = 0.0F;
            } else {
                y_36735 = 1.0F;
            }
            res_36736 = x_32560 * y_36735;
            res_36737 = acc_clone_36728 + res_36736;
            x_36744 = *(__local float *) &mem_37288[72];
            res_36745 = futrts_isnan32(x_36744);
            if (res_36745) {
                y_36746 = 0.0F;
            } else {
                y_36746 = 1.0F;
            }
            res_36747 = x_32560 * y_36746;
            res_36748 = acc_clone_36739 + res_36747;
            x_36755 = *(__local float *) &mem_37288[76];
            res_36756 = futrts_isnan32(x_36755);
            if (res_36756) {
                y_36757 = 0.0F;
            } else {
                y_36757 = 1.0F;
            }
            res_36758 = x_32560 * y_36757;
            res_36759 = acc_clone_36750 + res_36758;
            x_36766 = *(__local float *) &mem_37288[80];
            res_36767 = futrts_isnan32(x_36766);
            if (res_36767) {
                y_36768 = 0.0F;
            } else {
                y_36768 = 1.0F;
            }
            res_36769 = x_32560 * y_36768;
            res_36770 = acc_clone_36761 + res_36769;
            x_36777 = *(__local float *) &mem_37288[84];
            res_36778 = futrts_isnan32(x_36777);
            if (res_36778) {
                y_36779 = 0.0F;
            } else {
                y_36779 = 1.0F;
            }
            res_36780 = x_32560 * y_36779;
            res_36781 = acc_clone_36772 + res_36780;
            x_36788 = *(__local float *) &mem_37288[88];
            res_36789 = futrts_isnan32(x_36788);
            if (res_36789) {
                y_36790 = 0.0F;
            } else {
                y_36790 = 1.0F;
            }
            res_36791 = x_32560 * y_36790;
            res_36792 = acc_clone_36783 + res_36791;
            x_36799 = *(__local float *) &mem_37288[92];
            res_36800 = futrts_isnan32(x_36799);
            if (res_36800) {
                y_36801 = 0.0F;
            } else {
                y_36801 = 1.0F;
            }
            res_36802 = x_32560 * y_36801;
            res_36803 = acc_clone_36794 + res_36802;
            x_36810 = *(__local float *) &mem_37288[96];
            res_36811 = futrts_isnan32(x_36810);
            if (res_36811) {
                y_36812 = 0.0F;
            } else {
                y_36812 = 1.0F;
            }
            res_36813 = x_32560 * y_36812;
            res_36814 = acc_clone_36805 + res_36813;
            x_36821 = *(__local float *) &mem_37288[100];
            res_36822 = futrts_isnan32(x_36821);
            if (res_36822) {
                y_36823 = 0.0F;
            } else {
                y_36823 = 1.0F;
            }
            res_36824 = x_32560 * y_36823;
            res_36825 = acc_clone_36816 + res_36824;
            x_36832 = *(__local float *) &mem_37288[104];
            res_36833 = futrts_isnan32(x_36832);
            if (res_36833) {
                y_36834 = 0.0F;
            } else {
                y_36834 = 1.0F;
            }
            res_36835 = x_32560 * y_36834;
            res_36836 = acc_clone_36827 + res_36835;
            x_36843 = *(__local float *) &mem_37288[108];
            res_36844 = futrts_isnan32(x_36843);
            if (res_36844) {
                y_36845 = 0.0F;
            } else {
                y_36845 = 1.0F;
            }
            res_36846 = x_32560 * y_36845;
            res_36847 = acc_clone_36838 + res_36846;
            x_36854 = *(__local float *) &mem_37288[112];
            res_36855 = futrts_isnan32(x_36854);
            if (res_36855) {
                y_36856 = 0.0F;
            } else {
                y_36856 = 1.0F;
            }
            res_36857 = x_32560 * y_36856;
            res_36858 = acc_clone_36849 + res_36857;
            x_36865 = *(__local float *) &mem_37288[116];
            res_36866 = futrts_isnan32(x_36865);
            if (res_36866) {
                y_36867 = 0.0F;
            } else {
                y_36867 = 1.0F;
            }
            res_36868 = x_32560 * y_36867;
            res_36869 = acc_clone_36860 + res_36868;
            
            float acc_clone_tmp_37837 = res_36550;
            float acc_clone_tmp_37838 = res_36561;
            float acc_clone_tmp_37839 = res_36572;
            float acc_clone_tmp_37840 = res_36583;
            float acc_clone_tmp_37841 = res_36594;
            float acc_clone_tmp_37842 = res_36605;
            float acc_clone_tmp_37843 = res_36616;
            float acc_clone_tmp_37844 = res_36627;
            float acc_clone_tmp_37845 = res_36638;
            float acc_clone_tmp_37846 = res_36649;
            float acc_clone_tmp_37847 = res_36660;
            float acc_clone_tmp_37848 = res_36671;
            float acc_clone_tmp_37849 = res_36682;
            float acc_clone_tmp_37850 = res_36693;
            float acc_clone_tmp_37851 = res_36704;
            float acc_clone_tmp_37852 = res_36715;
            float acc_clone_tmp_37853 = res_36726;
            float acc_clone_tmp_37854 = res_36737;
            float acc_clone_tmp_37855 = res_36748;
            float acc_clone_tmp_37856 = res_36759;
            float acc_clone_tmp_37857 = res_36770;
            float acc_clone_tmp_37858 = res_36781;
            float acc_clone_tmp_37859 = res_36792;
            float acc_clone_tmp_37860 = res_36803;
            float acc_clone_tmp_37861 = res_36814;
            float acc_clone_tmp_37862 = res_36825;
            float acc_clone_tmp_37863 = res_36836;
            float acc_clone_tmp_37864 = res_36847;
            float acc_clone_tmp_37865 = res_36858;
            float acc_clone_tmp_37866;
            
            acc_clone_tmp_37866 = res_36869;
            acc_clone_36541 = acc_clone_tmp_37837;
            acc_clone_36552 = acc_clone_tmp_37838;
            acc_clone_36563 = acc_clone_tmp_37839;
            acc_clone_36574 = acc_clone_tmp_37840;
            acc_clone_36585 = acc_clone_tmp_37841;
            acc_clone_36596 = acc_clone_tmp_37842;
            acc_clone_36607 = acc_clone_tmp_37843;
            acc_clone_36618 = acc_clone_tmp_37844;
            acc_clone_36629 = acc_clone_tmp_37845;
            acc_clone_36640 = acc_clone_tmp_37846;
            acc_clone_36651 = acc_clone_tmp_37847;
            acc_clone_36662 = acc_clone_tmp_37848;
            acc_clone_36673 = acc_clone_tmp_37849;
            acc_clone_36684 = acc_clone_tmp_37850;
            acc_clone_36695 = acc_clone_tmp_37851;
            acc_clone_36706 = acc_clone_tmp_37852;
            acc_clone_36717 = acc_clone_tmp_37853;
            acc_clone_36728 = acc_clone_tmp_37854;
            acc_clone_36739 = acc_clone_tmp_37855;
            acc_clone_36750 = acc_clone_tmp_37856;
            acc_clone_36761 = acc_clone_tmp_37857;
            acc_clone_36772 = acc_clone_tmp_37858;
            acc_clone_36783 = acc_clone_tmp_37859;
            acc_clone_36794 = acc_clone_tmp_37860;
            acc_clone_36805 = acc_clone_tmp_37861;
            acc_clone_36816 = acc_clone_tmp_37862;
            acc_clone_36827 = acc_clone_tmp_37863;
            acc_clone_36838 = acc_clone_tmp_37864;
            acc_clone_36849 = acc_clone_tmp_37865;
            acc_clone_36860 = acc_clone_tmp_37866;
        }
        res_36871 = acc_clone_36541;
        res_36872 = acc_clone_36552;
        res_36873 = acc_clone_36563;
        res_36874 = acc_clone_36574;
        res_36875 = acc_clone_36585;
        res_36876 = acc_clone_36596;
        res_36877 = acc_clone_36607;
        res_36878 = acc_clone_36618;
        res_36879 = acc_clone_36629;
        res_36880 = acc_clone_36640;
        res_36881 = acc_clone_36651;
        res_36882 = acc_clone_36662;
        res_36883 = acc_clone_36673;
        res_36884 = acc_clone_36684;
        res_36885 = acc_clone_36695;
        res_36886 = acc_clone_36706;
        res_36887 = acc_clone_36717;
        res_36888 = acc_clone_36728;
        res_36889 = acc_clone_36739;
        res_36890 = acc_clone_36750;
        res_36891 = acc_clone_36761;
        res_36892 = acc_clone_36772;
        res_36893 = acc_clone_36783;
        res_36894 = acc_clone_36794;
        res_36895 = acc_clone_36805;
        res_36896 = acc_clone_36816;
        res_36897 = acc_clone_36827;
        res_36898 = acc_clone_36838;
        res_36899 = acc_clone_36849;
        res_36900 = acc_clone_36860;
        m_36906 = 1 + mm_36505;
        m_36909 = 2 + mm_36505;
        m_36912 = 3 + mm_36505;
        m_36915 = 4 + mm_36505;
        m_36918 = 5 + mm_36505;
        m_36921 = 6 + mm_36505;
        m_36924 = 7 + mm_36505;
        m_36927 = 8 + mm_36505;
        m_36930 = 9 + mm_36505;
        m_36933 = 10 + mm_36505;
        m_36936 = 11 + mm_36505;
        m_36939 = 12 + mm_36505;
        m_36942 = 13 + mm_36505;
        m_36945 = 14 + mm_36505;
        m_36948 = 15 + mm_36505;
        m_36951 = 16 + mm_36505;
        m_36954 = 17 + mm_36505;
        m_36957 = 18 + mm_36505;
        m_36960 = 19 + mm_36505;
        m_36963 = 20 + mm_36505;
        m_36966 = 21 + mm_36505;
        m_36969 = 22 + mm_36505;
        m_36972 = 23 + mm_36505;
        m_36975 = 24 + mm_36505;
        m_36978 = 25 + mm_36505;
        m_36981 = 26 + mm_36505;
        m_36984 = 27 + mm_36505;
        m_36987 = 28 + mm_36505;
        m_36990 = 29 + mm_36505;
    }
    if (((((slt32(gtid_32199, gidzz_range_36508) && slt32(gtid_32200,
                                                          res_30780)) &&
           slt32(gtid_32201, res_30780)) && (sle32(0, mm_36505) &&
                                             slt32(mm_36505, sizze_30757))) &&
         (sle32(0, gtid_32200) && slt32(gtid_32200, res_30780))) && (sle32(0,
                                                                           gtid_32201) &&
                                                                     slt32(gtid_32201,
                                                                           res_30780))) {
        *(__global float *) &mem_37281[(mm_36505 * (res_30780 * res_30780) +
                                        gtid_32200 * res_30780 + gtid_32201) *
                                       4] = res_36871;
    }
    if (((((slt32(gtid_32199, gidzz_range_36508) && slt32(gtid_32200,
                                                          res_30780)) &&
           slt32(gtid_32201, res_30780)) && (sle32(0, m_36906) && slt32(m_36906,
                                                                        sizze_30757))) &&
         (sle32(0, gtid_32200) && slt32(gtid_32200, res_30780))) && (sle32(0,
                                                                           gtid_32201) &&
                                                                     slt32(gtid_32201,
                                                                           res_30780))) {
        *(__global float *) &mem_37281[(m_36906 * (res_30780 * res_30780) +
                                        gtid_32200 * res_30780 + gtid_32201) *
                                       4] = res_36872;
    }
    if (((((slt32(gtid_32199, gidzz_range_36508) && slt32(gtid_32200,
                                                          res_30780)) &&
           slt32(gtid_32201, res_30780)) && (sle32(0, m_36909) && slt32(m_36909,
                                                                        sizze_30757))) &&
         (sle32(0, gtid_32200) && slt32(gtid_32200, res_30780))) && (sle32(0,
                                                                           gtid_32201) &&
                                                                     slt32(gtid_32201,
                                                                           res_30780))) {
        *(__global float *) &mem_37281[(m_36909 * (res_30780 * res_30780) +
                                        gtid_32200 * res_30780 + gtid_32201) *
                                       4] = res_36873;
    }
    if (((((slt32(gtid_32199, gidzz_range_36508) && slt32(gtid_32200,
                                                          res_30780)) &&
           slt32(gtid_32201, res_30780)) && (sle32(0, m_36912) && slt32(m_36912,
                                                                        sizze_30757))) &&
         (sle32(0, gtid_32200) && slt32(gtid_32200, res_30780))) && (sle32(0,
                                                                           gtid_32201) &&
                                                                     slt32(gtid_32201,
                                                                           res_30780))) {
        *(__global float *) &mem_37281[(m_36912 * (res_30780 * res_30780) +
                                        gtid_32200 * res_30780 + gtid_32201) *
                                       4] = res_36874;
    }
    if (((((slt32(gtid_32199, gidzz_range_36508) && slt32(gtid_32200,
                                                          res_30780)) &&
           slt32(gtid_32201, res_30780)) && (sle32(0, m_36915) && slt32(m_36915,
                                                                        sizze_30757))) &&
         (sle32(0, gtid_32200) && slt32(gtid_32200, res_30780))) && (sle32(0,
                                                                           gtid_32201) &&
                                                                     slt32(gtid_32201,
                                                                           res_30780))) {
        *(__global float *) &mem_37281[(m_36915 * (res_30780 * res_30780) +
                                        gtid_32200 * res_30780 + gtid_32201) *
                                       4] = res_36875;
    }
    if (((((slt32(gtid_32199, gidzz_range_36508) && slt32(gtid_32200,
                                                          res_30780)) &&
           slt32(gtid_32201, res_30780)) && (sle32(0, m_36918) && slt32(m_36918,
                                                                        sizze_30757))) &&
         (sle32(0, gtid_32200) && slt32(gtid_32200, res_30780))) && (sle32(0,
                                                                           gtid_32201) &&
                                                                     slt32(gtid_32201,
                                                                           res_30780))) {
        *(__global float *) &mem_37281[(m_36918 * (res_30780 * res_30780) +
                                        gtid_32200 * res_30780 + gtid_32201) *
                                       4] = res_36876;
    }
    if (((((slt32(gtid_32199, gidzz_range_36508) && slt32(gtid_32200,
                                                          res_30780)) &&
           slt32(gtid_32201, res_30780)) && (sle32(0, m_36921) && slt32(m_36921,
                                                                        sizze_30757))) &&
         (sle32(0, gtid_32200) && slt32(gtid_32200, res_30780))) && (sle32(0,
                                                                           gtid_32201) &&
                                                                     slt32(gtid_32201,
                                                                           res_30780))) {
        *(__global float *) &mem_37281[(m_36921 * (res_30780 * res_30780) +
                                        gtid_32200 * res_30780 + gtid_32201) *
                                       4] = res_36877;
    }
    if (((((slt32(gtid_32199, gidzz_range_36508) && slt32(gtid_32200,
                                                          res_30780)) &&
           slt32(gtid_32201, res_30780)) && (sle32(0, m_36924) && slt32(m_36924,
                                                                        sizze_30757))) &&
         (sle32(0, gtid_32200) && slt32(gtid_32200, res_30780))) && (sle32(0,
                                                                           gtid_32201) &&
                                                                     slt32(gtid_32201,
                                                                           res_30780))) {
        *(__global float *) &mem_37281[(m_36924 * (res_30780 * res_30780) +
                                        gtid_32200 * res_30780 + gtid_32201) *
                                       4] = res_36878;
    }
    if (((((slt32(gtid_32199, gidzz_range_36508) && slt32(gtid_32200,
                                                          res_30780)) &&
           slt32(gtid_32201, res_30780)) && (sle32(0, m_36927) && slt32(m_36927,
                                                                        sizze_30757))) &&
         (sle32(0, gtid_32200) && slt32(gtid_32200, res_30780))) && (sle32(0,
                                                                           gtid_32201) &&
                                                                     slt32(gtid_32201,
                                                                           res_30780))) {
        *(__global float *) &mem_37281[(m_36927 * (res_30780 * res_30780) +
                                        gtid_32200 * res_30780 + gtid_32201) *
                                       4] = res_36879;
    }
    if (((((slt32(gtid_32199, gidzz_range_36508) && slt32(gtid_32200,
                                                          res_30780)) &&
           slt32(gtid_32201, res_30780)) && (sle32(0, m_36930) && slt32(m_36930,
                                                                        sizze_30757))) &&
         (sle32(0, gtid_32200) && slt32(gtid_32200, res_30780))) && (sle32(0,
                                                                           gtid_32201) &&
                                                                     slt32(gtid_32201,
                                                                           res_30780))) {
        *(__global float *) &mem_37281[(m_36930 * (res_30780 * res_30780) +
                                        gtid_32200 * res_30780 + gtid_32201) *
                                       4] = res_36880;
    }
    if (((((slt32(gtid_32199, gidzz_range_36508) && slt32(gtid_32200,
                                                          res_30780)) &&
           slt32(gtid_32201, res_30780)) && (sle32(0, m_36933) && slt32(m_36933,
                                                                        sizze_30757))) &&
         (sle32(0, gtid_32200) && slt32(gtid_32200, res_30780))) && (sle32(0,
                                                                           gtid_32201) &&
                                                                     slt32(gtid_32201,
                                                                           res_30780))) {
        *(__global float *) &mem_37281[(m_36933 * (res_30780 * res_30780) +
                                        gtid_32200 * res_30780 + gtid_32201) *
                                       4] = res_36881;
    }
    if (((((slt32(gtid_32199, gidzz_range_36508) && slt32(gtid_32200,
                                                          res_30780)) &&
           slt32(gtid_32201, res_30780)) && (sle32(0, m_36936) && slt32(m_36936,
                                                                        sizze_30757))) &&
         (sle32(0, gtid_32200) && slt32(gtid_32200, res_30780))) && (sle32(0,
                                                                           gtid_32201) &&
                                                                     slt32(gtid_32201,
                                                                           res_30780))) {
        *(__global float *) &mem_37281[(m_36936 * (res_30780 * res_30780) +
                                        gtid_32200 * res_30780 + gtid_32201) *
                                       4] = res_36882;
    }
    if (((((slt32(gtid_32199, gidzz_range_36508) && slt32(gtid_32200,
                                                          res_30780)) &&
           slt32(gtid_32201, res_30780)) && (sle32(0, m_36939) && slt32(m_36939,
                                                                        sizze_30757))) &&
         (sle32(0, gtid_32200) && slt32(gtid_32200, res_30780))) && (sle32(0,
                                                                           gtid_32201) &&
                                                                     slt32(gtid_32201,
                                                                           res_30780))) {
        *(__global float *) &mem_37281[(m_36939 * (res_30780 * res_30780) +
                                        gtid_32200 * res_30780 + gtid_32201) *
                                       4] = res_36883;
    }
    if (((((slt32(gtid_32199, gidzz_range_36508) && slt32(gtid_32200,
                                                          res_30780)) &&
           slt32(gtid_32201, res_30780)) && (sle32(0, m_36942) && slt32(m_36942,
                                                                        sizze_30757))) &&
         (sle32(0, gtid_32200) && slt32(gtid_32200, res_30780))) && (sle32(0,
                                                                           gtid_32201) &&
                                                                     slt32(gtid_32201,
                                                                           res_30780))) {
        *(__global float *) &mem_37281[(m_36942 * (res_30780 * res_30780) +
                                        gtid_32200 * res_30780 + gtid_32201) *
                                       4] = res_36884;
    }
    if (((((slt32(gtid_32199, gidzz_range_36508) && slt32(gtid_32200,
                                                          res_30780)) &&
           slt32(gtid_32201, res_30780)) && (sle32(0, m_36945) && slt32(m_36945,
                                                                        sizze_30757))) &&
         (sle32(0, gtid_32200) && slt32(gtid_32200, res_30780))) && (sle32(0,
                                                                           gtid_32201) &&
                                                                     slt32(gtid_32201,
                                                                           res_30780))) {
        *(__global float *) &mem_37281[(m_36945 * (res_30780 * res_30780) +
                                        gtid_32200 * res_30780 + gtid_32201) *
                                       4] = res_36885;
    }
    if (((((slt32(gtid_32199, gidzz_range_36508) && slt32(gtid_32200,
                                                          res_30780)) &&
           slt32(gtid_32201, res_30780)) && (sle32(0, m_36948) && slt32(m_36948,
                                                                        sizze_30757))) &&
         (sle32(0, gtid_32200) && slt32(gtid_32200, res_30780))) && (sle32(0,
                                                                           gtid_32201) &&
                                                                     slt32(gtid_32201,
                                                                           res_30780))) {
        *(__global float *) &mem_37281[(m_36948 * (res_30780 * res_30780) +
                                        gtid_32200 * res_30780 + gtid_32201) *
                                       4] = res_36886;
    }
    if (((((slt32(gtid_32199, gidzz_range_36508) && slt32(gtid_32200,
                                                          res_30780)) &&
           slt32(gtid_32201, res_30780)) && (sle32(0, m_36951) && slt32(m_36951,
                                                                        sizze_30757))) &&
         (sle32(0, gtid_32200) && slt32(gtid_32200, res_30780))) && (sle32(0,
                                                                           gtid_32201) &&
                                                                     slt32(gtid_32201,
                                                                           res_30780))) {
        *(__global float *) &mem_37281[(m_36951 * (res_30780 * res_30780) +
                                        gtid_32200 * res_30780 + gtid_32201) *
                                       4] = res_36887;
    }
    if (((((slt32(gtid_32199, gidzz_range_36508) && slt32(gtid_32200,
                                                          res_30780)) &&
           slt32(gtid_32201, res_30780)) && (sle32(0, m_36954) && slt32(m_36954,
                                                                        sizze_30757))) &&
         (sle32(0, gtid_32200) && slt32(gtid_32200, res_30780))) && (sle32(0,
                                                                           gtid_32201) &&
                                                                     slt32(gtid_32201,
                                                                           res_30780))) {
        *(__global float *) &mem_37281[(m_36954 * (res_30780 * res_30780) +
                                        gtid_32200 * res_30780 + gtid_32201) *
                                       4] = res_36888;
    }
    if (((((slt32(gtid_32199, gidzz_range_36508) && slt32(gtid_32200,
                                                          res_30780)) &&
           slt32(gtid_32201, res_30780)) && (sle32(0, m_36957) && slt32(m_36957,
                                                                        sizze_30757))) &&
         (sle32(0, gtid_32200) && slt32(gtid_32200, res_30780))) && (sle32(0,
                                                                           gtid_32201) &&
                                                                     slt32(gtid_32201,
                                                                           res_30780))) {
        *(__global float *) &mem_37281[(m_36957 * (res_30780 * res_30780) +
                                        gtid_32200 * res_30780 + gtid_32201) *
                                       4] = res_36889;
    }
    if (((((slt32(gtid_32199, gidzz_range_36508) && slt32(gtid_32200,
                                                          res_30780)) &&
           slt32(gtid_32201, res_30780)) && (sle32(0, m_36960) && slt32(m_36960,
                                                                        sizze_30757))) &&
         (sle32(0, gtid_32200) && slt32(gtid_32200, res_30780))) && (sle32(0,
                                                                           gtid_32201) &&
                                                                     slt32(gtid_32201,
                                                                           res_30780))) {
        *(__global float *) &mem_37281[(m_36960 * (res_30780 * res_30780) +
                                        gtid_32200 * res_30780 + gtid_32201) *
                                       4] = res_36890;
    }
    if (((((slt32(gtid_32199, gidzz_range_36508) && slt32(gtid_32200,
                                                          res_30780)) &&
           slt32(gtid_32201, res_30780)) && (sle32(0, m_36963) && slt32(m_36963,
                                                                        sizze_30757))) &&
         (sle32(0, gtid_32200) && slt32(gtid_32200, res_30780))) && (sle32(0,
                                                                           gtid_32201) &&
                                                                     slt32(gtid_32201,
                                                                           res_30780))) {
        *(__global float *) &mem_37281[(m_36963 * (res_30780 * res_30780) +
                                        gtid_32200 * res_30780 + gtid_32201) *
                                       4] = res_36891;
    }
    if (((((slt32(gtid_32199, gidzz_range_36508) && slt32(gtid_32200,
                                                          res_30780)) &&
           slt32(gtid_32201, res_30780)) && (sle32(0, m_36966) && slt32(m_36966,
                                                                        sizze_30757))) &&
         (sle32(0, gtid_32200) && slt32(gtid_32200, res_30780))) && (sle32(0,
                                                                           gtid_32201) &&
                                                                     slt32(gtid_32201,
                                                                           res_30780))) {
        *(__global float *) &mem_37281[(m_36966 * (res_30780 * res_30780) +
                                        gtid_32200 * res_30780 + gtid_32201) *
                                       4] = res_36892;
    }
    if (((((slt32(gtid_32199, gidzz_range_36508) && slt32(gtid_32200,
                                                          res_30780)) &&
           slt32(gtid_32201, res_30780)) && (sle32(0, m_36969) && slt32(m_36969,
                                                                        sizze_30757))) &&
         (sle32(0, gtid_32200) && slt32(gtid_32200, res_30780))) && (sle32(0,
                                                                           gtid_32201) &&
                                                                     slt32(gtid_32201,
                                                                           res_30780))) {
        *(__global float *) &mem_37281[(m_36969 * (res_30780 * res_30780) +
                                        gtid_32200 * res_30780 + gtid_32201) *
                                       4] = res_36893;
    }
    if (((((slt32(gtid_32199, gidzz_range_36508) && slt32(gtid_32200,
                                                          res_30780)) &&
           slt32(gtid_32201, res_30780)) && (sle32(0, m_36972) && slt32(m_36972,
                                                                        sizze_30757))) &&
         (sle32(0, gtid_32200) && slt32(gtid_32200, res_30780))) && (sle32(0,
                                                                           gtid_32201) &&
                                                                     slt32(gtid_32201,
                                                                           res_30780))) {
        *(__global float *) &mem_37281[(m_36972 * (res_30780 * res_30780) +
                                        gtid_32200 * res_30780 + gtid_32201) *
                                       4] = res_36894;
    }
    if (((((slt32(gtid_32199, gidzz_range_36508) && slt32(gtid_32200,
                                                          res_30780)) &&
           slt32(gtid_32201, res_30780)) && (sle32(0, m_36975) && slt32(m_36975,
                                                                        sizze_30757))) &&
         (sle32(0, gtid_32200) && slt32(gtid_32200, res_30780))) && (sle32(0,
                                                                           gtid_32201) &&
                                                                     slt32(gtid_32201,
                                                                           res_30780))) {
        *(__global float *) &mem_37281[(m_36975 * (res_30780 * res_30780) +
                                        gtid_32200 * res_30780 + gtid_32201) *
                                       4] = res_36895;
    }
    if (((((slt32(gtid_32199, gidzz_range_36508) && slt32(gtid_32200,
                                                          res_30780)) &&
           slt32(gtid_32201, res_30780)) && (sle32(0, m_36978) && slt32(m_36978,
                                                                        sizze_30757))) &&
         (sle32(0, gtid_32200) && slt32(gtid_32200, res_30780))) && (sle32(0,
                                                                           gtid_32201) &&
                                                                     slt32(gtid_32201,
                                                                           res_30780))) {
        *(__global float *) &mem_37281[(m_36978 * (res_30780 * res_30780) +
                                        gtid_32200 * res_30780 + gtid_32201) *
                                       4] = res_36896;
    }
    if (((((slt32(gtid_32199, gidzz_range_36508) && slt32(gtid_32200,
                                                          res_30780)) &&
           slt32(gtid_32201, res_30780)) && (sle32(0, m_36981) && slt32(m_36981,
                                                                        sizze_30757))) &&
         (sle32(0, gtid_32200) && slt32(gtid_32200, res_30780))) && (sle32(0,
                                                                           gtid_32201) &&
                                                                     slt32(gtid_32201,
                                                                           res_30780))) {
        *(__global float *) &mem_37281[(m_36981 * (res_30780 * res_30780) +
                                        gtid_32200 * res_30780 + gtid_32201) *
                                       4] = res_36897;
    }
    if (((((slt32(gtid_32199, gidzz_range_36508) && slt32(gtid_32200,
                                                          res_30780)) &&
           slt32(gtid_32201, res_30780)) && (sle32(0, m_36984) && slt32(m_36984,
                                                                        sizze_30757))) &&
         (sle32(0, gtid_32200) && slt32(gtid_32200, res_30780))) && (sle32(0,
                                                                           gtid_32201) &&
                                                                     slt32(gtid_32201,
                                                                           res_30780))) {
        *(__global float *) &mem_37281[(m_36984 * (res_30780 * res_30780) +
                                        gtid_32200 * res_30780 + gtid_32201) *
                                       4] = res_36898;
    }
    if (((((slt32(gtid_32199, gidzz_range_36508) && slt32(gtid_32200,
                                                          res_30780)) &&
           slt32(gtid_32201, res_30780)) && (sle32(0, m_36987) && slt32(m_36987,
                                                                        sizze_30757))) &&
         (sle32(0, gtid_32200) && slt32(gtid_32200, res_30780))) && (sle32(0,
                                                                           gtid_32201) &&
                                                                     slt32(gtid_32201,
                                                                           res_30780))) {
        *(__global float *) &mem_37281[(m_36987 * (res_30780 * res_30780) +
                                        gtid_32200 * res_30780 + gtid_32201) *
                                       4] = res_36899;
    }
    if (((((slt32(gtid_32199, gidzz_range_36508) && slt32(gtid_32200,
                                                          res_30780)) &&
           slt32(gtid_32201, res_30780)) && (sle32(0, m_36990) && slt32(m_36990,
                                                                        sizze_30757))) &&
         (sle32(0, gtid_32200) && slt32(gtid_32200, res_30780))) && (sle32(0,
                                                                           gtid_32201) &&
                                                                     slt32(gtid_32201,
                                                                           res_30780))) {
        *(__global float *) &mem_37281[(m_36990 * (res_30780 * res_30780) +
                                        gtid_32200 * res_30780 + gtid_32201) *
                                       4] = res_36900;
    }
}
__kernel void map_32977(int32_t sizze_30757, int32_t res_30916, __global
                        unsigned char *mem_37330, __global
                        unsigned char *mem_37337)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_32977;
    int32_t local_tid_32978;
    int32_t group_sizze_37929;
    int32_t wave_sizze_37928;
    int32_t group_id_32979;
    
    global_tid_32977 = get_global_id(0);
    local_tid_32978 = get_local_id(0);
    group_sizze_37929 = get_local_size(0);
    wave_sizze_37928 = LOCKSTEP_WIDTH;
    group_id_32979 = get_group_id(0);
    
    int32_t gtid_32968;
    int32_t gtid_32969;
    
    gtid_32968 = squot32(global_tid_32977, res_30916);
    gtid_32969 = global_tid_32977 - squot32(global_tid_32977, res_30916) *
        res_30916;
    
    float write_value_33301;
    
    if (slt32(gtid_32968, sizze_30757) && slt32(gtid_32969, res_30916)) {
        write_value_33301 = *(__global float *) &mem_37337[(gtid_32968 *
                                                            res_30916 +
                                                            gtid_32969) * 4];
    }
    if (((slt32(gtid_32968, sizze_30757) && slt32(gtid_32969, res_30916)) &&
         (sle32(0, gtid_32968) && slt32(gtid_32968, sizze_30757))) && (sle32(0,
                                                                             gtid_32969) &&
                                                                       slt32(gtid_32969,
                                                                             res_30916))) {
        *(__global float *) &mem_37330[(gtid_32968 * res_30916 + gtid_32969) *
                                       4] = write_value_33301;
    }
}
__kernel void map_33027(int32_t sizze_30757, int32_t m_30862, int32_t j_30912,
                        int32_t res_30916, int32_t i_33242, __global
                        unsigned char *mem_37330, __global
                        unsigned char *mem_37333, __global
                        unsigned char *mem_37337)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_33027;
    int32_t local_tid_33028;
    int32_t group_sizze_37927;
    int32_t wave_sizze_37926;
    int32_t group_id_33029;
    
    global_tid_33027 = get_global_id(0);
    local_tid_33028 = get_local_id(0);
    group_sizze_37927 = get_local_size(0);
    wave_sizze_37926 = LOCKSTEP_WIDTH;
    group_id_33029 = get_group_id(0);
    
    int32_t gtid_33018;
    int32_t gtid_33019;
    
    gtid_33018 = squot32(global_tid_33027, res_30916);
    gtid_33019 = global_tid_33027 - squot32(global_tid_33027, res_30916) *
        res_30916;
    
    float res_33269;
    bool cond_33270;
    int32_t res_33272;
    int32_t res_33273;
    float res_33274;
    
    if (slt32(gtid_33018, sizze_30757) && slt32(gtid_33019, res_30916)) {
        res_33269 = *(__global float *) &mem_37330[(gtid_33018 * res_30916 +
                                                    i_33242) * 4];
        cond_33270 = *(__global bool *) &mem_37333[gtid_33018];
        res_33272 = sdiv32(gtid_33019, j_30912);
        res_33273 = smod32(gtid_33019, j_30912);
        if (cond_33270) {
            int32_t x_33275;
            int32_t i_33276;
            float res_33277;
            
            x_33275 = j_30912 * res_33272;
            i_33276 = res_33273 + x_33275;
            res_33277 = *(__global float *) &mem_37330[(gtid_33018 * res_30916 +
                                                        i_33276) * 4];
            res_33274 = res_33277;
        } else {
            float x_33278;
            float res_33279;
            bool cond_33280;
            float res_33281;
            
            x_33278 = *(__global float *) &mem_37330[(gtid_33018 * res_30916 +
                                                      res_33273) * 4];
            res_33279 = x_33278 / res_33269;
            cond_33280 = slt32(res_33272, m_30862);
            if (cond_33280) {
                int32_t x_33282;
                int32_t x_33283;
                int32_t i_33284;
                float x_33285;
                int32_t i_33286;
                float x_33287;
                float y_33288;
                float res_33289;
                
                x_33282 = 1 + res_33272;
                x_33283 = j_30912 * x_33282;
                i_33284 = res_33273 + x_33283;
                x_33285 = *(__global float *) &mem_37330[(gtid_33018 *
                                                          res_30916 + i_33284) *
                                                         4];
                i_33286 = i_33242 + x_33283;
                x_33287 = *(__global float *) &mem_37330[(gtid_33018 *
                                                          res_30916 + i_33286) *
                                                         4];
                y_33288 = res_33279 * x_33287;
                res_33289 = x_33285 - y_33288;
                res_33281 = res_33289;
            } else {
                res_33281 = res_33279;
            }
            res_33274 = res_33281;
        }
    }
    if (slt32(gtid_33018, sizze_30757) && slt32(gtid_33019, res_30916)) {
        *(__global float *) &mem_37337[(gtid_33018 * res_30916 + gtid_33019) *
                                       4] = res_33274;
    }
}
__kernel void map_33088(int32_t sizze_30757, int32_t res_30916, int32_t i_33242,
                        __global unsigned char *mem_37330, __global
                        unsigned char *mem_37333)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_33088;
    int32_t local_tid_33089;
    int32_t group_sizze_37925;
    int32_t wave_sizze_37924;
    int32_t group_id_33090;
    
    global_tid_33088 = get_global_id(0);
    local_tid_33089 = get_local_id(0);
    group_sizze_37925 = get_local_size(0);
    wave_sizze_37924 = LOCKSTEP_WIDTH;
    group_id_33090 = get_group_id(0);
    
    int32_t gtid_33081;
    
    gtid_33081 = global_tid_33088;
    
    float res_33252;
    bool cond_33253;
    
    if (slt32(gtid_33081, sizze_30757)) {
        res_33252 = *(__global float *) &mem_37330[(gtid_33081 * res_30916 +
                                                    i_33242) * 4];
        cond_33253 = res_33252 == 0.0F;
    }
    if (slt32(gtid_33081, sizze_30757)) {
        *(__global bool *) &mem_37333[gtid_33081] = cond_33253;
    }
}
__kernel void map_33185(int32_t sizze_30757, int32_t res_30780, int32_t j_30912,
                        int32_t res_30916, __global
                        unsigned char *res_mem_37313, __global
                        unsigned char *mem_37330)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_33185;
    int32_t local_tid_33186;
    int32_t group_sizze_37922;
    int32_t wave_sizze_37921;
    int32_t group_id_33187;
    
    global_tid_33185 = get_global_id(0);
    local_tid_33186 = get_local_id(0);
    group_sizze_37922 = get_local_size(0);
    wave_sizze_37921 = LOCKSTEP_WIDTH;
    group_id_33187 = get_group_id(0);
    
    int32_t gtid_33176;
    int32_t gtid_33177;
    
    gtid_33176 = squot32(global_tid_33185, res_30916);
    gtid_33177 = global_tid_33185 - squot32(global_tid_33185, res_30916) *
        res_30916;
    
    int32_t res_33231;
    int32_t res_33232;
    bool cond_33233;
    float res_33234;
    
    if (slt32(gtid_33176, sizze_30757) && slt32(gtid_33177, res_30916)) {
        res_33231 = sdiv32(gtid_33177, j_30912);
        res_33232 = smod32(gtid_33177, j_30912);
        cond_33233 = slt32(res_33232, res_30780);
        if (cond_33233) {
            float res_33235 = *(__global float *) &res_mem_37313[(gtid_33176 *
                                                                  (res_30780 *
                                                                   res_30780) +
                                                                  res_33231 *
                                                                  res_30780 +
                                                                  res_33232) *
                                                                 4];
            
            res_33234 = res_33235;
        } else {
            int32_t y_33236;
            bool cond_33237;
            float res_33238;
            
            y_33236 = res_30780 + res_33231;
            cond_33237 = res_33232 == y_33236;
            if (cond_33237) {
                res_33238 = 1.0F;
            } else {
                res_33238 = 0.0F;
            }
            res_33234 = res_33238;
        }
    }
    if (slt32(gtid_33176, sizze_30757) && slt32(gtid_33177, res_30916)) {
        *(__global float *) &mem_37330[(gtid_33176 * res_30916 + gtid_33177) *
                                       4] = res_33234;
    }
}
__kernel void map_33346(int32_t sizze_30756, int32_t sizze_30757,
                        int32_t n_30761, int32_t res_30780, __global
                        unsigned char *arg_mem_37210, __global
                        unsigned char *mem_37348, __global
                        unsigned char *mem_37351, __global
                        unsigned char *mem_37355)
{
    const int32_t group_sizze_33367 = mainzigroup_sizze_33340;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_33346;
    int32_t local_tid_33347;
    int32_t group_sizze_37936;
    int32_t wave_sizze_37935;
    int32_t group_id_33348;
    
    global_tid_33346 = get_global_id(0);
    local_tid_33347 = get_local_id(0);
    group_sizze_37936 = get_local_size(0);
    wave_sizze_37935 = LOCKSTEP_WIDTH;
    group_id_33348 = get_group_id(0);
    
    int32_t gtid_33339;
    
    gtid_33339 = global_tid_33346;
    if (slt32(gtid_33339, sizze_30757)) {
        for (int32_t i_33377 = 0; i_33377 < res_30780; i_33377++) {
            float res_33379;
            float redout_33380 = 0.0F;
            
            for (int32_t i_33381 = 0; i_33381 < n_30761; i_33381++) {
                float x_33382;
                float x_33383;
                bool res_33384;
                float res_33385;
                float res_33389;
                
                x_33382 = *(__global float *) &arg_mem_37210[(i_33377 *
                                                              sizze_30756 +
                                                              i_33381) * 4];
                x_33383 = *(__global float *) &mem_37348[(i_33381 *
                                                          sizze_30757 +
                                                          gtid_33339) * 4];
                res_33384 = futrts_isnan32(x_33383);
                if (res_33384) {
                    res_33385 = 0.0F;
                } else {
                    float res_33386 = x_33382 * x_33383;
                    
                    res_33385 = res_33386;
                }
                res_33389 = redout_33380 + res_33385;
                
                float redout_tmp_37938 = res_33389;
                
                redout_33380 = redout_tmp_37938;
            }
            res_33379 = redout_33380;
            *(__global float *) &mem_37351[(group_id_33348 *
                                            (group_sizze_33367 * res_30780) +
                                            local_tid_33347 + i_33377 *
                                            group_sizze_33367) * 4] = res_33379;
        }
    }
    if (slt32(gtid_33339, sizze_30757)) {
        for (int32_t i_37939 = 0; i_37939 < res_30780; i_37939++) {
            *(__global float *) &mem_37355[(gtid_33339 + i_37939 *
                                            sizze_30757) * 4] = *(__global
                                                                  float *) &mem_37351[(group_id_33348 *
                                                                                       (group_sizze_33367 *
                                                                                        res_30780) +
                                                                                       local_tid_33347 +
                                                                                       i_37939 *
                                                                                       group_sizze_33367) *
                                                                                      4];
        }
    }
}
__kernel void map_33440(int32_t sizze_30757, int32_t sizze_30758,
                        int32_t n_30761, int32_t res_30780, __global
                        unsigned char *images_mem_37201, __global
                        unsigned char *mem_37214, __global
                        unsigned char *mem_37379)
{
    const int32_t tile_sizze_37018 = mainzitile_sizze_37017;
    const int32_t tiled_group_sizze_37019 = mainzitile_sizze_37017 *
                  mainzitile_sizze_37017;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    ALIGNED_LOCAL_MEMORY(mem_37371_backing_0, 4 *
                         sext_i32_i64(mainzitile_sizze_37017 *
                         mainzitile_sizze_37017));
    ALIGNED_LOCAL_MEMORY(mem_37375_backing_1, 4 *
                         sext_i32_i64(mainzitile_sizze_37017 *
                         mainzitile_sizze_37017));
    
    int32_t global_tid_33440;
    int32_t local_tid_33441;
    int32_t group_sizze_37947;
    int32_t wave_sizze_37946;
    int32_t group_id_33442;
    
    global_tid_33440 = get_global_id(0);
    local_tid_33441 = get_local_id(0);
    group_sizze_37947 = get_local_size(0);
    wave_sizze_37946 = LOCKSTEP_WIDTH;
    group_id_33442 = get_group_id(0);
    
    int32_t gtid_33431;
    int32_t gtid_33432;
    int32_t ltid_37020;
    int32_t ltid_37021;
    
    gtid_33431 = squot32(srem32(global_tid_33440, tile_sizze_37018 *
                                tile_sizze_37018), tile_sizze_37018) +
        squot32(squot32(global_tid_33440, tile_sizze_37018 * tile_sizze_37018),
                squot32(res_30780 + tile_sizze_37018 - 1, tile_sizze_37018)) *
        tile_sizze_37018;
    gtid_33432 = srem32(global_tid_33440, tile_sizze_37018 * tile_sizze_37018) -
        squot32(srem32(global_tid_33440, tile_sizze_37018 * tile_sizze_37018),
                tile_sizze_37018) * tile_sizze_37018 +
        (squot32(global_tid_33440, tile_sizze_37018 * tile_sizze_37018) -
         squot32(squot32(global_tid_33440, tile_sizze_37018 * tile_sizze_37018),
                 squot32(res_30780 + tile_sizze_37018 - 1, tile_sizze_37018)) *
         squot32(res_30780 + tile_sizze_37018 - 1, tile_sizze_37018)) *
        tile_sizze_37018;
    ltid_37020 = squot32(srem32(global_tid_33440, tile_sizze_37018 *
                                tile_sizze_37018), tile_sizze_37018);
    ltid_37021 = srem32(global_tid_33440, tile_sizze_37018 * tile_sizze_37018) -
        squot32(srem32(global_tid_33440, tile_sizze_37018 * tile_sizze_37018),
                tile_sizze_37018) * tile_sizze_37018;
    if (slt32(gtid_33431, sizze_30757) && slt32(gtid_33432, res_30780)) { }
    
    __local char *mem_37371;
    __local char *mem_37375;
    float res_33578;
    
    mem_37371 = (__local char *) mem_37371_backing_0;
    mem_37375 = (__local char *) mem_37375_backing_1;
    
    float x_33581 = 0.0F;
    int32_t chunk_sizze_33579;
    int32_t chunk_offset_33580 = 0;
    
    while (slt32(chunk_offset_33580, n_30761)) {
        if (slt32(n_30761 - chunk_offset_33580, tile_sizze_37018)) {
            chunk_sizze_33579 = n_30761 - chunk_offset_33580;
        } else {
            chunk_sizze_33579 = tile_sizze_37018;
        }
        for (int32_t comb_iter_37948 = 0; comb_iter_37948 <
             squot32(tile_sizze_37018 * tile_sizze_37018 +
                     tiled_group_sizze_37019 - 1, tiled_group_sizze_37019);
             comb_iter_37948++) {
            int32_t cid_37033;
            int32_t cid_37034;
            int32_t flat_comb_id_37949 = comb_iter_37948 *
                    tiled_group_sizze_37019 + local_tid_33441;
            
            cid_37033 = squot32(flat_comb_id_37949, tile_sizze_37018);
            cid_37034 = flat_comb_id_37949 - squot32(flat_comb_id_37949,
                                                     tile_sizze_37018) *
                tile_sizze_37018;
            if ((slt32(cid_37033, chunk_sizze_33579) && slt32(cid_37034,
                                                              tile_sizze_37018)) &&
                slt32(gtid_33432, res_30780)) {
                float x_chunk_outer_elem_37032 = *(__global
                                                   float *) &mem_37214[(res_30780 *
                                                                        0 +
                                                                        gtid_33432 +
                                                                        res_30780 *
                                                                        chunk_offset_33580 +
                                                                        ltid_37020 *
                                                                        res_30780) *
                                                                       4];
                
                *(__local float *) &mem_37371[(cid_37033 * tile_sizze_37018 +
                                               cid_37034) * 4] =
                    x_chunk_outer_elem_37032;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (slt32(gtid_33431, sizze_30757) && slt32(gtid_33432, res_30780)) { }
        for (int32_t comb_iter_37950 = 0; comb_iter_37950 <
             squot32(tile_sizze_37018 * tile_sizze_37018 +
                     tiled_group_sizze_37019 - 1, tiled_group_sizze_37019);
             comb_iter_37950++) {
            int32_t cid_37038;
            int32_t cid_37039;
            int32_t flat_comb_id_37951 = comb_iter_37950 *
                    tiled_group_sizze_37019 + local_tid_33441;
            
            cid_37038 = squot32(flat_comb_id_37951, tile_sizze_37018);
            cid_37039 = flat_comb_id_37951 - squot32(flat_comb_id_37951,
                                                     tile_sizze_37018) *
                tile_sizze_37018;
            if ((slt32(cid_37038, tile_sizze_37018) && slt32(cid_37039,
                                                             chunk_sizze_33579)) &&
                slt32(gtid_33431, sizze_30757)) {
                float x_chunk_outer_elem_37037 = *(__global
                                                   float *) &images_mem_37201[(gtid_33431 *
                                                                               sizze_30758 +
                                                                               chunk_offset_33580 +
                                                                               ltid_37021) *
                                                                              4];
                
                *(__local float *) &mem_37375[(cid_37038 * tile_sizze_37018 +
                                               cid_37039) * 4] =
                    x_chunk_outer_elem_37037;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (slt32(gtid_33431, sizze_30757) && slt32(gtid_33432, res_30780)) { }
        
        float res_33584;
        float sync_37041;
        float acc_33587 = x_33581;
        int32_t groupstream_mapaccum_dummy_chunk_sizze_33585;
        
        groupstream_mapaccum_dummy_chunk_sizze_33585 = 1;
        if (slt32(gtid_33431, sizze_30757) && slt32(gtid_33432, res_30780)) {
            if (chunk_sizze_33579 == tile_sizze_37018) {
                for (int32_t i_33586 = 0; i_33586 < tile_sizze_37018;
                     i_33586++) {
                    float x_33590;
                    float x_33591;
                    bool res_33593;
                    float res_33594;
                    float res_33597;
                    
                    x_33590 = *(__local float *) &mem_37371[(tile_sizze_37018 *
                                                             0 + ltid_37021 +
                                                             tile_sizze_37018 *
                                                             i_33586 + 0 *
                                                             tile_sizze_37018) *
                                                            4];
                    x_33591 = *(__local float *) &mem_37375[(ltid_37020 *
                                                             tile_sizze_37018 +
                                                             i_33586) * 4];
                    res_33593 = futrts_isnan32(x_33591);
                    if (res_33593) {
                        res_33594 = 0.0F;
                    } else {
                        float res_33595 = x_33590 * x_33591;
                        
                        res_33594 = res_33595;
                    }
                    res_33597 = acc_33587 + res_33594;
                    
                    float acc_tmp_37952 = res_33597;
                    
                    acc_33587 = acc_tmp_37952;
                }
            } else {
                for (int32_t i_33586 = 0; i_33586 < chunk_sizze_33579;
                     i_33586++) {
                    float x_33590;
                    float x_33591;
                    bool res_33593;
                    float res_33594;
                    float res_33597;
                    
                    x_33590 = *(__local float *) &mem_37371[(tile_sizze_37018 *
                                                             0 + ltid_37021 +
                                                             tile_sizze_37018 *
                                                             i_33586 + 0 *
                                                             tile_sizze_37018) *
                                                            4];
                    x_33591 = *(__local float *) &mem_37375[(ltid_37020 *
                                                             tile_sizze_37018 +
                                                             i_33586) * 4];
                    res_33593 = futrts_isnan32(x_33591);
                    if (res_33593) {
                        res_33594 = 0.0F;
                    } else {
                        float res_33595 = x_33590 * x_33591;
                        
                        res_33594 = res_33595;
                    }
                    res_33597 = acc_33587 + res_33594;
                    
                    float acc_tmp_37953 = res_33597;
                    
                    acc_33587 = acc_tmp_37953;
                }
            }
        }
        res_33584 = acc_33587;
        sync_37041 = res_33584;
        barrier(CLK_LOCAL_MEM_FENCE);
        x_33581 = sync_37041;
        chunk_offset_33580 += tile_sizze_37018;
    }
    res_33578 = x_33581;
    if (slt32(gtid_33431, sizze_30757) && slt32(gtid_33432, res_30780)) {
        *(__global float *) &mem_37379[(gtid_33431 * res_30780 + gtid_33432) *
                                       4] = res_33578;
    }
}
__kernel void map_33682(int32_t sizze_30757, int32_t res_30780,
                        int32_t j_m_i_30913, __global unsigned char *mem_37397,
                        __global unsigned char *mem_37402, __global
                        unsigned char *mem_37405, __global
                        unsigned char *mem_37409)
{
    const int32_t group_sizze_33703 = mainzigroup_sizze_33676;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_33682;
    int32_t local_tid_33683;
    int32_t group_sizze_38002;
    int32_t wave_sizze_38001;
    int32_t group_id_33684;
    
    global_tid_33682 = get_global_id(0);
    local_tid_33683 = get_local_id(0);
    group_sizze_38002 = get_local_size(0);
    wave_sizze_38001 = LOCKSTEP_WIDTH;
    group_id_33684 = get_group_id(0);
    
    int32_t gtid_33675;
    
    gtid_33675 = global_tid_33682;
    if (slt32(gtid_33675, sizze_30757)) {
        for (int32_t i_33715 = 0; i_33715 < res_30780; i_33715++) {
            float res_33717;
            float redout_33718 = 0.0F;
            
            for (int32_t i_33719 = 0; i_33719 < j_m_i_30913; i_33719++) {
                float x_33720;
                float x_33721;
                float res_33722;
                float res_33725;
                
                x_33720 = *(__global float *) &mem_37397[(i_33719 *
                                                          sizze_30757 +
                                                          gtid_33675) * 4];
                x_33721 = *(__global float *) &mem_37402[(i_33719 *
                                                          (sizze_30757 *
                                                           res_30780) +
                                                          i_33715 *
                                                          sizze_30757 +
                                                          gtid_33675) * 4];
                res_33722 = x_33720 * x_33721;
                res_33725 = redout_33718 + res_33722;
                
                float redout_tmp_38004 = res_33725;
                
                redout_33718 = redout_tmp_38004;
            }
            res_33717 = redout_33718;
            *(__global float *) &mem_37405[(group_id_33684 *
                                            (group_sizze_33703 * res_30780) +
                                            local_tid_33683 + i_33715 *
                                            group_sizze_33703) * 4] = res_33717;
        }
    }
    if (slt32(gtid_33675, sizze_30757)) {
        for (int32_t i_38005 = 0; i_38005 < res_30780; i_38005++) {
            *(__global float *) &mem_37409[(gtid_33675 + i_38005 *
                                            sizze_30757) * 4] = *(__global
                                                                  float *) &mem_37405[(group_id_33684 *
                                                                                       (group_sizze_33703 *
                                                                                        res_30780) +
                                                                                       local_tid_33683 +
                                                                                       i_38005 *
                                                                                       group_sizze_33703) *
                                                                                      4];
        }
    }
}
__kernel void map_33777(int32_t sizze_30757, int32_t res_30780,
                        int32_t j_m_i_30913, __global
                        unsigned char *res_mem_37393, __global
                        unsigned char *mem_37431, __global
                        unsigned char *mem_37435)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_33777;
    int32_t local_tid_33778;
    int32_t group_sizze_38013;
    int32_t wave_sizze_38012;
    int32_t group_id_33779;
    
    global_tid_33777 = get_global_id(0);
    local_tid_33778 = get_local_id(0);
    group_sizze_38013 = get_local_size(0);
    wave_sizze_38012 = LOCKSTEP_WIDTH;
    group_id_33779 = get_group_id(0);
    
    int32_t gtid_33768;
    int32_t gtid_33769;
    
    gtid_33768 = squot32(global_tid_33777, res_30780);
    gtid_33769 = global_tid_33777 - squot32(global_tid_33777, res_30780) *
        res_30780;
    
    int32_t binop_x_37167;
    float res_33909;
    
    if (slt32(gtid_33768, sizze_30757) && slt32(gtid_33769, res_30780)) {
        binop_x_37167 = j_m_i_30913 * gtid_33768;
        
        float x_33912 = 0.0F;
        
        for (int32_t chunk_offset_33911 = 0; chunk_offset_33911 < j_m_i_30913;
             chunk_offset_33911++) {
            int32_t binop_x_37168;
            int32_t new_index_37169;
            int32_t binop_y_37175;
            int32_t new_index_37176;
            float x_33921;
            float x_33922;
            float res_33924;
            float res_33926;
            
            binop_x_37168 = chunk_offset_33911 + binop_x_37167;
            new_index_37169 = squot32(binop_x_37168, res_30780);
            binop_y_37175 = res_30780 * new_index_37169;
            new_index_37176 = binop_x_37168 - binop_y_37175;
            x_33921 = *(__global float *) &res_mem_37393[(new_index_37169 *
                                                          res_30780 +
                                                          new_index_37176) * 4];
            x_33922 = *(__global float *) &mem_37431[(chunk_offset_33911 *
                                                      (res_30780 *
                                                       sizze_30757) +
                                                      gtid_33768 * res_30780 +
                                                      gtid_33769) * 4];
            res_33924 = x_33921 * x_33922;
            res_33926 = x_33912 + res_33924;
            
            float x_tmp_38014 = res_33926;
            
            x_33912 = x_tmp_38014;
        }
        res_33909 = x_33912;
    }
    if (slt32(gtid_33768, sizze_30757) && slt32(gtid_33769, res_30780)) {
        *(__global float *) &mem_37435[(gtid_33768 * res_30780 + gtid_33769) *
                                       4] = res_33909;
    }
}
__kernel void map_34007(int32_t sizze_30756, int32_t sizze_30757,
                        int32_t res_30780, __global unsigned char *mem_37218,
                        __global unsigned char *mem_37453, __global
                        unsigned char *mem_37456, __global
                        unsigned char *mem_37460)
{
    const int32_t group_sizze_34026 = mainzigroup_sizze_34001;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_34007;
    int32_t local_tid_34008;
    int32_t group_sizze_38058;
    int32_t wave_sizze_38057;
    int32_t group_id_34009;
    
    global_tid_34007 = get_global_id(0);
    local_tid_34008 = get_local_id(0);
    group_sizze_38058 = get_local_size(0);
    wave_sizze_38057 = LOCKSTEP_WIDTH;
    group_id_34009 = get_group_id(0);
    
    int32_t gtid_34000;
    
    gtid_34000 = global_tid_34007;
    if (slt32(gtid_34000, sizze_30757)) {
        for (int32_t i_34036 = 0; i_34036 < sizze_30756; i_34036++) {
            float res_34038;
            float redout_34039 = 0.0F;
            
            for (int32_t i_34040 = 0; i_34040 < res_30780; i_34040++) {
                float x_34041;
                float x_34042;
                float res_34043;
                float res_34046;
                
                x_34041 = *(__global float *) &mem_37453[(i_34040 *
                                                          sizze_30757 +
                                                          gtid_34000) * 4];
                x_34042 = *(__global float *) &mem_37218[(i_34036 * res_30780 +
                                                          i_34040) * 4];
                res_34043 = x_34041 * x_34042;
                res_34046 = redout_34039 + res_34043;
                
                float redout_tmp_38060 = res_34046;
                
                redout_34039 = redout_tmp_38060;
            }
            res_34038 = redout_34039;
            *(__global float *) &mem_37456[(group_id_34009 *
                                            (group_sizze_34026 * sizze_30756) +
                                            local_tid_34008 + i_34036 *
                                            group_sizze_34026) * 4] = res_34038;
        }
    }
    if (slt32(gtid_34000, sizze_30757)) {
        for (int32_t i_38061 = 0; i_38061 < sizze_30756; i_38061++) {
            *(__global float *) &mem_37460[(gtid_34000 + i_38061 *
                                            sizze_30757) * 4] = *(__global
                                                                  float *) &mem_37456[(group_id_34009 *
                                                                                       (group_sizze_34026 *
                                                                                        sizze_30756) +
                                                                                       local_tid_34008 +
                                                                                       i_38061 *
                                                                                       group_sizze_34026) *
                                                                                      4];
        }
    }
}
__kernel void map_34095(int32_t sizze_30756, int32_t sizze_30757,
                        int32_t res_30780, __global
                        unsigned char *res_mem_37449, __global
                        unsigned char *mem_37480, __global
                        unsigned char *mem_37492)
{
    const int32_t tile_sizze_37068 = mainzitile_sizze_37067;
    const int32_t tiled_group_sizze_37069 = mainzitile_sizze_37067 *
                  mainzitile_sizze_37067;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    ALIGNED_LOCAL_MEMORY(mem_37484_backing_0, 4 *
                         sext_i32_i64(mainzitile_sizze_37067 *
                         mainzitile_sizze_37067));
    ALIGNED_LOCAL_MEMORY(mem_37488_backing_1, 4 *
                         sext_i32_i64(mainzitile_sizze_37067 *
                         mainzitile_sizze_37067));
    
    int32_t global_tid_34095;
    int32_t local_tid_34096;
    int32_t group_sizze_38069;
    int32_t wave_sizze_38068;
    int32_t group_id_34097;
    
    global_tid_34095 = get_global_id(0);
    local_tid_34096 = get_local_id(0);
    group_sizze_38069 = get_local_size(0);
    wave_sizze_38068 = LOCKSTEP_WIDTH;
    group_id_34097 = get_group_id(0);
    
    int32_t gtid_34086;
    int32_t gtid_34087;
    int32_t ltid_37070;
    int32_t ltid_37071;
    
    gtid_34086 = squot32(srem32(global_tid_34095, tile_sizze_37068 *
                                tile_sizze_37068), tile_sizze_37068) +
        squot32(squot32(global_tid_34095, tile_sizze_37068 * tile_sizze_37068),
                squot32(sizze_30756 + tile_sizze_37068 - 1, tile_sizze_37068)) *
        tile_sizze_37068;
    gtid_34087 = srem32(global_tid_34095, tile_sizze_37068 * tile_sizze_37068) -
        squot32(srem32(global_tid_34095, tile_sizze_37068 * tile_sizze_37068),
                tile_sizze_37068) * tile_sizze_37068 +
        (squot32(global_tid_34095, tile_sizze_37068 * tile_sizze_37068) -
         squot32(squot32(global_tid_34095, tile_sizze_37068 * tile_sizze_37068),
                 squot32(sizze_30756 + tile_sizze_37068 - 1,
                         tile_sizze_37068)) * squot32(sizze_30756 +
                                                      tile_sizze_37068 - 1,
                                                      tile_sizze_37068)) *
        tile_sizze_37068;
    ltid_37070 = squot32(srem32(global_tid_34095, tile_sizze_37068 *
                                tile_sizze_37068), tile_sizze_37068);
    ltid_37071 = srem32(global_tid_34095, tile_sizze_37068 * tile_sizze_37068) -
        squot32(srem32(global_tid_34095, tile_sizze_37068 * tile_sizze_37068),
                tile_sizze_37068) * tile_sizze_37068;
    if (slt32(gtid_34086, sizze_30757) && slt32(gtid_34087, sizze_30756)) { }
    
    __local char *mem_37484;
    __local char *mem_37488;
    float res_34227;
    
    mem_37484 = (__local char *) mem_37484_backing_0;
    mem_37488 = (__local char *) mem_37488_backing_1;
    
    float x_34230 = 0.0F;
    int32_t chunk_sizze_34228;
    int32_t chunk_offset_34229 = 0;
    
    while (slt32(chunk_offset_34229, res_30780)) {
        if (slt32(res_30780 - chunk_offset_34229, tile_sizze_37068)) {
            chunk_sizze_34228 = res_30780 - chunk_offset_34229;
        } else {
            chunk_sizze_34228 = tile_sizze_37068;
        }
        for (int32_t comb_iter_38070 = 0; comb_iter_38070 <
             squot32(tile_sizze_37068 * tile_sizze_37068 +
                     tiled_group_sizze_37069 - 1, tiled_group_sizze_37069);
             comb_iter_38070++) {
            int32_t cid_37083;
            int32_t cid_37084;
            int32_t flat_comb_id_38071 = comb_iter_38070 *
                    tiled_group_sizze_37069 + local_tid_34096;
            
            cid_37083 = squot32(flat_comb_id_38071, tile_sizze_37068);
            cid_37084 = flat_comb_id_38071 - squot32(flat_comb_id_38071,
                                                     tile_sizze_37068) *
                tile_sizze_37068;
            if ((slt32(cid_37083, tile_sizze_37068) && slt32(cid_37084,
                                                             chunk_sizze_34228)) &&
                slt32(gtid_34086, sizze_30757)) {
                float x_chunk_outer_elem_37082 = *(__global
                                                   float *) &res_mem_37449[(gtid_34086 *
                                                                            res_30780 +
                                                                            chunk_offset_34229 +
                                                                            ltid_37071) *
                                                                           4];
                
                *(__local float *) &mem_37484[(cid_37083 * tile_sizze_37068 +
                                               cid_37084) * 4] =
                    x_chunk_outer_elem_37082;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (slt32(gtid_34086, sizze_30757) && slt32(gtid_34087,
                                                    sizze_30756)) { }
        for (int32_t comb_iter_38072 = 0; comb_iter_38072 <
             squot32(tile_sizze_37068 * tile_sizze_37068 +
                     tiled_group_sizze_37069 - 1, tiled_group_sizze_37069);
             comb_iter_38072++) {
            int32_t cid_37088;
            int32_t cid_37089;
            int32_t flat_comb_id_38073 = comb_iter_38072 *
                    tiled_group_sizze_37069 + local_tid_34096;
            
            cid_37088 = squot32(flat_comb_id_38073, tile_sizze_37068);
            cid_37089 = flat_comb_id_38073 - squot32(flat_comb_id_38073,
                                                     tile_sizze_37068) *
                tile_sizze_37068;
            if ((slt32(cid_37088, chunk_sizze_34228) && slt32(cid_37089,
                                                              tile_sizze_37068)) &&
                slt32(gtid_34087, sizze_30756)) {
                float x_chunk_outer_elem_37087 = *(__global
                                                   float *) &mem_37480[(gtid_34087 +
                                                                        sizze_30756 *
                                                                        chunk_offset_34229 +
                                                                        ltid_37070 *
                                                                        sizze_30756) *
                                                                       4];
                
                *(__local float *) &mem_37488[(cid_37088 * tile_sizze_37068 +
                                               cid_37089) * 4] =
                    x_chunk_outer_elem_37087;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (slt32(gtid_34086, sizze_30757) && slt32(gtid_34087,
                                                    sizze_30756)) { }
        
        float res_34233;
        float sync_37091;
        float acc_34236 = x_34230;
        int32_t groupstream_mapaccum_dummy_chunk_sizze_34234;
        
        groupstream_mapaccum_dummy_chunk_sizze_34234 = 1;
        if (slt32(gtid_34086, sizze_30757) && slt32(gtid_34087, sizze_30756)) {
            if (chunk_sizze_34228 == tile_sizze_37068) {
                for (int32_t i_34235 = 0; i_34235 < tile_sizze_37068;
                     i_34235++) {
                    float x_34239;
                    float x_34240;
                    float res_34242;
                    float res_34244;
                    
                    x_34239 = *(__local float *) &mem_37484[(ltid_37070 *
                                                             tile_sizze_37068 +
                                                             i_34235) * 4];
                    x_34240 = *(__local float *) &mem_37488[(tile_sizze_37068 *
                                                             0 + ltid_37071 +
                                                             tile_sizze_37068 *
                                                             i_34235 + 0 *
                                                             tile_sizze_37068) *
                                                            4];
                    res_34242 = x_34239 * x_34240;
                    res_34244 = acc_34236 + res_34242;
                    
                    float acc_tmp_38074 = res_34244;
                    
                    acc_34236 = acc_tmp_38074;
                }
            } else {
                for (int32_t i_34235 = 0; i_34235 < chunk_sizze_34228;
                     i_34235++) {
                    float x_34239;
                    float x_34240;
                    float res_34242;
                    float res_34244;
                    
                    x_34239 = *(__local float *) &mem_37484[(ltid_37070 *
                                                             tile_sizze_37068 +
                                                             i_34235) * 4];
                    x_34240 = *(__local float *) &mem_37488[(tile_sizze_37068 *
                                                             0 + ltid_37071 +
                                                             tile_sizze_37068 *
                                                             i_34235 + 0 *
                                                             tile_sizze_37068) *
                                                            4];
                    res_34242 = x_34239 * x_34240;
                    res_34244 = acc_34236 + res_34242;
                    
                    float acc_tmp_38075 = res_34244;
                    
                    acc_34236 = acc_tmp_38075;
                }
            }
        }
        res_34233 = acc_34236;
        sync_37091 = res_34233;
        barrier(CLK_LOCAL_MEM_FENCE);
        x_34230 = sync_37091;
        chunk_offset_34229 += tile_sizze_37068;
    }
    res_34227 = x_34230;
    if (slt32(gtid_34086, sizze_30757) && slt32(gtid_34087, sizze_30756)) {
        *(__global float *) &mem_37492[(gtid_34086 * sizze_30756 + gtid_34087) *
                                       4] = res_34227;
    }
}
__kernel void map_34343(int32_t sizze_30756, int32_t sizze_30757,
                        int32_t i_31033, __global unsigned char *mem_37510,
                        __global unsigned char *mem_37514, __global
                        unsigned char *mem_37517, __global
                        unsigned char *mem_37520, __global
                        unsigned char *mem_37523, __global
                        unsigned char *mem_37526, __global
                        unsigned char *mem_37532, __global
                        unsigned char *mem_37536, __global
                        unsigned char *mem_37540)
{
    const int32_t group_sizze_34401 = mainzigroup_sizze_34337;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_34343;
    int32_t local_tid_34344;
    int32_t group_sizze_38119;
    int32_t wave_sizze_38118;
    int32_t group_id_34345;
    
    global_tid_34343 = get_global_id(0);
    local_tid_34344 = get_local_id(0);
    group_sizze_38119 = get_local_size(0);
    wave_sizze_38118 = LOCKSTEP_WIDTH;
    group_id_34345 = get_group_id(0);
    
    int32_t gtid_34336;
    
    gtid_34336 = global_tid_34343;
    
    int32_t discard_34413;
    int32_t res_34434;
    
    if (slt32(gtid_34336, sizze_30757)) {
        int32_t scanacc_34416 = 0;
        
        for (int32_t i_34419 = 0; i_34419 < sizze_30756; i_34419++) {
            float x_34420;
            float x_34421;
            bool res_34422;
            bool cond_34423;
            float res_34424;
            bool res_34426;
            bool res_34427;
            int32_t res_34428;
            int32_t res_34431;
            
            x_34420 = *(__global float *) &mem_37510[(i_34419 * sizze_30757 +
                                                      gtid_34336) * 4];
            x_34421 = *(__global float *) &mem_37514[(i_34419 * sizze_30757 +
                                                      gtid_34336) * 4];
            res_34422 = futrts_isnan32(x_34420);
            cond_34423 = !res_34422;
            if (cond_34423) {
                float res_34425 = x_34420 - x_34421;
                
                res_34424 = res_34425;
            } else {
                res_34424 = NAN;
            }
            res_34426 = futrts_isnan32(res_34424);
            res_34427 = !res_34426;
            if (res_34427) {
                res_34428 = 1;
            } else {
                res_34428 = 0;
            }
            res_34431 = scanacc_34416 + res_34428;
            *(__global int32_t *) &mem_37517[(group_id_34345 *
                                              (group_sizze_34401 *
                                               sizze_30756) + local_tid_34344 +
                                              i_34419 * group_sizze_34401) *
                                             4] = res_34431;
            *(__global float *) &mem_37520[(group_id_34345 *
                                            (group_sizze_34401 * sizze_30756) +
                                            local_tid_34344 + i_34419 *
                                            group_sizze_34401) * 4] = res_34424;
            
            int32_t scanacc_tmp_38120 = res_34431;
            
            scanacc_34416 = scanacc_tmp_38120;
        }
        discard_34413 = scanacc_34416;
        res_34434 = *(__global int32_t *) &mem_37517[(group_id_34345 *
                                                      (group_sizze_34401 *
                                                       sizze_30756) +
                                                      local_tid_34344 +
                                                      i_31033 *
                                                      group_sizze_34401) * 4];
        for (int32_t i_38123 = 0; i_38123 < sizze_30756; i_38123++) {
            *(__global float *) &mem_37523[(group_id_34345 *
                                            (group_sizze_34401 * sizze_30756) +
                                            local_tid_34344 + i_38123 *
                                            group_sizze_34401) * 4] = NAN;
        }
        for (int32_t i_38124 = 0; i_38124 < sizze_30756; i_38124++) {
            *(__global int32_t *) &mem_37526[(group_id_34345 *
                                              (group_sizze_34401 *
                                               sizze_30756) + local_tid_34344 +
                                              i_38124 * group_sizze_34401) *
                                             4] = 0;
        }
    }
    
    __private char *mem_37529;
    __private char mem_37529_backing_0[4];
    
    mem_37529 = mem_37529_backing_0;
    if (slt32(gtid_34336, sizze_30757)) {
        for (int32_t write_iter_34441 = 0; write_iter_34441 < sizze_30756;
             write_iter_34441++) {
            float write_iv_34442;
            int32_t write_iv_34443;
            bool res_34448;
            bool res_34449;
            int32_t res_34450;
            bool less_than_zzero_34452;
            bool greater_than_sizze_34453;
            bool outside_bounds_dim_34454;
            
            write_iv_34442 = *(__global float *) &mem_37520[(group_id_34345 *
                                                             (group_sizze_34401 *
                                                              sizze_30756) +
                                                             local_tid_34344 +
                                                             write_iter_34441 *
                                                             group_sizze_34401) *
                                                            4];
            write_iv_34443 = *(__global int32_t *) &mem_37517[(group_id_34345 *
                                                               (group_sizze_34401 *
                                                                sizze_30756) +
                                                               local_tid_34344 +
                                                               write_iter_34441 *
                                                               group_sizze_34401) *
                                                              4];
            res_34448 = futrts_isnan32(write_iv_34442);
            res_34449 = !res_34448;
            if (res_34449) {
                int32_t res_34451 = write_iv_34443 - 1;
                
                res_34450 = res_34451;
            } else {
                res_34450 = -1;
            }
            less_than_zzero_34452 = slt32(res_34450, 0);
            greater_than_sizze_34453 = sle32(sizze_30756, res_34450);
            outside_bounds_dim_34454 = less_than_zzero_34452 ||
                greater_than_sizze_34453;
            if (!outside_bounds_dim_34454) {
                int32_t x_38128;
                
                for (int32_t i_38127 = 0; i_38127 < 1; i_38127++) {
                    x_38128 = write_iter_34441 + sext_i32_i32(i_38127);
                    *(__private int32_t *) &mem_37529[i_38127 * 4] = x_38128;
                }
                for (int32_t i_38129 = 0; i_38129 < 1; i_38129++) {
                    *(__global int32_t *) &mem_37526[(group_id_34345 *
                                                      (group_sizze_34401 *
                                                       sizze_30756) +
                                                      local_tid_34344 +
                                                      group_sizze_34401 *
                                                      res_34450 + i_38129 *
                                                      group_sizze_34401) * 4] =
                        *(__private int32_t *) &mem_37529[i_38129 * 4];
                }
            }
            if (!outside_bounds_dim_34454) {
                for (int32_t i_38130 = 0; i_38130 < 1; i_38130++) {
                    *(__global float *) &mem_37523[(group_id_34345 *
                                                    (group_sizze_34401 *
                                                     sizze_30756) +
                                                    local_tid_34344 +
                                                    group_sizze_34401 *
                                                    res_34450 + i_38130 *
                                                    group_sizze_34401) * 4] =
                        *(__global float *) &mem_37520[(group_id_34345 *
                                                        (group_sizze_34401 *
                                                         sizze_30756) +
                                                        local_tid_34344 +
                                                        group_sizze_34401 *
                                                        write_iter_34441 +
                                                        i_38130 *
                                                        group_sizze_34401) * 4];
                }
            }
        }
    }
    if (slt32(gtid_34336, sizze_30757)) {
        *(__global int32_t *) &mem_37532[gtid_34336 * 4] = res_34434;
    }
    if (slt32(gtid_34336, sizze_30757)) {
        for (int32_t i_38131 = 0; i_38131 < sizze_30756; i_38131++) {
            *(__global float *) &mem_37536[(gtid_34336 + i_38131 *
                                            sizze_30757) * 4] = *(__global
                                                                  float *) &mem_37523[(group_id_34345 *
                                                                                       (group_sizze_34401 *
                                                                                        sizze_30756) +
                                                                                       local_tid_34344 +
                                                                                       i_38131 *
                                                                                       group_sizze_34401) *
                                                                                      4];
        }
    }
    if (slt32(gtid_34336, sizze_30757)) {
        for (int32_t i_38132 = 0; i_38132 < sizze_30756; i_38132++) {
            *(__global int32_t *) &mem_37540[(gtid_34336 + i_38132 *
                                              sizze_30757) * 4] = *(__global
                                                                    int32_t *) &mem_37526[(group_id_34345 *
                                                                                           (group_sizze_34401 *
                                                                                            sizze_30756) +
                                                                                           local_tid_34344 +
                                                                                           i_38132 *
                                                                                           group_sizze_34401) *
                                                                                          4];
        }
    }
}
__kernel void map_34518(int32_t sizze_30756, int32_t sizze_30757, __global
                        unsigned char *mem_37577, __global
                        unsigned char *mem_37581, __global
                        unsigned char *mem_37588, __global
                        unsigned char *mem_37592)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_34518;
    int32_t local_tid_34519;
    int32_t group_sizze_38208;
    int32_t wave_sizze_38207;
    int32_t group_id_34520;
    
    global_tid_34518 = get_global_id(0);
    local_tid_34519 = get_local_id(0);
    group_sizze_38208 = get_local_size(0);
    wave_sizze_38207 = LOCKSTEP_WIDTH;
    group_id_34520 = get_group_id(0);
    
    int32_t gtid_34509;
    int32_t gtid_34510;
    
    gtid_34509 = squot32(global_tid_34518, sizze_30756);
    gtid_34510 = global_tid_34518 - squot32(global_tid_34518, sizze_30756) *
        sizze_30756;
    
    float x_34720;
    int32_t x_34721;
    bool res_34723;
    bool res_34724;
    int32_t res_34725;
    
    if (slt32(gtid_34509, sizze_30757) && slt32(gtid_34510, sizze_30756)) {
        x_34720 = *(__global float *) &mem_37581[(gtid_34509 * sizze_30756 +
                                                  gtid_34510) * 4];
        x_34721 = *(__global int32_t *) &mem_37577[(gtid_34509 * sizze_30756 +
                                                    gtid_34510) * 4];
        res_34723 = futrts_isnan32(x_34720);
        res_34724 = !res_34723;
        if (res_34724) {
            int32_t res_34726 = x_34721 - 1;
            
            res_34725 = res_34726;
        } else {
            res_34725 = -1;
        }
    }
    if (((slt32(gtid_34509, sizze_30757) && slt32(gtid_34510, sizze_30756)) &&
         (sle32(0, gtid_34509) && slt32(gtid_34509, sizze_30757))) && (sle32(0,
                                                                             res_34725) &&
                                                                       slt32(res_34725,
                                                                             sizze_30756))) {
        *(__global int32_t *) &mem_37592[(gtid_34509 * sizze_30756 +
                                          res_34725) * 4] = gtid_34510;
    }
    if (((slt32(gtid_34509, sizze_30757) && slt32(gtid_34510, sizze_30756)) &&
         (sle32(0, gtid_34509) && slt32(gtid_34509, sizze_30757))) && (sle32(0,
                                                                             res_34725) &&
                                                                       slt32(res_34725,
                                                                             sizze_30756))) {
        *(__global float *) &mem_37588[(gtid_34509 * sizze_30756 + res_34725) *
                                       4] = x_34720;
    }
}
__kernel void map_34772(int32_t sizze_30757, int32_t n_30761, float hfrac_30763,
                        int32_t res_30778, __global unsigned char *mem_37602,
                        __global unsigned char *mem_37606, __global
                        unsigned char *mem_37609, __global
                        unsigned char *mem_37612, __global
                        unsigned char *mem_37615)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_34772;
    int32_t local_tid_34773;
    int32_t group_sizze_38210;
    int32_t wave_sizze_38209;
    int32_t group_id_34774;
    
    global_tid_34772 = get_global_id(0);
    local_tid_34773 = get_local_id(0);
    group_sizze_38210 = get_local_size(0);
    wave_sizze_38209 = LOCKSTEP_WIDTH;
    group_id_34774 = get_group_id(0);
    
    int32_t gtid_34765;
    
    gtid_34765 = global_tid_34772;
    
    int32_t res_34829;
    float res_34846;
    int32_t arg_34864;
    float res_34865;
    float arg_34866;
    float res_34867;
    float res_34868;
    float arg_34869;
    int32_t res_34870;
    
    if (slt32(gtid_34765, sizze_30757)) {
        int32_t x_34832 = 0;
        
        for (int32_t chunk_offset_34831 = 0; chunk_offset_34831 < n_30761;
             chunk_offset_34831++) {
            float x_34839;
            bool res_34841;
            bool cond_34842;
            int32_t res_34843;
            int32_t res_34845;
            
            x_34839 = *(__global float *) &mem_37602[(chunk_offset_34831 *
                                                      sizze_30757 +
                                                      gtid_34765) * 4];
            res_34841 = futrts_isnan32(x_34839);
            cond_34842 = !res_34841;
            if (cond_34842) {
                res_34843 = 1;
            } else {
                res_34843 = 0;
            }
            res_34845 = x_34832 + res_34843;
            
            int32_t x_tmp_38211 = res_34845;
            
            x_34832 = x_tmp_38211;
        }
        res_34829 = x_34832;
        
        float x_34849 = 0.0F;
        
        for (int32_t chunk_offset_34848 = 0; chunk_offset_34848 < n_30761;
             chunk_offset_34848++) {
            bool cond_34858;
            float res_34859;
            float res_34861;
            float res_34863;
            
            cond_34858 = slt32(chunk_offset_34848, res_34829);
            if (cond_34858) {
                float res_34860 = *(__global
                                    float *) &mem_37606[(chunk_offset_34848 *
                                                         sizze_30757 +
                                                         gtid_34765) * 4];
                
                res_34859 = res_34860;
            } else {
                res_34859 = 0.0F;
            }
            res_34861 = res_34859 * res_34859;
            res_34863 = x_34849 + res_34861;
            
            float x_tmp_38212 = res_34863;
            
            x_34849 = x_tmp_38212;
        }
        res_34846 = x_34849;
        arg_34864 = res_34829 - res_30778;
        res_34865 = sitofp_i32_f32(arg_34864);
        arg_34866 = res_34846 / res_34865;
        res_34867 = futrts_sqrt32(arg_34866);
        res_34868 = sitofp_i32_f32(res_34829);
        arg_34869 = hfrac_30763 * res_34868;
        res_34870 = fptosi_f32_i32(arg_34869);
    }
    if (slt32(gtid_34765, sizze_30757)) {
        *(__global int32_t *) &mem_37609[gtid_34765 * 4] = res_34870;
    }
    if (slt32(gtid_34765, sizze_30757)) {
        *(__global int32_t *) &mem_37612[gtid_34765 * 4] = res_34829;
    }
    if (slt32(gtid_34765, sizze_30757)) {
        *(__global float *) &mem_37615[gtid_34765 * 4] = res_34867;
    }
}
__kernel void map_34917(int32_t sizze_30757, float hfrac_30763,
                        int32_t res_30778, __global unsigned char *mem_37633,
                        __global unsigned char *mem_37636, __global
                        unsigned char *mem_37639, __global
                        unsigned char *mem_37642)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_34917;
    int32_t local_tid_34918;
    int32_t group_sizze_38297;
    int32_t wave_sizze_38296;
    int32_t group_id_34919;
    
    global_tid_34917 = get_global_id(0);
    local_tid_34918 = get_local_id(0);
    group_sizze_38297 = get_local_size(0);
    wave_sizze_38296 = LOCKSTEP_WIDTH;
    group_id_34919 = get_group_id(0);
    
    int32_t gtid_34910;
    
    gtid_34910 = global_tid_34917;
    
    int32_t res_35044;
    float res_35045;
    int32_t arg_35046;
    float res_35047;
    float arg_35048;
    float res_35049;
    float res_35050;
    float arg_35051;
    int32_t res_35052;
    
    if (slt32(gtid_34910, sizze_30757)) {
        res_35044 = *(__global int32_t *) &mem_37633[gtid_34910 * 4];
        res_35045 = *(__global float *) &mem_37636[gtid_34910 * 4];
        arg_35046 = res_35044 - res_30778;
        res_35047 = sitofp_i32_f32(arg_35046);
        arg_35048 = res_35045 / res_35047;
        res_35049 = futrts_sqrt32(arg_35048);
        res_35050 = sitofp_i32_f32(res_35044);
        arg_35051 = hfrac_30763 * res_35050;
        res_35052 = fptosi_f32_i32(arg_35051);
    }
    if (slt32(gtid_34910, sizze_30757)) {
        *(__global int32_t *) &mem_37639[gtid_34910 * 4] = res_35052;
    }
    if (slt32(gtid_34910, sizze_30757)) {
        *(__global float *) &mem_37642[gtid_34910 * 4] = res_35049;
    }
}
__kernel void map_35109(int32_t sizze_30756, int32_t sizze_30757,
                        int32_t res_31136, __global
                        unsigned char *res_mem_37597, __global
                        unsigned char *res_mem_37646, __global
                        unsigned char *res_mem_37647, __global
                        unsigned char *mem_37654)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_35109;
    int32_t local_tid_35110;
    int32_t group_sizze_38321;
    int32_t wave_sizze_38320;
    int32_t group_id_35111;
    
    global_tid_35109 = get_global_id(0);
    local_tid_35110 = get_local_id(0);
    group_sizze_38321 = get_local_size(0);
    wave_sizze_38320 = LOCKSTEP_WIDTH;
    group_id_35111 = get_group_id(0);
    
    int32_t gtid_35102;
    
    gtid_35102 = global_tid_35109;
    
    int32_t x_35142;
    int32_t x_35143;
    float res_35144;
    
    if (slt32(gtid_35102, sizze_30757)) {
        x_35142 = *(__global int32_t *) &res_mem_37647[gtid_35102 * 4];
        x_35143 = *(__global int32_t *) &res_mem_37646[gtid_35102 * 4];
        
        float x_35147 = 0.0F;
        
        for (int32_t chunk_offset_35146 = 0; chunk_offset_35146 < res_31136;
             chunk_offset_35146++) {
            bool cond_35156;
            float res_35157;
            float res_35163;
            
            cond_35156 = slt32(chunk_offset_35146, x_35143);
            if (cond_35156) {
                int32_t x_35158;
                int32_t x_35159;
                int32_t i_35160;
                float res_35161;
                
                x_35158 = x_35142 + chunk_offset_35146;
                x_35159 = x_35158 - x_35143;
                i_35160 = 1 + x_35159;
                res_35161 = *(__global float *) &res_mem_37597[(gtid_35102 *
                                                                sizze_30756 +
                                                                i_35160) * 4];
                res_35157 = res_35161;
            } else {
                res_35157 = 0.0F;
            }
            res_35163 = x_35147 + res_35157;
            
            float x_tmp_38322 = res_35163;
            
            x_35147 = x_tmp_38322;
        }
        res_35144 = x_35147;
    }
    if (slt32(gtid_35102, sizze_30757)) {
        *(__global float *) &mem_37654[gtid_35102 * 4] = res_35144;
    }
}
__kernel void map_35292(float lam_30764, int32_t arg_31158, int32_t x_31171,
                        float res_31174, __global
                        unsigned char *mappingindices_mem_37200, __global
                        unsigned char *mem_37668)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_35292;
    int32_t local_tid_35293;
    int32_t group_sizze_38366;
    int32_t wave_sizze_38365;
    int32_t group_id_35294;
    
    global_tid_35292 = get_global_id(0);
    local_tid_35293 = get_local_id(0);
    group_sizze_38366 = get_local_size(0);
    wave_sizze_38365 = LOCKSTEP_WIDTH;
    group_id_35294 = get_group_id(0);
    
    int32_t gtid_35285;
    
    gtid_35285 = global_tid_35292;
    
    int32_t res_35313;
    int32_t i_35314;
    int32_t res_35315;
    float res_35316;
    float arg_35317;
    bool cond_35318;
    float res_35319;
    float res_35321;
    float res_35322;
    
    if (slt32(gtid_35285, arg_31158)) {
        res_35313 = x_31171 + gtid_35285;
        i_35314 = res_35313 - 1;
        res_35315 = *(__global int32_t *) &mappingindices_mem_37200[i_35314 *
                                                                    4];
        res_35316 = sitofp_i32_f32(res_35315);
        arg_35317 = res_35316 / res_31174;
        cond_35318 = 2.7182817F < arg_35317;
        if (cond_35318) {
            float res_35320;
            
            res_35320 = futrts_log32(arg_35317);
            res_35319 = res_35320;
        } else {
            res_35319 = 1.0F;
        }
        res_35321 = futrts_sqrt32(res_35319);
        res_35322 = lam_30764 * res_35321;
    }
    if (slt32(gtid_35285, arg_31158)) {
        *(__global float *) &mem_37668[gtid_35285 * 4] = res_35322;
    }
}
__kernel void map_35371(int32_t sizze_30756, int32_t sizze_30757,
                        int32_t n_30761, int32_t arg_31158, __global
                        unsigned char *res_mem_37596, __global
                        unsigned char *res_mem_37597, __global
                        unsigned char *res_mem_37598, __global
                        unsigned char *res_mem_37646, __global
                        unsigned char *res_mem_37647, __global
                        unsigned char *res_mem_37648, __global
                        unsigned char *res_mem_37665, __global
                        unsigned char *mem_37668, __global
                        unsigned char *mem_37674, __global
                        unsigned char *mem_37677, __global
                        unsigned char *mem_37680)
{
    const int32_t group_sizze_35492 = mainzigroup_sizze_35365;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    ALIGNED_LOCAL_MEMORY(mem_37671_backing_0, 4 *
                         sext_i32_i64(mainzigroup_sizze_35365));
    
    int32_t global_tid_35371;
    int32_t local_tid_35372;
    int32_t group_sizze_38368;
    int32_t wave_sizze_38367;
    int32_t group_id_35373;
    
    global_tid_35371 = get_global_id(0);
    local_tid_35372 = get_local_id(0);
    group_sizze_38368 = get_local_size(0);
    wave_sizze_38367 = LOCKSTEP_WIDTH;
    group_id_35373 = get_group_id(0);
    
    int32_t gtid_35364;
    
    gtid_35364 = global_tid_35371;
    
    int32_t x_35499;
    int32_t x_35500;
    float x_35501;
    int32_t x_35502;
    float x_35503;
    int32_t y_35506;
    float res_35507;
    float res_35508;
    float y_35509;
    
    if (slt32(gtid_35364, sizze_30757)) {
        x_35499 = *(__global int32_t *) &res_mem_37596[gtid_35364 * 4];
        x_35500 = *(__global int32_t *) &res_mem_37647[gtid_35364 * 4];
        x_35501 = *(__global float *) &res_mem_37648[gtid_35364 * 4];
        x_35502 = *(__global int32_t *) &res_mem_37646[gtid_35364 * 4];
        x_35503 = *(__global float *) &res_mem_37665[gtid_35364 * 4];
        y_35506 = x_35499 - x_35500;
        res_35507 = sitofp_i32_f32(x_35500);
        res_35508 = futrts_sqrt32(res_35507);
        y_35509 = x_35501 * res_35508;
    }
    
    __local char *mem_37671;
    float inpacc_35510;
    bool res_35511;
    int32_t res_35512;
    float res_35513;
    
    mem_37671 = (__local char *) mem_37671_backing_0;
    
    float inpacc_35516;
    bool inpacc_35517;
    int32_t inpacc_35518;
    float inpacc_35519;
    
    inpacc_35516 = 0.0F;
    inpacc_35517 = 0;
    inpacc_35518 = -1;
    inpacc_35519 = 0.0F;
    
    int32_t chunk_35514;
    int32_t streamseq_chunk_offset_35515 = 0;
    
    while (slt32(streamseq_chunk_offset_35515, arg_31158)) {
        if (slt32(arg_31158 - streamseq_chunk_offset_35515,
                  group_sizze_35492)) {
            chunk_35514 = arg_31158 - streamseq_chunk_offset_35515;
        } else {
            chunk_35514 = group_sizze_35492;
        }
        for (int32_t comb_iter_38369 = 0; comb_iter_38369 < 1;
             comb_iter_38369++) {
            int32_t cid_37094;
            int32_t flat_comb_id_38370 = comb_iter_38369 * group_sizze_35492 +
                    local_tid_35372;
            
            cid_37094 = flat_comb_id_38370;
            if (slt32(cid_37094, chunk_35514) && 1) {
                float inp_outer_elem_37093 = *(__global
                                               float *) &mem_37668[(streamseq_chunk_offset_35515 +
                                                                    local_tid_35372) *
                                                                   4];
                
                *(__local float *) &mem_37671[cid_37094 * 4] =
                    inp_outer_elem_37093;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        float discard_35523;
        int32_t szzm1_35544;
        bool empty_arr_35545;
        float lstel_35546;
        float res_35548;
        
        if (slt32(gtid_35364, sizze_30757)) {
            float scanacc_35525 = 0.0F;
            
            for (int32_t i_35527 = 0; i_35527 < chunk_35514; i_35527++) {
                int32_t convop_x_36419;
                bool cond_35529;
                float res_35530;
                float res_35542;
                
                convop_x_36419 = streamseq_chunk_offset_35515 + i_35527;
                cond_35529 = sle32(y_35506, convop_x_36419);
                if (cond_35529) {
                    res_35530 = 0.0F;
                } else {
                    bool cond_35531;
                    float res_35532;
                    
                    cond_35531 = convop_x_36419 == 0;
                    if (cond_35531) {
                        res_35532 = x_35503;
                    } else {
                        int32_t x_35533;
                        int32_t i_35534;
                        float negate_arg_35535;
                        float x_35536;
                        int32_t i_35537;
                        float y_35538;
                        float res_35539;
                        
                        x_35533 = x_35500 - x_35502;
                        i_35534 = x_35533 + convop_x_36419;
                        negate_arg_35535 = *(__global
                                             float *) &res_mem_37597[(gtid_35364 *
                                                                      sizze_30756 +
                                                                      i_35534) *
                                                                     4];
                        x_35536 = 0.0F - negate_arg_35535;
                        i_35537 = x_35500 + convop_x_36419;
                        y_35538 = *(__global
                                    float *) &res_mem_37597[(gtid_35364 *
                                                             sizze_30756 +
                                                             i_35537) * 4];
                        res_35539 = x_35536 + y_35538;
                        res_35532 = res_35539;
                    }
                    res_35530 = res_35532;
                }
                res_35542 = scanacc_35525 + res_35530;
                *(__global float *) &mem_37674[(group_id_35373 *
                                                (group_sizze_35492 *
                                                 chunk_35514) +
                                                local_tid_35372 + i_35527 *
                                                group_sizze_35492) * 4] =
                    res_35542;
                
                float scanacc_tmp_38371 = res_35542;
                
                scanacc_35525 = scanacc_tmp_38371;
            }
            discard_35523 = scanacc_35525;
            szzm1_35544 = chunk_35514 - 1;
            empty_arr_35545 = slt32(szzm1_35544, 0);
            if (empty_arr_35545) {
                lstel_35546 = 0.0F;
            } else {
                float lstel_tmp_35547 = *(__global
                                          float *) &mem_37674[(group_id_35373 *
                                                               (group_sizze_35492 *
                                                                chunk_35514) +
                                                               local_tid_35372 +
                                                               szzm1_35544 *
                                                               group_sizze_35492) *
                                                              4];
                
                lstel_35546 = lstel_tmp_35547;
            }
            res_35548 = inpacc_35516 + lstel_35546;
        }
        
        bool acc0_35549;
        int32_t acc0_35550;
        float acc0_35551;
        bool x_35554;
        int32_t x_35555;
        float x_35556;
        
        x_35554 = 0;
        x_35555 = -1;
        x_35556 = 0.0F;
        
        int32_t chunk_sizze_35552;
        int32_t chunk_offset_35553 = 0;
        
        chunk_sizze_35552 = chunk_35514;
        
        bool acc0_35560;
        int32_t acc0_35561;
        float acc0_35562;
        bool acc_35565;
        int32_t acc_35566;
        float acc_35567;
        
        acc_35565 = x_35554;
        acc_35566 = x_35555;
        acc_35567 = x_35556;
        
        int32_t groupstream_mapaccum_dummy_chunk_sizze_35563;
        
        groupstream_mapaccum_dummy_chunk_sizze_35563 = 1;
        if (slt32(gtid_35364, sizze_30757)) {
            if (chunk_sizze_35552 == chunk_35514) {
                for (int32_t i_35564 = 0; i_35564 < chunk_35514; i_35564++) {
                    float x_35571;
                    float x_35572;
                    int32_t binop_y_36421;
                    int32_t convop_x_36422;
                    int32_t x_35573;
                    float res_35577;
                    float res_35578;
                    bool cond_35579;
                    bool res_35580;
                    bool res_35581;
                    bool x_35582;
                    float res_35583;
                    bool res_35584;
                    bool x_35585;
                    float res_35586;
                    bool res_35590;
                    int32_t res_35591;
                    float res_35596;
                    
                    x_35571 = *(__global float *) &mem_37674[(group_id_35373 *
                                                              (group_sizze_35492 *
                                                               chunk_35514) +
                                                              local_tid_35372 +
                                                              group_sizze_35492 *
                                                              (chunk_offset_35553 +
                                                               i_35564) + 0 *
                                                              group_sizze_35492) *
                                                             4];
                    x_35572 = *(__local
                                float *) &mem_37671[(chunk_offset_35553 +
                                                     i_35564) * 4];
                    binop_y_36421 = chunk_offset_35553 + i_35564;
                    convop_x_36422 = streamseq_chunk_offset_35515 +
                        binop_y_36421;
                    x_35573 = convop_x_36422;
                    res_35577 = inpacc_35516 + x_35571;
                    res_35578 = res_35577 / y_35509;
                    cond_35579 = slt32(convop_x_36422, y_35506);
                    res_35580 = futrts_isnan32(res_35578);
                    res_35581 = !res_35580;
                    x_35582 = cond_35579 && res_35581;
                    res_35583 = (float) fabs(res_35578);
                    res_35584 = x_35572 < res_35583;
                    x_35585 = x_35582 && res_35584;
                    if (cond_35579) {
                        res_35586 = res_35578;
                    } else {
                        res_35586 = 0.0F;
                    }
                    if (acc_35565) {
                        res_35590 = acc_35565;
                        res_35591 = acc_35566;
                    } else {
                        bool x_35592;
                        bool y_35593;
                        bool res_35594;
                        int32_t res_35595;
                        
                        x_35592 = !x_35585;
                        y_35593 = acc_35565 && x_35592;
                        res_35594 = x_35585 || y_35593;
                        if (x_35585) {
                            res_35595 = x_35573;
                        } else {
                            res_35595 = acc_35566;
                        }
                        res_35590 = res_35594;
                        res_35591 = res_35595;
                    }
                    res_35596 = acc_35567 + res_35586;
                    
                    bool acc_tmp_38373 = res_35590;
                    int32_t acc_tmp_38374 = res_35591;
                    float acc_tmp_38375;
                    
                    acc_tmp_38375 = res_35596;
                    acc_35565 = acc_tmp_38373;
                    acc_35566 = acc_tmp_38374;
                    acc_35567 = acc_tmp_38375;
                }
            } else {
                for (int32_t i_35564 = 0; i_35564 < chunk_sizze_35552;
                     i_35564++) {
                    float x_35571;
                    float x_35572;
                    int32_t binop_y_36421;
                    int32_t convop_x_36422;
                    int32_t x_35573;
                    float res_35577;
                    float res_35578;
                    bool cond_35579;
                    bool res_35580;
                    bool res_35581;
                    bool x_35582;
                    float res_35583;
                    bool res_35584;
                    bool x_35585;
                    float res_35586;
                    bool res_35590;
                    int32_t res_35591;
                    float res_35596;
                    
                    x_35571 = *(__global float *) &mem_37674[(group_id_35373 *
                                                              (group_sizze_35492 *
                                                               chunk_35514) +
                                                              local_tid_35372 +
                                                              group_sizze_35492 *
                                                              (chunk_offset_35553 +
                                                               i_35564) + 0 *
                                                              group_sizze_35492) *
                                                             4];
                    x_35572 = *(__local
                                float *) &mem_37671[(chunk_offset_35553 +
                                                     i_35564) * 4];
                    binop_y_36421 = chunk_offset_35553 + i_35564;
                    convop_x_36422 = streamseq_chunk_offset_35515 +
                        binop_y_36421;
                    x_35573 = convop_x_36422;
                    res_35577 = inpacc_35516 + x_35571;
                    res_35578 = res_35577 / y_35509;
                    cond_35579 = slt32(convop_x_36422, y_35506);
                    res_35580 = futrts_isnan32(res_35578);
                    res_35581 = !res_35580;
                    x_35582 = cond_35579 && res_35581;
                    res_35583 = (float) fabs(res_35578);
                    res_35584 = x_35572 < res_35583;
                    x_35585 = x_35582 && res_35584;
                    if (cond_35579) {
                        res_35586 = res_35578;
                    } else {
                        res_35586 = 0.0F;
                    }
                    if (acc_35565) {
                        res_35590 = acc_35565;
                        res_35591 = acc_35566;
                    } else {
                        bool x_35592;
                        bool y_35593;
                        bool res_35594;
                        int32_t res_35595;
                        
                        x_35592 = !x_35585;
                        y_35593 = acc_35565 && x_35592;
                        res_35594 = x_35585 || y_35593;
                        if (x_35585) {
                            res_35595 = x_35573;
                        } else {
                            res_35595 = acc_35566;
                        }
                        res_35590 = res_35594;
                        res_35591 = res_35595;
                    }
                    res_35596 = acc_35567 + res_35586;
                    
                    bool acc_tmp_38376 = res_35590;
                    int32_t acc_tmp_38377 = res_35591;
                    float acc_tmp_38378;
                    
                    acc_tmp_38378 = res_35596;
                    acc_35565 = acc_tmp_38376;
                    acc_35566 = acc_tmp_38377;
                    acc_35567 = acc_tmp_38378;
                }
            }
        }
        acc0_35560 = acc_35565;
        acc0_35561 = acc_35566;
        acc0_35562 = acc_35567;
        x_35554 = acc0_35560;
        x_35555 = acc0_35561;
        x_35556 = acc0_35562;
        acc0_35549 = x_35554;
        acc0_35550 = x_35555;
        acc0_35551 = x_35556;
        
        bool res_35597;
        int32_t res_35598;
        float res_35603;
        
        if (slt32(gtid_35364, sizze_30757)) {
            if (inpacc_35517) {
                res_35597 = inpacc_35517;
                res_35598 = inpacc_35518;
            } else {
                bool x_35599;
                bool y_35600;
                bool res_35601;
                int32_t res_35602;
                
                x_35599 = !acc0_35549;
                y_35600 = inpacc_35517 && x_35599;
                res_35601 = acc0_35549 || y_35600;
                if (acc0_35549) {
                    res_35602 = acc0_35550;
                } else {
                    res_35602 = inpacc_35518;
                }
                res_35597 = res_35601;
                res_35598 = res_35602;
            }
            res_35603 = inpacc_35519 + acc0_35551;
        }
        
        float sync_37095;
        bool sync_37096;
        int32_t sync_37097;
        float sync_37098;
        
        sync_37095 = res_35548;
        sync_37096 = res_35597;
        sync_37097 = res_35598;
        sync_37098 = res_35603;
        barrier(CLK_LOCAL_MEM_FENCE);
        inpacc_35516 = sync_37095;
        inpacc_35517 = sync_37096;
        inpacc_35518 = sync_37097;
        inpacc_35519 = sync_37098;
        streamseq_chunk_offset_35515 += group_sizze_35492;
    }
    inpacc_35510 = inpacc_35516;
    res_35511 = inpacc_35517;
    res_35512 = inpacc_35518;
    res_35513 = inpacc_35519;
    
    bool cond_35604;
    int32_t res_35605;
    bool cond_35611;
    bool res_35612;
    bool x_35613;
    bool y_35614;
    bool cond_35615;
    int32_t res_35616;
    
    if (slt32(gtid_35364, sizze_30757)) {
        cond_35604 = !res_35511;
        if (cond_35604) {
            res_35605 = -1;
        } else {
            bool cond_35606;
            int32_t res_35607;
            
            cond_35606 = slt32(res_35512, y_35506);
            if (cond_35606) {
                int32_t i_35608;
                int32_t x_35609;
                int32_t res_35610;
                
                i_35608 = x_35500 + res_35512;
                x_35609 = *(__global int32_t *) &res_mem_37598[(gtid_35364 *
                                                                sizze_30756 +
                                                                i_35608) * 4];
                res_35610 = x_35609 - n_30761;
                res_35607 = res_35610;
            } else {
                res_35607 = -1;
            }
            res_35605 = res_35607;
        }
        cond_35611 = sle32(x_35500, 5);
        res_35612 = sle32(y_35506, 5);
        x_35613 = !cond_35611;
        y_35614 = res_35612 && x_35613;
        cond_35615 = cond_35611 || y_35614;
        if (cond_35615) {
            res_35616 = -2;
        } else {
            res_35616 = res_35605;
        }
    }
    if (slt32(gtid_35364, sizze_30757)) {
        *(__global int32_t *) &mem_37677[gtid_35364 * 4] = res_35616;
    }
    if (slt32(gtid_35364, sizze_30757)) {
        *(__global float *) &mem_37680[gtid_35364 * 4] = res_35513;
    }
}
__kernel void map_35733(int32_t sizze_30756, int32_t sizze_30757,
                        int32_t n_30761, __global unsigned char *res_mem_37598,
                        __global unsigned char *res_mem_37647, __global
                        unsigned char *mem_37703, __global
                        unsigned char *mem_37709, __global
                        unsigned char *mem_37712, __global
                        unsigned char *mem_37721)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_35733;
    int32_t local_tid_35734;
    int32_t group_sizze_38507;
    int32_t wave_sizze_38506;
    int32_t group_id_35735;
    
    global_tid_35733 = get_global_id(0);
    local_tid_35734 = get_local_id(0);
    group_sizze_38507 = get_local_size(0);
    wave_sizze_38506 = LOCKSTEP_WIDTH;
    group_id_35735 = get_group_id(0);
    
    int32_t gtid_35726;
    
    gtid_35726 = global_tid_35733;
    
    int32_t x_36010;
    int32_t y_36012;
    bool acc0_36014;
    int32_t acc0_36015;
    int32_t res_36022;
    bool cond_36028;
    int32_t res_36029;
    bool cond_36035;
    bool res_36036;
    bool x_36037;
    bool y_36038;
    bool cond_36039;
    int32_t res_36040;
    
    if (slt32(gtid_35726, sizze_30757)) {
        x_36010 = *(__global int32_t *) &res_mem_37647[gtid_35726 * 4];
        y_36012 = *(__global int32_t *) &mem_37703[gtid_35726 * 4];
        acc0_36014 = *(__global bool *) &mem_37709[gtid_35726];
        acc0_36015 = *(__global int32_t *) &mem_37712[gtid_35726 * 4];
        if (acc0_36014) {
            res_36022 = acc0_36015;
        } else {
            res_36022 = -1;
        }
        cond_36028 = !acc0_36014;
        if (cond_36028) {
            res_36029 = -1;
        } else {
            bool cond_36030;
            int32_t res_36031;
            
            cond_36030 = slt32(res_36022, y_36012);
            if (cond_36030) {
                int32_t i_36032;
                int32_t x_36033;
                int32_t res_36034;
                
                i_36032 = x_36010 + res_36022;
                x_36033 = *(__global int32_t *) &res_mem_37598[(gtid_35726 *
                                                                sizze_30756 +
                                                                i_36032) * 4];
                res_36034 = x_36033 - n_30761;
                res_36031 = res_36034;
            } else {
                res_36031 = -1;
            }
            res_36029 = res_36031;
        }
        cond_36035 = sle32(x_36010, 5);
        res_36036 = sle32(y_36012, 5);
        x_36037 = !cond_36035;
        y_36038 = res_36036 && x_36037;
        cond_36039 = cond_36035 || y_36038;
        if (cond_36039) {
            res_36040 = -2;
        } else {
            res_36040 = res_36029;
        }
    }
    if (slt32(gtid_35726, sizze_30757)) {
        *(__global int32_t *) &mem_37721[gtid_35726 * 4] = res_36040;
    }
}
__kernel void map_35857(int32_t sizze_30757, __global
                        unsigned char *res_mem_37596, __global
                        unsigned char *res_mem_37647, __global
                        unsigned char *res_mem_37648, __global
                        unsigned char *mem_37700, __global
                        unsigned char *mem_37703)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_35857;
    int32_t local_tid_35858;
    int32_t group_sizze_38395;
    int32_t wave_sizze_38394;
    int32_t group_id_35859;
    
    global_tid_35857 = get_global_id(0);
    local_tid_35858 = get_local_id(0);
    group_sizze_38395 = get_local_size(0);
    wave_sizze_38394 = LOCKSTEP_WIDTH;
    group_id_35859 = get_group_id(0);
    
    int32_t gtid_35850;
    
    gtid_35850 = global_tid_35857;
    
    int32_t x_35883;
    int32_t x_35884;
    float x_35885;
    int32_t y_35886;
    float res_35887;
    float res_35888;
    float y_35889;
    
    if (slt32(gtid_35850, sizze_30757)) {
        x_35883 = *(__global int32_t *) &res_mem_37596[gtid_35850 * 4];
        x_35884 = *(__global int32_t *) &res_mem_37647[gtid_35850 * 4];
        x_35885 = *(__global float *) &res_mem_37648[gtid_35850 * 4];
        y_35886 = x_35883 - x_35884;
        res_35887 = sitofp_i32_f32(x_35884);
        res_35888 = futrts_sqrt32(res_35887);
        y_35889 = x_35885 * res_35888;
    }
    if (slt32(gtid_35850, sizze_30757)) {
        *(__global float *) &mem_37700[gtid_35850 * 4] = y_35889;
    }
    if (slt32(gtid_35850, sizze_30757)) {
        *(__global int32_t *) &mem_37703[gtid_35850 * 4] = y_35886;
    }
}
__kernel void map_intra_group_31978(__local volatile
                                    int64_t *mem_37250_backing_aligned_0,
                                    int32_t sizze_30756, int32_t sizze_30757,
                                    int32_t sizze_30758, int32_t n_30761,
                                    int32_t res_30780,
                                    int32_t computed_group_sizze_31976, __global
                                    unsigned char *images_mem_37201, __global
                                    unsigned char *arg_mem_37210, __global
                                    unsigned char *mem_37218, __global
                                    unsigned char *mem_37246, __global
                                    unsigned char *mem_37255)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict mem_37250_backing_0 =
                          mem_37250_backing_aligned_0;
    int32_t global_tid_31978;
    int32_t local_tid_31979;
    int32_t group_sizze_37817;
    int32_t wave_sizze_37816;
    int32_t group_id_31980;
    
    global_tid_31978 = get_global_id(0);
    local_tid_31979 = get_local_id(0);
    group_sizze_37817 = get_local_size(0);
    wave_sizze_37816 = LOCKSTEP_WIDTH;
    group_id_31980 = get_group_id(0);
    
    int32_t gtid_31964;
    int32_t ltid_31965;
    
    gtid_31964 = squot32(global_tid_31978, computed_group_sizze_31976);
    ltid_31965 = global_tid_31978 - squot32(global_tid_31978,
                                            computed_group_sizze_31976) *
        computed_group_sizze_31976;
    if (slt32(gtid_31964, sizze_30757) && slt32(ltid_31965,
                                                computed_group_sizze_31976)) { }
    
    __local char *mem_37250;
    
    mem_37250 = (__local char *) mem_37250_backing_0;
    for (int32_t comb_iter_37818 = 0; comb_iter_37818 < squot32(res_30780 +
                                                                computed_group_sizze_31976 -
                                                                1,
                                                                computed_group_sizze_31976);
         comb_iter_37818++) {
        int32_t ctid_31972;
        int32_t flat_comb_id_37819 = comb_iter_37818 *
                computed_group_sizze_31976 + local_tid_31979;
        
        ctid_31972 = flat_comb_id_37819;
        if (slt32(ctid_31972, res_30780) && 1) {
            for (int32_t i_32071 = 0; i_32071 < res_30780; i_32071++) {
                float res_32073;
                float redout_32074 = 0.0F;
                
                for (int32_t i_32075 = 0; i_32075 < n_30761; i_32075++) {
                    float x_32076;
                    float x_32077;
                    float x_32078;
                    float x_32079;
                    bool res_32080;
                    float y_32081;
                    float res_32082;
                    float res_32085;
                    
                    x_32076 = *(__global
                                float *) &images_mem_37201[(gtid_31964 *
                                                            sizze_30758 +
                                                            i_32075) * 4];
                    x_32077 = *(__global float *) &arg_mem_37210[(ltid_31965 *
                                                                  sizze_30756 +
                                                                  i_32075) * 4];
                    x_32078 = *(__global float *) &mem_37218[(i_32075 *
                                                              res_30780 +
                                                              i_32071) * 4];
                    x_32079 = x_32077 * x_32078;
                    res_32080 = futrts_isnan32(x_32076);
                    if (res_32080) {
                        y_32081 = 0.0F;
                    } else {
                        y_32081 = 1.0F;
                    }
                    res_32082 = x_32079 * y_32081;
                    res_32085 = redout_32074 + res_32082;
                    
                    float redout_tmp_37821 = res_32085;
                    
                    redout_32074 = redout_tmp_37821;
                }
                res_32073 = redout_32074;
                *(__global float *) &mem_37246[(group_id_31980 *
                                                (computed_group_sizze_31976 *
                                                 res_30780) + local_tid_31979 +
                                                i_32071 *
                                                computed_group_sizze_31976) *
                                               4] = res_32073;
            }
            for (int32_t i_37822 = 0; i_37822 < res_30780; i_37822++) {
                *(__local float *) &mem_37250[(ctid_31972 * res_30780 +
                                               i_37822) * 4] = *(__global
                                                                 float *) &mem_37246[(group_id_31980 *
                                                                                      (computed_group_sizze_31976 *
                                                                                       res_30780) +
                                                                                      local_tid_31979 +
                                                                                      i_37822 *
                                                                                      computed_group_sizze_31976) *
                                                                                     4];
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int32_t i_37823 = 0; i_37823 < squot32(res_30780 * res_30780 -
                                                local_tid_31979 +
                                                computed_group_sizze_31976 - 1,
                                                computed_group_sizze_31976);
         i_37823++) {
        *(__global float *) &mem_37255[(group_id_31980 * (res_30780 *
                                                          res_30780) +
                                        squot32(i_37823 *
                                                computed_group_sizze_31976 +
                                                local_tid_31979, res_30780) *
                                        res_30780 + (i_37823 *
                                                     computed_group_sizze_31976 +
                                                     local_tid_31979 -
                                                     squot32(i_37823 *
                                                             computed_group_sizze_31976 +
                                                             local_tid_31979,
                                                             res_30780) *
                                                     res_30780)) * 4] =
            *(__local float *) &mem_37250[(squot32(i_37823 *
                                                   computed_group_sizze_31976 +
                                                   local_tid_31979, res_30780) *
                                           res_30780 + (i_37823 *
                                                        computed_group_sizze_31976 +
                                                        local_tid_31979 -
                                                        squot32(i_37823 *
                                                                computed_group_sizze_31976 +
                                                                local_tid_31979,
                                                                res_30780) *
                                                        res_30780)) * 4];
    }
}
__kernel void map_intra_group_32126(__local volatile
                                    int64_t *mem_37272_backing_aligned_0,
                                    int32_t sizze_30757, int32_t sizze_30758,
                                    int32_t n_30761, int32_t res_30780, __global
                                    unsigned char *images_mem_37201, __global
                                    unsigned char *mem_37214, __global
                                    unsigned char *mem_37218, __global
                                    unsigned char *mem_37276)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict mem_37272_backing_0 =
                          mem_37272_backing_aligned_0;
    int32_t global_tid_32126;
    int32_t local_tid_32127;
    int32_t group_sizze_37830;
    int32_t wave_sizze_37829;
    int32_t group_id_32128;
    
    global_tid_32126 = get_global_id(0);
    local_tid_32127 = get_local_id(0);
    group_sizze_37830 = get_local_size(0);
    wave_sizze_37829 = LOCKSTEP_WIDTH;
    group_id_32128 = get_group_id(0);
    
    int32_t gtid_32107;
    int32_t gtid_32108;
    int32_t ltid_32110;
    
    gtid_32107 = squot32(global_tid_32126, res_30780 * res_30780);
    gtid_32108 = squot32(global_tid_32126 - squot32(global_tid_32126,
                                                    res_30780 * res_30780) *
                         (res_30780 * res_30780), res_30780);
    ltid_32110 = global_tid_32126 - squot32(global_tid_32126, res_30780 *
                                            res_30780) * (res_30780 *
                                                          res_30780) -
        squot32(global_tid_32126 - squot32(global_tid_32126, res_30780 *
                                           res_30780) * (res_30780 * res_30780),
                res_30780) * res_30780;
    
    float x_36103;
    
    if ((slt32(gtid_32107, sizze_30757) && slt32(gtid_32108, res_30780)) &&
        slt32(ltid_32110, res_30780)) {
        float x_32500 = 0.0F;
        
        for (int32_t chunk_offset_32499 = 0; chunk_offset_32499 < n_30761;
             chunk_offset_32499++) {
            float x_32511;
            float x_32512;
            float x_32513;
            float x_32515;
            bool res_32516;
            float y_32517;
            float res_32518;
            float res_32520;
            
            x_32511 = *(__global float *) &images_mem_37201[(gtid_32107 *
                                                             sizze_30758 +
                                                             chunk_offset_32499) *
                                                            4];
            x_32512 = *(__global float *) &mem_37214[(chunk_offset_32499 *
                                                      res_30780 + gtid_32108) *
                                                     4];
            x_32513 = *(__global float *) &mem_37218[(chunk_offset_32499 *
                                                      res_30780 + ltid_32110) *
                                                     4];
            x_32515 = x_32512 * x_32513;
            res_32516 = futrts_isnan32(x_32511);
            if (res_32516) {
                y_32517 = 0.0F;
            } else {
                y_32517 = 1.0F;
            }
            res_32518 = x_32515 * y_32517;
            res_32520 = x_32500 + res_32518;
            
            float x_tmp_37831 = res_32520;
            
            x_32500 = x_tmp_37831;
        }
        x_36103 = x_32500;
    }
    
    __local char *mem_37272;
    
    mem_37272 = (__local char *) mem_37272_backing_0;
    for (int32_t comb_iter_37832 = 0; comb_iter_37832 < 1; comb_iter_37832++) {
        int32_t ctid_32124;
        int32_t flat_comb_id_37833 = comb_iter_37832 * res_30780 +
                local_tid_32127;
        
        ctid_32124 = flat_comb_id_37833;
        if (slt32(ctid_32124, res_30780) && 1) {
            *(__local float *) &mem_37272[ctid_32124 * 4] = x_36103;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int32_t i_37834 = 0; i_37834 < squot32(res_30780 - local_tid_32127 +
                                                res_30780 - 1, res_30780);
         i_37834++) {
        *(__global float *) &mem_37276[(group_id_32128 * res_30780 + (i_37834 *
                                                                      res_30780 +
                                                                      local_tid_32127)) *
                                       4] = *(__local
                                              float *) &mem_37272[(i_37834 *
                                                                   res_30780 +
                                                                   local_tid_32127) *
                                                                  4];
    }
}
__kernel void map_intra_group_32223(__local volatile
                                    int64_t *mem_37295_backing_aligned_0,
                                    int32_t sizze_30756, int32_t sizze_30757,
                                    int32_t sizze_30758, int32_t n_30761,
                                    int32_t res_30780, __global
                                    unsigned char *images_mem_37201, __global
                                    unsigned char *arg_mem_37210, __global
                                    unsigned char *mem_37292, __global
                                    unsigned char *mem_37298)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict mem_37295_backing_0 =
                          mem_37295_backing_aligned_0;
    int32_t global_tid_32223;
    int32_t local_tid_32224;
    int32_t group_sizze_37870;
    int32_t wave_sizze_37869;
    int32_t group_id_32225;
    
    global_tid_32223 = get_global_id(0);
    local_tid_32224 = get_local_id(0);
    group_sizze_37870 = get_local_size(0);
    wave_sizze_37869 = LOCKSTEP_WIDTH;
    group_id_32225 = get_group_id(0);
    
    int32_t gtid_32214;
    int32_t gtid_32215;
    int32_t gtid_32216;
    int32_t ltid_32219;
    
    gtid_32214 = squot32(global_tid_32223, res_30780 * res_30780 * n_30761);
    gtid_32215 = squot32(global_tid_32223 - squot32(global_tid_32223,
                                                    res_30780 * res_30780 *
                                                    n_30761) * (res_30780 *
                                                                res_30780 *
                                                                n_30761),
                         res_30780 * n_30761);
    gtid_32216 = squot32(global_tid_32223 - squot32(global_tid_32223,
                                                    res_30780 * res_30780 *
                                                    n_30761) * (res_30780 *
                                                                res_30780 *
                                                                n_30761) -
                         squot32(global_tid_32223 - squot32(global_tid_32223,
                                                            res_30780 *
                                                            res_30780 *
                                                            n_30761) *
                                 (res_30780 * res_30780 * n_30761), res_30780 *
                                 n_30761) * (res_30780 * n_30761), n_30761);
    ltid_32219 = global_tid_32223 - squot32(global_tid_32223, res_30780 *
                                            res_30780 * n_30761) * (res_30780 *
                                                                    res_30780 *
                                                                    n_30761) -
        squot32(global_tid_32223 - squot32(global_tid_32223, res_30780 *
                                           res_30780 * n_30761) * (res_30780 *
                                                                   res_30780 *
                                                                   n_30761),
                res_30780 * n_30761) * (res_30780 * n_30761) -
        squot32(global_tid_32223 - squot32(global_tid_32223, res_30780 *
                                           res_30780 * n_30761) * (res_30780 *
                                                                   res_30780 *
                                                                   n_30761) -
                squot32(global_tid_32223 - squot32(global_tid_32223, res_30780 *
                                                   res_30780 * n_30761) *
                        (res_30780 * res_30780 * n_30761), res_30780 *
                        n_30761) * (res_30780 * n_30761), n_30761) * n_30761;
    
    float x_36139;
    float x_36141;
    float x_36143;
    float x_32575;
    bool res_32576;
    float y_32577;
    float res_32578;
    
    if (((slt32(gtid_32214, sizze_30757) && slt32(gtid_32215, res_30780)) &&
         slt32(gtid_32216, res_30780)) && slt32(ltid_32219, n_30761)) {
        x_36139 = *(__global float *) &images_mem_37201[(gtid_32214 *
                                                         sizze_30758 +
                                                         ltid_32219) * 4];
        x_36141 = *(__global float *) &arg_mem_37210[(gtid_32215 * sizze_30756 +
                                                      ltid_32219) * 4];
        x_36143 = *(__global float *) &mem_37292[(gtid_32216 * sizze_30756 +
                                                  ltid_32219) * 4];
        x_32575 = x_36141 * x_36143;
        res_32576 = futrts_isnan32(x_36139);
        if (res_32576) {
            y_32577 = 0.0F;
        } else {
            y_32577 = 1.0F;
        }
        res_32578 = x_32575 * y_32577;
    }
    
    __local char *mem_37295;
    float res_32579;
    
    mem_37295 = (__local char *) mem_37295_backing_0;
    for (int32_t comb_iter_37871 = 0; comb_iter_37871 < 1; comb_iter_37871++) {
        int32_t ctid_32221;
        int32_t flat_comb_id_37872 = comb_iter_37871 * n_30761 +
                local_tid_32224;
        
        ctid_32221 = flat_comb_id_37872;
        if (slt32(ctid_32221, n_30761) && 1) {
            *(__local float *) &mem_37295[ctid_32221 * 4] = res_32578;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_37873;
    int32_t skip_waves_37874;
    float x_32580;
    float x_32581;
    
    offset_37873 = 0;
    // participating threads read initial accumulator
    {
        if (slt32(local_tid_32224, n_30761)) {
            x_32580 = *(__local float *) &mem_37295[(local_tid_32224 +
                                                     offset_37873) * 4];
        }
    }
    offset_37873 = 1;
    while (slt32(offset_37873, wave_sizze_37869)) {
        if (slt32(local_tid_32224 + offset_37873, n_30761) &&
            ((local_tid_32224 - squot32(local_tid_32224, wave_sizze_37869) *
              wave_sizze_37869) & (2 * offset_37873 - 1)) == 0) {
            // read array element
            {
                x_32581 = *(volatile __local
                            float *) &mem_37295[(local_tid_32224 +
                                                 offset_37873) * 4];
            }
            // apply reduction operation
            {
                float res_32582;
                
                if (((slt32(gtid_32214, sizze_30757) && slt32(gtid_32215,
                                                              res_30780)) &&
                     slt32(gtid_32216, res_30780)) && slt32(ltid_32219,
                                                            n_30761)) {
                    res_32582 = x_32580 + x_32581;
                }
                x_32580 = res_32582;
            }
            // write result of operation
            {
                *(volatile __local float *) &mem_37295[local_tid_32224 * 4] =
                    x_32580;
            }
        }
        offset_37873 *= 2;
    }
    skip_waves_37874 = 1;
    while (slt32(skip_waves_37874, squot32(n_30761 + wave_sizze_37869 - 1,
                                           wave_sizze_37869))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_37873 = skip_waves_37874 * wave_sizze_37869;
        if (slt32(local_tid_32224 + offset_37873, n_30761) &&
            ((local_tid_32224 - squot32(local_tid_32224, wave_sizze_37869) *
              wave_sizze_37869) == 0 && (squot32(local_tid_32224,
                                                 wave_sizze_37869) & (2 *
                                                                      skip_waves_37874 -
                                                                      1)) ==
             0)) {
            // read array element
            {
                x_32581 = *(__local float *) &mem_37295[(local_tid_32224 +
                                                         offset_37873) * 4];
            }
            // apply reduction operation
            {
                float res_32582;
                
                if (((slt32(gtid_32214, sizze_30757) && slt32(gtid_32215,
                                                              res_30780)) &&
                     slt32(gtid_32216, res_30780)) && slt32(ltid_32219,
                                                            n_30761)) {
                    res_32582 = x_32580 + x_32581;
                }
                x_32580 = res_32582;
            }
            // write result of operation
            {
                *(__local float *) &mem_37295[local_tid_32224 * 4] = x_32580;
            }
        }
        skip_waves_37874 *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    res_32579 = *(__local float *) &mem_37295[0];
    if (local_tid_32224 == 0) {
        *(__global float *) &mem_37298[group_id_32225 * 4] = res_32579;
    }
}
__kernel void map_intra_group_32634(__local volatile
                                    int64_t *mem_37316_backing_aligned_0,
                                    __local volatile
                                    int64_t *mem_37320_backing_aligned_1,
                                    int32_t sizze_30757, int32_t res_30780,
                                    int32_t m_30862, int32_t j_30912,
                                    int32_t j_m_i_30913, int32_t res_30916,
                                    __global unsigned char *res_mem_37313,
                                    __global unsigned char *mem_37326)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict mem_37316_backing_0 =
                          mem_37316_backing_aligned_0;
    __local volatile char *restrict mem_37320_backing_1 =
                          mem_37320_backing_aligned_1;
    int32_t global_tid_32634;
    int32_t local_tid_32635;
    int32_t group_sizze_37912;
    int32_t wave_sizze_37911;
    int32_t group_id_32636;
    
    global_tid_32634 = get_global_id(0);
    local_tid_32635 = get_local_id(0);
    group_sizze_37912 = get_local_size(0);
    wave_sizze_37911 = LOCKSTEP_WIDTH;
    group_id_32636 = get_group_id(0);
    
    int32_t gtid_32628;
    int32_t ltid_32629;
    
    gtid_32628 = squot32(global_tid_32634, res_30916);
    ltid_32629 = global_tid_32634 - squot32(global_tid_32634, res_30916) *
        res_30916;
    
    int32_t x_36168;
    int32_t x_36170;
    bool cond_32803;
    float x_36172;
    
    if (slt32(gtid_32628, sizze_30757) && slt32(ltid_32629, res_30916)) {
        x_36168 = sdiv32(ltid_32629, j_30912);
        x_36170 = smod32(ltid_32629, j_30912);
        cond_32803 = slt32(x_36170, res_30780);
        if (cond_32803) {
            float res_32805 = *(__global float *) &res_mem_37313[(gtid_32628 *
                                                                  (res_30780 *
                                                                   res_30780) +
                                                                  x_36168 *
                                                                  res_30780 +
                                                                  x_36170) * 4];
            
            x_36172 = res_32805;
        } else {
            int32_t y_32806;
            bool cond_32807;
            float res_32808;
            
            y_32806 = res_30780 + x_36168;
            cond_32807 = x_36170 == y_32806;
            if (cond_32807) {
                res_32808 = 1.0F;
            } else {
                res_32808 = 0.0F;
            }
            x_36172 = res_32808;
        }
    }
    
    __local char *mem_37316;
    __local char *mem_37320;
    
    mem_37316 = (__local char *) mem_37316_backing_0;
    for (int32_t comb_iter_37913 = 0; comb_iter_37913 < 1; comb_iter_37913++) {
        int32_t ctid_32630;
        int32_t flat_comb_id_37914 = comb_iter_37913 * res_30916 +
                local_tid_32635;
        
        ctid_32630 = flat_comb_id_37914;
        if (slt32(ctid_32630, res_30916) && 1) {
            *(__local float *) &mem_37316[ctid_32630 * 4] = x_36172;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    mem_37320 = (__local char *) mem_37320_backing_1;
    if (slt32(gtid_32628, sizze_30757) && slt32(ltid_32629, res_30916)) {
        for (int32_t i_32811 = 0; i_32811 < res_30780; i_32811++) {
            float res_32812;
            bool cond_32813;
            float x_36179;
            
            res_32812 = *(__local float *) &mem_37316[i_32811 * 4];
            cond_32813 = res_32812 == 0.0F;
            if (cond_32813) {
                int32_t x_32819;
                int32_t i_32820;
                float res_32821;
                
                x_32819 = j_30912 * x_36168;
                i_32820 = x_32819 + x_36170;
                res_32821 = *(__local float *) &mem_37316[i_32820 * 4];
                x_36179 = res_32821;
            } else {
                float x_32822;
                float res_32823;
                bool cond_32824;
                float res_32825;
                
                x_32822 = *(__local float *) &mem_37316[x_36170 * 4];
                res_32823 = x_32822 / res_32812;
                cond_32824 = slt32(x_36168, m_30862);
                if (cond_32824) {
                    int32_t x_32826;
                    int32_t x_32827;
                    int32_t i_32828;
                    float x_32829;
                    int32_t i_32830;
                    float x_32831;
                    float y_32832;
                    float res_32833;
                    
                    x_32826 = 1 + x_36168;
                    x_32827 = j_30912 * x_32826;
                    i_32828 = x_32827 + x_36170;
                    x_32829 = *(__local float *) &mem_37316[i_32828 * 4];
                    i_32830 = i_32811 + x_32827;
                    x_32831 = *(__local float *) &mem_37316[i_32830 * 4];
                    y_32832 = res_32823 * x_32831;
                    res_32833 = x_32829 - y_32832;
                    res_32825 = res_32833;
                } else {
                    res_32825 = res_32823;
                }
                x_36179 = res_32825;
            }
            for (int32_t comb_iter_37916 = 0; comb_iter_37916 < 1;
                 comb_iter_37916++) {
                int32_t ctid_32631;
                int32_t flat_comb_id_37917 = comb_iter_37916 * res_30916 +
                        local_tid_32635;
                
                ctid_32631 = flat_comb_id_37917;
                if (slt32(ctid_32631, res_30916) && 1) {
                    *(__local float *) &mem_37320[ctid_32631 * 4] = x_36179;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            float x_36182 = *(__local float *) &mem_37320[ltid_32629 * 4];
            
            for (int32_t comb_iter_37918 = 0; comb_iter_37918 < 1;
                 comb_iter_37918++) {
                int32_t ctid_32632;
                int32_t flat_comb_id_37919 = comb_iter_37918 * res_30916 +
                        local_tid_32635;
                
                ctid_32632 = flat_comb_id_37919;
                if (slt32(ctid_32632, res_30916) && 1) {
                    if (sle32(0, ltid_32629) && slt32(ltid_32629, res_30916)) {
                        *(__local float *) &mem_37316[ltid_32629 * 4] = x_36182;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    for (int32_t i_37920 = 0; i_37920 < squot32(res_30780 * j_m_i_30913 -
                                                local_tid_32635 + res_30916 - 1,
                                                res_30916); i_37920++) {
        *(__global float *) &mem_37326[(group_id_32636 * (j_m_i_30913 *
                                                          res_30780) +
                                        squot32(i_37920 * res_30916 +
                                                local_tid_32635, j_m_i_30913) *
                                        j_m_i_30913 + (i_37920 * res_30916 +
                                                       local_tid_32635 -
                                                       squot32(i_37920 *
                                                               res_30916 +
                                                               local_tid_32635,
                                                               j_m_i_30913) *
                                                       j_m_i_30913)) * 4] =
            *(__local float *) &mem_37316[(res_30780 + (squot32(i_37920 *
                                                                res_30916 +
                                                                local_tid_32635,
                                                                j_m_i_30913) *
                                                        j_30912 + (i_37920 *
                                                                   res_30916 +
                                                                   local_tid_32635 -
                                                                   squot32(i_37920 *
                                                                           res_30916 +
                                                                           local_tid_32635,
                                                                           j_m_i_30913) *
                                                                   j_m_i_30913))) *
                                          4];
    }
}
__kernel void map_intra_group_33329(__local volatile
                                    int64_t *mem_37363_backing_aligned_0,
                                    int32_t sizze_30757, int32_t sizze_30758,
                                    int32_t n_30761, int32_t res_30780, __global
                                    unsigned char *images_mem_37201, __global
                                    unsigned char *mem_37214, __global
                                    unsigned char *mem_37367)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict mem_37363_backing_0 =
                          mem_37363_backing_aligned_0;
    int32_t global_tid_33329;
    int32_t local_tid_33330;
    int32_t group_sizze_37941;
    int32_t wave_sizze_37940;
    int32_t group_id_33331;
    
    global_tid_33329 = get_global_id(0);
    local_tid_33330 = get_local_id(0);
    group_sizze_37941 = get_local_size(0);
    wave_sizze_37940 = LOCKSTEP_WIDTH;
    group_id_33331 = get_group_id(0);
    
    int32_t gtid_33314;
    int32_t ltid_33315;
    
    gtid_33314 = squot32(global_tid_33329, res_30780);
    ltid_33315 = global_tid_33329 - squot32(global_tid_33329, res_30780) *
        res_30780;
    
    float x_36236;
    
    if (slt32(gtid_33314, sizze_30757) && slt32(ltid_33315, res_30780)) {
        float x_33402 = 0.0F;
        
        for (int32_t chunk_offset_33401 = 0; chunk_offset_33401 < n_30761;
             chunk_offset_33401++) {
            float x_33411;
            float x_33412;
            bool res_33414;
            float res_33415;
            float res_33418;
            
            x_33411 = *(__global float *) &mem_37214[(chunk_offset_33401 *
                                                      res_30780 + ltid_33315) *
                                                     4];
            x_33412 = *(__global float *) &images_mem_37201[(gtid_33314 *
                                                             sizze_30758 +
                                                             chunk_offset_33401) *
                                                            4];
            res_33414 = futrts_isnan32(x_33412);
            if (res_33414) {
                res_33415 = 0.0F;
            } else {
                float res_33416 = x_33411 * x_33412;
                
                res_33415 = res_33416;
            }
            res_33418 = x_33402 + res_33415;
            
            float x_tmp_37942 = res_33418;
            
            x_33402 = x_tmp_37942;
        }
        x_36236 = x_33402;
    }
    
    __local char *mem_37363;
    
    mem_37363 = (__local char *) mem_37363_backing_0;
    for (int32_t comb_iter_37943 = 0; comb_iter_37943 < 1; comb_iter_37943++) {
        int32_t ctid_33327;
        int32_t flat_comb_id_37944 = comb_iter_37943 * res_30780 +
                local_tid_33330;
        
        ctid_33327 = flat_comb_id_37944;
        if (slt32(ctid_33327, res_30780) && 1) {
            *(__local float *) &mem_37363[ctid_33327 * 4] = x_36236;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int32_t i_37945 = 0; i_37945 < squot32(res_30780 - local_tid_33330 +
                                                res_30780 - 1, res_30780);
         i_37945++) {
        *(__global float *) &mem_37367[(group_id_33331 * res_30780 + (i_37945 *
                                                                      res_30780 +
                                                                      local_tid_33330)) *
                                       4] = *(__local
                                              float *) &mem_37363[(i_37945 *
                                                                   res_30780 +
                                                                   local_tid_33330) *
                                                                  4];
    }
}
__kernel void map_intra_group_33451(__local volatile
                                    int64_t *mem_37382_backing_aligned_0,
                                    int32_t sizze_30756, int32_t sizze_30757,
                                    int32_t sizze_30758, int32_t n_30761,
                                    int32_t res_30780, __global
                                    unsigned char *images_mem_37201, __global
                                    unsigned char *arg_mem_37210, __global
                                    unsigned char *mem_37385)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict mem_37382_backing_0 =
                          mem_37382_backing_aligned_0;
    int32_t global_tid_33451;
    int32_t local_tid_33452;
    int32_t group_sizze_37955;
    int32_t wave_sizze_37954;
    int32_t group_id_33453;
    
    global_tid_33451 = get_global_id(0);
    local_tid_33452 = get_local_id(0);
    group_sizze_37955 = get_local_size(0);
    wave_sizze_37954 = LOCKSTEP_WIDTH;
    group_id_33453 = get_group_id(0);
    
    int32_t gtid_33444;
    int32_t gtid_33445;
    int32_t ltid_33447;
    
    gtid_33444 = squot32(global_tid_33451, res_30780 * n_30761);
    gtid_33445 = squot32(global_tid_33451 - squot32(global_tid_33451,
                                                    res_30780 * n_30761) *
                         (res_30780 * n_30761), n_30761);
    ltid_33447 = global_tid_33451 - squot32(global_tid_33451, res_30780 *
                                            n_30761) * (res_30780 * n_30761) -
        squot32(global_tid_33451 - squot32(global_tid_33451, res_30780 *
                                           n_30761) * (res_30780 * n_30761),
                n_30761) * n_30761;
    
    float x_36262;
    float x_36264;
    bool res_33605;
    float res_33606;
    
    if ((slt32(gtid_33444, sizze_30757) && slt32(gtid_33445, res_30780)) &&
        slt32(ltid_33447, n_30761)) {
        x_36262 = *(__global float *) &arg_mem_37210[(gtid_33445 * sizze_30756 +
                                                      ltid_33447) * 4];
        x_36264 = *(__global float *) &images_mem_37201[(gtid_33444 *
                                                         sizze_30758 +
                                                         ltid_33447) * 4];
        res_33605 = futrts_isnan32(x_36264);
        if (res_33605) {
            res_33606 = 0.0F;
        } else {
            float res_33607 = x_36262 * x_36264;
            
            res_33606 = res_33607;
        }
    }
    
    __local char *mem_37382;
    float res_33608;
    
    mem_37382 = (__local char *) mem_37382_backing_0;
    for (int32_t comb_iter_37956 = 0; comb_iter_37956 < 1; comb_iter_37956++) {
        int32_t ctid_33449;
        int32_t flat_comb_id_37957 = comb_iter_37956 * n_30761 +
                local_tid_33452;
        
        ctid_33449 = flat_comb_id_37957;
        if (slt32(ctid_33449, n_30761) && 1) {
            *(__local float *) &mem_37382[ctid_33449 * 4] = res_33606;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_37958;
    int32_t skip_waves_37959;
    float x_33609;
    float x_33610;
    
    offset_37958 = 0;
    // participating threads read initial accumulator
    {
        if (slt32(local_tid_33452, n_30761)) {
            x_33609 = *(__local float *) &mem_37382[(local_tid_33452 +
                                                     offset_37958) * 4];
        }
    }
    offset_37958 = 1;
    while (slt32(offset_37958, wave_sizze_37954)) {
        if (slt32(local_tid_33452 + offset_37958, n_30761) &&
            ((local_tid_33452 - squot32(local_tid_33452, wave_sizze_37954) *
              wave_sizze_37954) & (2 * offset_37958 - 1)) == 0) {
            // read array element
            {
                x_33610 = *(volatile __local
                            float *) &mem_37382[(local_tid_33452 +
                                                 offset_37958) * 4];
            }
            // apply reduction operation
            {
                float res_33611;
                
                if ((slt32(gtid_33444, sizze_30757) && slt32(gtid_33445,
                                                             res_30780)) &&
                    slt32(ltid_33447, n_30761)) {
                    res_33611 = x_33609 + x_33610;
                }
                x_33609 = res_33611;
            }
            // write result of operation
            {
                *(volatile __local float *) &mem_37382[local_tid_33452 * 4] =
                    x_33609;
            }
        }
        offset_37958 *= 2;
    }
    skip_waves_37959 = 1;
    while (slt32(skip_waves_37959, squot32(n_30761 + wave_sizze_37954 - 1,
                                           wave_sizze_37954))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_37958 = skip_waves_37959 * wave_sizze_37954;
        if (slt32(local_tid_33452 + offset_37958, n_30761) &&
            ((local_tid_33452 - squot32(local_tid_33452, wave_sizze_37954) *
              wave_sizze_37954) == 0 && (squot32(local_tid_33452,
                                                 wave_sizze_37954) & (2 *
                                                                      skip_waves_37959 -
                                                                      1)) ==
             0)) {
            // read array element
            {
                x_33610 = *(__local float *) &mem_37382[(local_tid_33452 +
                                                         offset_37958) * 4];
            }
            // apply reduction operation
            {
                float res_33611;
                
                if ((slt32(gtid_33444, sizze_30757) && slt32(gtid_33445,
                                                             res_30780)) &&
                    slt32(ltid_33447, n_30761)) {
                    res_33611 = x_33609 + x_33610;
                }
                x_33609 = res_33611;
            }
            // write result of operation
            {
                *(__local float *) &mem_37382[local_tid_33452 * 4] = x_33609;
            }
        }
        skip_waves_37959 *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    res_33608 = *(__local float *) &mem_37382[0];
    if (local_tid_33452 == 0) {
        *(__global float *) &mem_37385[group_id_33453 * 4] = res_33608;
    }
}
__kernel void map_intra_group_33665(__local volatile
                                    int64_t *mem_37422_backing_aligned_0,
                                    int32_t sizze_30757, int32_t res_30780,
                                    int32_t j_m_i_30913, __global
                                    unsigned char *res_mem_37393, __global
                                    unsigned char *mem_37419, __global
                                    unsigned char *mem_37426)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict mem_37422_backing_0 =
                          mem_37422_backing_aligned_0;
    int32_t global_tid_33665;
    int32_t local_tid_33666;
    int32_t group_sizze_38007;
    int32_t wave_sizze_38006;
    int32_t group_id_33667;
    
    global_tid_33665 = get_global_id(0);
    local_tid_33666 = get_local_id(0);
    group_sizze_38007 = get_local_size(0);
    wave_sizze_38006 = LOCKSTEP_WIDTH;
    group_id_33667 = get_group_id(0);
    
    int32_t gtid_33650;
    int32_t ltid_33651;
    
    gtid_33650 = squot32(global_tid_33665, res_30780);
    ltid_33651 = global_tid_33665 - squot32(global_tid_33665, res_30780) *
        res_30780;
    
    float x_36292;
    
    if (slt32(gtid_33650, sizze_30757) && slt32(ltid_33651, res_30780)) {
        float x_33740 = 0.0F;
        
        for (int32_t chunk_offset_33739 = 0; chunk_offset_33739 < j_m_i_30913;
             chunk_offset_33739++) {
            float x_33749;
            float x_33750;
            float res_33752;
            float res_33754;
            
            x_33749 = *(__global float *) &res_mem_37393[(gtid_33650 *
                                                          res_30780 +
                                                          chunk_offset_33739) *
                                                         4];
            x_33750 = *(__global float *) &mem_37419[(chunk_offset_33739 *
                                                      (res_30780 *
                                                       sizze_30757) +
                                                      gtid_33650 * res_30780 +
                                                      ltid_33651) * 4];
            res_33752 = x_33749 * x_33750;
            res_33754 = x_33740 + res_33752;
            
            float x_tmp_38008 = res_33754;
            
            x_33740 = x_tmp_38008;
        }
        x_36292 = x_33740;
    }
    
    __local char *mem_37422;
    
    mem_37422 = (__local char *) mem_37422_backing_0;
    for (int32_t comb_iter_38009 = 0; comb_iter_38009 < 1; comb_iter_38009++) {
        int32_t ctid_33663;
        int32_t flat_comb_id_38010 = comb_iter_38009 * res_30780 +
                local_tid_33666;
        
        ctid_33663 = flat_comb_id_38010;
        if (slt32(ctid_33663, res_30780) && 1) {
            *(__local float *) &mem_37422[ctid_33663 * 4] = x_36292;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int32_t i_38011 = 0; i_38011 < squot32(res_30780 - local_tid_33666 +
                                                res_30780 - 1, res_30780);
         i_38011++) {
        *(__global float *) &mem_37426[(group_id_33667 * res_30780 + (i_38011 *
                                                                      res_30780 +
                                                                      local_tid_33666)) *
                                       4] = *(__local
                                              float *) &mem_37422[(i_38011 *
                                                                   res_30780 +
                                                                   local_tid_33666) *
                                                                  4];
    }
}
__kernel void map_intra_group_33788(__local volatile
                                    int64_t *mem_37438_backing_aligned_0,
                                    int32_t sizze_30757, int32_t res_30780,
                                    int32_t j_m_i_30913, __global
                                    unsigned char *res_mem_37344, __global
                                    unsigned char *res_mem_37393, __global
                                    unsigned char *mem_37441)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict mem_37438_backing_0 =
                          mem_37438_backing_aligned_0;
    int32_t global_tid_33788;
    int32_t local_tid_33789;
    int32_t group_sizze_38016;
    int32_t wave_sizze_38015;
    int32_t group_id_33790;
    
    global_tid_33788 = get_global_id(0);
    local_tid_33789 = get_local_id(0);
    group_sizze_38016 = get_local_size(0);
    wave_sizze_38015 = LOCKSTEP_WIDTH;
    group_id_33790 = get_group_id(0);
    
    int32_t gtid_33781;
    int32_t gtid_33782;
    int32_t ltid_33784;
    
    gtid_33781 = squot32(global_tid_33788, res_30780 * j_m_i_30913);
    gtid_33782 = squot32(global_tid_33788 - squot32(global_tid_33788,
                                                    res_30780 * j_m_i_30913) *
                         (res_30780 * j_m_i_30913), j_m_i_30913);
    ltid_33784 = global_tid_33788 - squot32(global_tid_33788, res_30780 *
                                            j_m_i_30913) * (res_30780 *
                                                            j_m_i_30913) -
        squot32(global_tid_33788 - squot32(global_tid_33788, res_30780 *
                                           j_m_i_30913) * (res_30780 *
                                                           j_m_i_30913),
                j_m_i_30913) * j_m_i_30913;
    
    int32_t binop_x_36302;
    int32_t binop_x_36303;
    int32_t new_index_36304;
    int32_t binop_y_36310;
    int32_t new_index_36311;
    float x_36298;
    float x_36300;
    float res_33934;
    
    if ((slt32(gtid_33781, sizze_30757) && slt32(gtid_33782, res_30780)) &&
        slt32(ltid_33784, j_m_i_30913)) {
        binop_x_36302 = j_m_i_30913 * gtid_33781;
        binop_x_36303 = ltid_33784 + binop_x_36302;
        new_index_36304 = squot32(binop_x_36303, res_30780);
        binop_y_36310 = res_30780 * new_index_36304;
        new_index_36311 = binop_x_36303 - binop_y_36310;
        x_36298 = *(__global float *) &res_mem_37393[(new_index_36304 *
                                                      res_30780 +
                                                      new_index_36311) * 4];
        x_36300 = *(__global float *) &res_mem_37344[(gtid_33781 *
                                                      (j_m_i_30913 *
                                                       res_30780) + gtid_33782 *
                                                      j_m_i_30913 +
                                                      ltid_33784) * 4];
        res_33934 = x_36298 * x_36300;
    }
    
    __local char *mem_37438;
    float res_33935;
    
    mem_37438 = (__local char *) mem_37438_backing_0;
    for (int32_t comb_iter_38017 = 0; comb_iter_38017 < 1; comb_iter_38017++) {
        int32_t ctid_33786;
        int32_t flat_comb_id_38018 = comb_iter_38017 * j_m_i_30913 +
                local_tid_33789;
        
        ctid_33786 = flat_comb_id_38018;
        if (slt32(ctid_33786, j_m_i_30913) && 1) {
            *(__local float *) &mem_37438[ctid_33786 * 4] = res_33934;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_38019;
    int32_t skip_waves_38020;
    float x_33936;
    float x_33937;
    
    offset_38019 = 0;
    // participating threads read initial accumulator
    {
        if (slt32(local_tid_33789, j_m_i_30913)) {
            x_33936 = *(__local float *) &mem_37438[(local_tid_33789 +
                                                     offset_38019) * 4];
        }
    }
    offset_38019 = 1;
    while (slt32(offset_38019, wave_sizze_38015)) {
        if (slt32(local_tid_33789 + offset_38019, j_m_i_30913) &&
            ((local_tid_33789 - squot32(local_tid_33789, wave_sizze_38015) *
              wave_sizze_38015) & (2 * offset_38019 - 1)) == 0) {
            // read array element
            {
                x_33937 = *(volatile __local
                            float *) &mem_37438[(local_tid_33789 +
                                                 offset_38019) * 4];
            }
            // apply reduction operation
            {
                float res_33938;
                
                if ((slt32(gtid_33781, sizze_30757) && slt32(gtid_33782,
                                                             res_30780)) &&
                    slt32(ltid_33784, j_m_i_30913)) {
                    res_33938 = x_33936 + x_33937;
                }
                x_33936 = res_33938;
            }
            // write result of operation
            {
                *(volatile __local float *) &mem_37438[local_tid_33789 * 4] =
                    x_33936;
            }
        }
        offset_38019 *= 2;
    }
    skip_waves_38020 = 1;
    while (slt32(skip_waves_38020, squot32(j_m_i_30913 + wave_sizze_38015 - 1,
                                           wave_sizze_38015))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_38019 = skip_waves_38020 * wave_sizze_38015;
        if (slt32(local_tid_33789 + offset_38019, j_m_i_30913) &&
            ((local_tid_33789 - squot32(local_tid_33789, wave_sizze_38015) *
              wave_sizze_38015) == 0 && (squot32(local_tid_33789,
                                                 wave_sizze_38015) & (2 *
                                                                      skip_waves_38020 -
                                                                      1)) ==
             0)) {
            // read array element
            {
                x_33937 = *(__local float *) &mem_37438[(local_tid_33789 +
                                                         offset_38019) * 4];
            }
            // apply reduction operation
            {
                float res_33938;
                
                if ((slt32(gtid_33781, sizze_30757) && slt32(gtid_33782,
                                                             res_30780)) &&
                    slt32(ltid_33784, j_m_i_30913)) {
                    res_33938 = x_33936 + x_33937;
                }
                x_33936 = res_33938;
            }
            // write result of operation
            {
                *(__local float *) &mem_37438[local_tid_33789 * 4] = x_33936;
            }
        }
        skip_waves_38020 *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    res_33935 = *(__local float *) &mem_37438[0];
    if (local_tid_33789 == 0) {
        *(__global float *) &mem_37441[group_id_33790 * 4] = res_33935;
    }
}
__kernel void map_intra_group_33990(__local volatile
                                    int64_t *mem_37472_backing_aligned_0,
                                    int32_t sizze_30756, int32_t sizze_30757,
                                    int32_t res_30780, __global
                                    unsigned char *res_mem_37449, __global
                                    unsigned char *mem_37469, __global
                                    unsigned char *mem_37476)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict mem_37472_backing_0 =
                          mem_37472_backing_aligned_0;
    int32_t global_tid_33990;
    int32_t local_tid_33991;
    int32_t group_sizze_38063;
    int32_t wave_sizze_38062;
    int32_t group_id_33992;
    
    global_tid_33990 = get_global_id(0);
    local_tid_33991 = get_local_id(0);
    group_sizze_38063 = get_local_size(0);
    wave_sizze_38062 = LOCKSTEP_WIDTH;
    group_id_33992 = get_group_id(0);
    
    int32_t gtid_33975;
    int32_t ltid_33976;
    
    gtid_33975 = squot32(global_tid_33990, sizze_30756);
    ltid_33976 = global_tid_33990 - squot32(global_tid_33990, sizze_30756) *
        sizze_30756;
    
    float x_36322;
    
    if (slt32(gtid_33975, sizze_30757) && slt32(ltid_33976, sizze_30756)) {
        float x_34059 = 0.0F;
        
        for (int32_t chunk_offset_34058 = 0; chunk_offset_34058 < res_30780;
             chunk_offset_34058++) {
            float x_34068;
            float x_34069;
            float res_34071;
            float res_34073;
            
            x_34068 = *(__global float *) &res_mem_37449[(gtid_33975 *
                                                          res_30780 +
                                                          chunk_offset_34058) *
                                                         4];
            x_34069 = *(__global float *) &mem_37469[(chunk_offset_34058 *
                                                      sizze_30756 +
                                                      ltid_33976) * 4];
            res_34071 = x_34068 * x_34069;
            res_34073 = x_34059 + res_34071;
            
            float x_tmp_38064 = res_34073;
            
            x_34059 = x_tmp_38064;
        }
        x_36322 = x_34059;
    }
    
    __local char *mem_37472;
    
    mem_37472 = (__local char *) mem_37472_backing_0;
    for (int32_t comb_iter_38065 = 0; comb_iter_38065 < 1; comb_iter_38065++) {
        int32_t ctid_33988;
        int32_t flat_comb_id_38066 = comb_iter_38065 * sizze_30756 +
                local_tid_33991;
        
        ctid_33988 = flat_comb_id_38066;
        if (slt32(ctid_33988, sizze_30756) && 1) {
            *(__local float *) &mem_37472[ctid_33988 * 4] = x_36322;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int32_t i_38067 = 0; i_38067 < squot32(sizze_30756 - local_tid_33991 +
                                                sizze_30756 - 1, sizze_30756);
         i_38067++) {
        *(__global float *) &mem_37476[(group_id_33992 * sizze_30756 +
                                        (i_38067 * sizze_30756 +
                                         local_tid_33991)) * 4] = *(__local
                                                                    float *) &mem_37472[(i_38067 *
                                                                                         sizze_30756 +
                                                                                         local_tid_33991) *
                                                                                        4];
    }
}
__kernel void map_intra_group_34106(__local volatile
                                    int64_t *mem_37495_backing_aligned_0,
                                    int32_t sizze_30756, int32_t sizze_30757,
                                    int32_t res_30780, __global
                                    unsigned char *mem_37218, __global
                                    unsigned char *res_mem_37449, __global
                                    unsigned char *mem_37498)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict mem_37495_backing_0 =
                          mem_37495_backing_aligned_0;
    int32_t global_tid_34106;
    int32_t local_tid_34107;
    int32_t group_sizze_38077;
    int32_t wave_sizze_38076;
    int32_t group_id_34108;
    
    global_tid_34106 = get_global_id(0);
    local_tid_34107 = get_local_id(0);
    group_sizze_38077 = get_local_size(0);
    wave_sizze_38076 = LOCKSTEP_WIDTH;
    group_id_34108 = get_group_id(0);
    
    int32_t gtid_34099;
    int32_t gtid_34100;
    int32_t ltid_34102;
    
    gtid_34099 = squot32(global_tid_34106, sizze_30756 * res_30780);
    gtid_34100 = squot32(global_tid_34106 - squot32(global_tid_34106,
                                                    sizze_30756 * res_30780) *
                         (sizze_30756 * res_30780), res_30780);
    ltid_34102 = global_tid_34106 - squot32(global_tid_34106, sizze_30756 *
                                            res_30780) * (sizze_30756 *
                                                          res_30780) -
        squot32(global_tid_34106 - squot32(global_tid_34106, sizze_30756 *
                                           res_30780) * (sizze_30756 *
                                                         res_30780),
                res_30780) * res_30780;
    
    float x_36328;
    float x_36330;
    float res_34252;
    
    if ((slt32(gtid_34099, sizze_30757) && slt32(gtid_34100, sizze_30756)) &&
        slt32(ltid_34102, res_30780)) {
        x_36328 = *(__global float *) &res_mem_37449[(gtid_34099 * res_30780 +
                                                      ltid_34102) * 4];
        x_36330 = *(__global float *) &mem_37218[(gtid_34100 * res_30780 +
                                                  ltid_34102) * 4];
        res_34252 = x_36328 * x_36330;
    }
    
    __local char *mem_37495;
    float res_34253;
    
    mem_37495 = (__local char *) mem_37495_backing_0;
    for (int32_t comb_iter_38078 = 0; comb_iter_38078 < 1; comb_iter_38078++) {
        int32_t ctid_34104;
        int32_t flat_comb_id_38079 = comb_iter_38078 * res_30780 +
                local_tid_34107;
        
        ctid_34104 = flat_comb_id_38079;
        if (slt32(ctid_34104, res_30780) && 1) {
            *(__local float *) &mem_37495[ctid_34104 * 4] = res_34252;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_38080;
    int32_t skip_waves_38081;
    float x_34254;
    float x_34255;
    
    offset_38080 = 0;
    // participating threads read initial accumulator
    {
        if (slt32(local_tid_34107, res_30780)) {
            x_34254 = *(__local float *) &mem_37495[(local_tid_34107 +
                                                     offset_38080) * 4];
        }
    }
    offset_38080 = 1;
    while (slt32(offset_38080, wave_sizze_38076)) {
        if (slt32(local_tid_34107 + offset_38080, res_30780) &&
            ((local_tid_34107 - squot32(local_tid_34107, wave_sizze_38076) *
              wave_sizze_38076) & (2 * offset_38080 - 1)) == 0) {
            // read array element
            {
                x_34255 = *(volatile __local
                            float *) &mem_37495[(local_tid_34107 +
                                                 offset_38080) * 4];
            }
            // apply reduction operation
            {
                float res_34256;
                
                if ((slt32(gtid_34099, sizze_30757) && slt32(gtid_34100,
                                                             sizze_30756)) &&
                    slt32(ltid_34102, res_30780)) {
                    res_34256 = x_34254 + x_34255;
                }
                x_34254 = res_34256;
            }
            // write result of operation
            {
                *(volatile __local float *) &mem_37495[local_tid_34107 * 4] =
                    x_34254;
            }
        }
        offset_38080 *= 2;
    }
    skip_waves_38081 = 1;
    while (slt32(skip_waves_38081, squot32(res_30780 + wave_sizze_38076 - 1,
                                           wave_sizze_38076))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_38080 = skip_waves_38081 * wave_sizze_38076;
        if (slt32(local_tid_34107 + offset_38080, res_30780) &&
            ((local_tid_34107 - squot32(local_tid_34107, wave_sizze_38076) *
              wave_sizze_38076) == 0 && (squot32(local_tid_34107,
                                                 wave_sizze_38076) & (2 *
                                                                      skip_waves_38081 -
                                                                      1)) ==
             0)) {
            // read array element
            {
                x_34255 = *(__local float *) &mem_37495[(local_tid_34107 +
                                                         offset_38080) * 4];
            }
            // apply reduction operation
            {
                float res_34256;
                
                if ((slt32(gtid_34099, sizze_30757) && slt32(gtid_34100,
                                                             sizze_30756)) &&
                    slt32(ltid_34102, res_30780)) {
                    res_34256 = x_34254 + x_34255;
                }
                x_34254 = res_34256;
            }
            // write result of operation
            {
                *(__local float *) &mem_37495[local_tid_34107 * 4] = x_34254;
            }
        }
        skip_waves_38081 *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    res_34253 = *(__local float *) &mem_37495[0];
    if (local_tid_34107 == 0) {
        *(__global float *) &mem_37498[group_id_34108 * 4] = res_34253;
    }
}
__kernel void map_intra_group_34303(__local volatile
                                    int64_t *mem_37553_backing_aligned_0,
                                    __local volatile
                                    int64_t *mem_37556_backing_aligned_1,
                                    __local volatile
                                    int64_t *mem_37559_backing_aligned_2,
                                    __local volatile
                                    int64_t *mem_37562_backing_aligned_3,
                                    int32_t sizze_30756, int32_t sizze_30757,
                                    int32_t sizze_30758, int32_t i_31033,
                                    __global unsigned char *images_mem_37201,
                                    __global unsigned char *res_mem_37506,
                                    __global unsigned char *mem_37565, __global
                                    unsigned char *mem_37569, __global
                                    unsigned char *mem_37573)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict mem_37553_backing_0 =
                          mem_37553_backing_aligned_0;
    __local volatile char *restrict mem_37556_backing_1 =
                          mem_37556_backing_aligned_1;
    __local volatile char *restrict mem_37559_backing_2 =
                          mem_37559_backing_aligned_2;
    __local volatile char *restrict mem_37562_backing_3 =
                          mem_37562_backing_aligned_3;
    int32_t global_tid_34303;
    int32_t local_tid_34304;
    int32_t group_sizze_38134;
    int32_t wave_sizze_38133;
    int32_t group_id_34305;
    
    global_tid_34303 = get_global_id(0);
    local_tid_34304 = get_local_id(0);
    group_sizze_38134 = get_local_size(0);
    wave_sizze_38133 = LOCKSTEP_WIDTH;
    group_id_34305 = get_group_id(0);
    
    int32_t gtid_34293;
    int32_t ltid_34294;
    
    gtid_34293 = squot32(global_tid_34303, sizze_30756);
    ltid_34294 = global_tid_34303 - squot32(global_tid_34303, sizze_30756) *
        sizze_30756;
    
    float x_36349;
    float x_36351;
    bool res_34477;
    bool cond_34478;
    float res_34479;
    bool res_34481;
    bool res_34482;
    int32_t res_34483;
    
    if (slt32(gtid_34293, sizze_30757) && slt32(ltid_34294, sizze_30756)) {
        x_36349 = *(__global float *) &images_mem_37201[(gtid_34293 *
                                                         sizze_30758 +
                                                         ltid_34294) * 4];
        x_36351 = *(__global float *) &res_mem_37506[(gtid_34293 * sizze_30756 +
                                                      ltid_34294) * 4];
        res_34477 = futrts_isnan32(x_36349);
        cond_34478 = !res_34477;
        if (cond_34478) {
            float res_34480 = x_36349 - x_36351;
            
            res_34479 = res_34480;
        } else {
            res_34479 = NAN;
        }
        res_34481 = futrts_isnan32(res_34479);
        res_34482 = !res_34481;
        if (res_34482) {
            res_34483 = 1;
        } else {
            res_34483 = 0;
        }
    }
    
    __local char *mem_37553;
    __local char *mem_37556;
    
    mem_37553 = (__local char *) mem_37553_backing_0;
    mem_37556 = (__local char *) mem_37556_backing_1;
    for (int32_t comb_iter_38135 = 0; comb_iter_38135 < 1; comb_iter_38135++) {
        int32_t ctid_34296;
        int32_t flat_comb_id_38136 = comb_iter_38135 * sizze_30756 +
                local_tid_34304;
        
        ctid_34296 = flat_comb_id_38136;
        if (slt32(ctid_34296, sizze_30756) && 1) {
            *(__local int32_t *) &mem_37553[ctid_34296 * 4] = res_34483;
            *(__local float *) &mem_37556[ctid_34296 * 4] = res_34479;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t x_34485;
    int32_t x_34486;
    int32_t x_38137;
    int32_t x_38138;
    int32_t skip_threads_38140;
    
    if (slt32(local_tid_34304, sizze_30756)) {
        x_34486 = *(volatile __local int32_t *) &mem_37553[local_tid_34304 *
                                                           sizeof(int32_t)];
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_38140 = 1;
        while (slt32(skip_threads_38140, 32)) {
            if (sle32(skip_threads_38140, local_tid_34304 -
                      squot32(local_tid_34304, 32) * 32) &&
                slt32(local_tid_34304, sizze_30756)) {
                // read operands
                {
                    x_34485 = *(volatile __local
                                int32_t *) &mem_37553[(local_tid_34304 -
                                                       skip_threads_38140) *
                                                      sizeof(int32_t)];
                }
                // perform operation
                {
                    int32_t res_34487;
                    
                    if (slt32(gtid_34293, sizze_30757) && slt32(ltid_34294,
                                                                sizze_30756)) {
                        res_34487 = x_34485 + x_34486;
                    }
                    x_34486 = res_34487;
                }
            }
            if (sle32(wave_sizze_38133, skip_threads_38140)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_38140, local_tid_34304 -
                      squot32(local_tid_34304, 32) * 32) &&
                slt32(local_tid_34304, sizze_30756)) {
                // write result
                {
                    *(volatile __local int32_t *) &mem_37553[local_tid_34304 *
                                                             sizeof(int32_t)] =
                        x_34486;
                }
            }
            if (sle32(wave_sizze_38133, skip_threads_38140)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_38140 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_34304 - squot32(local_tid_34304, 32) * 32) == 31 &&
            slt32(local_tid_34304, sizze_30756)) {
            *(volatile __local int32_t *) &mem_37553[squot32(local_tid_34304,
                                                             32) *
                                                     sizeof(int32_t)] = x_34486;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'
    {
        int32_t skip_threads_38141;
        
        if (squot32(local_tid_34304, 32) == 0 && slt32(local_tid_34304,
                                                       sizze_30756)) {
            x_38138 = *(volatile __local int32_t *) &mem_37553[local_tid_34304 *
                                                               sizeof(int32_t)];
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_38141 = 1;
            while (slt32(skip_threads_38141, 32)) {
                if (sle32(skip_threads_38141, local_tid_34304 -
                          squot32(local_tid_34304, 32) * 32) &&
                    (squot32(local_tid_34304, 32) == 0 && slt32(local_tid_34304,
                                                                sizze_30756))) {
                    // read operands
                    {
                        x_38137 = *(volatile __local
                                    int32_t *) &mem_37553[(local_tid_34304 -
                                                           skip_threads_38141) *
                                                          sizeof(int32_t)];
                    }
                    // perform operation
                    {
                        int32_t res_38139;
                        
                        if (slt32(gtid_34293, sizze_30757) && slt32(ltid_34294,
                                                                    sizze_30756)) {
                            res_38139 = x_38137 + x_38138;
                        }
                        x_38138 = res_38139;
                    }
                }
                if (sle32(wave_sizze_38133, skip_threads_38141)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_38141, local_tid_34304 -
                          squot32(local_tid_34304, 32) * 32) &&
                    (squot32(local_tid_34304, 32) == 0 && slt32(local_tid_34304,
                                                                sizze_30756))) {
                    // write result
                    {
                        *(volatile __local
                          int32_t *) &mem_37553[local_tid_34304 *
                                                sizeof(int32_t)] = x_38138;
                    }
                }
                if (sle32(wave_sizze_38133, skip_threads_38141)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_38141 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_34304, 32) == 0 || !slt32(local_tid_34304,
                                                          sizze_30756))) {
            // read operands
            {
                x_34485 = *(volatile __local
                            int32_t *) &mem_37553[(squot32(local_tid_34304,
                                                           32) - 1) *
                                                  sizeof(int32_t)];
            }
            // perform operation
            {
                int32_t res_34487;
                
                if (slt32(gtid_34293, sizze_30757) && slt32(ltid_34294,
                                                            sizze_30756)) {
                    res_34487 = x_34485 + x_34486;
                }
                x_34486 = res_34487;
            }
            // write final result
            {
                *(volatile __local int32_t *) &mem_37553[local_tid_34304 *
                                                         sizeof(int32_t)] =
                    x_34486;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_34304, 32) == 0) {
            *(volatile __local int32_t *) &mem_37553[local_tid_34304 *
                                                     sizeof(int32_t)] = x_34486;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t res_34488;
    
    if (slt32(gtid_34293, sizze_30757) && slt32(ltid_34294, sizze_30756)) {
        res_34488 = *(__local int32_t *) &mem_37553[i_31033 * 4];
    }
    
    __local char *mem_37559;
    __local char *mem_37562;
    
    mem_37559 = (__local char *) mem_37559_backing_2;
    for (int32_t comb_iter_38142 = 0; comb_iter_38142 < 1; comb_iter_38142++) {
        int32_t new_local_index_34297;
        int32_t flat_comb_id_38143 = comb_iter_38142 * sizze_30756 +
                local_tid_34304;
        
        new_local_index_34297 = flat_comb_id_38143;
        if (slt32(new_local_index_34297, sizze_30756) && 1) {
            *(__local float *) &mem_37559[new_local_index_34297 * 4] = NAN;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    mem_37562 = (__local char *) mem_37562_backing_3;
    for (int32_t comb_iter_38144 = 0; comb_iter_38144 < 1; comb_iter_38144++) {
        int32_t new_local_index_34299;
        int32_t flat_comb_id_38145 = comb_iter_38144 * sizze_30756 +
                local_tid_34304;
        
        new_local_index_34299 = flat_comb_id_38145;
        if (slt32(new_local_index_34299, sizze_30756) && 1) {
            *(__local int32_t *) &mem_37562[new_local_index_34299 * 4] = 0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    float x_36354;
    int32_t x_36356;
    bool res_34498;
    bool res_34499;
    int32_t res_34500;
    
    if (slt32(gtid_34293, sizze_30757) && slt32(ltid_34294, sizze_30756)) {
        x_36354 = *(__local float *) &mem_37556[ltid_34294 * 4];
        x_36356 = *(__local int32_t *) &mem_37553[ltid_34294 * 4];
        res_34498 = futrts_isnan32(x_36354);
        res_34499 = !res_34498;
        if (res_34499) {
            int32_t res_34501 = x_36356 - 1;
            
            res_34500 = res_34501;
        } else {
            res_34500 = -1;
        }
    }
    for (int32_t comb_iter_38146 = 0; comb_iter_38146 < 1; comb_iter_38146++) {
        int32_t ctid_34301;
        int32_t flat_comb_id_38147 = comb_iter_38146 * sizze_30756 +
                local_tid_34304;
        
        ctid_34301 = flat_comb_id_38147;
        if (slt32(ctid_34301, sizze_30756) && 1) {
            if (sle32(0, res_34500) && slt32(res_34500, sizze_30756)) {
                *(__local int32_t *) &mem_37562[res_34500 * 4] = ltid_34294;
            }
            if (sle32(0, res_34500) && slt32(res_34500, sizze_30756)) {
                *(__local float *) &mem_37559[res_34500 * 4] = x_36354;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_tid_34304 == 0) {
        *(__global int32_t *) &mem_37565[group_id_34305 * 4] = res_34488;
    }
    for (int32_t i_38149 = 0; i_38149 < squot32(sizze_30756 - local_tid_34304 +
                                                sizze_30756 - 1, sizze_30756);
         i_38149++) {
        *(__global float *) &mem_37569[(group_id_34305 * sizze_30756 +
                                        (i_38149 * sizze_30756 +
                                         local_tid_34304)) * 4] = *(__local
                                                                    float *) &mem_37559[(i_38149 *
                                                                                         sizze_30756 +
                                                                                         local_tid_34304) *
                                                                                        4];
    }
    for (int32_t i_38150 = 0; i_38150 < squot32(sizze_30756 - local_tid_34304 +
                                                sizze_30756 - 1, sizze_30756);
         i_38150++) {
        *(__global int32_t *) &mem_37573[(group_id_34305 * sizze_30756 +
                                          (i_38150 * sizze_30756 +
                                           local_tid_34304)) * 4] = *(__local
                                                                      int32_t *) &mem_37562[(i_38150 *
                                                                                             sizze_30756 +
                                                                                             local_tid_34304) *
                                                                                            4];
    }
}
__kernel void map_intra_group_34741(__local volatile
                                    int64_t *mem_37618_backing_aligned_0,
                                    __local volatile
                                    int64_t *mem_37621_backing_aligned_1,
                                    int32_t sizze_30756, int32_t sizze_30757,
                                    int32_t sizze_30758, int32_t n_30761,
                                    float hfrac_30763, int32_t res_30778,
                                    __global unsigned char *images_mem_37201,
                                    __global unsigned char *res_mem_37597,
                                    __global unsigned char *mem_37624, __global
                                    unsigned char *mem_37627, __global
                                    unsigned char *mem_37630)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict mem_37618_backing_0 =
                          mem_37618_backing_aligned_0;
    __local volatile char *restrict mem_37621_backing_1 =
                          mem_37621_backing_aligned_1;
    int32_t global_tid_34741;
    int32_t local_tid_34742;
    int32_t group_sizze_38214;
    int32_t wave_sizze_38213;
    int32_t group_id_34743;
    
    global_tid_34741 = get_global_id(0);
    local_tid_34742 = get_local_id(0);
    group_sizze_38214 = get_local_size(0);
    wave_sizze_38213 = LOCKSTEP_WIDTH;
    group_id_34743 = get_group_id(0);
    
    int32_t gtid_34734;
    int32_t ltid_34735;
    
    gtid_34734 = squot32(global_tid_34741, n_30761);
    ltid_34735 = global_tid_34741 - squot32(global_tid_34741, n_30761) *
        n_30761;
    
    float x_36384;
    bool res_34882;
    bool cond_34883;
    int32_t res_34884;
    
    if (slt32(gtid_34734, sizze_30757) && slt32(ltid_34735, n_30761)) {
        x_36384 = *(__global float *) &images_mem_37201[(gtid_34734 *
                                                         sizze_30758 +
                                                         ltid_34735) * 4];
        res_34882 = futrts_isnan32(x_36384);
        cond_34883 = !res_34882;
        if (cond_34883) {
            res_34884 = 1;
        } else {
            res_34884 = 0;
        }
    }
    
    __local char *mem_37618;
    int32_t res_34885;
    
    mem_37618 = (__local char *) mem_37618_backing_0;
    for (int32_t comb_iter_38215 = 0; comb_iter_38215 < 1; comb_iter_38215++) {
        int32_t ctid_34737;
        int32_t flat_comb_id_38216 = comb_iter_38215 * n_30761 +
                local_tid_34742;
        
        ctid_34737 = flat_comb_id_38216;
        if (slt32(ctid_34737, n_30761) && 1) {
            *(__local int32_t *) &mem_37618[ctid_34737 * 4] = res_34884;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_38217;
    int32_t skip_waves_38218;
    int32_t x_34886;
    int32_t x_34887;
    
    offset_38217 = 0;
    // participating threads read initial accumulator
    {
        if (slt32(local_tid_34742, n_30761)) {
            x_34886 = *(__local int32_t *) &mem_37618[(local_tid_34742 +
                                                       offset_38217) * 4];
        }
    }
    offset_38217 = 1;
    while (slt32(offset_38217, wave_sizze_38213)) {
        if (slt32(local_tid_34742 + offset_38217, n_30761) &&
            ((local_tid_34742 - squot32(local_tid_34742, wave_sizze_38213) *
              wave_sizze_38213) & (2 * offset_38217 - 1)) == 0) {
            // read array element
            {
                x_34887 = *(volatile __local
                            int32_t *) &mem_37618[(local_tid_34742 +
                                                   offset_38217) * 4];
            }
            // apply reduction operation
            {
                int32_t res_34888;
                
                if (slt32(gtid_34734, sizze_30757) && slt32(ltid_34735,
                                                            n_30761)) {
                    res_34888 = x_34886 + x_34887;
                }
                x_34886 = res_34888;
            }
            // write result of operation
            {
                *(volatile __local int32_t *) &mem_37618[local_tid_34742 * 4] =
                    x_34886;
            }
        }
        offset_38217 *= 2;
    }
    skip_waves_38218 = 1;
    while (slt32(skip_waves_38218, squot32(n_30761 + wave_sizze_38213 - 1,
                                           wave_sizze_38213))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_38217 = skip_waves_38218 * wave_sizze_38213;
        if (slt32(local_tid_34742 + offset_38217, n_30761) &&
            ((local_tid_34742 - squot32(local_tid_34742, wave_sizze_38213) *
              wave_sizze_38213) == 0 && (squot32(local_tid_34742,
                                                 wave_sizze_38213) & (2 *
                                                                      skip_waves_38218 -
                                                                      1)) ==
             0)) {
            // read array element
            {
                x_34887 = *(__local int32_t *) &mem_37618[(local_tid_34742 +
                                                           offset_38217) * 4];
            }
            // apply reduction operation
            {
                int32_t res_34888;
                
                if (slt32(gtid_34734, sizze_30757) && slt32(ltid_34735,
                                                            n_30761)) {
                    res_34888 = x_34886 + x_34887;
                }
                x_34886 = res_34888;
            }
            // write result of operation
            {
                *(__local int32_t *) &mem_37618[local_tid_34742 * 4] = x_34886;
            }
        }
        skip_waves_38218 *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    res_34885 = *(__local int32_t *) &mem_37618[0];
    
    bool cond_34891;
    float x_36389;
    float res_34894;
    
    if (slt32(gtid_34734, sizze_30757) && slt32(ltid_34735, n_30761)) {
        cond_34891 = slt32(ltid_34735, res_34885);
        if (cond_34891) {
            float res_34893 = *(__global float *) &res_mem_37597[(gtid_34734 *
                                                                  sizze_30756 +
                                                                  ltid_34735) *
                                                                 4];
            
            x_36389 = res_34893;
        } else {
            x_36389 = 0.0F;
        }
        res_34894 = x_36389 * x_36389;
    }
    
    __local char *mem_37621;
    float res_34895;
    
    mem_37621 = (__local char *) mem_37621_backing_1;
    for (int32_t comb_iter_38219 = 0; comb_iter_38219 < 1; comb_iter_38219++) {
        int32_t ctid_34739;
        int32_t flat_comb_id_38220 = comb_iter_38219 * n_30761 +
                local_tid_34742;
        
        ctid_34739 = flat_comb_id_38220;
        if (slt32(ctid_34739, n_30761) && 1) {
            *(__local float *) &mem_37621[ctid_34739 * 4] = res_34894;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_38221;
    int32_t skip_waves_38222;
    float x_34896;
    float x_34897;
    
    offset_38221 = 0;
    // participating threads read initial accumulator
    {
        if (slt32(local_tid_34742, n_30761)) {
            x_34896 = *(__local float *) &mem_37621[(local_tid_34742 +
                                                     offset_38221) * 4];
        }
    }
    offset_38221 = 1;
    while (slt32(offset_38221, wave_sizze_38213)) {
        if (slt32(local_tid_34742 + offset_38221, n_30761) &&
            ((local_tid_34742 - squot32(local_tid_34742, wave_sizze_38213) *
              wave_sizze_38213) & (2 * offset_38221 - 1)) == 0) {
            // read array element
            {
                x_34897 = *(volatile __local
                            float *) &mem_37621[(local_tid_34742 +
                                                 offset_38221) * 4];
            }
            // apply reduction operation
            {
                float res_34898;
                
                if (slt32(gtid_34734, sizze_30757) && slt32(ltid_34735,
                                                            n_30761)) {
                    res_34898 = x_34896 + x_34897;
                }
                x_34896 = res_34898;
            }
            // write result of operation
            {
                *(volatile __local float *) &mem_37621[local_tid_34742 * 4] =
                    x_34896;
            }
        }
        offset_38221 *= 2;
    }
    skip_waves_38222 = 1;
    while (slt32(skip_waves_38222, squot32(n_30761 + wave_sizze_38213 - 1,
                                           wave_sizze_38213))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_38221 = skip_waves_38222 * wave_sizze_38213;
        if (slt32(local_tid_34742 + offset_38221, n_30761) &&
            ((local_tid_34742 - squot32(local_tid_34742, wave_sizze_38213) *
              wave_sizze_38213) == 0 && (squot32(local_tid_34742,
                                                 wave_sizze_38213) & (2 *
                                                                      skip_waves_38222 -
                                                                      1)) ==
             0)) {
            // read array element
            {
                x_34897 = *(__local float *) &mem_37621[(local_tid_34742 +
                                                         offset_38221) * 4];
            }
            // apply reduction operation
            {
                float res_34898;
                
                if (slt32(gtid_34734, sizze_30757) && slt32(ltid_34735,
                                                            n_30761)) {
                    res_34898 = x_34896 + x_34897;
                }
                x_34896 = res_34898;
            }
            // write result of operation
            {
                *(__local float *) &mem_37621[local_tid_34742 * 4] = x_34896;
            }
        }
        skip_waves_38222 *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    res_34895 = *(__local float *) &mem_37621[0];
    
    int32_t arg_34899;
    float res_34900;
    float arg_34901;
    float res_34902;
    float res_34903;
    float arg_34904;
    int32_t res_34905;
    
    if (slt32(gtid_34734, sizze_30757) && slt32(ltid_34735, n_30761)) {
        arg_34899 = res_34885 - res_30778;
        res_34900 = sitofp_i32_f32(arg_34899);
        arg_34901 = res_34895 / res_34900;
        res_34902 = futrts_sqrt32(arg_34901);
        res_34903 = sitofp_i32_f32(res_34885);
        arg_34904 = hfrac_30763 * res_34903;
        res_34905 = fptosi_f32_i32(arg_34904);
    }
    if (local_tid_34742 == 0) {
        *(__global int32_t *) &mem_37624[group_id_34743 * 4] = res_34905;
    }
    if (local_tid_34742 == 0) {
        *(__global int32_t *) &mem_37627[group_id_34743 * 4] = res_34885;
    }
    if (local_tid_34742 == 0) {
        *(__global float *) &mem_37630[group_id_34743 * 4] = res_34902;
    }
}
__kernel void map_intra_group_35089(__local volatile
                                    int64_t *mem_37657_backing_aligned_0,
                                    int32_t sizze_30756, int32_t sizze_30757,
                                    int32_t res_31136, __global
                                    unsigned char *res_mem_37597, __global
                                    unsigned char *res_mem_37646, __global
                                    unsigned char *res_mem_37647, __global
                                    unsigned char *mem_37660)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict mem_37657_backing_0 =
                          mem_37657_backing_aligned_0;
    int32_t global_tid_35089;
    int32_t local_tid_35090;
    int32_t group_sizze_38324;
    int32_t wave_sizze_38323;
    int32_t group_id_35091;
    
    global_tid_35089 = get_global_id(0);
    local_tid_35090 = get_local_id(0);
    group_sizze_38324 = get_local_size(0);
    wave_sizze_38323 = LOCKSTEP_WIDTH;
    group_id_35091 = get_group_id(0);
    
    int32_t gtid_35084;
    int32_t ltid_35085;
    
    gtid_35084 = squot32(global_tid_35089, res_31136);
    ltid_35085 = global_tid_35089 - squot32(global_tid_35089, res_31136) *
        res_31136;
    
    int32_t x_35170;
    int32_t x_35171;
    bool cond_35174;
    float x_36409;
    
    if (slt32(gtid_35084, sizze_30757) && slt32(ltid_35085, res_31136)) {
        x_35170 = *(__global int32_t *) &res_mem_37647[gtid_35084 * 4];
        x_35171 = *(__global int32_t *) &res_mem_37646[gtid_35084 * 4];
        cond_35174 = slt32(ltid_35085, x_35171);
        if (cond_35174) {
            int32_t x_35176;
            int32_t x_35177;
            int32_t i_35178;
            float res_35179;
            
            x_35176 = ltid_35085 + x_35170;
            x_35177 = x_35176 - x_35171;
            i_35178 = 1 + x_35177;
            res_35179 = *(__global float *) &res_mem_37597[(gtid_35084 *
                                                            sizze_30756 +
                                                            i_35178) * 4];
            x_36409 = res_35179;
        } else {
            x_36409 = 0.0F;
        }
    }
    
    __local char *mem_37657;
    float res_35180;
    
    mem_37657 = (__local char *) mem_37657_backing_0;
    for (int32_t comb_iter_38325 = 0; comb_iter_38325 < 1; comb_iter_38325++) {
        int32_t ctid_35087;
        int32_t flat_comb_id_38326 = comb_iter_38325 * res_31136 +
                local_tid_35090;
        
        ctid_35087 = flat_comb_id_38326;
        if (slt32(ctid_35087, res_31136) && 1) {
            *(__local float *) &mem_37657[ctid_35087 * 4] = x_36409;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_38327;
    int32_t skip_waves_38328;
    float x_35181;
    float x_35182;
    
    offset_38327 = 0;
    // participating threads read initial accumulator
    {
        if (slt32(local_tid_35090, res_31136)) {
            x_35181 = *(__local float *) &mem_37657[(local_tid_35090 +
                                                     offset_38327) * 4];
        }
    }
    offset_38327 = 1;
    while (slt32(offset_38327, wave_sizze_38323)) {
        if (slt32(local_tid_35090 + offset_38327, res_31136) &&
            ((local_tid_35090 - squot32(local_tid_35090, wave_sizze_38323) *
              wave_sizze_38323) & (2 * offset_38327 - 1)) == 0) {
            // read array element
            {
                x_35182 = *(volatile __local
                            float *) &mem_37657[(local_tid_35090 +
                                                 offset_38327) * 4];
            }
            // apply reduction operation
            {
                float res_35183;
                
                if (slt32(gtid_35084, sizze_30757) && slt32(ltid_35085,
                                                            res_31136)) {
                    res_35183 = x_35181 + x_35182;
                }
                x_35181 = res_35183;
            }
            // write result of operation
            {
                *(volatile __local float *) &mem_37657[local_tid_35090 * 4] =
                    x_35181;
            }
        }
        offset_38327 *= 2;
    }
    skip_waves_38328 = 1;
    while (slt32(skip_waves_38328, squot32(res_31136 + wave_sizze_38323 - 1,
                                           wave_sizze_38323))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_38327 = skip_waves_38328 * wave_sizze_38323;
        if (slt32(local_tid_35090 + offset_38327, res_31136) &&
            ((local_tid_35090 - squot32(local_tid_35090, wave_sizze_38323) *
              wave_sizze_38323) == 0 && (squot32(local_tid_35090,
                                                 wave_sizze_38323) & (2 *
                                                                      skip_waves_38328 -
                                                                      1)) ==
             0)) {
            // read array element
            {
                x_35182 = *(__local float *) &mem_37657[(local_tid_35090 +
                                                         offset_38327) * 4];
            }
            // apply reduction operation
            {
                float res_35183;
                
                if (slt32(gtid_35084, sizze_30757) && slt32(ltid_35085,
                                                            res_31136)) {
                    res_35183 = x_35181 + x_35182;
                }
                x_35181 = res_35183;
            }
            // write result of operation
            {
                *(__local float *) &mem_37657[local_tid_35090 * 4] = x_35181;
            }
        }
        skip_waves_38328 *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    res_35180 = *(__local float *) &mem_37657[0];
    if (local_tid_35090 == 0) {
        *(__global float *) &mem_37660[group_id_35091 * 4] = res_35180;
    }
}
__kernel void map_intra_group_35333(__local volatile
                                    int64_t *mem_37683_backing_aligned_0,
                                    __local volatile
                                    int64_t *mem_37685_backing_aligned_1,
                                    __local volatile
                                    int64_t *mem_37688_backing_aligned_2,
                                    __local volatile
                                    int64_t *mem_37691_backing_aligned_3,
                                    int32_t sizze_30756, int32_t sizze_30757,
                                    int32_t n_30761, int32_t arg_31158, __global
                                    unsigned char *res_mem_37596, __global
                                    unsigned char *res_mem_37597, __global
                                    unsigned char *res_mem_37598, __global
                                    unsigned char *res_mem_37646, __global
                                    unsigned char *res_mem_37647, __global
                                    unsigned char *res_mem_37648, __global
                                    unsigned char *res_mem_37665, __global
                                    unsigned char *mem_37668, __global
                                    unsigned char *mem_37694, __global
                                    unsigned char *mem_37697)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict mem_37683_backing_0 =
                          mem_37683_backing_aligned_0;
    __local volatile char *restrict mem_37685_backing_1 =
                          mem_37685_backing_aligned_1;
    __local volatile char *restrict mem_37688_backing_2 =
                          mem_37688_backing_aligned_2;
    __local volatile char *restrict mem_37691_backing_3 =
                          mem_37691_backing_aligned_3;
    int32_t global_tid_35333;
    int32_t local_tid_35334;
    int32_t group_sizze_38380;
    int32_t wave_sizze_38379;
    int32_t group_id_35335;
    
    global_tid_35333 = get_global_id(0);
    local_tid_35334 = get_local_id(0);
    group_sizze_38380 = get_local_size(0);
    wave_sizze_38379 = LOCKSTEP_WIDTH;
    group_id_35335 = get_group_id(0);
    
    int32_t gtid_35324;
    int32_t ltid_35325;
    
    gtid_35324 = squot32(global_tid_35333, arg_31158);
    ltid_35325 = global_tid_35333 - squot32(global_tid_35333, arg_31158) *
        arg_31158;
    
    int32_t x_35623;
    int32_t x_35624;
    float x_35625;
    int32_t x_35626;
    float x_35627;
    int32_t y_35630;
    float res_35631;
    float res_35632;
    float y_35633;
    bool cond_35643;
    float x_36478;
    
    if (slt32(gtid_35324, sizze_30757) && slt32(ltid_35325, arg_31158)) {
        x_35623 = *(__global int32_t *) &res_mem_37596[gtid_35324 * 4];
        x_35624 = *(__global int32_t *) &res_mem_37647[gtid_35324 * 4];
        x_35625 = *(__global float *) &res_mem_37648[gtid_35324 * 4];
        x_35626 = *(__global int32_t *) &res_mem_37646[gtid_35324 * 4];
        x_35627 = *(__global float *) &res_mem_37665[gtid_35324 * 4];
        y_35630 = x_35623 - x_35624;
        res_35631 = sitofp_i32_f32(x_35624);
        res_35632 = futrts_sqrt32(res_35631);
        y_35633 = x_35625 * res_35632;
        cond_35643 = sle32(y_35630, ltid_35325);
        if (cond_35643) {
            x_36478 = 0.0F;
        } else {
            bool cond_35645;
            float res_35646;
            
            cond_35645 = ltid_35325 == 0;
            if (cond_35645) {
                res_35646 = x_35627;
            } else {
                int32_t x_35647;
                int32_t i_35648;
                float negate_arg_35649;
                float x_35650;
                int32_t i_35651;
                float y_35652;
                float res_35653;
                
                x_35647 = x_35624 - x_35626;
                i_35648 = ltid_35325 + x_35647;
                negate_arg_35649 = *(__global
                                     float *) &res_mem_37597[(gtid_35324 *
                                                              sizze_30756 +
                                                              i_35648) * 4];
                x_35650 = 0.0F - negate_arg_35649;
                i_35651 = ltid_35325 + x_35624;
                y_35652 = *(__global float *) &res_mem_37597[(gtid_35324 *
                                                              sizze_30756 +
                                                              i_35651) * 4];
                res_35653 = x_35650 + y_35652;
                res_35646 = res_35653;
            }
            x_36478 = res_35646;
        }
    }
    
    __local char *mem_37683;
    
    mem_37683 = (__local char *) mem_37683_backing_0;
    for (int32_t comb_iter_38381 = 0; comb_iter_38381 < 1; comb_iter_38381++) {
        int32_t ctid_35327;
        int32_t flat_comb_id_38382 = comb_iter_38381 * arg_31158 +
                local_tid_35334;
        
        ctid_35327 = flat_comb_id_38382;
        if (slt32(ctid_35327, arg_31158) && 1) {
            *(__local float *) &mem_37683[ctid_35327 * 4] = x_36478;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    float x_35655;
    float x_35656;
    float x_38383;
    float x_38384;
    int32_t skip_threads_38386;
    
    if (slt32(local_tid_35334, arg_31158)) {
        x_35656 = *(volatile __local float *) &mem_37683[local_tid_35334 *
                                                         sizeof(float)];
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_38386 = 1;
        while (slt32(skip_threads_38386, 32)) {
            if (sle32(skip_threads_38386, local_tid_35334 -
                      squot32(local_tid_35334, 32) * 32) &&
                slt32(local_tid_35334, arg_31158)) {
                // read operands
                {
                    x_35655 = *(volatile __local
                                float *) &mem_37683[(local_tid_35334 -
                                                     skip_threads_38386) *
                                                    sizeof(float)];
                }
                // perform operation
                {
                    float res_35657;
                    
                    if (slt32(gtid_35324, sizze_30757) && slt32(ltid_35325,
                                                                arg_31158)) {
                        res_35657 = x_35655 + x_35656;
                    }
                    x_35656 = res_35657;
                }
            }
            if (sle32(wave_sizze_38379, skip_threads_38386)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_38386, local_tid_35334 -
                      squot32(local_tid_35334, 32) * 32) &&
                slt32(local_tid_35334, arg_31158)) {
                // write result
                {
                    *(volatile __local float *) &mem_37683[local_tid_35334 *
                                                           sizeof(float)] =
                        x_35656;
                }
            }
            if (sle32(wave_sizze_38379, skip_threads_38386)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_38386 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_35334 - squot32(local_tid_35334, 32) * 32) == 31 &&
            slt32(local_tid_35334, arg_31158)) {
            *(volatile __local float *) &mem_37683[squot32(local_tid_35334,
                                                           32) *
                                                   sizeof(float)] = x_35656;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'
    {
        int32_t skip_threads_38387;
        
        if (squot32(local_tid_35334, 32) == 0 && slt32(local_tid_35334,
                                                       arg_31158)) {
            x_38384 = *(volatile __local float *) &mem_37683[local_tid_35334 *
                                                             sizeof(float)];
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_38387 = 1;
            while (slt32(skip_threads_38387, 32)) {
                if (sle32(skip_threads_38387, local_tid_35334 -
                          squot32(local_tid_35334, 32) * 32) &&
                    (squot32(local_tid_35334, 32) == 0 && slt32(local_tid_35334,
                                                                arg_31158))) {
                    // read operands
                    {
                        x_38383 = *(volatile __local
                                    float *) &mem_37683[(local_tid_35334 -
                                                         skip_threads_38387) *
                                                        sizeof(float)];
                    }
                    // perform operation
                    {
                        float res_38385;
                        
                        if (slt32(gtid_35324, sizze_30757) && slt32(ltid_35325,
                                                                    arg_31158)) {
                            res_38385 = x_38383 + x_38384;
                        }
                        x_38384 = res_38385;
                    }
                }
                if (sle32(wave_sizze_38379, skip_threads_38387)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_38387, local_tid_35334 -
                          squot32(local_tid_35334, 32) * 32) &&
                    (squot32(local_tid_35334, 32) == 0 && slt32(local_tid_35334,
                                                                arg_31158))) {
                    // write result
                    {
                        *(volatile __local float *) &mem_37683[local_tid_35334 *
                                                               sizeof(float)] =
                            x_38384;
                    }
                }
                if (sle32(wave_sizze_38379, skip_threads_38387)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_38387 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_35334, 32) == 0 || !slt32(local_tid_35334,
                                                          arg_31158))) {
            // read operands
            {
                x_35655 = *(volatile __local
                            float *) &mem_37683[(squot32(local_tid_35334, 32) -
                                                 1) * sizeof(float)];
            }
            // perform operation
            {
                float res_35657;
                
                if (slt32(gtid_35324, sizze_30757) && slt32(ltid_35325,
                                                            arg_31158)) {
                    res_35657 = x_35655 + x_35656;
                }
                x_35656 = res_35657;
            }
            // write final result
            {
                *(volatile __local float *) &mem_37683[local_tid_35334 *
                                                       sizeof(float)] = x_35656;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_35334, 32) == 0) {
            *(volatile __local float *) &mem_37683[local_tid_35334 *
                                                   sizeof(float)] = x_35656;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t x_35668;
    bool cond_35671;
    float x_36480;
    float x_36482;
    float res_35670;
    bool res_35672;
    bool res_35673;
    bool x_35674;
    float res_35675;
    bool res_35676;
    bool x_35677;
    float res_35678;
    
    if (slt32(gtid_35324, sizze_30757) && slt32(ltid_35325, arg_31158)) {
        x_35668 = ltid_35325;
        cond_35671 = slt32(ltid_35325, y_35630);
        x_36480 = *(__local float *) &mem_37683[ltid_35325 * 4];
        x_36482 = *(__global float *) &mem_37668[ltid_35325 * 4];
        res_35670 = x_36480 / y_35633;
        res_35672 = futrts_isnan32(res_35670);
        res_35673 = !res_35672;
        x_35674 = cond_35671 && res_35673;
        res_35675 = (float) fabs(res_35670);
        res_35676 = x_36482 < res_35675;
        x_35677 = x_35674 && res_35676;
        if (cond_35671) {
            res_35678 = res_35670;
        } else {
            res_35678 = 0.0F;
        }
    }
    
    __local char *mem_37685;
    __local char *mem_37688;
    __local char *mem_37691;
    bool acc0_35679;
    int32_t acc0_35680;
    float acc0_35681;
    
    mem_37685 = (__local char *) mem_37685_backing_1;
    mem_37688 = (__local char *) mem_37688_backing_2;
    mem_37691 = (__local char *) mem_37691_backing_3;
    for (int32_t comb_iter_38388 = 0; comb_iter_38388 < 1; comb_iter_38388++) {
        int32_t ctid_35331;
        int32_t flat_comb_id_38389 = comb_iter_38388 * arg_31158 +
                local_tid_35334;
        
        ctid_35331 = flat_comb_id_38389;
        if (slt32(ctid_35331, arg_31158) && 1) {
            *(__local bool *) &mem_37685[ctid_35331] = x_35677;
            *(__local int32_t *) &mem_37688[ctid_35331 * 4] = x_35668;
            *(__local float *) &mem_37691[ctid_35331 * 4] = res_35678;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_38390;
    int32_t skip_waves_38391;
    bool x_35682;
    int32_t x_35683;
    float x_35684;
    bool x_35685;
    int32_t x_35686;
    float x_35687;
    
    offset_38390 = 0;
    // participating threads read initial accumulator
    {
        if (slt32(local_tid_35334, arg_31158)) {
            x_35682 = *(__local bool *) &mem_37685[local_tid_35334 +
                                                   offset_38390];
            x_35683 = *(__local int32_t *) &mem_37688[(local_tid_35334 +
                                                       offset_38390) * 4];
            x_35684 = *(__local float *) &mem_37691[(local_tid_35334 +
                                                     offset_38390) * 4];
        }
    }
    offset_38390 = 1;
    while (slt32(offset_38390, wave_sizze_38379)) {
        if (slt32(local_tid_35334 + offset_38390, arg_31158) &&
            ((local_tid_35334 - squot32(local_tid_35334, wave_sizze_38379) *
              wave_sizze_38379) & (2 * offset_38390 - 1)) == 0) {
            // read array element
            {
                x_35685 = *(volatile __local
                            bool *) &mem_37685[local_tid_35334 + offset_38390];
                x_35686 = *(volatile __local
                            int32_t *) &mem_37688[(local_tid_35334 +
                                                   offset_38390) * 4];
                x_35687 = *(volatile __local
                            float *) &mem_37691[(local_tid_35334 +
                                                 offset_38390) * 4];
            }
            // apply reduction operation
            {
                bool res_35688;
                int32_t res_35689;
                float res_35694;
                
                if (slt32(gtid_35324, sizze_30757) && slt32(ltid_35325,
                                                            arg_31158)) {
                    if (x_35682) {
                        res_35688 = x_35682;
                        res_35689 = x_35683;
                    } else {
                        bool x_35690;
                        bool y_35691;
                        bool res_35692;
                        int32_t res_35693;
                        
                        x_35690 = !x_35685;
                        y_35691 = x_35682 && x_35690;
                        res_35692 = x_35685 || y_35691;
                        if (x_35685) {
                            res_35693 = x_35686;
                        } else {
                            res_35693 = x_35683;
                        }
                        res_35688 = res_35692;
                        res_35689 = res_35693;
                    }
                    res_35694 = x_35684 + x_35687;
                }
                x_35682 = res_35688;
                x_35683 = res_35689;
                x_35684 = res_35694;
            }
            // write result of operation
            {
                *(volatile __local bool *) &mem_37685[local_tid_35334] =
                    x_35682;
                *(volatile __local int32_t *) &mem_37688[local_tid_35334 * 4] =
                    x_35683;
                *(volatile __local float *) &mem_37691[local_tid_35334 * 4] =
                    x_35684;
            }
        }
        offset_38390 *= 2;
    }
    skip_waves_38391 = 1;
    while (slt32(skip_waves_38391, squot32(arg_31158 + wave_sizze_38379 - 1,
                                           wave_sizze_38379))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_38390 = skip_waves_38391 * wave_sizze_38379;
        if (slt32(local_tid_35334 + offset_38390, arg_31158) &&
            ((local_tid_35334 - squot32(local_tid_35334, wave_sizze_38379) *
              wave_sizze_38379) == 0 && (squot32(local_tid_35334,
                                                 wave_sizze_38379) & (2 *
                                                                      skip_waves_38391 -
                                                                      1)) ==
             0)) {
            // read array element
            {
                x_35685 = *(__local bool *) &mem_37685[local_tid_35334 +
                                                       offset_38390];
                x_35686 = *(__local int32_t *) &mem_37688[(local_tid_35334 +
                                                           offset_38390) * 4];
                x_35687 = *(__local float *) &mem_37691[(local_tid_35334 +
                                                         offset_38390) * 4];
            }
            // apply reduction operation
            {
                bool res_35688;
                int32_t res_35689;
                float res_35694;
                
                if (slt32(gtid_35324, sizze_30757) && slt32(ltid_35325,
                                                            arg_31158)) {
                    if (x_35682) {
                        res_35688 = x_35682;
                        res_35689 = x_35683;
                    } else {
                        bool x_35690;
                        bool y_35691;
                        bool res_35692;
                        int32_t res_35693;
                        
                        x_35690 = !x_35685;
                        y_35691 = x_35682 && x_35690;
                        res_35692 = x_35685 || y_35691;
                        if (x_35685) {
                            res_35693 = x_35686;
                        } else {
                            res_35693 = x_35683;
                        }
                        res_35688 = res_35692;
                        res_35689 = res_35693;
                    }
                    res_35694 = x_35684 + x_35687;
                }
                x_35682 = res_35688;
                x_35683 = res_35689;
                x_35684 = res_35694;
            }
            // write result of operation
            {
                *(__local bool *) &mem_37685[local_tid_35334] = x_35682;
                *(__local int32_t *) &mem_37688[local_tid_35334 * 4] = x_35683;
                *(__local float *) &mem_37691[local_tid_35334 * 4] = x_35684;
            }
        }
        skip_waves_38391 *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    acc0_35679 = *(__local bool *) &mem_37685[0];
    acc0_35680 = *(__local int32_t *) &mem_37688[0];
    acc0_35681 = *(__local float *) &mem_37691[0];
    
    int32_t res_35700;
    bool cond_35706;
    int32_t res_35707;
    bool cond_35713;
    bool res_35714;
    bool x_35715;
    bool y_35716;
    bool cond_35717;
    int32_t res_35718;
    
    if (slt32(gtid_35324, sizze_30757) && slt32(ltid_35325, arg_31158)) {
        if (acc0_35679) {
            res_35700 = acc0_35680;
        } else {
            res_35700 = -1;
        }
        cond_35706 = !acc0_35679;
        if (cond_35706) {
            res_35707 = -1;
        } else {
            bool cond_35708;
            int32_t res_35709;
            
            cond_35708 = slt32(res_35700, y_35630);
            if (cond_35708) {
                int32_t i_35710;
                int32_t x_35711;
                int32_t res_35712;
                
                i_35710 = x_35624 + res_35700;
                x_35711 = *(__global int32_t *) &res_mem_37598[(gtid_35324 *
                                                                sizze_30756 +
                                                                i_35710) * 4];
                res_35712 = x_35711 - n_30761;
                res_35709 = res_35712;
            } else {
                res_35709 = -1;
            }
            res_35707 = res_35709;
        }
        cond_35713 = sle32(x_35624, 5);
        res_35714 = sle32(y_35630, 5);
        x_35715 = !cond_35713;
        y_35716 = res_35714 && x_35715;
        cond_35717 = cond_35713 || y_35716;
        if (cond_35717) {
            res_35718 = -2;
        } else {
            res_35718 = res_35707;
        }
    }
    if (local_tid_35334 == 0) {
        *(__global int32_t *) &mem_37694[group_id_35335 * 4] = res_35718;
    }
    if (local_tid_35334 == 0) {
        *(__global float *) &mem_37697[group_id_35335 * 4] = acc0_35681;
    }
}
__kernel void map_transpose_f32(int32_t destoffset_1, int32_t srcoffset_3,
                                int32_t num_arrays_4, int32_t x_elems_5,
                                int32_t y_elems_6, int32_t in_elems_7,
                                int32_t out_elems_8, int32_t mulx_9,
                                int32_t muly_10, __global
                                unsigned char *destmem_0, __global
                                unsigned char *srcmem_2)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    ALIGNED_LOCAL_MEMORY(block_11_backing_0, 4224);
    
    __local char *block_11;
    
    block_11 = (__local char *) block_11_backing_0;
    
    int32_t get_global_id_0_37;
    
    get_global_id_0_37 = get_global_id(0);
    
    int32_t get_local_id_0_38;
    
    get_local_id_0_38 = get_local_id(0);
    
    int32_t get_local_id_1_39;
    
    get_local_id_1_39 = get_local_id(1);
    
    int32_t get_group_id_0_40;
    
    get_group_id_0_40 = get_group_id(0);
    
    int32_t get_group_id_1_41;
    
    get_group_id_1_41 = get_group_id(1);
    
    int32_t get_group_id_2_42;
    
    get_group_id_2_42 = get_group_id(2);
    
    int32_t our_array_offset_30 = get_group_id_2_42 * x_elems_5 * y_elems_6;
    int32_t odata_offset_33 = squot32(destoffset_1, 4) + our_array_offset_30;
    int32_t idata_offset_34 = squot32(srcoffset_3, 4) + our_array_offset_30;
    int32_t x_index_31 = get_global_id_0_37;
    int32_t y_index_32 = get_group_id_1_41 * 32 + get_local_id_1_39;
    
    if (slt32(x_index_31, x_elems_5)) {
        for (int32_t j_43 = 0; j_43 < 4; j_43++) {
            int32_t index_in_35 = (y_index_32 + j_43 * 8) * x_elems_5 +
                    x_index_31;
            
            if (slt32(y_index_32 + j_43 * 8, y_elems_6) && slt32(index_in_35,
                                                                 in_elems_7)) {
                *(__local float *) &block_11[((get_local_id_1_39 + j_43 * 8) *
                                              33 + get_local_id_0_38) *
                                             sizeof(float)] = *(__global
                                                                float *) &srcmem_2[(idata_offset_34 +
                                                                                    index_in_35) *
                                                                                   sizeof(float)];
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    x_index_31 = get_group_id_1_41 * 32 + get_local_id_0_38;
    y_index_32 = get_group_id_0_40 * 32 + get_local_id_1_39;
    if (slt32(x_index_31, y_elems_6)) {
        for (int32_t j_43 = 0; j_43 < 4; j_43++) {
            int32_t index_out_36 = (y_index_32 + j_43 * 8) * y_elems_6 +
                    x_index_31;
            
            if (slt32(y_index_32 + j_43 * 8, x_elems_5) && slt32(index_out_36,
                                                                 out_elems_8)) {
                *(__global float *) &destmem_0[(odata_offset_33 +
                                                index_out_36) * sizeof(float)] =
                    *(__local float *) &block_11[(get_local_id_0_38 * 33 +
                                                  get_local_id_1_39 + j_43 *
                                                  8) * sizeof(float)];
            }
        }
    }
}
__kernel void map_transpose_f32_low_height(int32_t destoffset_1,
                                           int32_t srcoffset_3,
                                           int32_t num_arrays_4,
                                           int32_t x_elems_5, int32_t y_elems_6,
                                           int32_t in_elems_7,
                                           int32_t out_elems_8, int32_t mulx_9,
                                           int32_t muly_10, __global
                                           unsigned char *destmem_0, __global
                                           unsigned char *srcmem_2)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    ALIGNED_LOCAL_MEMORY(block_11_backing_0, 1088);
    
    __local char *block_11;
    
    block_11 = (__local char *) block_11_backing_0;
    
    int32_t get_global_id_0_37;
    
    get_global_id_0_37 = get_global_id(0);
    
    int32_t get_local_id_0_38;
    
    get_local_id_0_38 = get_local_id(0);
    
    int32_t get_local_id_1_39;
    
    get_local_id_1_39 = get_local_id(1);
    
    int32_t get_group_id_0_40;
    
    get_group_id_0_40 = get_group_id(0);
    
    int32_t get_group_id_1_41;
    
    get_group_id_1_41 = get_group_id(1);
    
    int32_t get_group_id_2_42;
    
    get_group_id_2_42 = get_group_id(2);
    
    int32_t our_array_offset_30 = get_group_id_2_42 * x_elems_5 * y_elems_6;
    int32_t odata_offset_33 = squot32(destoffset_1, 4) + our_array_offset_30;
    int32_t idata_offset_34 = squot32(srcoffset_3, 4) + our_array_offset_30;
    int32_t x_index_31 = get_group_id_0_40 * 16 * mulx_9 + get_local_id_0_38 +
            srem32(get_local_id_1_39, mulx_9) * 16;
    int32_t y_index_32 = get_group_id_1_41 * 16 + squot32(get_local_id_1_39,
                                                          mulx_9);
    int32_t index_in_35 = y_index_32 * x_elems_5 + x_index_31;
    
    if (slt32(x_index_31, x_elems_5) && (slt32(y_index_32, y_elems_6) &&
                                         slt32(index_in_35, in_elems_7))) {
        *(__local float *) &block_11[(get_local_id_1_39 * 17 +
                                      get_local_id_0_38) * sizeof(float)] =
            *(__global float *) &srcmem_2[(idata_offset_34 + index_in_35) *
                                          sizeof(float)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    x_index_31 = get_group_id_1_41 * 16 + squot32(get_local_id_0_38, mulx_9);
    y_index_32 = get_group_id_0_40 * 16 * mulx_9 + get_local_id_1_39 +
        srem32(get_local_id_0_38, mulx_9) * 16;
    
    int32_t index_out_36 = y_index_32 * y_elems_6 + x_index_31;
    
    if (slt32(x_index_31, y_elems_6) && (slt32(y_index_32, x_elems_5) &&
                                         slt32(index_out_36, out_elems_8))) {
        *(__global float *) &destmem_0[(odata_offset_33 + index_out_36) *
                                       sizeof(float)] = *(__local
                                                          float *) &block_11[(get_local_id_0_38 *
                                                                              17 +
                                                                              get_local_id_1_39) *
                                                                             sizeof(float)];
    }
}
__kernel void map_transpose_f32_low_width(int32_t destoffset_1,
                                          int32_t srcoffset_3,
                                          int32_t num_arrays_4,
                                          int32_t x_elems_5, int32_t y_elems_6,
                                          int32_t in_elems_7,
                                          int32_t out_elems_8, int32_t mulx_9,
                                          int32_t muly_10, __global
                                          unsigned char *destmem_0, __global
                                          unsigned char *srcmem_2)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    ALIGNED_LOCAL_MEMORY(block_11_backing_0, 1088);
    
    __local char *block_11;
    
    block_11 = (__local char *) block_11_backing_0;
    
    int32_t get_global_id_0_37;
    
    get_global_id_0_37 = get_global_id(0);
    
    int32_t get_local_id_0_38;
    
    get_local_id_0_38 = get_local_id(0);
    
    int32_t get_local_id_1_39;
    
    get_local_id_1_39 = get_local_id(1);
    
    int32_t get_group_id_0_40;
    
    get_group_id_0_40 = get_group_id(0);
    
    int32_t get_group_id_1_41;
    
    get_group_id_1_41 = get_group_id(1);
    
    int32_t get_group_id_2_42;
    
    get_group_id_2_42 = get_group_id(2);
    
    int32_t our_array_offset_30 = get_group_id_2_42 * x_elems_5 * y_elems_6;
    int32_t odata_offset_33 = squot32(destoffset_1, 4) + our_array_offset_30;
    int32_t idata_offset_34 = squot32(srcoffset_3, 4) + our_array_offset_30;
    int32_t x_index_31 = get_group_id_0_40 * 16 + squot32(get_local_id_0_38,
                                                          muly_10);
    int32_t y_index_32 = get_group_id_1_41 * 16 * muly_10 + get_local_id_1_39 +
            srem32(get_local_id_0_38, muly_10) * 16;
    int32_t index_in_35 = y_index_32 * x_elems_5 + x_index_31;
    
    if (slt32(x_index_31, x_elems_5) && (slt32(y_index_32, y_elems_6) &&
                                         slt32(index_in_35, in_elems_7))) {
        *(__local float *) &block_11[(get_local_id_1_39 * 17 +
                                      get_local_id_0_38) * sizeof(float)] =
            *(__global float *) &srcmem_2[(idata_offset_34 + index_in_35) *
                                          sizeof(float)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    x_index_31 = get_group_id_1_41 * 16 * muly_10 + get_local_id_0_38 +
        srem32(get_local_id_1_39, muly_10) * 16;
    y_index_32 = get_group_id_0_40 * 16 + squot32(get_local_id_1_39, muly_10);
    
    int32_t index_out_36 = y_index_32 * y_elems_6 + x_index_31;
    
    if (slt32(x_index_31, y_elems_6) && (slt32(y_index_32, x_elems_5) &&
                                         slt32(index_out_36, out_elems_8))) {
        *(__global float *) &destmem_0[(odata_offset_33 + index_out_36) *
                                       sizeof(float)] = *(__local
                                                          float *) &block_11[(get_local_id_0_38 *
                                                                              17 +
                                                                              get_local_id_1_39) *
                                                                             sizeof(float)];
    }
}
__kernel void map_transpose_f32_small(int32_t destoffset_1, int32_t srcoffset_3,
                                      int32_t num_arrays_4, int32_t x_elems_5,
                                      int32_t y_elems_6, int32_t in_elems_7,
                                      int32_t out_elems_8, int32_t mulx_9,
                                      int32_t muly_10, __global
                                      unsigned char *destmem_0, __global
                                      unsigned char *srcmem_2)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    ALIGNED_LOCAL_MEMORY(block_11_backing_0, 1);
    
    __local char *block_11;
    
    block_11 = (__local char *) block_11_backing_0;
    
    int32_t get_global_id_0_37;
    
    get_global_id_0_37 = get_global_id(0);
    
    int32_t get_local_id_0_38;
    
    get_local_id_0_38 = get_local_id(0);
    
    int32_t get_local_id_1_39;
    
    get_local_id_1_39 = get_local_id(1);
    
    int32_t get_group_id_0_40;
    
    get_group_id_0_40 = get_group_id(0);
    
    int32_t get_group_id_1_41;
    
    get_group_id_1_41 = get_group_id(1);
    
    int32_t get_group_id_2_42;
    
    get_group_id_2_42 = get_group_id(2);
    
    int32_t our_array_offset_30 = squot32(get_global_id_0_37, y_elems_6 *
                                          x_elems_5) * (y_elems_6 * x_elems_5);
    int32_t x_index_31 = squot32(srem32(get_global_id_0_37, y_elems_6 *
                                        x_elems_5), y_elems_6);
    int32_t y_index_32 = srem32(get_global_id_0_37, y_elems_6);
    int32_t odata_offset_33 = squot32(destoffset_1, 4) + our_array_offset_30;
    int32_t idata_offset_34 = squot32(srcoffset_3, 4) + our_array_offset_30;
    int32_t index_in_35 = y_index_32 * x_elems_5 + x_index_31;
    int32_t index_out_36 = x_index_31 * y_elems_6 + y_index_32;
    
    if (slt32(get_global_id_0_37, in_elems_7)) {
        *(__global float *) &destmem_0[(odata_offset_33 + index_out_36) *
                                       sizeof(float)] = *(__global
                                                          float *) &srcmem_2[(idata_offset_34 +
                                                                              index_in_35) *
                                                                             sizeof(float)];
    }
}
__kernel void map_transpose_i32(int32_t destoffset_1, int32_t srcoffset_3,
                                int32_t num_arrays_4, int32_t x_elems_5,
                                int32_t y_elems_6, int32_t in_elems_7,
                                int32_t out_elems_8, int32_t mulx_9,
                                int32_t muly_10, __global
                                unsigned char *destmem_0, __global
                                unsigned char *srcmem_2)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    ALIGNED_LOCAL_MEMORY(block_11_backing_0, 4224);
    
    __local char *block_11;
    
    block_11 = (__local char *) block_11_backing_0;
    
    int32_t get_global_id_0_37;
    
    get_global_id_0_37 = get_global_id(0);
    
    int32_t get_local_id_0_38;
    
    get_local_id_0_38 = get_local_id(0);
    
    int32_t get_local_id_1_39;
    
    get_local_id_1_39 = get_local_id(1);
    
    int32_t get_group_id_0_40;
    
    get_group_id_0_40 = get_group_id(0);
    
    int32_t get_group_id_1_41;
    
    get_group_id_1_41 = get_group_id(1);
    
    int32_t get_group_id_2_42;
    
    get_group_id_2_42 = get_group_id(2);
    
    int32_t our_array_offset_30 = get_group_id_2_42 * x_elems_5 * y_elems_6;
    int32_t odata_offset_33 = squot32(destoffset_1, 4) + our_array_offset_30;
    int32_t idata_offset_34 = squot32(srcoffset_3, 4) + our_array_offset_30;
    int32_t x_index_31 = get_global_id_0_37;
    int32_t y_index_32 = get_group_id_1_41 * 32 + get_local_id_1_39;
    
    if (slt32(x_index_31, x_elems_5)) {
        for (int32_t j_43 = 0; j_43 < 4; j_43++) {
            int32_t index_in_35 = (y_index_32 + j_43 * 8) * x_elems_5 +
                    x_index_31;
            
            if (slt32(y_index_32 + j_43 * 8, y_elems_6) && slt32(index_in_35,
                                                                 in_elems_7)) {
                *(__local int32_t *) &block_11[((get_local_id_1_39 + j_43 * 8) *
                                                33 + get_local_id_0_38) *
                                               sizeof(int32_t)] = *(__global
                                                                    int32_t *) &srcmem_2[(idata_offset_34 +
                                                                                          index_in_35) *
                                                                                         sizeof(int32_t)];
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    x_index_31 = get_group_id_1_41 * 32 + get_local_id_0_38;
    y_index_32 = get_group_id_0_40 * 32 + get_local_id_1_39;
    if (slt32(x_index_31, y_elems_6)) {
        for (int32_t j_43 = 0; j_43 < 4; j_43++) {
            int32_t index_out_36 = (y_index_32 + j_43 * 8) * y_elems_6 +
                    x_index_31;
            
            if (slt32(y_index_32 + j_43 * 8, x_elems_5) && slt32(index_out_36,
                                                                 out_elems_8)) {
                *(__global int32_t *) &destmem_0[(odata_offset_33 +
                                                  index_out_36) *
                                                 sizeof(int32_t)] = *(__local
                                                                      int32_t *) &block_11[(get_local_id_0_38 *
                                                                                            33 +
                                                                                            get_local_id_1_39 +
                                                                                            j_43 *
                                                                                            8) *
                                                                                           sizeof(int32_t)];
            }
        }
    }
}
__kernel void map_transpose_i32_low_height(int32_t destoffset_1,
                                           int32_t srcoffset_3,
                                           int32_t num_arrays_4,
                                           int32_t x_elems_5, int32_t y_elems_6,
                                           int32_t in_elems_7,
                                           int32_t out_elems_8, int32_t mulx_9,
                                           int32_t muly_10, __global
                                           unsigned char *destmem_0, __global
                                           unsigned char *srcmem_2)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    ALIGNED_LOCAL_MEMORY(block_11_backing_0, 1088);
    
    __local char *block_11;
    
    block_11 = (__local char *) block_11_backing_0;
    
    int32_t get_global_id_0_37;
    
    get_global_id_0_37 = get_global_id(0);
    
    int32_t get_local_id_0_38;
    
    get_local_id_0_38 = get_local_id(0);
    
    int32_t get_local_id_1_39;
    
    get_local_id_1_39 = get_local_id(1);
    
    int32_t get_group_id_0_40;
    
    get_group_id_0_40 = get_group_id(0);
    
    int32_t get_group_id_1_41;
    
    get_group_id_1_41 = get_group_id(1);
    
    int32_t get_group_id_2_42;
    
    get_group_id_2_42 = get_group_id(2);
    
    int32_t our_array_offset_30 = get_group_id_2_42 * x_elems_5 * y_elems_6;
    int32_t odata_offset_33 = squot32(destoffset_1, 4) + our_array_offset_30;
    int32_t idata_offset_34 = squot32(srcoffset_3, 4) + our_array_offset_30;
    int32_t x_index_31 = get_group_id_0_40 * 16 * mulx_9 + get_local_id_0_38 +
            srem32(get_local_id_1_39, mulx_9) * 16;
    int32_t y_index_32 = get_group_id_1_41 * 16 + squot32(get_local_id_1_39,
                                                          mulx_9);
    int32_t index_in_35 = y_index_32 * x_elems_5 + x_index_31;
    
    if (slt32(x_index_31, x_elems_5) && (slt32(y_index_32, y_elems_6) &&
                                         slt32(index_in_35, in_elems_7))) {
        *(__local int32_t *) &block_11[(get_local_id_1_39 * 17 +
                                        get_local_id_0_38) * sizeof(int32_t)] =
            *(__global int32_t *) &srcmem_2[(idata_offset_34 + index_in_35) *
                                            sizeof(int32_t)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    x_index_31 = get_group_id_1_41 * 16 + squot32(get_local_id_0_38, mulx_9);
    y_index_32 = get_group_id_0_40 * 16 * mulx_9 + get_local_id_1_39 +
        srem32(get_local_id_0_38, mulx_9) * 16;
    
    int32_t index_out_36 = y_index_32 * y_elems_6 + x_index_31;
    
    if (slt32(x_index_31, y_elems_6) && (slt32(y_index_32, x_elems_5) &&
                                         slt32(index_out_36, out_elems_8))) {
        *(__global int32_t *) &destmem_0[(odata_offset_33 + index_out_36) *
                                         sizeof(int32_t)] = *(__local
                                                              int32_t *) &block_11[(get_local_id_0_38 *
                                                                                    17 +
                                                                                    get_local_id_1_39) *
                                                                                   sizeof(int32_t)];
    }
}
__kernel void map_transpose_i32_low_width(int32_t destoffset_1,
                                          int32_t srcoffset_3,
                                          int32_t num_arrays_4,
                                          int32_t x_elems_5, int32_t y_elems_6,
                                          int32_t in_elems_7,
                                          int32_t out_elems_8, int32_t mulx_9,
                                          int32_t muly_10, __global
                                          unsigned char *destmem_0, __global
                                          unsigned char *srcmem_2)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    ALIGNED_LOCAL_MEMORY(block_11_backing_0, 1088);
    
    __local char *block_11;
    
    block_11 = (__local char *) block_11_backing_0;
    
    int32_t get_global_id_0_37;
    
    get_global_id_0_37 = get_global_id(0);
    
    int32_t get_local_id_0_38;
    
    get_local_id_0_38 = get_local_id(0);
    
    int32_t get_local_id_1_39;
    
    get_local_id_1_39 = get_local_id(1);
    
    int32_t get_group_id_0_40;
    
    get_group_id_0_40 = get_group_id(0);
    
    int32_t get_group_id_1_41;
    
    get_group_id_1_41 = get_group_id(1);
    
    int32_t get_group_id_2_42;
    
    get_group_id_2_42 = get_group_id(2);
    
    int32_t our_array_offset_30 = get_group_id_2_42 * x_elems_5 * y_elems_6;
    int32_t odata_offset_33 = squot32(destoffset_1, 4) + our_array_offset_30;
    int32_t idata_offset_34 = squot32(srcoffset_3, 4) + our_array_offset_30;
    int32_t x_index_31 = get_group_id_0_40 * 16 + squot32(get_local_id_0_38,
                                                          muly_10);
    int32_t y_index_32 = get_group_id_1_41 * 16 * muly_10 + get_local_id_1_39 +
            srem32(get_local_id_0_38, muly_10) * 16;
    int32_t index_in_35 = y_index_32 * x_elems_5 + x_index_31;
    
    if (slt32(x_index_31, x_elems_5) && (slt32(y_index_32, y_elems_6) &&
                                         slt32(index_in_35, in_elems_7))) {
        *(__local int32_t *) &block_11[(get_local_id_1_39 * 17 +
                                        get_local_id_0_38) * sizeof(int32_t)] =
            *(__global int32_t *) &srcmem_2[(idata_offset_34 + index_in_35) *
                                            sizeof(int32_t)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    x_index_31 = get_group_id_1_41 * 16 * muly_10 + get_local_id_0_38 +
        srem32(get_local_id_1_39, muly_10) * 16;
    y_index_32 = get_group_id_0_40 * 16 + squot32(get_local_id_1_39, muly_10);
    
    int32_t index_out_36 = y_index_32 * y_elems_6 + x_index_31;
    
    if (slt32(x_index_31, y_elems_6) && (slt32(y_index_32, x_elems_5) &&
                                         slt32(index_out_36, out_elems_8))) {
        *(__global int32_t *) &destmem_0[(odata_offset_33 + index_out_36) *
                                         sizeof(int32_t)] = *(__local
                                                              int32_t *) &block_11[(get_local_id_0_38 *
                                                                                    17 +
                                                                                    get_local_id_1_39) *
                                                                                   sizeof(int32_t)];
    }
}
__kernel void map_transpose_i32_small(int32_t destoffset_1, int32_t srcoffset_3,
                                      int32_t num_arrays_4, int32_t x_elems_5,
                                      int32_t y_elems_6, int32_t in_elems_7,
                                      int32_t out_elems_8, int32_t mulx_9,
                                      int32_t muly_10, __global
                                      unsigned char *destmem_0, __global
                                      unsigned char *srcmem_2)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    ALIGNED_LOCAL_MEMORY(block_11_backing_0, 1);
    
    __local char *block_11;
    
    block_11 = (__local char *) block_11_backing_0;
    
    int32_t get_global_id_0_37;
    
    get_global_id_0_37 = get_global_id(0);
    
    int32_t get_local_id_0_38;
    
    get_local_id_0_38 = get_local_id(0);
    
    int32_t get_local_id_1_39;
    
    get_local_id_1_39 = get_local_id(1);
    
    int32_t get_group_id_0_40;
    
    get_group_id_0_40 = get_group_id(0);
    
    int32_t get_group_id_1_41;
    
    get_group_id_1_41 = get_group_id(1);
    
    int32_t get_group_id_2_42;
    
    get_group_id_2_42 = get_group_id(2);
    
    int32_t our_array_offset_30 = squot32(get_global_id_0_37, y_elems_6 *
                                          x_elems_5) * (y_elems_6 * x_elems_5);
    int32_t x_index_31 = squot32(srem32(get_global_id_0_37, y_elems_6 *
                                        x_elems_5), y_elems_6);
    int32_t y_index_32 = srem32(get_global_id_0_37, y_elems_6);
    int32_t odata_offset_33 = squot32(destoffset_1, 4) + our_array_offset_30;
    int32_t idata_offset_34 = squot32(srcoffset_3, 4) + our_array_offset_30;
    int32_t index_in_35 = y_index_32 * x_elems_5 + x_index_31;
    int32_t index_out_36 = x_index_31 * y_elems_6 + y_index_32;
    
    if (slt32(get_global_id_0_37, in_elems_7)) {
        *(__global int32_t *) &destmem_0[(odata_offset_33 + index_out_36) *
                                         sizeof(int32_t)] = *(__global
                                                              int32_t *) &srcmem_2[(idata_offset_34 +
                                                                                    index_in_35) *
                                                                                   sizeof(int32_t)];
    }
}
__kernel void replicate_38197(int32_t sizze_30756, int32_t sizze_30757, __global
                              unsigned char *mem_37588)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t replicate_gtid_38197;
    int32_t replicate_ltid_38198;
    int32_t replicate_gid_38199;
    
    replicate_gtid_38197 = get_global_id(0);
    replicate_ltid_38198 = get_local_id(0);
    replicate_gid_38199 = get_group_id(0);
    if (slt32(replicate_gtid_38197, sizze_30757 * sizze_30756)) {
        *(__global float *) &mem_37588[(squot32(replicate_gtid_38197,
                                                sizze_30756) * sizze_30756 +
                                        (replicate_gtid_38197 -
                                         squot32(replicate_gtid_38197,
                                                 sizze_30756) * sizze_30756)) *
                                       4] = NAN;
    }
}
__kernel void replicate_38202(int32_t sizze_30756, int32_t sizze_30757, __global
                              unsigned char *mem_37592)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t replicate_gtid_38202;
    int32_t replicate_ltid_38203;
    int32_t replicate_gid_38204;
    
    replicate_gtid_38202 = get_global_id(0);
    replicate_ltid_38203 = get_local_id(0);
    replicate_gid_38204 = get_group_id(0);
    if (slt32(replicate_gtid_38202, sizze_30757 * sizze_30756)) {
        *(__global int32_t *) &mem_37592[(squot32(replicate_gtid_38202,
                                                  sizze_30756) * sizze_30756 +
                                          (replicate_gtid_38202 -
                                           squot32(replicate_gtid_38202,
                                                   sizze_30756) *
                                           sizze_30756)) * 4] = 0;
    }
}
__kernel void scan_stage1_34630(int32_t sizze_30756, int32_t sizze_30757,
                                int32_t sizze_30758, int32_t num_groups_34647,
                                __global unsigned char *images_mem_37201,
                                __global unsigned char *res_mem_37506, __global
                                unsigned char *mem_37577, __global
                                unsigned char *mem_37581)
{
    const int32_t group_sizze_34637 = mainzigroup_sizze_34612;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    ALIGNED_LOCAL_MEMORY(scan_arr_mem_38156_backing_0, 4 *
                         mainzigroup_sizze_34612);
    
    int32_t global_tid_34630;
    int32_t local_tid_34631;
    int32_t group_sizze_38152;
    int32_t wave_sizze_38151;
    int32_t group_id_34632;
    
    global_tid_34630 = get_global_id(0);
    local_tid_34631 = get_local_id(0);
    group_sizze_38152 = get_local_size(0);
    wave_sizze_38151 = LOCKSTEP_WIDTH;
    group_id_34632 = get_group_id(0);
    
    int32_t gtid_34607;
    int32_t gtid_34629;
    __local char *scan_arr_mem_38156;
    
    scan_arr_mem_38156 = (__local char *) scan_arr_mem_38156_backing_0;
    
    int32_t x_34654;
    int32_t x_34655;
    
    x_34654 = 0;
    for (int32_t j_38158 = 0; j_38158 < squot32(sizze_30757 * sizze_30756 +
                                                group_sizze_34637 *
                                                num_groups_34647 - 1,
                                                group_sizze_34637 *
                                                num_groups_34647); j_38158++) {
        int32_t chunk_offset_38159 = group_sizze_34637 * j_38158 +
                group_id_34632 * (group_sizze_34637 * squot32(sizze_30757 *
                                                              sizze_30756 +
                                                              group_sizze_34637 *
                                                              num_groups_34647 -
                                                              1,
                                                              group_sizze_34637 *
                                                              num_groups_34647));
        int32_t flat_idx_38160 = chunk_offset_38159 + local_tid_34631;
        
        gtid_34607 = squot32(flat_idx_38160, sizze_30756);
        gtid_34629 = flat_idx_38160 - squot32(flat_idx_38160, sizze_30756) *
            sizze_30756;
        // threads in bounds read input; others get neutral element
        {
            if (slt32(gtid_34607, sizze_30757) && slt32(gtid_34629,
                                                        sizze_30756)) {
                float x_34659;
                float x_34660;
                bool res_34661;
                bool cond_34662;
                float res_34663;
                bool res_34665;
                bool res_34666;
                int32_t res_34667;
                
                x_34659 = *(__global float *) &images_mem_37201[(gtid_34607 *
                                                                 sizze_30758 +
                                                                 gtid_34629) *
                                                                4];
                x_34660 = *(__global float *) &res_mem_37506[(gtid_34607 *
                                                              sizze_30756 +
                                                              gtid_34629) * 4];
                res_34661 = futrts_isnan32(x_34659);
                cond_34662 = !res_34661;
                if (cond_34662) {
                    float res_34664 = x_34659 - x_34660;
                    
                    res_34663 = res_34664;
                } else {
                    res_34663 = NAN;
                }
                res_34665 = futrts_isnan32(res_34663);
                res_34666 = !res_34665;
                if (res_34666) {
                    res_34667 = 1;
                } else {
                    res_34667 = 0;
                }
                // write to-scan values to parameters
                {
                    x_34655 = res_34667;
                }
                // write mapped values results to global memory
                {
                    *(__global float *) &mem_37581[(gtid_34607 * sizze_30756 +
                                                    gtid_34629) * 4] =
                        res_34663;
                }
            } else {
                x_34655 = 0;
            }
        }
        // combine with carry and write to local memory
        {
            int32_t res_34656 = x_34654 + x_34655;
            
            *(__local int32_t *) &scan_arr_mem_38156[local_tid_34631 * 4] =
                res_34656;
        }
        
        int32_t x_38153;
        int32_t x_38154;
        int32_t x_38161;
        int32_t x_38162;
        int32_t skip_threads_38164;
        
        if (slt32(local_tid_34631, group_sizze_34637)) {
            x_38154 = *(volatile __local
                        int32_t *) &scan_arr_mem_38156[local_tid_34631 *
                                                       sizeof(int32_t)];
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_38164 = 1;
            while (slt32(skip_threads_38164, 32)) {
                if (sle32(skip_threads_38164, local_tid_34631 -
                          squot32(local_tid_34631, 32) * 32) &&
                    slt32(local_tid_34631, group_sizze_34637)) {
                    // read operands
                    {
                        x_38153 = *(volatile __local
                                    int32_t *) &scan_arr_mem_38156[(local_tid_34631 -
                                                                    skip_threads_38164) *
                                                                   sizeof(int32_t)];
                    }
                    // perform operation
                    {
                        if (!slt32(srem32(local_tid_34631 + chunk_offset_38159,
                                          sizze_30756), local_tid_34631 +
                                   chunk_offset_38159 - (local_tid_34631 -
                                                         skip_threads_38164 +
                                                         chunk_offset_38159))) {
                            int32_t res_38155 = x_38153 + x_38154;
                            
                            x_38154 = res_38155;
                        }
                    }
                }
                if (sle32(wave_sizze_38151, skip_threads_38164)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_38164, local_tid_34631 -
                          squot32(local_tid_34631, 32) * 32) &&
                    slt32(local_tid_34631, group_sizze_34637)) {
                    // write result
                    {
                        *(volatile __local
                          int32_t *) &scan_arr_mem_38156[local_tid_34631 *
                                                         sizeof(int32_t)] =
                            x_38154;
                    }
                }
                if (sle32(wave_sizze_38151, skip_threads_38164)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_38164 *= 2;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // last thread of block 'i' writes its result to offset 'i'
        {
            if ((local_tid_34631 - squot32(local_tid_34631, 32) * 32) == 31 &&
                slt32(local_tid_34631, group_sizze_34637)) {
                *(volatile __local
                  int32_t *) &scan_arr_mem_38156[squot32(local_tid_34631, 32) *
                                                 sizeof(int32_t)] = x_38154;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'
        {
            int32_t skip_threads_38165;
            
            if (squot32(local_tid_34631, 32) == 0 && slt32(local_tid_34631,
                                                           group_sizze_34637)) {
                x_38162 = *(volatile __local
                            int32_t *) &scan_arr_mem_38156[local_tid_34631 *
                                                           sizeof(int32_t)];
            }
            // in-block scan (hopefully no barriers needed)
            {
                skip_threads_38165 = 1;
                while (slt32(skip_threads_38165, 32)) {
                    if (sle32(skip_threads_38165, local_tid_34631 -
                              squot32(local_tid_34631, 32) * 32) &&
                        (squot32(local_tid_34631, 32) == 0 &&
                         slt32(local_tid_34631, group_sizze_34637))) {
                        // read operands
                        {
                            x_38161 = *(volatile __local
                                        int32_t *) &scan_arr_mem_38156[(local_tid_34631 -
                                                                        skip_threads_38165) *
                                                                       sizeof(int32_t)];
                        }
                        // perform operation
                        {
                            if (!slt32(srem32(local_tid_34631 * 32 + 32 - 1 +
                                              chunk_offset_38159, sizze_30756),
                                       local_tid_34631 * 32 + 32 - 1 +
                                       chunk_offset_38159 - ((local_tid_34631 -
                                                              skip_threads_38165) *
                                                             32 + 32 - 1 +
                                                             chunk_offset_38159))) {
                                int32_t res_38163 = x_38161 + x_38162;
                                
                                x_38162 = res_38163;
                            }
                        }
                    }
                    if (sle32(wave_sizze_38151, skip_threads_38165)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    if (sle32(skip_threads_38165, local_tid_34631 -
                              squot32(local_tid_34631, 32) * 32) &&
                        (squot32(local_tid_34631, 32) == 0 &&
                         slt32(local_tid_34631, group_sizze_34637))) {
                        // write result
                        {
                            *(volatile __local
                              int32_t *) &scan_arr_mem_38156[local_tid_34631 *
                                                             sizeof(int32_t)] =
                                x_38162;
                        }
                    }
                    if (sle32(wave_sizze_38151, skip_threads_38165)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    skip_threads_38165 *= 2;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // carry-in for every block except the first
        {
            if (!(squot32(local_tid_34631, 32) == 0 || !slt32(local_tid_34631,
                                                              group_sizze_34637))) {
                // read operands
                {
                    x_38153 = *(volatile __local
                                int32_t *) &scan_arr_mem_38156[(squot32(local_tid_34631,
                                                                        32) -
                                                                1) *
                                                               sizeof(int32_t)];
                }
                // perform operation
                {
                    if (!slt32(srem32(local_tid_34631 + chunk_offset_38159,
                                      sizze_30756), local_tid_34631 +
                               chunk_offset_38159 - (squot32(local_tid_34631,
                                                             32) * 32 - 1 +
                                                     chunk_offset_38159))) {
                        int32_t res_38155 = x_38153 + x_38154;
                        
                        x_38154 = res_38155;
                    }
                }
                // write final result
                {
                    *(volatile __local
                      int32_t *) &scan_arr_mem_38156[local_tid_34631 *
                                                     sizeof(int32_t)] = x_38154;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // restore correct values for first block
        {
            if (squot32(local_tid_34631, 32) == 0) {
                *(volatile __local
                  int32_t *) &scan_arr_mem_38156[local_tid_34631 *
                                                 sizeof(int32_t)] = x_38154;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // threads in bounds write partial scan result
        {
            if (slt32(gtid_34607, sizze_30757) && slt32(gtid_34629,
                                                        sizze_30756)) {
                *(__global int32_t *) &mem_37577[(gtid_34607 * sizze_30756 +
                                                  gtid_34629) * 4] = *(__local
                                                                       int32_t *) &scan_arr_mem_38156[local_tid_34631 *
                                                                                                      4];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // first thread reads last element as carry-in for next iteration
        {
            if (local_tid_34631 == 0) {
                if (slt32(srem32(chunk_offset_38159 + group_sizze_34637,
                                 sizze_30756), chunk_offset_38159 +
                          group_sizze_34637 - (chunk_offset_38159 +
                                               group_sizze_34637 - 1))) {
                    x_34654 = 0;
                } else {
                    x_34654 = *(__local
                                int32_t *) &scan_arr_mem_38156[(group_sizze_34637 -
                                                                1) * 4];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
__kernel void scan_stage1_35836(int32_t sizze_30756, int32_t sizze_30757,
                                int32_t arg_31158, int32_t num_groups_35910,
                                __global unsigned char *res_mem_37597, __global
                                unsigned char *res_mem_37646, __global
                                unsigned char *res_mem_37647, __global
                                unsigned char *res_mem_37665, __global
                                unsigned char *mem_37703, __global
                                unsigned char *mem_37707)
{
    const int32_t group_sizze_35900 = mainzigroup_sizze_35818;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    ALIGNED_LOCAL_MEMORY(scan_arr_mem_38401_backing_0, 4 *
                         mainzigroup_sizze_35818);
    
    int32_t global_tid_35836;
    int32_t local_tid_35837;
    int32_t group_sizze_38397;
    int32_t wave_sizze_38396;
    int32_t group_id_35838;
    
    global_tid_35836 = get_global_id(0);
    local_tid_35837 = get_local_id(0);
    group_sizze_38397 = get_local_size(0);
    wave_sizze_38396 = LOCKSTEP_WIDTH;
    group_id_35838 = get_group_id(0);
    
    int32_t gtid_35814;
    int32_t gtid_35835;
    __local char *scan_arr_mem_38401;
    
    scan_arr_mem_38401 = (__local char *) scan_arr_mem_38401_backing_0;
    
    float x_35916;
    float x_35917;
    
    x_35916 = 0.0F;
    for (int32_t j_38403 = 0; j_38403 < squot32(sizze_30757 * arg_31158 +
                                                group_sizze_35900 *
                                                num_groups_35910 - 1,
                                                group_sizze_35900 *
                                                num_groups_35910); j_38403++) {
        int32_t chunk_offset_38404 = group_sizze_35900 * j_38403 +
                group_id_35838 * (group_sizze_35900 * squot32(sizze_30757 *
                                                              arg_31158 +
                                                              group_sizze_35900 *
                                                              num_groups_35910 -
                                                              1,
                                                              group_sizze_35900 *
                                                              num_groups_35910));
        int32_t flat_idx_38405 = chunk_offset_38404 + local_tid_35837;
        
        gtid_35814 = squot32(flat_idx_38405, arg_31158);
        gtid_35835 = flat_idx_38405 - squot32(flat_idx_38405, arg_31158) *
            arg_31158;
        // threads in bounds read input; others get neutral element
        {
            if (slt32(gtid_35814, sizze_30757) && slt32(gtid_35835,
                                                        arg_31158)) {
                int32_t x_35919;
                int32_t x_35920;
                float x_35921;
                int32_t y_35923;
                bool cond_35926;
                float res_35927;
                
                x_35919 = *(__global int32_t *) &res_mem_37647[gtid_35814 * 4];
                x_35920 = *(__global int32_t *) &res_mem_37646[gtid_35814 * 4];
                x_35921 = *(__global float *) &res_mem_37665[gtid_35814 * 4];
                y_35923 = *(__global int32_t *) &mem_37703[gtid_35814 * 4];
                cond_35926 = sle32(y_35923, gtid_35835);
                if (cond_35926) {
                    res_35927 = 0.0F;
                } else {
                    bool cond_35928;
                    float res_35929;
                    
                    cond_35928 = gtid_35835 == 0;
                    if (cond_35928) {
                        res_35929 = x_35921;
                    } else {
                        int32_t x_35930;
                        int32_t i_35931;
                        float negate_arg_35932;
                        float x_35933;
                        int32_t i_35934;
                        float y_35935;
                        float res_35936;
                        
                        x_35930 = x_35919 - x_35920;
                        i_35931 = gtid_35835 + x_35930;
                        negate_arg_35932 = *(__global
                                             float *) &res_mem_37597[(gtid_35814 *
                                                                      sizze_30756 +
                                                                      i_35931) *
                                                                     4];
                        x_35933 = 0.0F - negate_arg_35932;
                        i_35934 = gtid_35835 + x_35919;
                        y_35935 = *(__global
                                    float *) &res_mem_37597[(gtid_35814 *
                                                             sizze_30756 +
                                                             i_35934) * 4];
                        res_35936 = x_35933 + y_35935;
                        res_35929 = res_35936;
                    }
                    res_35927 = res_35929;
                }
                // write to-scan values to parameters
                {
                    x_35917 = res_35927;
                }
                // write mapped values results to global memory
                { }
            } else {
                x_35917 = 0.0F;
            }
        }
        // combine with carry and write to local memory
        {
            float res_35918 = x_35916 + x_35917;
            
            *(__local float *) &scan_arr_mem_38401[local_tid_35837 * 4] =
                res_35918;
        }
        
        float x_38398;
        float x_38399;
        float x_38406;
        float x_38407;
        int32_t skip_threads_38409;
        
        if (slt32(local_tid_35837, group_sizze_35900)) {
            x_38399 = *(volatile __local
                        float *) &scan_arr_mem_38401[local_tid_35837 *
                                                     sizeof(float)];
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_38409 = 1;
            while (slt32(skip_threads_38409, 32)) {
                if (sle32(skip_threads_38409, local_tid_35837 -
                          squot32(local_tid_35837, 32) * 32) &&
                    slt32(local_tid_35837, group_sizze_35900)) {
                    // read operands
                    {
                        x_38398 = *(volatile __local
                                    float *) &scan_arr_mem_38401[(local_tid_35837 -
                                                                  skip_threads_38409) *
                                                                 sizeof(float)];
                    }
                    // perform operation
                    {
                        if (!slt32(srem32(local_tid_35837 + chunk_offset_38404,
                                          arg_31158), local_tid_35837 +
                                   chunk_offset_38404 - (local_tid_35837 -
                                                         skip_threads_38409 +
                                                         chunk_offset_38404))) {
                            float res_38400 = x_38398 + x_38399;
                            
                            x_38399 = res_38400;
                        }
                    }
                }
                if (sle32(wave_sizze_38396, skip_threads_38409)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_38409, local_tid_35837 -
                          squot32(local_tid_35837, 32) * 32) &&
                    slt32(local_tid_35837, group_sizze_35900)) {
                    // write result
                    {
                        *(volatile __local
                          float *) &scan_arr_mem_38401[local_tid_35837 *
                                                       sizeof(float)] = x_38399;
                    }
                }
                if (sle32(wave_sizze_38396, skip_threads_38409)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_38409 *= 2;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // last thread of block 'i' writes its result to offset 'i'
        {
            if ((local_tid_35837 - squot32(local_tid_35837, 32) * 32) == 31 &&
                slt32(local_tid_35837, group_sizze_35900)) {
                *(volatile __local
                  float *) &scan_arr_mem_38401[squot32(local_tid_35837, 32) *
                                               sizeof(float)] = x_38399;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'
        {
            int32_t skip_threads_38410;
            
            if (squot32(local_tid_35837, 32) == 0 && slt32(local_tid_35837,
                                                           group_sizze_35900)) {
                x_38407 = *(volatile __local
                            float *) &scan_arr_mem_38401[local_tid_35837 *
                                                         sizeof(float)];
            }
            // in-block scan (hopefully no barriers needed)
            {
                skip_threads_38410 = 1;
                while (slt32(skip_threads_38410, 32)) {
                    if (sle32(skip_threads_38410, local_tid_35837 -
                              squot32(local_tid_35837, 32) * 32) &&
                        (squot32(local_tid_35837, 32) == 0 &&
                         slt32(local_tid_35837, group_sizze_35900))) {
                        // read operands
                        {
                            x_38406 = *(volatile __local
                                        float *) &scan_arr_mem_38401[(local_tid_35837 -
                                                                      skip_threads_38410) *
                                                                     sizeof(float)];
                        }
                        // perform operation
                        {
                            if (!slt32(srem32(local_tid_35837 * 32 + 32 - 1 +
                                              chunk_offset_38404, arg_31158),
                                       local_tid_35837 * 32 + 32 - 1 +
                                       chunk_offset_38404 - ((local_tid_35837 -
                                                              skip_threads_38410) *
                                                             32 + 32 - 1 +
                                                             chunk_offset_38404))) {
                                float res_38408 = x_38406 + x_38407;
                                
                                x_38407 = res_38408;
                            }
                        }
                    }
                    if (sle32(wave_sizze_38396, skip_threads_38410)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    if (sle32(skip_threads_38410, local_tid_35837 -
                              squot32(local_tid_35837, 32) * 32) &&
                        (squot32(local_tid_35837, 32) == 0 &&
                         slt32(local_tid_35837, group_sizze_35900))) {
                        // write result
                        {
                            *(volatile __local
                              float *) &scan_arr_mem_38401[local_tid_35837 *
                                                           sizeof(float)] =
                                x_38407;
                        }
                    }
                    if (sle32(wave_sizze_38396, skip_threads_38410)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    skip_threads_38410 *= 2;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // carry-in for every block except the first
        {
            if (!(squot32(local_tid_35837, 32) == 0 || !slt32(local_tid_35837,
                                                              group_sizze_35900))) {
                // read operands
                {
                    x_38398 = *(volatile __local
                                float *) &scan_arr_mem_38401[(squot32(local_tid_35837,
                                                                      32) - 1) *
                                                             sizeof(float)];
                }
                // perform operation
                {
                    if (!slt32(srem32(local_tid_35837 + chunk_offset_38404,
                                      arg_31158), local_tid_35837 +
                               chunk_offset_38404 - (squot32(local_tid_35837,
                                                             32) * 32 - 1 +
                                                     chunk_offset_38404))) {
                        float res_38400 = x_38398 + x_38399;
                        
                        x_38399 = res_38400;
                    }
                }
                // write final result
                {
                    *(volatile __local
                      float *) &scan_arr_mem_38401[local_tid_35837 *
                                                   sizeof(float)] = x_38399;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // restore correct values for first block
        {
            if (squot32(local_tid_35837, 32) == 0) {
                *(volatile __local
                  float *) &scan_arr_mem_38401[local_tid_35837 *
                                               sizeof(float)] = x_38399;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // threads in bounds write partial scan result
        {
            if (slt32(gtid_35814, sizze_30757) && slt32(gtid_35835,
                                                        arg_31158)) {
                *(__global float *) &mem_37707[(gtid_35814 * arg_31158 +
                                                gtid_35835) * 4] = *(__local
                                                                     float *) &scan_arr_mem_38401[local_tid_35837 *
                                                                                                  4];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // first thread reads last element as carry-in for next iteration
        {
            if (local_tid_35837 == 0) {
                if (slt32(srem32(chunk_offset_38404 + group_sizze_35900,
                                 arg_31158), chunk_offset_38404 +
                          group_sizze_35900 - (chunk_offset_38404 +
                                               group_sizze_35900 - 1))) {
                    x_35916 = 0.0F;
                } else {
                    x_35916 = *(__local
                                float *) &scan_arr_mem_38401[(group_sizze_35900 -
                                                              1) * 4];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
__kernel void scan_stage2_38172(__local volatile
                                int64_t *scan_arr_mem_38177_backing_aligned_0,
                                int32_t sizze_30756, int32_t sizze_30757,
                                int32_t num_groups_34647, __global
                                unsigned char *mem_37577)
{
    const int32_t group_sizze_34637 = mainzigroup_sizze_34612;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_38177_backing_0 =
                          scan_arr_mem_38177_backing_aligned_0;
    int32_t global_tid_38172;
    int32_t local_tid_38173;
    int32_t group_sizze_38176;
    int32_t wave_sizze_38175;
    int32_t group_id_38174;
    
    global_tid_38172 = get_global_id(0);
    local_tid_38173 = get_local_id(0);
    group_sizze_38176 = get_local_size(0);
    wave_sizze_38175 = LOCKSTEP_WIDTH;
    group_id_38174 = get_group_id(0);
    
    __local char *scan_arr_mem_38177;
    
    scan_arr_mem_38177 = (__local char *) scan_arr_mem_38177_backing_0;
    
    int32_t flat_idx_38179 = (local_tid_38173 + 1) * (group_sizze_34637 *
                                                      squot32(sizze_30757 *
                                                              sizze_30756 +
                                                              group_sizze_34637 *
                                                              num_groups_34647 -
                                                              1,
                                                              group_sizze_34637 *
                                                              num_groups_34647)) -
            1;
    int32_t gtid_34607 = squot32(flat_idx_38179, sizze_30756);
    int32_t gtid_34629;
    
    gtid_34629 = flat_idx_38179 - squot32(flat_idx_38179, sizze_30756) *
        sizze_30756;
    // threads in bound read carries; others get neutral element
    {
        if (slt32(gtid_34607, sizze_30757) && slt32(gtid_34629, sizze_30756)) {
            *(__local int32_t *) &scan_arr_mem_38177[local_tid_38173 * 4] =
                *(__global int32_t *) &mem_37577[(gtid_34607 * sizze_30756 +
                                                  gtid_34629) * 4];
        } else {
            *(__local int32_t *) &scan_arr_mem_38177[local_tid_38173 * 4] = 0;
        }
    }
    
    int32_t x_38166;
    int32_t x_38167;
    int32_t x_38180;
    int32_t x_38181;
    int32_t skip_threads_38183;
    
    if (slt32(local_tid_38173, num_groups_34647)) {
        x_38167 = *(volatile __local
                    int32_t *) &scan_arr_mem_38177[local_tid_38173 *
                                                   sizeof(int32_t)];
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_38183 = 1;
        while (slt32(skip_threads_38183, 32)) {
            if (sle32(skip_threads_38183, local_tid_38173 -
                      squot32(local_tid_38173, 32) * 32) &&
                slt32(local_tid_38173, num_groups_34647)) {
                // read operands
                {
                    x_38166 = *(volatile __local
                                int32_t *) &scan_arr_mem_38177[(local_tid_38173 -
                                                                skip_threads_38183) *
                                                               sizeof(int32_t)];
                }
                // perform operation
                {
                    if (!slt32(srem32((local_tid_38173 + 1) *
                                      (group_sizze_34637 * squot32(sizze_30757 *
                                                                   sizze_30756 +
                                                                   group_sizze_34637 *
                                                                   num_groups_34647 -
                                                                   1,
                                                                   group_sizze_34637 *
                                                                   num_groups_34647)) -
                                      1, sizze_30756), (local_tid_38173 + 1) *
                               (group_sizze_34637 * squot32(sizze_30757 *
                                                            sizze_30756 +
                                                            group_sizze_34637 *
                                                            num_groups_34647 -
                                                            1,
                                                            group_sizze_34637 *
                                                            num_groups_34647)) -
                               1 - ((local_tid_38173 - skip_threads_38183 + 1) *
                                    (group_sizze_34637 * squot32(sizze_30757 *
                                                                 sizze_30756 +
                                                                 group_sizze_34637 *
                                                                 num_groups_34647 -
                                                                 1,
                                                                 group_sizze_34637 *
                                                                 num_groups_34647)) -
                                    1))) {
                        int32_t res_38168 = x_38166 + x_38167;
                        
                        x_38167 = res_38168;
                    }
                }
            }
            if (sle32(wave_sizze_38175, skip_threads_38183)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_38183, local_tid_38173 -
                      squot32(local_tid_38173, 32) * 32) &&
                slt32(local_tid_38173, num_groups_34647)) {
                // write result
                {
                    *(volatile __local
                      int32_t *) &scan_arr_mem_38177[local_tid_38173 *
                                                     sizeof(int32_t)] = x_38167;
                }
            }
            if (sle32(wave_sizze_38175, skip_threads_38183)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_38183 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_38173 - squot32(local_tid_38173, 32) * 32) == 31 &&
            slt32(local_tid_38173, num_groups_34647)) {
            *(volatile __local
              int32_t *) &scan_arr_mem_38177[squot32(local_tid_38173, 32) *
                                             sizeof(int32_t)] = x_38167;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'
    {
        int32_t skip_threads_38184;
        
        if (squot32(local_tid_38173, 32) == 0 && slt32(local_tid_38173,
                                                       num_groups_34647)) {
            x_38181 = *(volatile __local
                        int32_t *) &scan_arr_mem_38177[local_tid_38173 *
                                                       sizeof(int32_t)];
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_38184 = 1;
            while (slt32(skip_threads_38184, 32)) {
                if (sle32(skip_threads_38184, local_tid_38173 -
                          squot32(local_tid_38173, 32) * 32) &&
                    (squot32(local_tid_38173, 32) == 0 && slt32(local_tid_38173,
                                                                num_groups_34647))) {
                    // read operands
                    {
                        x_38180 = *(volatile __local
                                    int32_t *) &scan_arr_mem_38177[(local_tid_38173 -
                                                                    skip_threads_38184) *
                                                                   sizeof(int32_t)];
                    }
                    // perform operation
                    {
                        if (!slt32(srem32((local_tid_38173 * 32 + 32 - 1 + 1) *
                                          (group_sizze_34637 *
                                           squot32(sizze_30757 * sizze_30756 +
                                                   group_sizze_34637 *
                                                   num_groups_34647 - 1,
                                                   group_sizze_34637 *
                                                   num_groups_34647)) - 1,
                                          sizze_30756), (local_tid_38173 * 32 +
                                                         32 - 1 + 1) *
                                   (group_sizze_34637 * squot32(sizze_30757 *
                                                                sizze_30756 +
                                                                group_sizze_34637 *
                                                                num_groups_34647 -
                                                                1,
                                                                group_sizze_34637 *
                                                                num_groups_34647)) -
                                   1 - (((local_tid_38173 -
                                          skip_threads_38184) * 32 + 32 - 1 +
                                         1) * (group_sizze_34637 *
                                               squot32(sizze_30757 *
                                                       sizze_30756 +
                                                       group_sizze_34637 *
                                                       num_groups_34647 - 1,
                                                       group_sizze_34637 *
                                                       num_groups_34647)) -
                                        1))) {
                            int32_t res_38182 = x_38180 + x_38181;
                            
                            x_38181 = res_38182;
                        }
                    }
                }
                if (sle32(wave_sizze_38175, skip_threads_38184)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_38184, local_tid_38173 -
                          squot32(local_tid_38173, 32) * 32) &&
                    (squot32(local_tid_38173, 32) == 0 && slt32(local_tid_38173,
                                                                num_groups_34647))) {
                    // write result
                    {
                        *(volatile __local
                          int32_t *) &scan_arr_mem_38177[local_tid_38173 *
                                                         sizeof(int32_t)] =
                            x_38181;
                    }
                }
                if (sle32(wave_sizze_38175, skip_threads_38184)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_38184 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_38173, 32) == 0 || !slt32(local_tid_38173,
                                                          num_groups_34647))) {
            // read operands
            {
                x_38166 = *(volatile __local
                            int32_t *) &scan_arr_mem_38177[(squot32(local_tid_38173,
                                                                    32) - 1) *
                                                           sizeof(int32_t)];
            }
            // perform operation
            {
                if (!slt32(srem32((local_tid_38173 + 1) * (group_sizze_34637 *
                                                           squot32(sizze_30757 *
                                                                   sizze_30756 +
                                                                   group_sizze_34637 *
                                                                   num_groups_34647 -
                                                                   1,
                                                                   group_sizze_34637 *
                                                                   num_groups_34647)) -
                                  1, sizze_30756), (local_tid_38173 + 1) *
                           (group_sizze_34637 * squot32(sizze_30757 *
                                                        sizze_30756 +
                                                        group_sizze_34637 *
                                                        num_groups_34647 - 1,
                                                        group_sizze_34637 *
                                                        num_groups_34647)) - 1 -
                           ((squot32(local_tid_38173, 32) * 32 - 1 + 1) *
                            (group_sizze_34637 * squot32(sizze_30757 *
                                                         sizze_30756 +
                                                         group_sizze_34637 *
                                                         num_groups_34647 - 1,
                                                         group_sizze_34637 *
                                                         num_groups_34647)) -
                            1))) {
                    int32_t res_38168 = x_38166 + x_38167;
                    
                    x_38167 = res_38168;
                }
            }
            // write final result
            {
                *(volatile __local
                  int32_t *) &scan_arr_mem_38177[local_tid_38173 *
                                                 sizeof(int32_t)] = x_38167;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_38173, 32) == 0) {
            *(volatile __local int32_t *) &scan_arr_mem_38177[local_tid_38173 *
                                                              sizeof(int32_t)] =
                x_38167;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // threads in bounds write scanned carries
    {
        if (slt32(gtid_34607, sizze_30757) && slt32(gtid_34629, sizze_30756)) {
            *(__global int32_t *) &mem_37577[(gtid_34607 * sizze_30756 +
                                              gtid_34629) * 4] = *(__local
                                                                   int32_t *) &scan_arr_mem_38177[local_tid_38173 *
                                                                                                  4];
        }
    }
}
__kernel void scan_stage2_38417(__local volatile
                                int64_t *scan_arr_mem_38422_backing_aligned_0,
                                int32_t sizze_30757, int32_t arg_31158,
                                int32_t num_groups_35910, __global
                                unsigned char *mem_37707)
{
    const int32_t group_sizze_35900 = mainzigroup_sizze_35818;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_38422_backing_0 =
                          scan_arr_mem_38422_backing_aligned_0;
    int32_t global_tid_38417;
    int32_t local_tid_38418;
    int32_t group_sizze_38421;
    int32_t wave_sizze_38420;
    int32_t group_id_38419;
    
    global_tid_38417 = get_global_id(0);
    local_tid_38418 = get_local_id(0);
    group_sizze_38421 = get_local_size(0);
    wave_sizze_38420 = LOCKSTEP_WIDTH;
    group_id_38419 = get_group_id(0);
    
    __local char *scan_arr_mem_38422;
    
    scan_arr_mem_38422 = (__local char *) scan_arr_mem_38422_backing_0;
    
    int32_t flat_idx_38424 = (local_tid_38418 + 1) * (group_sizze_35900 *
                                                      squot32(sizze_30757 *
                                                              arg_31158 +
                                                              group_sizze_35900 *
                                                              num_groups_35910 -
                                                              1,
                                                              group_sizze_35900 *
                                                              num_groups_35910)) -
            1;
    int32_t gtid_35814 = squot32(flat_idx_38424, arg_31158);
    int32_t gtid_35835;
    
    gtid_35835 = flat_idx_38424 - squot32(flat_idx_38424, arg_31158) *
        arg_31158;
    // threads in bound read carries; others get neutral element
    {
        if (slt32(gtid_35814, sizze_30757) && slt32(gtid_35835, arg_31158)) {
            *(__local float *) &scan_arr_mem_38422[local_tid_38418 * 4] =
                *(__global float *) &mem_37707[(gtid_35814 * arg_31158 +
                                                gtid_35835) * 4];
        } else {
            *(__local float *) &scan_arr_mem_38422[local_tid_38418 * 4] = 0.0F;
        }
    }
    
    float x_38411;
    float x_38412;
    float x_38425;
    float x_38426;
    int32_t skip_threads_38428;
    
    if (slt32(local_tid_38418, num_groups_35910)) {
        x_38412 = *(volatile __local
                    float *) &scan_arr_mem_38422[local_tid_38418 *
                                                 sizeof(float)];
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_38428 = 1;
        while (slt32(skip_threads_38428, 32)) {
            if (sle32(skip_threads_38428, local_tid_38418 -
                      squot32(local_tid_38418, 32) * 32) &&
                slt32(local_tid_38418, num_groups_35910)) {
                // read operands
                {
                    x_38411 = *(volatile __local
                                float *) &scan_arr_mem_38422[(local_tid_38418 -
                                                              skip_threads_38428) *
                                                             sizeof(float)];
                }
                // perform operation
                {
                    if (!slt32(srem32((local_tid_38418 + 1) *
                                      (group_sizze_35900 * squot32(sizze_30757 *
                                                                   arg_31158 +
                                                                   group_sizze_35900 *
                                                                   num_groups_35910 -
                                                                   1,
                                                                   group_sizze_35900 *
                                                                   num_groups_35910)) -
                                      1, arg_31158), (local_tid_38418 + 1) *
                               (group_sizze_35900 * squot32(sizze_30757 *
                                                            arg_31158 +
                                                            group_sizze_35900 *
                                                            num_groups_35910 -
                                                            1,
                                                            group_sizze_35900 *
                                                            num_groups_35910)) -
                               1 - ((local_tid_38418 - skip_threads_38428 + 1) *
                                    (group_sizze_35900 * squot32(sizze_30757 *
                                                                 arg_31158 +
                                                                 group_sizze_35900 *
                                                                 num_groups_35910 -
                                                                 1,
                                                                 group_sizze_35900 *
                                                                 num_groups_35910)) -
                                    1))) {
                        float res_38413 = x_38411 + x_38412;
                        
                        x_38412 = res_38413;
                    }
                }
            }
            if (sle32(wave_sizze_38420, skip_threads_38428)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_38428, local_tid_38418 -
                      squot32(local_tid_38418, 32) * 32) &&
                slt32(local_tid_38418, num_groups_35910)) {
                // write result
                {
                    *(volatile __local
                      float *) &scan_arr_mem_38422[local_tid_38418 *
                                                   sizeof(float)] = x_38412;
                }
            }
            if (sle32(wave_sizze_38420, skip_threads_38428)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_38428 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_38418 - squot32(local_tid_38418, 32) * 32) == 31 &&
            slt32(local_tid_38418, num_groups_35910)) {
            *(volatile __local
              float *) &scan_arr_mem_38422[squot32(local_tid_38418, 32) *
                                           sizeof(float)] = x_38412;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'
    {
        int32_t skip_threads_38429;
        
        if (squot32(local_tid_38418, 32) == 0 && slt32(local_tid_38418,
                                                       num_groups_35910)) {
            x_38426 = *(volatile __local
                        float *) &scan_arr_mem_38422[local_tid_38418 *
                                                     sizeof(float)];
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_38429 = 1;
            while (slt32(skip_threads_38429, 32)) {
                if (sle32(skip_threads_38429, local_tid_38418 -
                          squot32(local_tid_38418, 32) * 32) &&
                    (squot32(local_tid_38418, 32) == 0 && slt32(local_tid_38418,
                                                                num_groups_35910))) {
                    // read operands
                    {
                        x_38425 = *(volatile __local
                                    float *) &scan_arr_mem_38422[(local_tid_38418 -
                                                                  skip_threads_38429) *
                                                                 sizeof(float)];
                    }
                    // perform operation
                    {
                        if (!slt32(srem32((local_tid_38418 * 32 + 32 - 1 + 1) *
                                          (group_sizze_35900 *
                                           squot32(sizze_30757 * arg_31158 +
                                                   group_sizze_35900 *
                                                   num_groups_35910 - 1,
                                                   group_sizze_35900 *
                                                   num_groups_35910)) - 1,
                                          arg_31158), (local_tid_38418 * 32 +
                                                       32 - 1 + 1) *
                                   (group_sizze_35900 * squot32(sizze_30757 *
                                                                arg_31158 +
                                                                group_sizze_35900 *
                                                                num_groups_35910 -
                                                                1,
                                                                group_sizze_35900 *
                                                                num_groups_35910)) -
                                   1 - (((local_tid_38418 -
                                          skip_threads_38429) * 32 + 32 - 1 +
                                         1) * (group_sizze_35900 *
                                               squot32(sizze_30757 * arg_31158 +
                                                       group_sizze_35900 *
                                                       num_groups_35910 - 1,
                                                       group_sizze_35900 *
                                                       num_groups_35910)) -
                                        1))) {
                            float res_38427 = x_38425 + x_38426;
                            
                            x_38426 = res_38427;
                        }
                    }
                }
                if (sle32(wave_sizze_38420, skip_threads_38429)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_38429, local_tid_38418 -
                          squot32(local_tid_38418, 32) * 32) &&
                    (squot32(local_tid_38418, 32) == 0 && slt32(local_tid_38418,
                                                                num_groups_35910))) {
                    // write result
                    {
                        *(volatile __local
                          float *) &scan_arr_mem_38422[local_tid_38418 *
                                                       sizeof(float)] = x_38426;
                    }
                }
                if (sle32(wave_sizze_38420, skip_threads_38429)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_38429 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_38418, 32) == 0 || !slt32(local_tid_38418,
                                                          num_groups_35910))) {
            // read operands
            {
                x_38411 = *(volatile __local
                            float *) &scan_arr_mem_38422[(squot32(local_tid_38418,
                                                                  32) - 1) *
                                                         sizeof(float)];
            }
            // perform operation
            {
                if (!slt32(srem32((local_tid_38418 + 1) * (group_sizze_35900 *
                                                           squot32(sizze_30757 *
                                                                   arg_31158 +
                                                                   group_sizze_35900 *
                                                                   num_groups_35910 -
                                                                   1,
                                                                   group_sizze_35900 *
                                                                   num_groups_35910)) -
                                  1, arg_31158), (local_tid_38418 + 1) *
                           (group_sizze_35900 * squot32(sizze_30757 *
                                                        arg_31158 +
                                                        group_sizze_35900 *
                                                        num_groups_35910 - 1,
                                                        group_sizze_35900 *
                                                        num_groups_35910)) - 1 -
                           ((squot32(local_tid_38418, 32) * 32 - 1 + 1) *
                            (group_sizze_35900 * squot32(sizze_30757 *
                                                         arg_31158 +
                                                         group_sizze_35900 *
                                                         num_groups_35910 - 1,
                                                         group_sizze_35900 *
                                                         num_groups_35910)) -
                            1))) {
                    float res_38413 = x_38411 + x_38412;
                    
                    x_38412 = res_38413;
                }
            }
            // write final result
            {
                *(volatile __local
                  float *) &scan_arr_mem_38422[local_tid_38418 *
                                               sizeof(float)] = x_38412;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_38418, 32) == 0) {
            *(volatile __local float *) &scan_arr_mem_38422[local_tid_38418 *
                                                            sizeof(float)] =
                x_38412;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // threads in bounds write scanned carries
    {
        if (slt32(gtid_35814, sizze_30757) && slt32(gtid_35835, arg_31158)) {
            *(__global float *) &mem_37707[(gtid_35814 * arg_31158 +
                                            gtid_35835) * 4] = *(__local
                                                                 float *) &scan_arr_mem_38422[local_tid_38418 *
                                                                                              4];
        }
    }
}
__kernel void scan_stage3_38185(int32_t sizze_30756, int32_t sizze_30757,
                                int32_t num_groups_34647, __global
                                unsigned char *mem_37577)
{
    const int32_t group_sizze_34637 = mainzigroup_sizze_34612;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t scan_gtid_38185;
    int32_t scan_ltid_38186;
    int32_t scan_gid_38187;
    
    scan_gtid_38185 = get_global_id(0);
    scan_ltid_38186 = get_local_id(0);
    scan_gid_38187 = get_group_id(0);
    
    int32_t gtid_34607 = squot32(scan_gtid_38185, sizze_30756);
    int32_t gtid_34629;
    
    gtid_34629 = scan_gtid_38185 - squot32(scan_gtid_38185, sizze_30756) *
        sizze_30756;
    
    int32_t orig_group_38190 = squot32(scan_gtid_38185, group_sizze_34637 *
                                       squot32(sizze_30757 * sizze_30756 +
                                               group_sizze_34637 *
                                               num_groups_34647 - 1,
                                               group_sizze_34637 *
                                               num_groups_34647));
    int32_t carry_in_flat_idx_38191 = orig_group_38190 * (group_sizze_34637 *
                                                          squot32(sizze_30757 *
                                                                  sizze_30756 +
                                                                  group_sizze_34637 *
                                                                  num_groups_34647 -
                                                                  1,
                                                                  group_sizze_34637 *
                                                                  num_groups_34647)) -
            1;
    
    if (slt32(scan_gtid_38185, sizze_30757 * sizze_30756)) {
        if (!(orig_group_38190 == 0 || (scan_gtid_38185 == (orig_group_38190 +
                                                            1) *
                                        (group_sizze_34637 *
                                         squot32(sizze_30757 * sizze_30756 +
                                                 group_sizze_34637 *
                                                 num_groups_34647 - 1,
                                                 group_sizze_34637 *
                                                 num_groups_34647)) - 1 ||
                                        slt32(srem32(scan_gtid_38185,
                                                     sizze_30756),
                                              scan_gtid_38185 -
                                              carry_in_flat_idx_38191)))) {
            int32_t x_38169;
            int32_t x_38170;
            
            x_38169 = *(__global
                        int32_t *) &mem_37577[(squot32(carry_in_flat_idx_38191,
                                                       sizze_30756) *
                                               sizze_30756 +
                                               (carry_in_flat_idx_38191 -
                                                squot32(carry_in_flat_idx_38191,
                                                        sizze_30756) *
                                                sizze_30756)) * 4];
            x_38170 = *(__global int32_t *) &mem_37577[(gtid_34607 *
                                                        sizze_30756 +
                                                        gtid_34629) * 4];
            
            int32_t res_38171;
            
            if (slt32(scan_gtid_38185, sizze_30757 * sizze_30756)) {
                res_38171 = x_38169 + x_38170;
            }
            x_38169 = res_38171;
            *(__global int32_t *) &mem_37577[(gtid_34607 * sizze_30756 +
                                              gtid_34629) * 4] = x_38169;
        }
    }
}
__kernel void scan_stage3_38430(int32_t sizze_30757, int32_t arg_31158,
                                int32_t num_groups_35910, __global
                                unsigned char *mem_37707)
{
    const int32_t group_sizze_35900 = mainzigroup_sizze_35818;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t scan_gtid_38430;
    int32_t scan_ltid_38431;
    int32_t scan_gid_38432;
    
    scan_gtid_38430 = get_global_id(0);
    scan_ltid_38431 = get_local_id(0);
    scan_gid_38432 = get_group_id(0);
    
    int32_t gtid_35814 = squot32(scan_gtid_38430, arg_31158);
    int32_t gtid_35835;
    
    gtid_35835 = scan_gtid_38430 - squot32(scan_gtid_38430, arg_31158) *
        arg_31158;
    
    int32_t orig_group_38435 = squot32(scan_gtid_38430, group_sizze_35900 *
                                       squot32(sizze_30757 * arg_31158 +
                                               group_sizze_35900 *
                                               num_groups_35910 - 1,
                                               group_sizze_35900 *
                                               num_groups_35910));
    int32_t carry_in_flat_idx_38436 = orig_group_38435 * (group_sizze_35900 *
                                                          squot32(sizze_30757 *
                                                                  arg_31158 +
                                                                  group_sizze_35900 *
                                                                  num_groups_35910 -
                                                                  1,
                                                                  group_sizze_35900 *
                                                                  num_groups_35910)) -
            1;
    
    if (slt32(scan_gtid_38430, sizze_30757 * arg_31158)) {
        if (!(orig_group_38435 == 0 || (scan_gtid_38430 == (orig_group_38435 +
                                                            1) *
                                        (group_sizze_35900 *
                                         squot32(sizze_30757 * arg_31158 +
                                                 group_sizze_35900 *
                                                 num_groups_35910 - 1,
                                                 group_sizze_35900 *
                                                 num_groups_35910)) - 1 ||
                                        slt32(srem32(scan_gtid_38430,
                                                     arg_31158),
                                              scan_gtid_38430 -
                                              carry_in_flat_idx_38436)))) {
            float x_38414;
            float x_38415;
            
            x_38414 = *(__global
                        float *) &mem_37707[(squot32(carry_in_flat_idx_38436,
                                                     arg_31158) * arg_31158 +
                                             (carry_in_flat_idx_38436 -
                                              squot32(carry_in_flat_idx_38436,
                                                      arg_31158) * arg_31158)) *
                                            4];
            x_38415 = *(__global float *) &mem_37707[(gtid_35814 * arg_31158 +
                                                      gtid_35835) * 4];
            
            float res_38416;
            
            if (slt32(scan_gtid_38430, sizze_30757 * arg_31158)) {
                res_38416 = x_38414 + x_38415;
            }
            x_38414 = res_38416;
            *(__global float *) &mem_37707[(gtid_35814 * arg_31158 +
                                            gtid_35835) * 4] = x_38414;
        }
    }
}
__kernel void segred_large_32307(int32_t sizze_30756, int32_t sizze_30757,
                                 int32_t sizze_30758, int32_t n_30761,
                                 int32_t res_30780, int32_t num_groups_32604,
                                 __global unsigned char *images_mem_37201,
                                 __global unsigned char *arg_mem_37210, __global
                                 unsigned char *mem_37302, __global
                                 unsigned char *mem_37307,
                                 int32_t thread_per_segment_37891, __global
                                 unsigned char *group_res_arr_mem_37892,
                                 __global unsigned char *counter_mem_37894)
{
    const int32_t group_sizze_32594 = mainzigroup_sizze_32289;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    ALIGNED_LOCAL_MEMORY(red_arr_mem_37896_backing_0, 4 *
                         mainzigroup_sizze_32289);
    ALIGNED_LOCAL_MEMORY(sync_arr_mem_37898_backing_1, 1);
    
    int32_t global_tid_32307;
    int32_t local_tid_32308;
    int32_t group_sizze_37888;
    int32_t wave_sizze_37887;
    int32_t group_id_32309;
    
    global_tid_32307 = get_global_id(0);
    local_tid_32308 = get_local_id(0);
    group_sizze_37888 = get_local_size(0);
    wave_sizze_37887 = LOCKSTEP_WIDTH;
    group_id_32309 = get_group_id(0);
    
    int32_t gtid_32276;
    int32_t gtid_32277;
    int32_t gtid_32278;
    int32_t gtid_32306;
    __local char *red_arr_mem_37896;
    
    red_arr_mem_37896 = (__local char *) red_arr_mem_37896_backing_0;
    
    __local char *sync_arr_mem_37898;
    
    sync_arr_mem_37898 = (__local char *) sync_arr_mem_37898_backing_1;
    gtid_32276 = squot32(squot32(group_id_32309, squot32(num_groups_32604 +
                                                         smax32(1, sizze_30757 *
                                                                res_30780 *
                                                                res_30780) - 1,
                                                         smax32(1, sizze_30757 *
                                                                res_30780 *
                                                                res_30780))),
                         res_30780 * res_30780);
    gtid_32277 = squot32(squot32(group_id_32309, squot32(num_groups_32604 +
                                                         smax32(1, sizze_30757 *
                                                                res_30780 *
                                                                res_30780) - 1,
                                                         smax32(1, sizze_30757 *
                                                                res_30780 *
                                                                res_30780))) -
                         squot32(squot32(group_id_32309,
                                         squot32(num_groups_32604 + smax32(1,
                                                                           sizze_30757 *
                                                                           res_30780 *
                                                                           res_30780) -
                                                 1, smax32(1, sizze_30757 *
                                                           res_30780 *
                                                           res_30780))),
                                 res_30780 * res_30780) * (res_30780 *
                                                           res_30780),
                         res_30780);
    gtid_32278 = squot32(group_id_32309, squot32(num_groups_32604 + smax32(1,
                                                                           sizze_30757 *
                                                                           res_30780 *
                                                                           res_30780) -
                                                 1, smax32(1, sizze_30757 *
                                                           res_30780 *
                                                           res_30780))) -
        squot32(squot32(group_id_32309, squot32(num_groups_32604 + smax32(1,
                                                                          sizze_30757 *
                                                                          res_30780 *
                                                                          res_30780) -
                                                1, smax32(1, sizze_30757 *
                                                          res_30780 *
                                                          res_30780))),
                res_30780 * res_30780) * (res_30780 * res_30780) -
        squot32(squot32(group_id_32309, squot32(num_groups_32604 + smax32(1,
                                                                          sizze_30757 *
                                                                          res_30780 *
                                                                          res_30780) -
                                                1, smax32(1, sizze_30757 *
                                                          res_30780 *
                                                          res_30780))) -
                squot32(squot32(group_id_32309, squot32(num_groups_32604 +
                                                        smax32(1, sizze_30757 *
                                                               res_30780 *
                                                               res_30780) - 1,
                                                        smax32(1, sizze_30757 *
                                                               res_30780 *
                                                               res_30780))),
                        res_30780 * res_30780) * (res_30780 * res_30780),
                res_30780) * res_30780;
    
    int32_t chunk_sizze_37900 = smin32(squot32(n_30761 + group_sizze_32594 *
                                               squot32(num_groups_32604 +
                                                       smax32(1, sizze_30757 *
                                                              res_30780 *
                                                              res_30780) - 1,
                                                       smax32(1, sizze_30757 *
                                                              res_30780 *
                                                              res_30780)) - 1,
                                               group_sizze_32594 *
                                               squot32(num_groups_32604 +
                                                       smax32(1, sizze_30757 *
                                                              res_30780 *
                                                              res_30780) - 1,
                                                       smax32(1, sizze_30757 *
                                                              res_30780 *
                                                              res_30780))),
                                       squot32(n_30761 -
                                               srem32(global_tid_32307,
                                                      group_sizze_32594 *
                                                      squot32(num_groups_32604 +
                                                              smax32(1,
                                                                     sizze_30757 *
                                                                     res_30780 *
                                                                     res_30780) -
                                                              1, smax32(1,
                                                                        sizze_30757 *
                                                                        res_30780 *
                                                                        res_30780))) +
                                               thread_per_segment_37891 - 1,
                                               thread_per_segment_37891));
    float x_32610;
    float x_32611;
    
    x_32610 = 0.0F;
    for (int32_t i_37904 = 0; i_37904 < chunk_sizze_37900; i_37904++) {
        gtid_32306 = srem32(global_tid_32307, group_sizze_32594 *
                            squot32(num_groups_32604 + smax32(1, sizze_30757 *
                                                              res_30780 *
                                                              res_30780) - 1,
                                    smax32(1, sizze_30757 * res_30780 *
                                           res_30780))) +
            thread_per_segment_37891 * i_37904;
        // apply map function
        {
            float x_32616;
            float x_32617;
            float x_32618;
            float x_32619;
            bool res_32620;
            float y_32621;
            float res_32622;
            
            x_32616 = *(__global float *) &images_mem_37201[(gtid_32276 *
                                                             sizze_30758 +
                                                             gtid_32306) * 4];
            x_32617 = *(__global float *) &arg_mem_37210[(gtid_32277 *
                                                          sizze_30756 +
                                                          gtid_32306) * 4];
            x_32618 = *(__global float *) &mem_37302[(gtid_32278 * sizze_30756 +
                                                      gtid_32306) * 4];
            x_32619 = x_32617 * x_32618;
            res_32620 = futrts_isnan32(x_32616);
            if (res_32620) {
                y_32621 = 0.0F;
            } else {
                y_32621 = 1.0F;
            }
            res_32622 = x_32619 * y_32621;
            // save results to be reduced
            {
                x_32611 = res_32622;
            }
            // save map-out results
            { }
            // apply reduction operator
            {
                float res_32612 = x_32610 + x_32611;
                
                x_32610 = res_32612;
            }
        }
    }
    // to reduce current chunk, first store our result to memory
    {
        *(__local float *) &red_arr_mem_37896[local_tid_32308 * 4] = x_32610;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_37905;
    int32_t skip_waves_37906;
    float x_37901;
    float x_37902;
    
    offset_37905 = 0;
    // participating threads read initial accumulator
    {
        if (slt32(local_tid_32308, group_sizze_32594)) {
            x_37901 = *(__local float *) &red_arr_mem_37896[(local_tid_32308 +
                                                             offset_37905) * 4];
        }
    }
    offset_37905 = 1;
    while (slt32(offset_37905, wave_sizze_37887)) {
        if (slt32(local_tid_32308 + offset_37905, group_sizze_32594) &&
            ((local_tid_32308 - squot32(local_tid_32308, wave_sizze_37887) *
              wave_sizze_37887) & (2 * offset_37905 - 1)) == 0) {
            // read array element
            {
                x_37902 = *(volatile __local
                            float *) &red_arr_mem_37896[(local_tid_32308 +
                                                         offset_37905) * 4];
            }
            // apply reduction operation
            {
                float res_37903 = x_37901 + x_37902;
                
                x_37901 = res_37903;
            }
            // write result of operation
            {
                *(volatile __local float *) &red_arr_mem_37896[local_tid_32308 *
                                                               4] = x_37901;
            }
        }
        offset_37905 *= 2;
    }
    skip_waves_37906 = 1;
    while (slt32(skip_waves_37906, squot32(group_sizze_32594 +
                                           wave_sizze_37887 - 1,
                                           wave_sizze_37887))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_37905 = skip_waves_37906 * wave_sizze_37887;
        if (slt32(local_tid_32308 + offset_37905, group_sizze_32594) &&
            ((local_tid_32308 - squot32(local_tid_32308, wave_sizze_37887) *
              wave_sizze_37887) == 0 && (squot32(local_tid_32308,
                                                 wave_sizze_37887) & (2 *
                                                                      skip_waves_37906 -
                                                                      1)) ==
             0)) {
            // read array element
            {
                x_37902 = *(__local
                            float *) &red_arr_mem_37896[(local_tid_32308 +
                                                         offset_37905) * 4];
            }
            // apply reduction operation
            {
                float res_37903 = x_37901 + x_37902;
                
                x_37901 = res_37903;
            }
            // write result of operation
            {
                *(__local float *) &red_arr_mem_37896[local_tid_32308 * 4] =
                    x_37901;
            }
        }
        skip_waves_37906 *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (squot32(num_groups_32604 + smax32(1, sizze_30757 * res_30780 *
                                          res_30780) - 1, smax32(1,
                                                                 sizze_30757 *
                                                                 res_30780 *
                                                                 res_30780)) ==
        1) {
        // first thread in group saves final result to memory
        {
            if (local_tid_32308 == 0) {
                *(__global float *) &mem_37307[(gtid_32276 * (res_30780 *
                                                              res_30780) +
                                                gtid_32277 * res_30780 +
                                                gtid_32278) * 4] = x_37901;
            }
        }
    } else {
        int32_t old_counter_37907;
        
        // first thread in group saves group result to memory
        {
            if (local_tid_32308 == 0) {
                *(__global float *) &group_res_arr_mem_37892[group_id_32309 *
                                                             4] = x_37901;
                mem_fence_global();
                old_counter_37907 = atomic_add((volatile __global
                                                int *) &counter_mem_37894[srem32(squot32(group_id_32309,
                                                                                         squot32(num_groups_32604 +
                                                                                                 smax32(1,
                                                                                                        sizze_30757 *
                                                                                                        res_30780 *
                                                                                                        res_30780) -
                                                                                                 1,
                                                                                                 smax32(1,
                                                                                                        sizze_30757 *
                                                                                                        res_30780 *
                                                                                                        res_30780))),
                                                                                 1024) *
                                                                          4],
                                               1);
                *(__local bool *) &sync_arr_mem_37898[0] = old_counter_37907 ==
                    squot32(num_groups_32604 + smax32(1, sizze_30757 *
                                                      res_30780 * res_30780) -
                            1, smax32(1, sizze_30757 * res_30780 * res_30780)) -
                    1;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        bool is_last_group_37908 = *(__local bool *) &sync_arr_mem_37898[0];
        
        if (is_last_group_37908) {
            if (local_tid_32308 == 0) {
                old_counter_37907 = atomic_add((volatile __global
                                                int *) &counter_mem_37894[srem32(squot32(group_id_32309,
                                                                                         squot32(num_groups_32604 +
                                                                                                 smax32(1,
                                                                                                        sizze_30757 *
                                                                                                        res_30780 *
                                                                                                        res_30780) -
                                                                                                 1,
                                                                                                 smax32(1,
                                                                                                        sizze_30757 *
                                                                                                        res_30780 *
                                                                                                        res_30780))),
                                                                                 1024) *
                                                                          4],
                                               0 - squot32(num_groups_32604 +
                                                           smax32(1,
                                                                  sizze_30757 *
                                                                  res_30780 *
                                                                  res_30780) -
                                                           1, smax32(1,
                                                                     sizze_30757 *
                                                                     res_30780 *
                                                                     res_30780)));
            }
            // read in the per-group-results
            {
                if (slt32(local_tid_32308, squot32(num_groups_32604 + smax32(1,
                                                                             sizze_30757 *
                                                                             res_30780 *
                                                                             res_30780) -
                                                   1, smax32(1, sizze_30757 *
                                                             res_30780 *
                                                             res_30780)))) {
                    x_32610 = *(__global
                                float *) &group_res_arr_mem_37892[(squot32(group_id_32309,
                                                                           squot32(num_groups_32604 +
                                                                                   smax32(1,
                                                                                          sizze_30757 *
                                                                                          res_30780 *
                                                                                          res_30780) -
                                                                                   1,
                                                                                   smax32(1,
                                                                                          sizze_30757 *
                                                                                          res_30780 *
                                                                                          res_30780))) *
                                                                   squot32(num_groups_32604 +
                                                                           smax32(1,
                                                                                  sizze_30757 *
                                                                                  res_30780 *
                                                                                  res_30780) -
                                                                           1,
                                                                           smax32(1,
                                                                                  sizze_30757 *
                                                                                  res_30780 *
                                                                                  res_30780)) +
                                                                   local_tid_32308) *
                                                                  4];
                } else {
                    x_32610 = 0.0F;
                }
                *(__local float *) &red_arr_mem_37896[local_tid_32308 * 4] =
                    x_32610;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // reduce the per-group results
            {
                int32_t offset_37909;
                int32_t skip_waves_37910;
                float x_37901;
                float x_37902;
                
                offset_37909 = 0;
                // participating threads read initial accumulator
                {
                    if (slt32(local_tid_32308, group_sizze_32594)) {
                        x_37901 = *(__local
                                    float *) &red_arr_mem_37896[(local_tid_32308 +
                                                                 offset_37909) *
                                                                4];
                    }
                }
                offset_37909 = 1;
                while (slt32(offset_37909, wave_sizze_37887)) {
                    if (slt32(local_tid_32308 + offset_37909,
                              group_sizze_32594) && ((local_tid_32308 -
                                                      squot32(local_tid_32308,
                                                              wave_sizze_37887) *
                                                      wave_sizze_37887) & (2 *
                                                                           offset_37909 -
                                                                           1)) ==
                        0) {
                        // read array element
                        {
                            x_37902 = *(volatile __local
                                        float *) &red_arr_mem_37896[(local_tid_32308 +
                                                                     offset_37909) *
                                                                    4];
                        }
                        // apply reduction operation
                        {
                            float res_37903 = x_37901 + x_37902;
                            
                            x_37901 = res_37903;
                        }
                        // write result of operation
                        {
                            *(volatile __local
                              float *) &red_arr_mem_37896[local_tid_32308 * 4] =
                                x_37901;
                        }
                    }
                    offset_37909 *= 2;
                }
                skip_waves_37910 = 1;
                while (slt32(skip_waves_37910, squot32(group_sizze_32594 +
                                                       wave_sizze_37887 - 1,
                                                       wave_sizze_37887))) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                    offset_37909 = skip_waves_37910 * wave_sizze_37887;
                    if (slt32(local_tid_32308 + offset_37909,
                              group_sizze_32594) && ((local_tid_32308 -
                                                      squot32(local_tid_32308,
                                                              wave_sizze_37887) *
                                                      wave_sizze_37887) == 0 &&
                                                     (squot32(local_tid_32308,
                                                              wave_sizze_37887) &
                                                      (2 * skip_waves_37910 -
                                                       1)) == 0)) {
                        // read array element
                        {
                            x_37902 = *(__local
                                        float *) &red_arr_mem_37896[(local_tid_32308 +
                                                                     offset_37909) *
                                                                    4];
                        }
                        // apply reduction operation
                        {
                            float res_37903 = x_37901 + x_37902;
                            
                            x_37901 = res_37903;
                        }
                        // write result of operation
                        {
                            *(__local
                              float *) &red_arr_mem_37896[local_tid_32308 * 4] =
                                x_37901;
                        }
                    }
                    skip_waves_37910 *= 2;
                }
                // and back to memory with the final result
                {
                    if (local_tid_32308 == 0) {
                        *(__global float *) &mem_37307[(gtid_32276 *
                                                        (res_30780 *
                                                         res_30780) +
                                                        gtid_32277 * res_30780 +
                                                        gtid_32278) * 4] =
                            x_37901;
                    }
                }
            }
        }
    }
}
__kernel void segred_large_33523(int32_t sizze_30756, int32_t sizze_30757,
                                 int32_t sizze_30758, int32_t n_30761,
                                 int32_t res_30780, int32_t num_groups_33630,
                                 __global unsigned char *images_mem_37201,
                                 __global unsigned char *arg_mem_37210, __global
                                 unsigned char *mem_37389,
                                 int32_t thread_per_segment_37976, __global
                                 unsigned char *group_res_arr_mem_37977,
                                 __global unsigned char *counter_mem_37979)
{
    const int32_t group_sizze_33620 = mainzigroup_sizze_33505;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    ALIGNED_LOCAL_MEMORY(red_arr_mem_37981_backing_0, 4 *
                         mainzigroup_sizze_33505);
    ALIGNED_LOCAL_MEMORY(sync_arr_mem_37983_backing_1, 1);
    
    int32_t global_tid_33523;
    int32_t local_tid_33524;
    int32_t group_sizze_37973;
    int32_t wave_sizze_37972;
    int32_t group_id_33525;
    
    global_tid_33523 = get_global_id(0);
    local_tid_33524 = get_local_id(0);
    group_sizze_37973 = get_local_size(0);
    wave_sizze_37972 = LOCKSTEP_WIDTH;
    group_id_33525 = get_group_id(0);
    
    int32_t gtid_33496;
    int32_t gtid_33497;
    int32_t gtid_33522;
    __local char *red_arr_mem_37981;
    
    red_arr_mem_37981 = (__local char *) red_arr_mem_37981_backing_0;
    
    __local char *sync_arr_mem_37983;
    
    sync_arr_mem_37983 = (__local char *) sync_arr_mem_37983_backing_1;
    gtid_33496 = squot32(squot32(group_id_33525, squot32(num_groups_33630 +
                                                         smax32(1, sizze_30757 *
                                                                res_30780) - 1,
                                                         smax32(1, sizze_30757 *
                                                                res_30780))),
                         res_30780);
    gtid_33497 = squot32(group_id_33525, squot32(num_groups_33630 + smax32(1,
                                                                           sizze_30757 *
                                                                           res_30780) -
                                                 1, smax32(1, sizze_30757 *
                                                           res_30780))) -
        squot32(squot32(group_id_33525, squot32(num_groups_33630 + smax32(1,
                                                                          sizze_30757 *
                                                                          res_30780) -
                                                1, smax32(1, sizze_30757 *
                                                          res_30780))),
                res_30780) * res_30780;
    
    int32_t chunk_sizze_37985 = smin32(squot32(n_30761 + group_sizze_33620 *
                                               squot32(num_groups_33630 +
                                                       smax32(1, sizze_30757 *
                                                              res_30780) - 1,
                                                       smax32(1, sizze_30757 *
                                                              res_30780)) - 1,
                                               group_sizze_33620 *
                                               squot32(num_groups_33630 +
                                                       smax32(1, sizze_30757 *
                                                              res_30780) - 1,
                                                       smax32(1, sizze_30757 *
                                                              res_30780))),
                                       squot32(n_30761 -
                                               srem32(global_tid_33523,
                                                      group_sizze_33620 *
                                                      squot32(num_groups_33630 +
                                                              smax32(1,
                                                                     sizze_30757 *
                                                                     res_30780) -
                                                              1, smax32(1,
                                                                        sizze_30757 *
                                                                        res_30780))) +
                                               thread_per_segment_37976 - 1,
                                               thread_per_segment_37976));
    float x_33636;
    float x_33637;
    
    x_33636 = 0.0F;
    for (int32_t i_37989 = 0; i_37989 < chunk_sizze_37985; i_37989++) {
        gtid_33522 = srem32(global_tid_33523, group_sizze_33620 *
                            squot32(num_groups_33630 + smax32(1, sizze_30757 *
                                                              res_30780) - 1,
                                    smax32(1, sizze_30757 * res_30780))) +
            thread_per_segment_37976 * i_37989;
        // apply map function
        {
            float x_33641;
            float x_33642;
            bool res_33643;
            float res_33644;
            
            x_33641 = *(__global float *) &arg_mem_37210[(gtid_33497 *
                                                          sizze_30756 +
                                                          gtid_33522) * 4];
            x_33642 = *(__global float *) &images_mem_37201[(gtid_33496 *
                                                             sizze_30758 +
                                                             gtid_33522) * 4];
            res_33643 = futrts_isnan32(x_33642);
            if (res_33643) {
                res_33644 = 0.0F;
            } else {
                float res_33645 = x_33641 * x_33642;
                
                res_33644 = res_33645;
            }
            // save results to be reduced
            {
                x_33637 = res_33644;
            }
            // save map-out results
            { }
            // apply reduction operator
            {
                float res_33638 = x_33636 + x_33637;
                
                x_33636 = res_33638;
            }
        }
    }
    // to reduce current chunk, first store our result to memory
    {
        *(__local float *) &red_arr_mem_37981[local_tid_33524 * 4] = x_33636;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_37990;
    int32_t skip_waves_37991;
    float x_37986;
    float x_37987;
    
    offset_37990 = 0;
    // participating threads read initial accumulator
    {
        if (slt32(local_tid_33524, group_sizze_33620)) {
            x_37986 = *(__local float *) &red_arr_mem_37981[(local_tid_33524 +
                                                             offset_37990) * 4];
        }
    }
    offset_37990 = 1;
    while (slt32(offset_37990, wave_sizze_37972)) {
        if (slt32(local_tid_33524 + offset_37990, group_sizze_33620) &&
            ((local_tid_33524 - squot32(local_tid_33524, wave_sizze_37972) *
              wave_sizze_37972) & (2 * offset_37990 - 1)) == 0) {
            // read array element
            {
                x_37987 = *(volatile __local
                            float *) &red_arr_mem_37981[(local_tid_33524 +
                                                         offset_37990) * 4];
            }
            // apply reduction operation
            {
                float res_37988 = x_37986 + x_37987;
                
                x_37986 = res_37988;
            }
            // write result of operation
            {
                *(volatile __local float *) &red_arr_mem_37981[local_tid_33524 *
                                                               4] = x_37986;
            }
        }
        offset_37990 *= 2;
    }
    skip_waves_37991 = 1;
    while (slt32(skip_waves_37991, squot32(group_sizze_33620 +
                                           wave_sizze_37972 - 1,
                                           wave_sizze_37972))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_37990 = skip_waves_37991 * wave_sizze_37972;
        if (slt32(local_tid_33524 + offset_37990, group_sizze_33620) &&
            ((local_tid_33524 - squot32(local_tid_33524, wave_sizze_37972) *
              wave_sizze_37972) == 0 && (squot32(local_tid_33524,
                                                 wave_sizze_37972) & (2 *
                                                                      skip_waves_37991 -
                                                                      1)) ==
             0)) {
            // read array element
            {
                x_37987 = *(__local
                            float *) &red_arr_mem_37981[(local_tid_33524 +
                                                         offset_37990) * 4];
            }
            // apply reduction operation
            {
                float res_37988 = x_37986 + x_37987;
                
                x_37986 = res_37988;
            }
            // write result of operation
            {
                *(__local float *) &red_arr_mem_37981[local_tid_33524 * 4] =
                    x_37986;
            }
        }
        skip_waves_37991 *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (squot32(num_groups_33630 + smax32(1, sizze_30757 * res_30780) - 1,
                smax32(1, sizze_30757 * res_30780)) == 1) {
        // first thread in group saves final result to memory
        {
            if (local_tid_33524 == 0) {
                *(__global float *) &mem_37389[(gtid_33496 * res_30780 +
                                                gtid_33497) * 4] = x_37986;
            }
        }
    } else {
        int32_t old_counter_37992;
        
        // first thread in group saves group result to memory
        {
            if (local_tid_33524 == 0) {
                *(__global float *) &group_res_arr_mem_37977[group_id_33525 *
                                                             4] = x_37986;
                mem_fence_global();
                old_counter_37992 = atomic_add((volatile __global
                                                int *) &counter_mem_37979[srem32(squot32(group_id_33525,
                                                                                         squot32(num_groups_33630 +
                                                                                                 smax32(1,
                                                                                                        sizze_30757 *
                                                                                                        res_30780) -
                                                                                                 1,
                                                                                                 smax32(1,
                                                                                                        sizze_30757 *
                                                                                                        res_30780))),
                                                                                 1024) *
                                                                          4],
                                               1);
                *(__local bool *) &sync_arr_mem_37983[0] = old_counter_37992 ==
                    squot32(num_groups_33630 + smax32(1, sizze_30757 *
                                                      res_30780) - 1, smax32(1,
                                                                             sizze_30757 *
                                                                             res_30780)) -
                    1;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        bool is_last_group_37993 = *(__local bool *) &sync_arr_mem_37983[0];
        
        if (is_last_group_37993) {
            if (local_tid_33524 == 0) {
                old_counter_37992 = atomic_add((volatile __global
                                                int *) &counter_mem_37979[srem32(squot32(group_id_33525,
                                                                                         squot32(num_groups_33630 +
                                                                                                 smax32(1,
                                                                                                        sizze_30757 *
                                                                                                        res_30780) -
                                                                                                 1,
                                                                                                 smax32(1,
                                                                                                        sizze_30757 *
                                                                                                        res_30780))),
                                                                                 1024) *
                                                                          4],
                                               0 - squot32(num_groups_33630 +
                                                           smax32(1,
                                                                  sizze_30757 *
                                                                  res_30780) -
                                                           1, smax32(1,
                                                                     sizze_30757 *
                                                                     res_30780)));
            }
            // read in the per-group-results
            {
                if (slt32(local_tid_33524, squot32(num_groups_33630 + smax32(1,
                                                                             sizze_30757 *
                                                                             res_30780) -
                                                   1, smax32(1, sizze_30757 *
                                                             res_30780)))) {
                    x_33636 = *(__global
                                float *) &group_res_arr_mem_37977[(squot32(group_id_33525,
                                                                           squot32(num_groups_33630 +
                                                                                   smax32(1,
                                                                                          sizze_30757 *
                                                                                          res_30780) -
                                                                                   1,
                                                                                   smax32(1,
                                                                                          sizze_30757 *
                                                                                          res_30780))) *
                                                                   squot32(num_groups_33630 +
                                                                           smax32(1,
                                                                                  sizze_30757 *
                                                                                  res_30780) -
                                                                           1,
                                                                           smax32(1,
                                                                                  sizze_30757 *
                                                                                  res_30780)) +
                                                                   local_tid_33524) *
                                                                  4];
                } else {
                    x_33636 = 0.0F;
                }
                *(__local float *) &red_arr_mem_37981[local_tid_33524 * 4] =
                    x_33636;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // reduce the per-group results
            {
                int32_t offset_37994;
                int32_t skip_waves_37995;
                float x_37986;
                float x_37987;
                
                offset_37994 = 0;
                // participating threads read initial accumulator
                {
                    if (slt32(local_tid_33524, group_sizze_33620)) {
                        x_37986 = *(__local
                                    float *) &red_arr_mem_37981[(local_tid_33524 +
                                                                 offset_37994) *
                                                                4];
                    }
                }
                offset_37994 = 1;
                while (slt32(offset_37994, wave_sizze_37972)) {
                    if (slt32(local_tid_33524 + offset_37994,
                              group_sizze_33620) && ((local_tid_33524 -
                                                      squot32(local_tid_33524,
                                                              wave_sizze_37972) *
                                                      wave_sizze_37972) & (2 *
                                                                           offset_37994 -
                                                                           1)) ==
                        0) {
                        // read array element
                        {
                            x_37987 = *(volatile __local
                                        float *) &red_arr_mem_37981[(local_tid_33524 +
                                                                     offset_37994) *
                                                                    4];
                        }
                        // apply reduction operation
                        {
                            float res_37988 = x_37986 + x_37987;
                            
                            x_37986 = res_37988;
                        }
                        // write result of operation
                        {
                            *(volatile __local
                              float *) &red_arr_mem_37981[local_tid_33524 * 4] =
                                x_37986;
                        }
                    }
                    offset_37994 *= 2;
                }
                skip_waves_37995 = 1;
                while (slt32(skip_waves_37995, squot32(group_sizze_33620 +
                                                       wave_sizze_37972 - 1,
                                                       wave_sizze_37972))) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                    offset_37994 = skip_waves_37995 * wave_sizze_37972;
                    if (slt32(local_tid_33524 + offset_37994,
                              group_sizze_33620) && ((local_tid_33524 -
                                                      squot32(local_tid_33524,
                                                              wave_sizze_37972) *
                                                      wave_sizze_37972) == 0 &&
                                                     (squot32(local_tid_33524,
                                                              wave_sizze_37972) &
                                                      (2 * skip_waves_37995 -
                                                       1)) == 0)) {
                        // read array element
                        {
                            x_37987 = *(__local
                                        float *) &red_arr_mem_37981[(local_tid_33524 +
                                                                     offset_37994) *
                                                                    4];
                        }
                        // apply reduction operation
                        {
                            float res_37988 = x_37986 + x_37987;
                            
                            x_37986 = res_37988;
                        }
                        // write result of operation
                        {
                            *(__local
                              float *) &red_arr_mem_37981[local_tid_33524 * 4] =
                                x_37986;
                        }
                    }
                    skip_waves_37995 *= 2;
                }
                // and back to memory with the final result
                {
                    if (local_tid_33524 == 0) {
                        *(__global float *) &mem_37389[(gtid_33496 * res_30780 +
                                                        gtid_33497) * 4] =
                            x_37986;
                    }
                }
            }
        }
    }
}
__kernel void segred_large_33855(int32_t sizze_30757, int32_t res_30780,
                                 int32_t j_m_i_30913, int32_t num_groups_33956,
                                 __global unsigned char *res_mem_37344, __global
                                 unsigned char *res_mem_37393, __global
                                 unsigned char *mem_37445,
                                 int32_t thread_per_segment_38037, __global
                                 unsigned char *group_res_arr_mem_38038,
                                 __global unsigned char *counter_mem_38040)
{
    const int32_t group_sizze_33946 = mainzigroup_sizze_33837;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    ALIGNED_LOCAL_MEMORY(red_arr_mem_38042_backing_0, 4 *
                         mainzigroup_sizze_33837);
    ALIGNED_LOCAL_MEMORY(sync_arr_mem_38044_backing_1, 1);
    
    int32_t global_tid_33855;
    int32_t local_tid_33856;
    int32_t group_sizze_38034;
    int32_t wave_sizze_38033;
    int32_t group_id_33857;
    
    global_tid_33855 = get_global_id(0);
    local_tid_33856 = get_local_id(0);
    group_sizze_38034 = get_local_size(0);
    wave_sizze_38033 = LOCKSTEP_WIDTH;
    group_id_33857 = get_group_id(0);
    
    int32_t gtid_33829;
    int32_t gtid_33830;
    int32_t gtid_33854;
    __local char *red_arr_mem_38042;
    
    red_arr_mem_38042 = (__local char *) red_arr_mem_38042_backing_0;
    
    __local char *sync_arr_mem_38044;
    
    sync_arr_mem_38044 = (__local char *) sync_arr_mem_38044_backing_1;
    gtid_33829 = squot32(squot32(group_id_33857, squot32(num_groups_33956 +
                                                         smax32(1, sizze_30757 *
                                                                res_30780) - 1,
                                                         smax32(1, sizze_30757 *
                                                                res_30780))),
                         res_30780);
    gtid_33830 = squot32(group_id_33857, squot32(num_groups_33956 + smax32(1,
                                                                           sizze_30757 *
                                                                           res_30780) -
                                                 1, smax32(1, sizze_30757 *
                                                           res_30780))) -
        squot32(squot32(group_id_33857, squot32(num_groups_33956 + smax32(1,
                                                                          sizze_30757 *
                                                                          res_30780) -
                                                1, smax32(1, sizze_30757 *
                                                          res_30780))),
                res_30780) * res_30780;
    
    int32_t chunk_sizze_38046 = smin32(squot32(j_m_i_30913 + group_sizze_33946 *
                                               squot32(num_groups_33956 +
                                                       smax32(1, sizze_30757 *
                                                              res_30780) - 1,
                                                       smax32(1, sizze_30757 *
                                                              res_30780)) - 1,
                                               group_sizze_33946 *
                                               squot32(num_groups_33956 +
                                                       smax32(1, sizze_30757 *
                                                              res_30780) - 1,
                                                       smax32(1, sizze_30757 *
                                                              res_30780))),
                                       squot32(j_m_i_30913 -
                                               srem32(global_tid_33855,
                                                      group_sizze_33946 *
                                                      squot32(num_groups_33956 +
                                                              smax32(1,
                                                                     sizze_30757 *
                                                                     res_30780) -
                                                              1, smax32(1,
                                                                        sizze_30757 *
                                                                        res_30780))) +
                                               thread_per_segment_38037 - 1,
                                               thread_per_segment_38037));
    float x_33962;
    float x_33963;
    
    x_33962 = 0.0F;
    for (int32_t i_38050 = 0; i_38050 < chunk_sizze_38046; i_38050++) {
        gtid_33854 = srem32(global_tid_33855, group_sizze_33946 *
                            squot32(num_groups_33956 + smax32(1, sizze_30757 *
                                                              res_30780) - 1,
                                    smax32(1, sizze_30757 * res_30780))) +
            thread_per_segment_38037 * i_38050;
        // apply map function
        {
            int32_t binop_x_36468;
            int32_t binop_x_36469;
            int32_t new_index_36470;
            int32_t binop_y_36476;
            int32_t new_index_36477;
            float x_33968;
            float x_33969;
            float res_33970;
            
            binop_x_36468 = j_m_i_30913 * gtid_33829;
            binop_x_36469 = gtid_33854 + binop_x_36468;
            new_index_36470 = squot32(binop_x_36469, res_30780);
            binop_y_36476 = res_30780 * new_index_36470;
            new_index_36477 = binop_x_36469 - binop_y_36476;
            x_33968 = *(__global float *) &res_mem_37393[(new_index_36470 *
                                                          res_30780 +
                                                          new_index_36477) * 4];
            x_33969 = *(__global float *) &res_mem_37344[(gtid_33829 *
                                                          (j_m_i_30913 *
                                                           res_30780) +
                                                          gtid_33830 *
                                                          j_m_i_30913 +
                                                          gtid_33854) * 4];
            res_33970 = x_33968 * x_33969;
            // save results to be reduced
            {
                x_33963 = res_33970;
            }
            // save map-out results
            { }
            // apply reduction operator
            {
                float res_33964 = x_33962 + x_33963;
                
                x_33962 = res_33964;
            }
        }
    }
    // to reduce current chunk, first store our result to memory
    {
        *(__local float *) &red_arr_mem_38042[local_tid_33856 * 4] = x_33962;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_38051;
    int32_t skip_waves_38052;
    float x_38047;
    float x_38048;
    
    offset_38051 = 0;
    // participating threads read initial accumulator
    {
        if (slt32(local_tid_33856, group_sizze_33946)) {
            x_38047 = *(__local float *) &red_arr_mem_38042[(local_tid_33856 +
                                                             offset_38051) * 4];
        }
    }
    offset_38051 = 1;
    while (slt32(offset_38051, wave_sizze_38033)) {
        if (slt32(local_tid_33856 + offset_38051, group_sizze_33946) &&
            ((local_tid_33856 - squot32(local_tid_33856, wave_sizze_38033) *
              wave_sizze_38033) & (2 * offset_38051 - 1)) == 0) {
            // read array element
            {
                x_38048 = *(volatile __local
                            float *) &red_arr_mem_38042[(local_tid_33856 +
                                                         offset_38051) * 4];
            }
            // apply reduction operation
            {
                float res_38049 = x_38047 + x_38048;
                
                x_38047 = res_38049;
            }
            // write result of operation
            {
                *(volatile __local float *) &red_arr_mem_38042[local_tid_33856 *
                                                               4] = x_38047;
            }
        }
        offset_38051 *= 2;
    }
    skip_waves_38052 = 1;
    while (slt32(skip_waves_38052, squot32(group_sizze_33946 +
                                           wave_sizze_38033 - 1,
                                           wave_sizze_38033))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_38051 = skip_waves_38052 * wave_sizze_38033;
        if (slt32(local_tid_33856 + offset_38051, group_sizze_33946) &&
            ((local_tid_33856 - squot32(local_tid_33856, wave_sizze_38033) *
              wave_sizze_38033) == 0 && (squot32(local_tid_33856,
                                                 wave_sizze_38033) & (2 *
                                                                      skip_waves_38052 -
                                                                      1)) ==
             0)) {
            // read array element
            {
                x_38048 = *(__local
                            float *) &red_arr_mem_38042[(local_tid_33856 +
                                                         offset_38051) * 4];
            }
            // apply reduction operation
            {
                float res_38049 = x_38047 + x_38048;
                
                x_38047 = res_38049;
            }
            // write result of operation
            {
                *(__local float *) &red_arr_mem_38042[local_tid_33856 * 4] =
                    x_38047;
            }
        }
        skip_waves_38052 *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (squot32(num_groups_33956 + smax32(1, sizze_30757 * res_30780) - 1,
                smax32(1, sizze_30757 * res_30780)) == 1) {
        // first thread in group saves final result to memory
        {
            if (local_tid_33856 == 0) {
                *(__global float *) &mem_37445[(gtid_33829 * res_30780 +
                                                gtid_33830) * 4] = x_38047;
            }
        }
    } else {
        int32_t old_counter_38053;
        
        // first thread in group saves group result to memory
        {
            if (local_tid_33856 == 0) {
                *(__global float *) &group_res_arr_mem_38038[group_id_33857 *
                                                             4] = x_38047;
                mem_fence_global();
                old_counter_38053 = atomic_add((volatile __global
                                                int *) &counter_mem_38040[srem32(squot32(group_id_33857,
                                                                                         squot32(num_groups_33956 +
                                                                                                 smax32(1,
                                                                                                        sizze_30757 *
                                                                                                        res_30780) -
                                                                                                 1,
                                                                                                 smax32(1,
                                                                                                        sizze_30757 *
                                                                                                        res_30780))),
                                                                                 1024) *
                                                                          4],
                                               1);
                *(__local bool *) &sync_arr_mem_38044[0] = old_counter_38053 ==
                    squot32(num_groups_33956 + smax32(1, sizze_30757 *
                                                      res_30780) - 1, smax32(1,
                                                                             sizze_30757 *
                                                                             res_30780)) -
                    1;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        bool is_last_group_38054 = *(__local bool *) &sync_arr_mem_38044[0];
        
        if (is_last_group_38054) {
            if (local_tid_33856 == 0) {
                old_counter_38053 = atomic_add((volatile __global
                                                int *) &counter_mem_38040[srem32(squot32(group_id_33857,
                                                                                         squot32(num_groups_33956 +
                                                                                                 smax32(1,
                                                                                                        sizze_30757 *
                                                                                                        res_30780) -
                                                                                                 1,
                                                                                                 smax32(1,
                                                                                                        sizze_30757 *
                                                                                                        res_30780))),
                                                                                 1024) *
                                                                          4],
                                               0 - squot32(num_groups_33956 +
                                                           smax32(1,
                                                                  sizze_30757 *
                                                                  res_30780) -
                                                           1, smax32(1,
                                                                     sizze_30757 *
                                                                     res_30780)));
            }
            // read in the per-group-results
            {
                if (slt32(local_tid_33856, squot32(num_groups_33956 + smax32(1,
                                                                             sizze_30757 *
                                                                             res_30780) -
                                                   1, smax32(1, sizze_30757 *
                                                             res_30780)))) {
                    x_33962 = *(__global
                                float *) &group_res_arr_mem_38038[(squot32(group_id_33857,
                                                                           squot32(num_groups_33956 +
                                                                                   smax32(1,
                                                                                          sizze_30757 *
                                                                                          res_30780) -
                                                                                   1,
                                                                                   smax32(1,
                                                                                          sizze_30757 *
                                                                                          res_30780))) *
                                                                   squot32(num_groups_33956 +
                                                                           smax32(1,
                                                                                  sizze_30757 *
                                                                                  res_30780) -
                                                                           1,
                                                                           smax32(1,
                                                                                  sizze_30757 *
                                                                                  res_30780)) +
                                                                   local_tid_33856) *
                                                                  4];
                } else {
                    x_33962 = 0.0F;
                }
                *(__local float *) &red_arr_mem_38042[local_tid_33856 * 4] =
                    x_33962;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // reduce the per-group results
            {
                int32_t offset_38055;
                int32_t skip_waves_38056;
                float x_38047;
                float x_38048;
                
                offset_38055 = 0;
                // participating threads read initial accumulator
                {
                    if (slt32(local_tid_33856, group_sizze_33946)) {
                        x_38047 = *(__local
                                    float *) &red_arr_mem_38042[(local_tid_33856 +
                                                                 offset_38055) *
                                                                4];
                    }
                }
                offset_38055 = 1;
                while (slt32(offset_38055, wave_sizze_38033)) {
                    if (slt32(local_tid_33856 + offset_38055,
                              group_sizze_33946) && ((local_tid_33856 -
                                                      squot32(local_tid_33856,
                                                              wave_sizze_38033) *
                                                      wave_sizze_38033) & (2 *
                                                                           offset_38055 -
                                                                           1)) ==
                        0) {
                        // read array element
                        {
                            x_38048 = *(volatile __local
                                        float *) &red_arr_mem_38042[(local_tid_33856 +
                                                                     offset_38055) *
                                                                    4];
                        }
                        // apply reduction operation
                        {
                            float res_38049 = x_38047 + x_38048;
                            
                            x_38047 = res_38049;
                        }
                        // write result of operation
                        {
                            *(volatile __local
                              float *) &red_arr_mem_38042[local_tid_33856 * 4] =
                                x_38047;
                        }
                    }
                    offset_38055 *= 2;
                }
                skip_waves_38056 = 1;
                while (slt32(skip_waves_38056, squot32(group_sizze_33946 +
                                                       wave_sizze_38033 - 1,
                                                       wave_sizze_38033))) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                    offset_38055 = skip_waves_38056 * wave_sizze_38033;
                    if (slt32(local_tid_33856 + offset_38055,
                              group_sizze_33946) && ((local_tid_33856 -
                                                      squot32(local_tid_33856,
                                                              wave_sizze_38033) *
                                                      wave_sizze_38033) == 0 &&
                                                     (squot32(local_tid_33856,
                                                              wave_sizze_38033) &
                                                      (2 * skip_waves_38056 -
                                                       1)) == 0)) {
                        // read array element
                        {
                            x_38048 = *(__local
                                        float *) &red_arr_mem_38042[(local_tid_33856 +
                                                                     offset_38055) *
                                                                    4];
                        }
                        // apply reduction operation
                        {
                            float res_38049 = x_38047 + x_38048;
                            
                            x_38047 = res_38049;
                        }
                        // write result of operation
                        {
                            *(__local
                              float *) &red_arr_mem_38042[local_tid_33856 * 4] =
                                x_38047;
                        }
                    }
                    skip_waves_38056 *= 2;
                }
                // and back to memory with the final result
                {
                    if (local_tid_33856 == 0) {
                        *(__global float *) &mem_37445[(gtid_33829 * res_30780 +
                                                        gtid_33830) * 4] =
                            x_38047;
                    }
                }
            }
        }
    }
}
__kernel void segred_large_34174(int32_t sizze_30756, int32_t sizze_30757,
                                 int32_t res_30780, int32_t num_groups_34275,
                                 __global unsigned char *mem_37218, __global
                                 unsigned char *res_mem_37449, __global
                                 unsigned char *mem_37502,
                                 int32_t thread_per_segment_38098, __global
                                 unsigned char *group_res_arr_mem_38099,
                                 __global unsigned char *counter_mem_38101)
{
    const int32_t group_sizze_34265 = mainzigroup_sizze_34156;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    ALIGNED_LOCAL_MEMORY(red_arr_mem_38103_backing_0, 4 *
                         mainzigroup_sizze_34156);
    ALIGNED_LOCAL_MEMORY(sync_arr_mem_38105_backing_1, 1);
    
    int32_t global_tid_34174;
    int32_t local_tid_34175;
    int32_t group_sizze_38095;
    int32_t wave_sizze_38094;
    int32_t group_id_34176;
    
    global_tid_34174 = get_global_id(0);
    local_tid_34175 = get_local_id(0);
    group_sizze_38095 = get_local_size(0);
    wave_sizze_38094 = LOCKSTEP_WIDTH;
    group_id_34176 = get_group_id(0);
    
    int32_t gtid_34147;
    int32_t gtid_34148;
    int32_t gtid_34173;
    __local char *red_arr_mem_38103;
    
    red_arr_mem_38103 = (__local char *) red_arr_mem_38103_backing_0;
    
    __local char *sync_arr_mem_38105;
    
    sync_arr_mem_38105 = (__local char *) sync_arr_mem_38105_backing_1;
    gtid_34147 = squot32(squot32(group_id_34176, squot32(num_groups_34275 +
                                                         smax32(1, sizze_30757 *
                                                                sizze_30756) -
                                                         1, smax32(1,
                                                                   sizze_30757 *
                                                                   sizze_30756))),
                         sizze_30756);
    gtid_34148 = squot32(group_id_34176, squot32(num_groups_34275 + smax32(1,
                                                                           sizze_30757 *
                                                                           sizze_30756) -
                                                 1, smax32(1, sizze_30757 *
                                                           sizze_30756))) -
        squot32(squot32(group_id_34176, squot32(num_groups_34275 + smax32(1,
                                                                          sizze_30757 *
                                                                          sizze_30756) -
                                                1, smax32(1, sizze_30757 *
                                                          sizze_30756))),
                sizze_30756) * sizze_30756;
    
    int32_t chunk_sizze_38107 = smin32(squot32(res_30780 + group_sizze_34265 *
                                               squot32(num_groups_34275 +
                                                       smax32(1, sizze_30757 *
                                                              sizze_30756) - 1,
                                                       smax32(1, sizze_30757 *
                                                              sizze_30756)) - 1,
                                               group_sizze_34265 *
                                               squot32(num_groups_34275 +
                                                       smax32(1, sizze_30757 *
                                                              sizze_30756) - 1,
                                                       smax32(1, sizze_30757 *
                                                              sizze_30756))),
                                       squot32(res_30780 -
                                               srem32(global_tid_34174,
                                                      group_sizze_34265 *
                                                      squot32(num_groups_34275 +
                                                              smax32(1,
                                                                     sizze_30757 *
                                                                     sizze_30756) -
                                                              1, smax32(1,
                                                                        sizze_30757 *
                                                                        sizze_30756))) +
                                               thread_per_segment_38098 - 1,
                                               thread_per_segment_38098));
    float x_34281;
    float x_34282;
    
    x_34281 = 0.0F;
    for (int32_t i_38111 = 0; i_38111 < chunk_sizze_38107; i_38111++) {
        gtid_34173 = srem32(global_tid_34174, group_sizze_34265 *
                            squot32(num_groups_34275 + smax32(1, sizze_30757 *
                                                              sizze_30756) - 1,
                                    smax32(1, sizze_30757 * sizze_30756))) +
            thread_per_segment_38098 * i_38111;
        // apply map function
        {
            float x_34286;
            float x_34287;
            float res_34288;
            
            x_34286 = *(__global float *) &res_mem_37449[(gtid_34147 *
                                                          res_30780 +
                                                          gtid_34173) * 4];
            x_34287 = *(__global float *) &mem_37218[(gtid_34148 * res_30780 +
                                                      gtid_34173) * 4];
            res_34288 = x_34286 * x_34287;
            // save results to be reduced
            {
                x_34282 = res_34288;
            }
            // save map-out results
            { }
            // apply reduction operator
            {
                float res_34283 = x_34281 + x_34282;
                
                x_34281 = res_34283;
            }
        }
    }
    // to reduce current chunk, first store our result to memory
    {
        *(__local float *) &red_arr_mem_38103[local_tid_34175 * 4] = x_34281;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_38112;
    int32_t skip_waves_38113;
    float x_38108;
    float x_38109;
    
    offset_38112 = 0;
    // participating threads read initial accumulator
    {
        if (slt32(local_tid_34175, group_sizze_34265)) {
            x_38108 = *(__local float *) &red_arr_mem_38103[(local_tid_34175 +
                                                             offset_38112) * 4];
        }
    }
    offset_38112 = 1;
    while (slt32(offset_38112, wave_sizze_38094)) {
        if (slt32(local_tid_34175 + offset_38112, group_sizze_34265) &&
            ((local_tid_34175 - squot32(local_tid_34175, wave_sizze_38094) *
              wave_sizze_38094) & (2 * offset_38112 - 1)) == 0) {
            // read array element
            {
                x_38109 = *(volatile __local
                            float *) &red_arr_mem_38103[(local_tid_34175 +
                                                         offset_38112) * 4];
            }
            // apply reduction operation
            {
                float res_38110 = x_38108 + x_38109;
                
                x_38108 = res_38110;
            }
            // write result of operation
            {
                *(volatile __local float *) &red_arr_mem_38103[local_tid_34175 *
                                                               4] = x_38108;
            }
        }
        offset_38112 *= 2;
    }
    skip_waves_38113 = 1;
    while (slt32(skip_waves_38113, squot32(group_sizze_34265 +
                                           wave_sizze_38094 - 1,
                                           wave_sizze_38094))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_38112 = skip_waves_38113 * wave_sizze_38094;
        if (slt32(local_tid_34175 + offset_38112, group_sizze_34265) &&
            ((local_tid_34175 - squot32(local_tid_34175, wave_sizze_38094) *
              wave_sizze_38094) == 0 && (squot32(local_tid_34175,
                                                 wave_sizze_38094) & (2 *
                                                                      skip_waves_38113 -
                                                                      1)) ==
             0)) {
            // read array element
            {
                x_38109 = *(__local
                            float *) &red_arr_mem_38103[(local_tid_34175 +
                                                         offset_38112) * 4];
            }
            // apply reduction operation
            {
                float res_38110 = x_38108 + x_38109;
                
                x_38108 = res_38110;
            }
            // write result of operation
            {
                *(__local float *) &red_arr_mem_38103[local_tid_34175 * 4] =
                    x_38108;
            }
        }
        skip_waves_38113 *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (squot32(num_groups_34275 + smax32(1, sizze_30757 * sizze_30756) - 1,
                smax32(1, sizze_30757 * sizze_30756)) == 1) {
        // first thread in group saves final result to memory
        {
            if (local_tid_34175 == 0) {
                *(__global float *) &mem_37502[(gtid_34147 * sizze_30756 +
                                                gtid_34148) * 4] = x_38108;
            }
        }
    } else {
        int32_t old_counter_38114;
        
        // first thread in group saves group result to memory
        {
            if (local_tid_34175 == 0) {
                *(__global float *) &group_res_arr_mem_38099[group_id_34176 *
                                                             4] = x_38108;
                mem_fence_global();
                old_counter_38114 = atomic_add((volatile __global
                                                int *) &counter_mem_38101[srem32(squot32(group_id_34176,
                                                                                         squot32(num_groups_34275 +
                                                                                                 smax32(1,
                                                                                                        sizze_30757 *
                                                                                                        sizze_30756) -
                                                                                                 1,
                                                                                                 smax32(1,
                                                                                                        sizze_30757 *
                                                                                                        sizze_30756))),
                                                                                 1024) *
                                                                          4],
                                               1);
                *(__local bool *) &sync_arr_mem_38105[0] = old_counter_38114 ==
                    squot32(num_groups_34275 + smax32(1, sizze_30757 *
                                                      sizze_30756) - 1,
                            smax32(1, sizze_30757 * sizze_30756)) - 1;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        bool is_last_group_38115 = *(__local bool *) &sync_arr_mem_38105[0];
        
        if (is_last_group_38115) {
            if (local_tid_34175 == 0) {
                old_counter_38114 = atomic_add((volatile __global
                                                int *) &counter_mem_38101[srem32(squot32(group_id_34176,
                                                                                         squot32(num_groups_34275 +
                                                                                                 smax32(1,
                                                                                                        sizze_30757 *
                                                                                                        sizze_30756) -
                                                                                                 1,
                                                                                                 smax32(1,
                                                                                                        sizze_30757 *
                                                                                                        sizze_30756))),
                                                                                 1024) *
                                                                          4],
                                               0 - squot32(num_groups_34275 +
                                                           smax32(1,
                                                                  sizze_30757 *
                                                                  sizze_30756) -
                                                           1, smax32(1,
                                                                     sizze_30757 *
                                                                     sizze_30756)));
            }
            // read in the per-group-results
            {
                if (slt32(local_tid_34175, squot32(num_groups_34275 + smax32(1,
                                                                             sizze_30757 *
                                                                             sizze_30756) -
                                                   1, smax32(1, sizze_30757 *
                                                             sizze_30756)))) {
                    x_34281 = *(__global
                                float *) &group_res_arr_mem_38099[(squot32(group_id_34176,
                                                                           squot32(num_groups_34275 +
                                                                                   smax32(1,
                                                                                          sizze_30757 *
                                                                                          sizze_30756) -
                                                                                   1,
                                                                                   smax32(1,
                                                                                          sizze_30757 *
                                                                                          sizze_30756))) *
                                                                   squot32(num_groups_34275 +
                                                                           smax32(1,
                                                                                  sizze_30757 *
                                                                                  sizze_30756) -
                                                                           1,
                                                                           smax32(1,
                                                                                  sizze_30757 *
                                                                                  sizze_30756)) +
                                                                   local_tid_34175) *
                                                                  4];
                } else {
                    x_34281 = 0.0F;
                }
                *(__local float *) &red_arr_mem_38103[local_tid_34175 * 4] =
                    x_34281;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // reduce the per-group results
            {
                int32_t offset_38116;
                int32_t skip_waves_38117;
                float x_38108;
                float x_38109;
                
                offset_38116 = 0;
                // participating threads read initial accumulator
                {
                    if (slt32(local_tid_34175, group_sizze_34265)) {
                        x_38108 = *(__local
                                    float *) &red_arr_mem_38103[(local_tid_34175 +
                                                                 offset_38116) *
                                                                4];
                    }
                }
                offset_38116 = 1;
                while (slt32(offset_38116, wave_sizze_38094)) {
                    if (slt32(local_tid_34175 + offset_38116,
                              group_sizze_34265) && ((local_tid_34175 -
                                                      squot32(local_tid_34175,
                                                              wave_sizze_38094) *
                                                      wave_sizze_38094) & (2 *
                                                                           offset_38116 -
                                                                           1)) ==
                        0) {
                        // read array element
                        {
                            x_38109 = *(volatile __local
                                        float *) &red_arr_mem_38103[(local_tid_34175 +
                                                                     offset_38116) *
                                                                    4];
                        }
                        // apply reduction operation
                        {
                            float res_38110 = x_38108 + x_38109;
                            
                            x_38108 = res_38110;
                        }
                        // write result of operation
                        {
                            *(volatile __local
                              float *) &red_arr_mem_38103[local_tid_34175 * 4] =
                                x_38108;
                        }
                    }
                    offset_38116 *= 2;
                }
                skip_waves_38117 = 1;
                while (slt32(skip_waves_38117, squot32(group_sizze_34265 +
                                                       wave_sizze_38094 - 1,
                                                       wave_sizze_38094))) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                    offset_38116 = skip_waves_38117 * wave_sizze_38094;
                    if (slt32(local_tid_34175 + offset_38116,
                              group_sizze_34265) && ((local_tid_34175 -
                                                      squot32(local_tid_34175,
                                                              wave_sizze_38094) *
                                                      wave_sizze_38094) == 0 &&
                                                     (squot32(local_tid_34175,
                                                              wave_sizze_38094) &
                                                      (2 * skip_waves_38117 -
                                                       1)) == 0)) {
                        // read array element
                        {
                            x_38109 = *(__local
                                        float *) &red_arr_mem_38103[(local_tid_34175 +
                                                                     offset_38116) *
                                                                    4];
                        }
                        // apply reduction operation
                        {
                            float res_38110 = x_38108 + x_38109;
                            
                            x_38108 = res_38110;
                        }
                        // write result of operation
                        {
                            *(__local
                              float *) &red_arr_mem_38103[local_tid_34175 * 4] =
                                x_38108;
                        }
                    }
                    skip_waves_38117 *= 2;
                }
                // and back to memory with the final result
                {
                    if (local_tid_34175 == 0) {
                        *(__global float *) &mem_37502[(gtid_34147 *
                                                        sizze_30756 +
                                                        gtid_34148) * 4] =
                            x_38108;
                    }
                }
            }
        }
    }
}
__kernel void segred_large_34952(int32_t sizze_30756, int32_t sizze_30757,
                                 int32_t n_30761, int32_t num_groups_35021,
                                 __global unsigned char *res_mem_37597, __global
                                 unsigned char *mem_37633, __global
                                 unsigned char *mem_37636,
                                 int32_t thread_per_segment_38276, __global
                                 unsigned char *group_res_arr_mem_38277,
                                 __global unsigned char *counter_mem_38279)
{
    const int32_t group_sizze_35011 = mainzigroup_sizze_34934;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    ALIGNED_LOCAL_MEMORY(red_arr_mem_38281_backing_0, 4 *
                         mainzigroup_sizze_34934);
    ALIGNED_LOCAL_MEMORY(sync_arr_mem_38283_backing_1, 1);
    
    int32_t global_tid_34952;
    int32_t local_tid_34953;
    int32_t group_sizze_38273;
    int32_t wave_sizze_38272;
    int32_t group_id_34954;
    
    global_tid_34952 = get_global_id(0);
    local_tid_34953 = get_local_id(0);
    group_sizze_38273 = get_local_size(0);
    wave_sizze_38272 = LOCKSTEP_WIDTH;
    group_id_34954 = get_group_id(0);
    
    int32_t gtid_34929;
    int32_t gtid_34951;
    __local char *red_arr_mem_38281;
    
    red_arr_mem_38281 = (__local char *) red_arr_mem_38281_backing_0;
    
    __local char *sync_arr_mem_38283;
    
    sync_arr_mem_38283 = (__local char *) sync_arr_mem_38283_backing_1;
    gtid_34929 = squot32(group_id_34954, squot32(num_groups_35021 + smax32(1,
                                                                           sizze_30757) -
                                                 1, smax32(1, sizze_30757)));
    
    int32_t chunk_sizze_38285 = smin32(squot32(n_30761 + group_sizze_35011 *
                                               squot32(num_groups_35021 +
                                                       smax32(1, sizze_30757) -
                                                       1, smax32(1,
                                                                 sizze_30757)) -
                                               1, group_sizze_35011 *
                                               squot32(num_groups_35021 +
                                                       smax32(1, sizze_30757) -
                                                       1, smax32(1,
                                                                 sizze_30757))),
                                       squot32(n_30761 -
                                               srem32(global_tid_34952,
                                                      group_sizze_35011 *
                                                      squot32(num_groups_35021 +
                                                              smax32(1,
                                                                     sizze_30757) -
                                                              1, smax32(1,
                                                                        sizze_30757))) +
                                               thread_per_segment_38276 - 1,
                                               thread_per_segment_38276));
    float x_35027;
    float x_35028;
    
    x_35027 = 0.0F;
    for (int32_t i_38289 = 0; i_38289 < chunk_sizze_38285; i_38289++) {
        gtid_34951 = srem32(global_tid_34952, group_sizze_35011 *
                            squot32(num_groups_35021 + smax32(1, sizze_30757) -
                                    1, smax32(1, sizze_30757))) +
            thread_per_segment_38276 * i_38289;
        // apply map function
        {
            int32_t res_35031;
            bool cond_35033;
            float res_35034;
            float res_35036;
            
            res_35031 = *(__global int32_t *) &mem_37633[gtid_34929 * 4];
            cond_35033 = slt32(gtid_34951, res_35031);
            if (cond_35033) {
                float res_35035 = *(__global
                                    float *) &res_mem_37597[(gtid_34929 *
                                                             sizze_30756 +
                                                             gtid_34951) * 4];
                
                res_35034 = res_35035;
            } else {
                res_35034 = 0.0F;
            }
            res_35036 = res_35034 * res_35034;
            // save results to be reduced
            {
                x_35028 = res_35036;
            }
            // save map-out results
            { }
            // apply reduction operator
            {
                float res_35029 = x_35027 + x_35028;
                
                x_35027 = res_35029;
            }
        }
    }
    // to reduce current chunk, first store our result to memory
    {
        *(__local float *) &red_arr_mem_38281[local_tid_34953 * 4] = x_35027;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_38290;
    int32_t skip_waves_38291;
    float x_38286;
    float x_38287;
    
    offset_38290 = 0;
    // participating threads read initial accumulator
    {
        if (slt32(local_tid_34953, group_sizze_35011)) {
            x_38286 = *(__local float *) &red_arr_mem_38281[(local_tid_34953 +
                                                             offset_38290) * 4];
        }
    }
    offset_38290 = 1;
    while (slt32(offset_38290, wave_sizze_38272)) {
        if (slt32(local_tid_34953 + offset_38290, group_sizze_35011) &&
            ((local_tid_34953 - squot32(local_tid_34953, wave_sizze_38272) *
              wave_sizze_38272) & (2 * offset_38290 - 1)) == 0) {
            // read array element
            {
                x_38287 = *(volatile __local
                            float *) &red_arr_mem_38281[(local_tid_34953 +
                                                         offset_38290) * 4];
            }
            // apply reduction operation
            {
                float res_38288 = x_38286 + x_38287;
                
                x_38286 = res_38288;
            }
            // write result of operation
            {
                *(volatile __local float *) &red_arr_mem_38281[local_tid_34953 *
                                                               4] = x_38286;
            }
        }
        offset_38290 *= 2;
    }
    skip_waves_38291 = 1;
    while (slt32(skip_waves_38291, squot32(group_sizze_35011 +
                                           wave_sizze_38272 - 1,
                                           wave_sizze_38272))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_38290 = skip_waves_38291 * wave_sizze_38272;
        if (slt32(local_tid_34953 + offset_38290, group_sizze_35011) &&
            ((local_tid_34953 - squot32(local_tid_34953, wave_sizze_38272) *
              wave_sizze_38272) == 0 && (squot32(local_tid_34953,
                                                 wave_sizze_38272) & (2 *
                                                                      skip_waves_38291 -
                                                                      1)) ==
             0)) {
            // read array element
            {
                x_38287 = *(__local
                            float *) &red_arr_mem_38281[(local_tid_34953 +
                                                         offset_38290) * 4];
            }
            // apply reduction operation
            {
                float res_38288 = x_38286 + x_38287;
                
                x_38286 = res_38288;
            }
            // write result of operation
            {
                *(__local float *) &red_arr_mem_38281[local_tid_34953 * 4] =
                    x_38286;
            }
        }
        skip_waves_38291 *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (squot32(num_groups_35021 + smax32(1, sizze_30757) - 1, smax32(1,
                                                                      sizze_30757)) ==
        1) {
        // first thread in group saves final result to memory
        {
            if (local_tid_34953 == 0) {
                *(__global float *) &mem_37636[gtid_34929 * 4] = x_38286;
            }
        }
    } else {
        int32_t old_counter_38292;
        
        // first thread in group saves group result to memory
        {
            if (local_tid_34953 == 0) {
                *(__global float *) &group_res_arr_mem_38277[group_id_34954 *
                                                             4] = x_38286;
                mem_fence_global();
                old_counter_38292 = atomic_add((volatile __global
                                                int *) &counter_mem_38279[srem32(squot32(group_id_34954,
                                                                                         squot32(num_groups_35021 +
                                                                                                 smax32(1,
                                                                                                        sizze_30757) -
                                                                                                 1,
                                                                                                 smax32(1,
                                                                                                        sizze_30757))),
                                                                                 1024) *
                                                                          4],
                                               1);
                *(__local bool *) &sync_arr_mem_38283[0] = old_counter_38292 ==
                    squot32(num_groups_35021 + smax32(1, sizze_30757) - 1,
                            smax32(1, sizze_30757)) - 1;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        bool is_last_group_38293 = *(__local bool *) &sync_arr_mem_38283[0];
        
        if (is_last_group_38293) {
            if (local_tid_34953 == 0) {
                old_counter_38292 = atomic_add((volatile __global
                                                int *) &counter_mem_38279[srem32(squot32(group_id_34954,
                                                                                         squot32(num_groups_35021 +
                                                                                                 smax32(1,
                                                                                                        sizze_30757) -
                                                                                                 1,
                                                                                                 smax32(1,
                                                                                                        sizze_30757))),
                                                                                 1024) *
                                                                          4],
                                               0 - squot32(num_groups_35021 +
                                                           smax32(1,
                                                                  sizze_30757) -
                                                           1, smax32(1,
                                                                     sizze_30757)));
            }
            // read in the per-group-results
            {
                if (slt32(local_tid_34953, squot32(num_groups_35021 + smax32(1,
                                                                             sizze_30757) -
                                                   1, smax32(1,
                                                             sizze_30757)))) {
                    x_35027 = *(__global
                                float *) &group_res_arr_mem_38277[(squot32(group_id_34954,
                                                                           squot32(num_groups_35021 +
                                                                                   smax32(1,
                                                                                          sizze_30757) -
                                                                                   1,
                                                                                   smax32(1,
                                                                                          sizze_30757))) *
                                                                   squot32(num_groups_35021 +
                                                                           smax32(1,
                                                                                  sizze_30757) -
                                                                           1,
                                                                           smax32(1,
                                                                                  sizze_30757)) +
                                                                   local_tid_34953) *
                                                                  4];
                } else {
                    x_35027 = 0.0F;
                }
                *(__local float *) &red_arr_mem_38281[local_tid_34953 * 4] =
                    x_35027;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // reduce the per-group results
            {
                int32_t offset_38294;
                int32_t skip_waves_38295;
                float x_38286;
                float x_38287;
                
                offset_38294 = 0;
                // participating threads read initial accumulator
                {
                    if (slt32(local_tid_34953, group_sizze_35011)) {
                        x_38286 = *(__local
                                    float *) &red_arr_mem_38281[(local_tid_34953 +
                                                                 offset_38294) *
                                                                4];
                    }
                }
                offset_38294 = 1;
                while (slt32(offset_38294, wave_sizze_38272)) {
                    if (slt32(local_tid_34953 + offset_38294,
                              group_sizze_35011) && ((local_tid_34953 -
                                                      squot32(local_tid_34953,
                                                              wave_sizze_38272) *
                                                      wave_sizze_38272) & (2 *
                                                                           offset_38294 -
                                                                           1)) ==
                        0) {
                        // read array element
                        {
                            x_38287 = *(volatile __local
                                        float *) &red_arr_mem_38281[(local_tid_34953 +
                                                                     offset_38294) *
                                                                    4];
                        }
                        // apply reduction operation
                        {
                            float res_38288 = x_38286 + x_38287;
                            
                            x_38286 = res_38288;
                        }
                        // write result of operation
                        {
                            *(volatile __local
                              float *) &red_arr_mem_38281[local_tid_34953 * 4] =
                                x_38286;
                        }
                    }
                    offset_38294 *= 2;
                }
                skip_waves_38295 = 1;
                while (slt32(skip_waves_38295, squot32(group_sizze_35011 +
                                                       wave_sizze_38272 - 1,
                                                       wave_sizze_38272))) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                    offset_38294 = skip_waves_38295 * wave_sizze_38272;
                    if (slt32(local_tid_34953 + offset_38294,
                              group_sizze_35011) && ((local_tid_34953 -
                                                      squot32(local_tid_34953,
                                                              wave_sizze_38272) *
                                                      wave_sizze_38272) == 0 &&
                                                     (squot32(local_tid_34953,
                                                              wave_sizze_38272) &
                                                      (2 * skip_waves_38295 -
                                                       1)) == 0)) {
                        // read array element
                        {
                            x_38287 = *(__local
                                        float *) &red_arr_mem_38281[(local_tid_34953 +
                                                                     offset_38294) *
                                                                    4];
                        }
                        // apply reduction operation
                        {
                            float res_38288 = x_38286 + x_38287;
                            
                            x_38286 = res_38288;
                        }
                        // write result of operation
                        {
                            *(__local
                              float *) &red_arr_mem_38281[local_tid_34953 * 4] =
                                x_38286;
                        }
                    }
                    skip_waves_38295 *= 2;
                }
                // and back to memory with the final result
                {
                    if (local_tid_34953 == 0) {
                        *(__global float *) &mem_37636[gtid_34929 * 4] =
                            x_38286;
                    }
                }
            }
        }
    }
}
__kernel void segred_large_34977(int32_t sizze_30757, int32_t sizze_30758,
                                 int32_t n_30761, int32_t num_groups_34993,
                                 __global unsigned char *images_mem_37201,
                                 __global unsigned char *mem_37633,
                                 int32_t thread_per_segment_38241, __global
                                 unsigned char *group_res_arr_mem_38242,
                                 __global unsigned char *counter_mem_38244)
{
    const int32_t group_sizze_34983 = mainzigroup_sizze_34959;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    ALIGNED_LOCAL_MEMORY(red_arr_mem_38246_backing_0, 4 *
                         mainzigroup_sizze_34959);
    ALIGNED_LOCAL_MEMORY(sync_arr_mem_38248_backing_1, 1);
    
    int32_t global_tid_34977;
    int32_t local_tid_34978;
    int32_t group_sizze_38238;
    int32_t wave_sizze_38237;
    int32_t group_id_34979;
    
    global_tid_34977 = get_global_id(0);
    local_tid_34978 = get_local_id(0);
    group_sizze_38238 = get_local_size(0);
    wave_sizze_38237 = LOCKSTEP_WIDTH;
    group_id_34979 = get_group_id(0);
    
    int32_t gtid_34955;
    int32_t gtid_34976;
    __local char *red_arr_mem_38246;
    
    red_arr_mem_38246 = (__local char *) red_arr_mem_38246_backing_0;
    
    __local char *sync_arr_mem_38248;
    
    sync_arr_mem_38248 = (__local char *) sync_arr_mem_38248_backing_1;
    gtid_34955 = squot32(group_id_34979, squot32(num_groups_34993 + smax32(1,
                                                                           sizze_30757) -
                                                 1, smax32(1, sizze_30757)));
    
    int32_t chunk_sizze_38250 = smin32(squot32(n_30761 + group_sizze_34983 *
                                               squot32(num_groups_34993 +
                                                       smax32(1, sizze_30757) -
                                                       1, smax32(1,
                                                                 sizze_30757)) -
                                               1, group_sizze_34983 *
                                               squot32(num_groups_34993 +
                                                       smax32(1, sizze_30757) -
                                                       1, smax32(1,
                                                                 sizze_30757))),
                                       squot32(n_30761 -
                                               srem32(global_tid_34977,
                                                      group_sizze_34983 *
                                                      squot32(num_groups_34993 +
                                                              smax32(1,
                                                                     sizze_30757) -
                                                              1, smax32(1,
                                                                        sizze_30757))) +
                                               thread_per_segment_38241 - 1,
                                               thread_per_segment_38241));
    int32_t x_34999;
    int32_t x_35000;
    
    x_34999 = 0;
    for (int32_t i_38254 = 0; i_38254 < chunk_sizze_38250; i_38254++) {
        gtid_34976 = srem32(global_tid_34977, group_sizze_34983 *
                            squot32(num_groups_34993 + smax32(1, sizze_30757) -
                                    1, smax32(1, sizze_30757))) +
            thread_per_segment_38241 * i_38254;
        // apply map function
        {
            float x_35003;
            bool res_35004;
            bool cond_35005;
            int32_t res_35006;
            
            x_35003 = *(__global float *) &images_mem_37201[(gtid_34955 *
                                                             sizze_30758 +
                                                             gtid_34976) * 4];
            res_35004 = futrts_isnan32(x_35003);
            cond_35005 = !res_35004;
            if (cond_35005) {
                res_35006 = 1;
            } else {
                res_35006 = 0;
            }
            // save results to be reduced
            {
                x_35000 = res_35006;
            }
            // save map-out results
            { }
            // apply reduction operator
            {
                int32_t res_35001 = x_34999 + x_35000;
                
                x_34999 = res_35001;
            }
        }
    }
    // to reduce current chunk, first store our result to memory
    {
        *(__local int32_t *) &red_arr_mem_38246[local_tid_34978 * 4] = x_34999;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_38255;
    int32_t skip_waves_38256;
    int32_t x_38251;
    int32_t x_38252;
    
    offset_38255 = 0;
    // participating threads read initial accumulator
    {
        if (slt32(local_tid_34978, group_sizze_34983)) {
            x_38251 = *(__local int32_t *) &red_arr_mem_38246[(local_tid_34978 +
                                                               offset_38255) *
                                                              4];
        }
    }
    offset_38255 = 1;
    while (slt32(offset_38255, wave_sizze_38237)) {
        if (slt32(local_tid_34978 + offset_38255, group_sizze_34983) &&
            ((local_tid_34978 - squot32(local_tid_34978, wave_sizze_38237) *
              wave_sizze_38237) & (2 * offset_38255 - 1)) == 0) {
            // read array element
            {
                x_38252 = *(volatile __local
                            int32_t *) &red_arr_mem_38246[(local_tid_34978 +
                                                           offset_38255) * 4];
            }
            // apply reduction operation
            {
                int32_t res_38253 = x_38251 + x_38252;
                
                x_38251 = res_38253;
            }
            // write result of operation
            {
                *(volatile __local
                  int32_t *) &red_arr_mem_38246[local_tid_34978 * 4] = x_38251;
            }
        }
        offset_38255 *= 2;
    }
    skip_waves_38256 = 1;
    while (slt32(skip_waves_38256, squot32(group_sizze_34983 +
                                           wave_sizze_38237 - 1,
                                           wave_sizze_38237))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_38255 = skip_waves_38256 * wave_sizze_38237;
        if (slt32(local_tid_34978 + offset_38255, group_sizze_34983) &&
            ((local_tid_34978 - squot32(local_tid_34978, wave_sizze_38237) *
              wave_sizze_38237) == 0 && (squot32(local_tid_34978,
                                                 wave_sizze_38237) & (2 *
                                                                      skip_waves_38256 -
                                                                      1)) ==
             0)) {
            // read array element
            {
                x_38252 = *(__local
                            int32_t *) &red_arr_mem_38246[(local_tid_34978 +
                                                           offset_38255) * 4];
            }
            // apply reduction operation
            {
                int32_t res_38253 = x_38251 + x_38252;
                
                x_38251 = res_38253;
            }
            // write result of operation
            {
                *(__local int32_t *) &red_arr_mem_38246[local_tid_34978 * 4] =
                    x_38251;
            }
        }
        skip_waves_38256 *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (squot32(num_groups_34993 + smax32(1, sizze_30757) - 1, smax32(1,
                                                                      sizze_30757)) ==
        1) {
        // first thread in group saves final result to memory
        {
            if (local_tid_34978 == 0) {
                *(__global int32_t *) &mem_37633[gtid_34955 * 4] = x_38251;
            }
        }
    } else {
        int32_t old_counter_38257;
        
        // first thread in group saves group result to memory
        {
            if (local_tid_34978 == 0) {
                *(__global int32_t *) &group_res_arr_mem_38242[group_id_34979 *
                                                               4] = x_38251;
                mem_fence_global();
                old_counter_38257 = atomic_add((volatile __global
                                                int *) &counter_mem_38244[srem32(squot32(group_id_34979,
                                                                                         squot32(num_groups_34993 +
                                                                                                 smax32(1,
                                                                                                        sizze_30757) -
                                                                                                 1,
                                                                                                 smax32(1,
                                                                                                        sizze_30757))),
                                                                                 1024) *
                                                                          4],
                                               1);
                *(__local bool *) &sync_arr_mem_38248[0] = old_counter_38257 ==
                    squot32(num_groups_34993 + smax32(1, sizze_30757) - 1,
                            smax32(1, sizze_30757)) - 1;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        bool is_last_group_38258 = *(__local bool *) &sync_arr_mem_38248[0];
        
        if (is_last_group_38258) {
            if (local_tid_34978 == 0) {
                old_counter_38257 = atomic_add((volatile __global
                                                int *) &counter_mem_38244[srem32(squot32(group_id_34979,
                                                                                         squot32(num_groups_34993 +
                                                                                                 smax32(1,
                                                                                                        sizze_30757) -
                                                                                                 1,
                                                                                                 smax32(1,
                                                                                                        sizze_30757))),
                                                                                 1024) *
                                                                          4],
                                               0 - squot32(num_groups_34993 +
                                                           smax32(1,
                                                                  sizze_30757) -
                                                           1, smax32(1,
                                                                     sizze_30757)));
            }
            // read in the per-group-results
            {
                if (slt32(local_tid_34978, squot32(num_groups_34993 + smax32(1,
                                                                             sizze_30757) -
                                                   1, smax32(1,
                                                             sizze_30757)))) {
                    x_34999 = *(__global
                                int32_t *) &group_res_arr_mem_38242[(squot32(group_id_34979,
                                                                             squot32(num_groups_34993 +
                                                                                     smax32(1,
                                                                                            sizze_30757) -
                                                                                     1,
                                                                                     smax32(1,
                                                                                            sizze_30757))) *
                                                                     squot32(num_groups_34993 +
                                                                             smax32(1,
                                                                                    sizze_30757) -
                                                                             1,
                                                                             smax32(1,
                                                                                    sizze_30757)) +
                                                                     local_tid_34978) *
                                                                    4];
                } else {
                    x_34999 = 0;
                }
                *(__local int32_t *) &red_arr_mem_38246[local_tid_34978 * 4] =
                    x_34999;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // reduce the per-group results
            {
                int32_t offset_38259;
                int32_t skip_waves_38260;
                int32_t x_38251;
                int32_t x_38252;
                
                offset_38259 = 0;
                // participating threads read initial accumulator
                {
                    if (slt32(local_tid_34978, group_sizze_34983)) {
                        x_38251 = *(__local
                                    int32_t *) &red_arr_mem_38246[(local_tid_34978 +
                                                                   offset_38259) *
                                                                  4];
                    }
                }
                offset_38259 = 1;
                while (slt32(offset_38259, wave_sizze_38237)) {
                    if (slt32(local_tid_34978 + offset_38259,
                              group_sizze_34983) && ((local_tid_34978 -
                                                      squot32(local_tid_34978,
                                                              wave_sizze_38237) *
                                                      wave_sizze_38237) & (2 *
                                                                           offset_38259 -
                                                                           1)) ==
                        0) {
                        // read array element
                        {
                            x_38252 = *(volatile __local
                                        int32_t *) &red_arr_mem_38246[(local_tid_34978 +
                                                                       offset_38259) *
                                                                      4];
                        }
                        // apply reduction operation
                        {
                            int32_t res_38253 = x_38251 + x_38252;
                            
                            x_38251 = res_38253;
                        }
                        // write result of operation
                        {
                            *(volatile __local
                              int32_t *) &red_arr_mem_38246[local_tid_34978 *
                                                            4] = x_38251;
                        }
                    }
                    offset_38259 *= 2;
                }
                skip_waves_38260 = 1;
                while (slt32(skip_waves_38260, squot32(group_sizze_34983 +
                                                       wave_sizze_38237 - 1,
                                                       wave_sizze_38237))) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                    offset_38259 = skip_waves_38260 * wave_sizze_38237;
                    if (slt32(local_tid_34978 + offset_38259,
                              group_sizze_34983) && ((local_tid_34978 -
                                                      squot32(local_tid_34978,
                                                              wave_sizze_38237) *
                                                      wave_sizze_38237) == 0 &&
                                                     (squot32(local_tid_34978,
                                                              wave_sizze_38237) &
                                                      (2 * skip_waves_38260 -
                                                       1)) == 0)) {
                        // read array element
                        {
                            x_38252 = *(__local
                                        int32_t *) &red_arr_mem_38246[(local_tid_34978 +
                                                                       offset_38259) *
                                                                      4];
                        }
                        // apply reduction operation
                        {
                            int32_t res_38253 = x_38251 + x_38252;
                            
                            x_38251 = res_38253;
                        }
                        // write result of operation
                        {
                            *(__local
                              int32_t *) &red_arr_mem_38246[local_tid_34978 *
                                                            4] = x_38251;
                        }
                    }
                    skip_waves_38260 *= 2;
                }
                // and back to memory with the final result
                {
                    if (local_tid_34978 == 0) {
                        *(__global int32_t *) &mem_37633[gtid_34955 * 4] =
                            x_38251;
                    }
                }
            }
        }
    }
}
__kernel void segred_large_35208(int32_t sizze_30756, int32_t sizze_30757,
                                 int32_t res_31136, int32_t num_groups_35225,
                                 __global unsigned char *res_mem_37597, __global
                                 unsigned char *res_mem_37646, __global
                                 unsigned char *res_mem_37647, __global
                                 unsigned char *mem_37663,
                                 int32_t thread_per_segment_38345, __global
                                 unsigned char *group_res_arr_mem_38346,
                                 __global unsigned char *counter_mem_38348)
{
    const int32_t group_sizze_35215 = mainzigroup_sizze_35190;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    ALIGNED_LOCAL_MEMORY(red_arr_mem_38350_backing_0, 4 *
                         mainzigroup_sizze_35190);
    ALIGNED_LOCAL_MEMORY(sync_arr_mem_38352_backing_1, 1);
    
    int32_t global_tid_35208;
    int32_t local_tid_35209;
    int32_t group_sizze_38342;
    int32_t wave_sizze_38341;
    int32_t group_id_35210;
    
    global_tid_35208 = get_global_id(0);
    local_tid_35209 = get_local_id(0);
    group_sizze_38342 = get_local_size(0);
    wave_sizze_38341 = LOCKSTEP_WIDTH;
    group_id_35210 = get_group_id(0);
    
    int32_t gtid_35185;
    int32_t gtid_35207;
    __local char *red_arr_mem_38350;
    
    red_arr_mem_38350 = (__local char *) red_arr_mem_38350_backing_0;
    
    __local char *sync_arr_mem_38352;
    
    sync_arr_mem_38352 = (__local char *) sync_arr_mem_38352_backing_1;
    gtid_35185 = squot32(group_id_35210, squot32(num_groups_35225 + smax32(1,
                                                                           sizze_30757) -
                                                 1, smax32(1, sizze_30757)));
    
    int32_t chunk_sizze_38354 = smin32(squot32(res_31136 + group_sizze_35215 *
                                               squot32(num_groups_35225 +
                                                       smax32(1, sizze_30757) -
                                                       1, smax32(1,
                                                                 sizze_30757)) -
                                               1, group_sizze_35215 *
                                               squot32(num_groups_35225 +
                                                       smax32(1, sizze_30757) -
                                                       1, smax32(1,
                                                                 sizze_30757))),
                                       squot32(res_31136 -
                                               srem32(global_tid_35208,
                                                      group_sizze_35215 *
                                                      squot32(num_groups_35225 +
                                                              smax32(1,
                                                                     sizze_30757) -
                                                              1, smax32(1,
                                                                        sizze_30757))) +
                                               thread_per_segment_38345 - 1,
                                               thread_per_segment_38345));
    float x_35231;
    float x_35232;
    
    x_35231 = 0.0F;
    for (int32_t i_38358 = 0; i_38358 < chunk_sizze_38354; i_38358++) {
        gtid_35207 = srem32(global_tid_35208, group_sizze_35215 *
                            squot32(num_groups_35225 + smax32(1, sizze_30757) -
                                    1, smax32(1, sizze_30757))) +
            thread_per_segment_38345 * i_38358;
        // apply map function
        {
            int32_t x_35235;
            int32_t x_35236;
            bool cond_35238;
            float res_35239;
            
            x_35235 = *(__global int32_t *) &res_mem_37647[gtid_35185 * 4];
            x_35236 = *(__global int32_t *) &res_mem_37646[gtid_35185 * 4];
            cond_35238 = slt32(gtid_35207, x_35236);
            if (cond_35238) {
                int32_t x_35240;
                int32_t x_35241;
                int32_t i_35242;
                float res_35243;
                
                x_35240 = gtid_35207 + x_35235;
                x_35241 = x_35240 - x_35236;
                i_35242 = 1 + x_35241;
                res_35243 = *(__global float *) &res_mem_37597[(gtid_35185 *
                                                                sizze_30756 +
                                                                i_35242) * 4];
                res_35239 = res_35243;
            } else {
                res_35239 = 0.0F;
            }
            // save results to be reduced
            {
                x_35232 = res_35239;
            }
            // save map-out results
            { }
            // apply reduction operator
            {
                float res_35233 = x_35231 + x_35232;
                
                x_35231 = res_35233;
            }
        }
    }
    // to reduce current chunk, first store our result to memory
    {
        *(__local float *) &red_arr_mem_38350[local_tid_35209 * 4] = x_35231;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_38359;
    int32_t skip_waves_38360;
    float x_38355;
    float x_38356;
    
    offset_38359 = 0;
    // participating threads read initial accumulator
    {
        if (slt32(local_tid_35209, group_sizze_35215)) {
            x_38355 = *(__local float *) &red_arr_mem_38350[(local_tid_35209 +
                                                             offset_38359) * 4];
        }
    }
    offset_38359 = 1;
    while (slt32(offset_38359, wave_sizze_38341)) {
        if (slt32(local_tid_35209 + offset_38359, group_sizze_35215) &&
            ((local_tid_35209 - squot32(local_tid_35209, wave_sizze_38341) *
              wave_sizze_38341) & (2 * offset_38359 - 1)) == 0) {
            // read array element
            {
                x_38356 = *(volatile __local
                            float *) &red_arr_mem_38350[(local_tid_35209 +
                                                         offset_38359) * 4];
            }
            // apply reduction operation
            {
                float res_38357 = x_38355 + x_38356;
                
                x_38355 = res_38357;
            }
            // write result of operation
            {
                *(volatile __local float *) &red_arr_mem_38350[local_tid_35209 *
                                                               4] = x_38355;
            }
        }
        offset_38359 *= 2;
    }
    skip_waves_38360 = 1;
    while (slt32(skip_waves_38360, squot32(group_sizze_35215 +
                                           wave_sizze_38341 - 1,
                                           wave_sizze_38341))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_38359 = skip_waves_38360 * wave_sizze_38341;
        if (slt32(local_tid_35209 + offset_38359, group_sizze_35215) &&
            ((local_tid_35209 - squot32(local_tid_35209, wave_sizze_38341) *
              wave_sizze_38341) == 0 && (squot32(local_tid_35209,
                                                 wave_sizze_38341) & (2 *
                                                                      skip_waves_38360 -
                                                                      1)) ==
             0)) {
            // read array element
            {
                x_38356 = *(__local
                            float *) &red_arr_mem_38350[(local_tid_35209 +
                                                         offset_38359) * 4];
            }
            // apply reduction operation
            {
                float res_38357 = x_38355 + x_38356;
                
                x_38355 = res_38357;
            }
            // write result of operation
            {
                *(__local float *) &red_arr_mem_38350[local_tid_35209 * 4] =
                    x_38355;
            }
        }
        skip_waves_38360 *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (squot32(num_groups_35225 + smax32(1, sizze_30757) - 1, smax32(1,
                                                                      sizze_30757)) ==
        1) {
        // first thread in group saves final result to memory
        {
            if (local_tid_35209 == 0) {
                *(__global float *) &mem_37663[gtid_35185 * 4] = x_38355;
            }
        }
    } else {
        int32_t old_counter_38361;
        
        // first thread in group saves group result to memory
        {
            if (local_tid_35209 == 0) {
                *(__global float *) &group_res_arr_mem_38346[group_id_35210 *
                                                             4] = x_38355;
                mem_fence_global();
                old_counter_38361 = atomic_add((volatile __global
                                                int *) &counter_mem_38348[srem32(squot32(group_id_35210,
                                                                                         squot32(num_groups_35225 +
                                                                                                 smax32(1,
                                                                                                        sizze_30757) -
                                                                                                 1,
                                                                                                 smax32(1,
                                                                                                        sizze_30757))),
                                                                                 1024) *
                                                                          4],
                                               1);
                *(__local bool *) &sync_arr_mem_38352[0] = old_counter_38361 ==
                    squot32(num_groups_35225 + smax32(1, sizze_30757) - 1,
                            smax32(1, sizze_30757)) - 1;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        bool is_last_group_38362 = *(__local bool *) &sync_arr_mem_38352[0];
        
        if (is_last_group_38362) {
            if (local_tid_35209 == 0) {
                old_counter_38361 = atomic_add((volatile __global
                                                int *) &counter_mem_38348[srem32(squot32(group_id_35210,
                                                                                         squot32(num_groups_35225 +
                                                                                                 smax32(1,
                                                                                                        sizze_30757) -
                                                                                                 1,
                                                                                                 smax32(1,
                                                                                                        sizze_30757))),
                                                                                 1024) *
                                                                          4],
                                               0 - squot32(num_groups_35225 +
                                                           smax32(1,
                                                                  sizze_30757) -
                                                           1, smax32(1,
                                                                     sizze_30757)));
            }
            // read in the per-group-results
            {
                if (slt32(local_tid_35209, squot32(num_groups_35225 + smax32(1,
                                                                             sizze_30757) -
                                                   1, smax32(1,
                                                             sizze_30757)))) {
                    x_35231 = *(__global
                                float *) &group_res_arr_mem_38346[(squot32(group_id_35210,
                                                                           squot32(num_groups_35225 +
                                                                                   smax32(1,
                                                                                          sizze_30757) -
                                                                                   1,
                                                                                   smax32(1,
                                                                                          sizze_30757))) *
                                                                   squot32(num_groups_35225 +
                                                                           smax32(1,
                                                                                  sizze_30757) -
                                                                           1,
                                                                           smax32(1,
                                                                                  sizze_30757)) +
                                                                   local_tid_35209) *
                                                                  4];
                } else {
                    x_35231 = 0.0F;
                }
                *(__local float *) &red_arr_mem_38350[local_tid_35209 * 4] =
                    x_35231;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // reduce the per-group results
            {
                int32_t offset_38363;
                int32_t skip_waves_38364;
                float x_38355;
                float x_38356;
                
                offset_38363 = 0;
                // participating threads read initial accumulator
                {
                    if (slt32(local_tid_35209, group_sizze_35215)) {
                        x_38355 = *(__local
                                    float *) &red_arr_mem_38350[(local_tid_35209 +
                                                                 offset_38363) *
                                                                4];
                    }
                }
                offset_38363 = 1;
                while (slt32(offset_38363, wave_sizze_38341)) {
                    if (slt32(local_tid_35209 + offset_38363,
                              group_sizze_35215) && ((local_tid_35209 -
                                                      squot32(local_tid_35209,
                                                              wave_sizze_38341) *
                                                      wave_sizze_38341) & (2 *
                                                                           offset_38363 -
                                                                           1)) ==
                        0) {
                        // read array element
                        {
                            x_38356 = *(volatile __local
                                        float *) &red_arr_mem_38350[(local_tid_35209 +
                                                                     offset_38363) *
                                                                    4];
                        }
                        // apply reduction operation
                        {
                            float res_38357 = x_38355 + x_38356;
                            
                            x_38355 = res_38357;
                        }
                        // write result of operation
                        {
                            *(volatile __local
                              float *) &red_arr_mem_38350[local_tid_35209 * 4] =
                                x_38355;
                        }
                    }
                    offset_38363 *= 2;
                }
                skip_waves_38364 = 1;
                while (slt32(skip_waves_38364, squot32(group_sizze_35215 +
                                                       wave_sizze_38341 - 1,
                                                       wave_sizze_38341))) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                    offset_38363 = skip_waves_38364 * wave_sizze_38341;
                    if (slt32(local_tid_35209 + offset_38363,
                              group_sizze_35215) && ((local_tid_35209 -
                                                      squot32(local_tid_35209,
                                                              wave_sizze_38341) *
                                                      wave_sizze_38341) == 0 &&
                                                     (squot32(local_tid_35209,
                                                              wave_sizze_38341) &
                                                      (2 * skip_waves_38364 -
                                                       1)) == 0)) {
                        // read array element
                        {
                            x_38356 = *(__local
                                        float *) &red_arr_mem_38350[(local_tid_35209 +
                                                                     offset_38363) *
                                                                    4];
                        }
                        // apply reduction operation
                        {
                            float res_38357 = x_38355 + x_38356;
                            
                            x_38355 = res_38357;
                        }
                        // write result of operation
                        {
                            *(__local
                              float *) &red_arr_mem_38350[local_tid_35209 * 4] =
                                x_38355;
                        }
                    }
                    skip_waves_38364 *= 2;
                }
                // and back to memory with the final result
                {
                    if (local_tid_35209 == 0) {
                        *(__global float *) &mem_37663[gtid_35185 * 4] =
                            x_38355;
                    }
                }
            }
        }
    }
}
__kernel void segred_large_35795(int32_t sizze_30757, int32_t arg_31158,
                                 int32_t num_groups_35964, __global
                                 unsigned char *mem_37668, __global
                                 unsigned char *mem_37700, __global
                                 unsigned char *mem_37703, __global
                                 unsigned char *mem_37707, __global
                                 unsigned char *mem_37709, __global
                                 unsigned char *mem_37712, __global
                                 unsigned char *mem_37715, __global
                                 unsigned char *group_res_arr_mem_38467,
                                 __global
                                 unsigned char *group_res_arr_mem_38469,
                                 __global
                                 unsigned char *group_res_arr_mem_38471,
                                 __global unsigned char *counter_mem_38473)
{
    const int32_t group_sizze_35954 = mainzigroup_sizze_35777;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    ALIGNED_LOCAL_MEMORY(red_arr_mem_38475_backing_0, mainzigroup_sizze_35777);
    ALIGNED_LOCAL_MEMORY(red_arr_mem_38477_backing_1, 4 *
                         mainzigroup_sizze_35777);
    ALIGNED_LOCAL_MEMORY(red_arr_mem_38479_backing_2, 4 *
                         mainzigroup_sizze_35777);
    ALIGNED_LOCAL_MEMORY(sync_arr_mem_38481_backing_3, 1);
    
    int32_t global_tid_35795;
    int32_t local_tid_35796;
    int32_t group_sizze_38463;
    int32_t wave_sizze_38462;
    int32_t group_id_35797;
    
    global_tid_35795 = get_global_id(0);
    local_tid_35796 = get_local_id(0);
    group_sizze_38463 = get_local_size(0);
    wave_sizze_38462 = LOCKSTEP_WIDTH;
    group_id_35797 = get_group_id(0);
    
    int32_t gtid_35771;
    int32_t gtid_35794;
    __local char *red_arr_mem_38475;
    
    red_arr_mem_38475 = (__local char *) red_arr_mem_38475_backing_0;
    
    __local char *red_arr_mem_38477;
    
    red_arr_mem_38477 = (__local char *) red_arr_mem_38477_backing_1;
    
    __local char *red_arr_mem_38479;
    
    red_arr_mem_38479 = (__local char *) red_arr_mem_38479_backing_2;
    
    __local char *sync_arr_mem_38481;
    
    sync_arr_mem_38481 = (__local char *) sync_arr_mem_38481_backing_3;
    gtid_35771 = squot32(group_id_35797, squot32(num_groups_35964 + smax32(1,
                                                                           sizze_30757) -
                                                 1, smax32(1, sizze_30757)));
    
    int32_t chunk_sizze_38483;
    int32_t starting_point_38484 = srem32(global_tid_35795, group_sizze_35954 *
                                          squot32(num_groups_35964 + smax32(1,
                                                                            sizze_30757) -
                                                  1, smax32(1, sizze_30757))) *
            squot32(arg_31158 + group_sizze_35954 * squot32(num_groups_35964 +
                                                            smax32(1,
                                                                   sizze_30757) -
                                                            1, smax32(1,
                                                                      sizze_30757)) -
                    1, group_sizze_35954 * squot32(num_groups_35964 + smax32(1,
                                                                             sizze_30757) -
                                                   1, smax32(1, sizze_30757)));
    int32_t remaining_elements_38485 = arg_31158 - starting_point_38484;
    
    if (sle32(remaining_elements_38485, 0) || sle32(arg_31158,
                                                    starting_point_38484)) {
        chunk_sizze_38483 = 0;
    } else {
        if (slt32(arg_31158, (srem32(global_tid_35795, group_sizze_35954 *
                                     squot32(num_groups_35964 + smax32(1,
                                                                       sizze_30757) -
                                             1, smax32(1, sizze_30757))) + 1) *
                  squot32(arg_31158 + group_sizze_35954 *
                          squot32(num_groups_35964 + smax32(1, sizze_30757) - 1,
                                  smax32(1, sizze_30757)) - 1,
                          group_sizze_35954 * squot32(num_groups_35964 +
                                                      smax32(1, sizze_30757) -
                                                      1, smax32(1,
                                                                sizze_30757))))) {
            chunk_sizze_38483 = arg_31158 - srem32(global_tid_35795,
                                                   group_sizze_35954 *
                                                   squot32(num_groups_35964 +
                                                           smax32(1,
                                                                  sizze_30757) -
                                                           1, smax32(1,
                                                                     sizze_30757))) *
                squot32(arg_31158 + group_sizze_35954 *
                        squot32(num_groups_35964 + smax32(1, sizze_30757) - 1,
                                smax32(1, sizze_30757)) - 1, group_sizze_35954 *
                        squot32(num_groups_35964 + smax32(1, sizze_30757) - 1,
                                smax32(1, sizze_30757)));
        } else {
            chunk_sizze_38483 = squot32(arg_31158 + group_sizze_35954 *
                                        squot32(num_groups_35964 + smax32(1,
                                                                          sizze_30757) -
                                                1, smax32(1, sizze_30757)) - 1,
                                        group_sizze_35954 *
                                        squot32(num_groups_35964 + smax32(1,
                                                                          sizze_30757) -
                                                1, smax32(1, sizze_30757)));
        }
    }
    
    bool x_35972;
    int32_t x_35973;
    float x_35974;
    bool x_35975;
    int32_t x_35976;
    float x_35977;
    
    x_35972 = 0;
    x_35973 = -1;
    x_35974 = 0.0F;
    for (int32_t i_38499 = 0; i_38499 < squot32(arg_31158 + group_sizze_35954 *
                                                squot32(num_groups_35964 +
                                                        smax32(1, sizze_30757) -
                                                        1, smax32(1,
                                                                  sizze_30757)) -
                                                1, group_sizze_35954 *
                                                squot32(num_groups_35964 +
                                                        smax32(1, sizze_30757) -
                                                        1, smax32(1,
                                                                  sizze_30757)));
         i_38499++) {
        gtid_35794 = local_tid_35796 + (squot32(srem32(global_tid_35795,
                                                       group_sizze_35954 *
                                                       squot32(num_groups_35964 +
                                                               smax32(1,
                                                                      sizze_30757) -
                                                               1, smax32(1,
                                                                         sizze_30757))),
                                                group_sizze_35954) *
                                        squot32(arg_31158 + group_sizze_35954 *
                                                squot32(num_groups_35964 +
                                                        smax32(1, sizze_30757) -
                                                        1, smax32(1,
                                                                  sizze_30757)) -
                                                1, group_sizze_35954 *
                                                squot32(num_groups_35964 +
                                                        smax32(1, sizze_30757) -
                                                        1, smax32(1,
                                                                  sizze_30757))) +
                                        i_38499) * group_sizze_35954;
        if (slt32(gtid_35794, arg_31158)) {
            // apply map function
            {
                int32_t y_35985;
                float y_35986;
                float x_35990;
                float x_35991;
                float res_35994;
                bool cond_35995;
                bool res_35996;
                bool res_35997;
                bool x_35998;
                float res_35999;
                bool res_36000;
                bool x_36001;
                float res_36002;
                
                y_35985 = *(__global int32_t *) &mem_37703[gtid_35771 * 4];
                y_35986 = *(__global float *) &mem_37700[gtid_35771 * 4];
                x_35990 = *(__global float *) &mem_37707[(gtid_35771 *
                                                          arg_31158 +
                                                          gtid_35794) * 4];
                x_35991 = *(__global float *) &mem_37668[gtid_35794 * 4];
                res_35994 = x_35990 / y_35986;
                cond_35995 = slt32(gtid_35794, y_35985);
                res_35996 = futrts_isnan32(res_35994);
                res_35997 = !res_35996;
                x_35998 = cond_35995 && res_35997;
                res_35999 = (float) fabs(res_35994);
                res_36000 = x_35991 < res_35999;
                x_36001 = x_35998 && res_36000;
                if (cond_35995) {
                    res_36002 = res_35994;
                } else {
                    res_36002 = 0.0F;
                }
                // save results to be reduced
                {
                    x_35975 = x_36001;
                    x_35976 = gtid_35794;
                    x_35977 = res_36002;
                }
                // save map-out results
                { }
                // apply reduction operator
                {
                    bool res_35978;
                    int32_t res_35979;
                    float res_35984;
                    
                    if (x_35972) {
                        res_35978 = x_35972;
                        res_35979 = x_35973;
                    } else {
                        bool x_35980;
                        bool y_35981;
                        bool res_35982;
                        int32_t res_35983;
                        
                        x_35980 = !x_35975;
                        y_35981 = x_35972 && x_35980;
                        res_35982 = x_35975 || y_35981;
                        if (x_35975) {
                            res_35983 = x_35976;
                        } else {
                            res_35983 = x_35973;
                        }
                        res_35978 = res_35982;
                        res_35979 = res_35983;
                    }
                    res_35984 = x_35974 + x_35977;
                    x_35972 = res_35978;
                    x_35973 = res_35979;
                    x_35974 = res_35984;
                }
            }
        }
        // to reduce current chunk, first store our result to memory
        {
            *(__local bool *) &red_arr_mem_38475[local_tid_35796] = x_35972;
            *(__local int32_t *) &red_arr_mem_38477[local_tid_35796 * 4] =
                x_35973;
            *(__local float *) &red_arr_mem_38479[local_tid_35796 * 4] =
                x_35974;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t offset_38500;
        int32_t skip_waves_38501;
        bool x_38486;
        int32_t x_38487;
        float x_38488;
        bool x_38489;
        int32_t x_38490;
        float x_38491;
        
        offset_38500 = 0;
        // participating threads read initial accumulator
        {
            if (slt32(local_tid_35796, group_sizze_35954)) {
                x_38486 = *(__local bool *) &red_arr_mem_38475[local_tid_35796 +
                                                               offset_38500];
                x_38487 = *(__local
                            int32_t *) &red_arr_mem_38477[(local_tid_35796 +
                                                           offset_38500) * 4];
                x_38488 = *(__local
                            float *) &red_arr_mem_38479[(local_tid_35796 +
                                                         offset_38500) * 4];
            }
        }
        offset_38500 = 1;
        while (slt32(offset_38500, wave_sizze_38462)) {
            if (slt32(local_tid_35796 + offset_38500, group_sizze_35954) &&
                ((local_tid_35796 - squot32(local_tid_35796, wave_sizze_38462) *
                  wave_sizze_38462) & (2 * offset_38500 - 1)) == 0) {
                // read array element
                {
                    x_38489 = *(volatile __local
                                bool *) &red_arr_mem_38475[local_tid_35796 +
                                                           offset_38500];
                    x_38490 = *(volatile __local
                                int32_t *) &red_arr_mem_38477[(local_tid_35796 +
                                                               offset_38500) *
                                                              4];
                    x_38491 = *(volatile __local
                                float *) &red_arr_mem_38479[(local_tid_35796 +
                                                             offset_38500) * 4];
                }
                // apply reduction operation
                {
                    bool res_38492;
                    int32_t res_38493;
                    float res_38498;
                    
                    if (x_38486) {
                        res_38492 = x_38486;
                        res_38493 = x_38487;
                    } else {
                        bool x_38494;
                        bool y_38495;
                        bool res_38496;
                        int32_t res_38497;
                        
                        x_38494 = !x_38489;
                        y_38495 = x_38486 && x_38494;
                        res_38496 = x_38489 || y_38495;
                        if (x_38489) {
                            res_38497 = x_38490;
                        } else {
                            res_38497 = x_38487;
                        }
                        res_38492 = res_38496;
                        res_38493 = res_38497;
                    }
                    res_38498 = x_38488 + x_38491;
                    x_38486 = res_38492;
                    x_38487 = res_38493;
                    x_38488 = res_38498;
                }
                // write result of operation
                {
                    *(volatile __local
                      bool *) &red_arr_mem_38475[local_tid_35796] = x_38486;
                    *(volatile __local
                      int32_t *) &red_arr_mem_38477[local_tid_35796 * 4] =
                        x_38487;
                    *(volatile __local
                      float *) &red_arr_mem_38479[local_tid_35796 * 4] =
                        x_38488;
                }
            }
            offset_38500 *= 2;
        }
        skip_waves_38501 = 1;
        while (slt32(skip_waves_38501, squot32(group_sizze_35954 +
                                               wave_sizze_38462 - 1,
                                               wave_sizze_38462))) {
            barrier(CLK_LOCAL_MEM_FENCE);
            offset_38500 = skip_waves_38501 * wave_sizze_38462;
            if (slt32(local_tid_35796 + offset_38500, group_sizze_35954) &&
                ((local_tid_35796 - squot32(local_tid_35796, wave_sizze_38462) *
                  wave_sizze_38462) == 0 && (squot32(local_tid_35796,
                                                     wave_sizze_38462) & (2 *
                                                                          skip_waves_38501 -
                                                                          1)) ==
                 0)) {
                // read array element
                {
                    x_38489 = *(__local
                                bool *) &red_arr_mem_38475[local_tid_35796 +
                                                           offset_38500];
                    x_38490 = *(__local
                                int32_t *) &red_arr_mem_38477[(local_tid_35796 +
                                                               offset_38500) *
                                                              4];
                    x_38491 = *(__local
                                float *) &red_arr_mem_38479[(local_tid_35796 +
                                                             offset_38500) * 4];
                }
                // apply reduction operation
                {
                    bool res_38492;
                    int32_t res_38493;
                    float res_38498;
                    
                    if (x_38486) {
                        res_38492 = x_38486;
                        res_38493 = x_38487;
                    } else {
                        bool x_38494;
                        bool y_38495;
                        bool res_38496;
                        int32_t res_38497;
                        
                        x_38494 = !x_38489;
                        y_38495 = x_38486 && x_38494;
                        res_38496 = x_38489 || y_38495;
                        if (x_38489) {
                            res_38497 = x_38490;
                        } else {
                            res_38497 = x_38487;
                        }
                        res_38492 = res_38496;
                        res_38493 = res_38497;
                    }
                    res_38498 = x_38488 + x_38491;
                    x_38486 = res_38492;
                    x_38487 = res_38493;
                    x_38488 = res_38498;
                }
                // write result of operation
                {
                    *(__local bool *) &red_arr_mem_38475[local_tid_35796] =
                        x_38486;
                    *(__local int32_t *) &red_arr_mem_38477[local_tid_35796 *
                                                            4] = x_38487;
                    *(__local float *) &red_arr_mem_38479[local_tid_35796 * 4] =
                        x_38488;
                }
            }
            skip_waves_38501 *= 2;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // first thread takes carry-out; others neutral element
        {
            if (local_tid_35796 == 0) {
                x_35972 = x_38486;
                x_35973 = x_38487;
                x_35974 = x_38488;
            } else {
                x_35972 = 0;
                x_35973 = -1;
                x_35974 = 0.0F;
            }
        }
    }
    if (squot32(num_groups_35964 + smax32(1, sizze_30757) - 1, smax32(1,
                                                                      sizze_30757)) ==
        1) {
        // first thread in group saves final result to memory
        {
            if (local_tid_35796 == 0) {
                *(__global bool *) &mem_37709[gtid_35771] = x_35972;
                *(__global int32_t *) &mem_37712[gtid_35771 * 4] = x_35973;
                *(__global float *) &mem_37715[gtid_35771 * 4] = x_35974;
            }
        }
    } else {
        int32_t old_counter_38502;
        
        // first thread in group saves group result to memory
        {
            if (local_tid_35796 == 0) {
                *(__global bool *) &group_res_arr_mem_38467[group_id_35797] =
                    x_35972;
                *(__global int32_t *) &group_res_arr_mem_38469[group_id_35797 *
                                                               4] = x_35973;
                *(__global float *) &group_res_arr_mem_38471[group_id_35797 *
                                                             4] = x_35974;
                mem_fence_global();
                old_counter_38502 = atomic_add((volatile __global
                                                int *) &counter_mem_38473[srem32(squot32(group_id_35797,
                                                                                         squot32(num_groups_35964 +
                                                                                                 smax32(1,
                                                                                                        sizze_30757) -
                                                                                                 1,
                                                                                                 smax32(1,
                                                                                                        sizze_30757))),
                                                                                 1024) *
                                                                          4],
                                               1);
                *(__local bool *) &sync_arr_mem_38481[0] = old_counter_38502 ==
                    squot32(num_groups_35964 + smax32(1, sizze_30757) - 1,
                            smax32(1, sizze_30757)) - 1;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        bool is_last_group_38503 = *(__local bool *) &sync_arr_mem_38481[0];
        
        if (is_last_group_38503) {
            if (local_tid_35796 == 0) {
                old_counter_38502 = atomic_add((volatile __global
                                                int *) &counter_mem_38473[srem32(squot32(group_id_35797,
                                                                                         squot32(num_groups_35964 +
                                                                                                 smax32(1,
                                                                                                        sizze_30757) -
                                                                                                 1,
                                                                                                 smax32(1,
                                                                                                        sizze_30757))),
                                                                                 1024) *
                                                                          4],
                                               0 - squot32(num_groups_35964 +
                                                           smax32(1,
                                                                  sizze_30757) -
                                                           1, smax32(1,
                                                                     sizze_30757)));
            }
            // read in the per-group-results
            {
                if (slt32(local_tid_35796, squot32(num_groups_35964 + smax32(1,
                                                                             sizze_30757) -
                                                   1, smax32(1,
                                                             sizze_30757)))) {
                    x_35972 = *(__global
                                bool *) &group_res_arr_mem_38467[squot32(group_id_35797,
                                                                         squot32(num_groups_35964 +
                                                                                 smax32(1,
                                                                                        sizze_30757) -
                                                                                 1,
                                                                                 smax32(1,
                                                                                        sizze_30757))) *
                                                                 squot32(num_groups_35964 +
                                                                         smax32(1,
                                                                                sizze_30757) -
                                                                         1,
                                                                         smax32(1,
                                                                                sizze_30757)) +
                                                                 local_tid_35796];
                } else {
                    x_35972 = 0;
                }
                *(__local bool *) &red_arr_mem_38475[local_tid_35796] = x_35972;
                if (slt32(local_tid_35796, squot32(num_groups_35964 + smax32(1,
                                                                             sizze_30757) -
                                                   1, smax32(1,
                                                             sizze_30757)))) {
                    x_35973 = *(__global
                                int32_t *) &group_res_arr_mem_38469[(squot32(group_id_35797,
                                                                             squot32(num_groups_35964 +
                                                                                     smax32(1,
                                                                                            sizze_30757) -
                                                                                     1,
                                                                                     smax32(1,
                                                                                            sizze_30757))) *
                                                                     squot32(num_groups_35964 +
                                                                             smax32(1,
                                                                                    sizze_30757) -
                                                                             1,
                                                                             smax32(1,
                                                                                    sizze_30757)) +
                                                                     local_tid_35796) *
                                                                    4];
                } else {
                    x_35973 = -1;
                }
                *(__local int32_t *) &red_arr_mem_38477[local_tid_35796 * 4] =
                    x_35973;
                if (slt32(local_tid_35796, squot32(num_groups_35964 + smax32(1,
                                                                             sizze_30757) -
                                                   1, smax32(1,
                                                             sizze_30757)))) {
                    x_35974 = *(__global
                                float *) &group_res_arr_mem_38471[(squot32(group_id_35797,
                                                                           squot32(num_groups_35964 +
                                                                                   smax32(1,
                                                                                          sizze_30757) -
                                                                                   1,
                                                                                   smax32(1,
                                                                                          sizze_30757))) *
                                                                   squot32(num_groups_35964 +
                                                                           smax32(1,
                                                                                  sizze_30757) -
                                                                           1,
                                                                           smax32(1,
                                                                                  sizze_30757)) +
                                                                   local_tid_35796) *
                                                                  4];
                } else {
                    x_35974 = 0.0F;
                }
                *(__local float *) &red_arr_mem_38479[local_tid_35796 * 4] =
                    x_35974;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // reduce the per-group results
            {
                int32_t offset_38504;
                int32_t skip_waves_38505;
                bool x_38486;
                int32_t x_38487;
                float x_38488;
                bool x_38489;
                int32_t x_38490;
                float x_38491;
                
                offset_38504 = 0;
                // participating threads read initial accumulator
                {
                    if (slt32(local_tid_35796, group_sizze_35954)) {
                        x_38486 = *(__local
                                    bool *) &red_arr_mem_38475[local_tid_35796 +
                                                               offset_38504];
                        x_38487 = *(__local
                                    int32_t *) &red_arr_mem_38477[(local_tid_35796 +
                                                                   offset_38504) *
                                                                  4];
                        x_38488 = *(__local
                                    float *) &red_arr_mem_38479[(local_tid_35796 +
                                                                 offset_38504) *
                                                                4];
                    }
                }
                offset_38504 = 1;
                while (slt32(offset_38504, wave_sizze_38462)) {
                    if (slt32(local_tid_35796 + offset_38504,
                              group_sizze_35954) && ((local_tid_35796 -
                                                      squot32(local_tid_35796,
                                                              wave_sizze_38462) *
                                                      wave_sizze_38462) & (2 *
                                                                           offset_38504 -
                                                                           1)) ==
                        0) {
                        // read array element
                        {
                            x_38489 = *(volatile __local
                                        bool *) &red_arr_mem_38475[local_tid_35796 +
                                                                   offset_38504];
                            x_38490 = *(volatile __local
                                        int32_t *) &red_arr_mem_38477[(local_tid_35796 +
                                                                       offset_38504) *
                                                                      4];
                            x_38491 = *(volatile __local
                                        float *) &red_arr_mem_38479[(local_tid_35796 +
                                                                     offset_38504) *
                                                                    4];
                        }
                        // apply reduction operation
                        {
                            bool res_38492;
                            int32_t res_38493;
                            float res_38498;
                            
                            if (x_38486) {
                                res_38492 = x_38486;
                                res_38493 = x_38487;
                            } else {
                                bool x_38494;
                                bool y_38495;
                                bool res_38496;
                                int32_t res_38497;
                                
                                x_38494 = !x_38489;
                                y_38495 = x_38486 && x_38494;
                                res_38496 = x_38489 || y_38495;
                                if (x_38489) {
                                    res_38497 = x_38490;
                                } else {
                                    res_38497 = x_38487;
                                }
                                res_38492 = res_38496;
                                res_38493 = res_38497;
                            }
                            res_38498 = x_38488 + x_38491;
                            x_38486 = res_38492;
                            x_38487 = res_38493;
                            x_38488 = res_38498;
                        }
                        // write result of operation
                        {
                            *(volatile __local
                              bool *) &red_arr_mem_38475[local_tid_35796] =
                                x_38486;
                            *(volatile __local
                              int32_t *) &red_arr_mem_38477[local_tid_35796 *
                                                            4] = x_38487;
                            *(volatile __local
                              float *) &red_arr_mem_38479[local_tid_35796 * 4] =
                                x_38488;
                        }
                    }
                    offset_38504 *= 2;
                }
                skip_waves_38505 = 1;
                while (slt32(skip_waves_38505, squot32(group_sizze_35954 +
                                                       wave_sizze_38462 - 1,
                                                       wave_sizze_38462))) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                    offset_38504 = skip_waves_38505 * wave_sizze_38462;
                    if (slt32(local_tid_35796 + offset_38504,
                              group_sizze_35954) && ((local_tid_35796 -
                                                      squot32(local_tid_35796,
                                                              wave_sizze_38462) *
                                                      wave_sizze_38462) == 0 &&
                                                     (squot32(local_tid_35796,
                                                              wave_sizze_38462) &
                                                      (2 * skip_waves_38505 -
                                                       1)) == 0)) {
                        // read array element
                        {
                            x_38489 = *(__local
                                        bool *) &red_arr_mem_38475[local_tid_35796 +
                                                                   offset_38504];
                            x_38490 = *(__local
                                        int32_t *) &red_arr_mem_38477[(local_tid_35796 +
                                                                       offset_38504) *
                                                                      4];
                            x_38491 = *(__local
                                        float *) &red_arr_mem_38479[(local_tid_35796 +
                                                                     offset_38504) *
                                                                    4];
                        }
                        // apply reduction operation
                        {
                            bool res_38492;
                            int32_t res_38493;
                            float res_38498;
                            
                            if (x_38486) {
                                res_38492 = x_38486;
                                res_38493 = x_38487;
                            } else {
                                bool x_38494;
                                bool y_38495;
                                bool res_38496;
                                int32_t res_38497;
                                
                                x_38494 = !x_38489;
                                y_38495 = x_38486 && x_38494;
                                res_38496 = x_38489 || y_38495;
                                if (x_38489) {
                                    res_38497 = x_38490;
                                } else {
                                    res_38497 = x_38487;
                                }
                                res_38492 = res_38496;
                                res_38493 = res_38497;
                            }
                            res_38498 = x_38488 + x_38491;
                            x_38486 = res_38492;
                            x_38487 = res_38493;
                            x_38488 = res_38498;
                        }
                        // write result of operation
                        {
                            *(__local
                              bool *) &red_arr_mem_38475[local_tid_35796] =
                                x_38486;
                            *(__local
                              int32_t *) &red_arr_mem_38477[local_tid_35796 *
                                                            4] = x_38487;
                            *(__local
                              float *) &red_arr_mem_38479[local_tid_35796 * 4] =
                                x_38488;
                        }
                    }
                    skip_waves_38505 *= 2;
                }
                // and back to memory with the final result
                {
                    if (local_tid_35796 == 0) {
                        *(__global bool *) &mem_37709[gtid_35771] = x_38486;
                        *(__global int32_t *) &mem_37712[gtid_35771 * 4] =
                            x_38487;
                        *(__global float *) &mem_37715[gtid_35771 * 4] =
                            x_38488;
                    }
                }
            }
        }
    }
}
__kernel void segred_nonseg_35080(int32_t sizze_30757, int32_t num_groups_35074,
                                  __global unsigned char *res_mem_37646,
                                  __global unsigned char *mem_37651, __global
                                  unsigned char *counter_mem_38300, __global
                                  unsigned char *group_res_arr_mem_38302,
                                  int32_t num_threads_38304)
{
    const int32_t group_sizze_35063 = mainzigroup_sizze_35062;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    ALIGNED_LOCAL_MEMORY(red_arr_mem_38305_backing_0, 4 *
                         mainzigroup_sizze_35062);
    ALIGNED_LOCAL_MEMORY(sync_arr_mem_38307_backing_1, 1);
    
    int32_t global_tid_35080;
    int32_t local_tid_35081;
    int32_t group_sizze_38299;
    int32_t wave_sizze_38298;
    int32_t group_id_35082;
    
    global_tid_35080 = get_global_id(0);
    local_tid_35081 = get_local_id(0);
    group_sizze_38299 = get_local_size(0);
    wave_sizze_38298 = LOCKSTEP_WIDTH;
    group_id_35082 = get_group_id(0);
    
    int32_t dummy_35060;
    int32_t gtid_35079;
    __local char *red_arr_mem_38305;
    
    red_arr_mem_38305 = (__local char *) red_arr_mem_38305_backing_0;
    
    __local char *sync_arr_mem_38307;
    
    sync_arr_mem_38307 = (__local char *) sync_arr_mem_38307_backing_1;
    dummy_35060 = 0;
    
    int32_t chunk_sizze_38309 = smin32(squot32(sizze_30757 + group_sizze_35063 *
                                               num_groups_35074 - 1,
                                               group_sizze_35063 *
                                               num_groups_35074),
                                       squot32(sizze_30757 - global_tid_35080 +
                                               num_threads_38304 - 1,
                                               num_threads_38304));
    int32_t x_31137;
    int32_t x_31138;
    
    x_31137 = 0;
    for (int32_t i_38313 = 0; i_38313 < chunk_sizze_38309; i_38313++) {
        gtid_35079 = global_tid_35080 + num_threads_38304 * i_38313;
        // apply map function
        {
            int32_t x_31140 = *(__global int32_t *) &res_mem_37646[gtid_35079 *
                                                                   4];
            
            // save results to be reduced
            {
                x_31138 = x_31140;
            }
            // save map-out results
            { }
            // apply reduction operator
            {
                int32_t res_31139 = smax32(x_31137, x_31138);
                
                x_31137 = res_31139;
            }
        }
    }
    // to reduce current chunk, first store our result to memory
    {
        *(__local int32_t *) &red_arr_mem_38305[local_tid_35081 * 4] = x_31137;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_38314;
    int32_t skip_waves_38315;
    int32_t x_38310;
    int32_t x_38311;
    
    offset_38314 = 0;
    // participating threads read initial accumulator
    {
        if (slt32(local_tid_35081, group_sizze_35063)) {
            x_38310 = *(__local int32_t *) &red_arr_mem_38305[(local_tid_35081 +
                                                               offset_38314) *
                                                              4];
        }
    }
    offset_38314 = 1;
    while (slt32(offset_38314, wave_sizze_38298)) {
        if (slt32(local_tid_35081 + offset_38314, group_sizze_35063) &&
            ((local_tid_35081 - squot32(local_tid_35081, wave_sizze_38298) *
              wave_sizze_38298) & (2 * offset_38314 - 1)) == 0) {
            // read array element
            {
                x_38311 = *(volatile __local
                            int32_t *) &red_arr_mem_38305[(local_tid_35081 +
                                                           offset_38314) * 4];
            }
            // apply reduction operation
            {
                int32_t res_38312 = smax32(x_38310, x_38311);
                
                x_38310 = res_38312;
            }
            // write result of operation
            {
                *(volatile __local
                  int32_t *) &red_arr_mem_38305[local_tid_35081 * 4] = x_38310;
            }
        }
        offset_38314 *= 2;
    }
    skip_waves_38315 = 1;
    while (slt32(skip_waves_38315, squot32(group_sizze_35063 +
                                           wave_sizze_38298 - 1,
                                           wave_sizze_38298))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_38314 = skip_waves_38315 * wave_sizze_38298;
        if (slt32(local_tid_35081 + offset_38314, group_sizze_35063) &&
            ((local_tid_35081 - squot32(local_tid_35081, wave_sizze_38298) *
              wave_sizze_38298) == 0 && (squot32(local_tid_35081,
                                                 wave_sizze_38298) & (2 *
                                                                      skip_waves_38315 -
                                                                      1)) ==
             0)) {
            // read array element
            {
                x_38311 = *(__local
                            int32_t *) &red_arr_mem_38305[(local_tid_35081 +
                                                           offset_38314) * 4];
            }
            // apply reduction operation
            {
                int32_t res_38312 = smax32(x_38310, x_38311);
                
                x_38310 = res_38312;
            }
            // write result of operation
            {
                *(__local int32_t *) &red_arr_mem_38305[local_tid_35081 * 4] =
                    x_38310;
            }
        }
        skip_waves_38315 *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t old_counter_38316;
    
    // first thread in group saves group result to memory
    {
        if (local_tid_35081 == 0) {
            *(__global int32_t *) &group_res_arr_mem_38302[group_id_35082 * 4] =
                x_38310;
            mem_fence_global();
            old_counter_38316 = atomic_add((volatile __global
                                            int *) &counter_mem_38300[0], 1);
            *(__local bool *) &sync_arr_mem_38307[0] = old_counter_38316 ==
                num_groups_35074 - 1;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    bool is_last_group_38317 = *(__local bool *) &sync_arr_mem_38307[0];
    
    if (is_last_group_38317) {
        if (local_tid_35081 == 0) {
            old_counter_38316 = atomic_add((volatile __global
                                            int *) &counter_mem_38300[0], 0 -
                                           num_groups_35074);
        }
        // read in the per-group-results
        {
            if (slt32(local_tid_35081, num_groups_35074)) {
                x_31137 = *(__global
                            int32_t *) &group_res_arr_mem_38302[local_tid_35081 *
                                                                4];
            } else {
                x_31137 = 0;
            }
            *(__local int32_t *) &red_arr_mem_38305[local_tid_35081 * 4] =
                x_31137;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // reduce the per-group results
        {
            int32_t offset_38318;
            int32_t skip_waves_38319;
            int32_t x_38310;
            int32_t x_38311;
            
            offset_38318 = 0;
            // participating threads read initial accumulator
            {
                if (slt32(local_tid_35081, group_sizze_35063)) {
                    x_38310 = *(__local
                                int32_t *) &red_arr_mem_38305[(local_tid_35081 +
                                                               offset_38318) *
                                                              4];
                }
            }
            offset_38318 = 1;
            while (slt32(offset_38318, wave_sizze_38298)) {
                if (slt32(local_tid_35081 + offset_38318, group_sizze_35063) &&
                    ((local_tid_35081 - squot32(local_tid_35081,
                                                wave_sizze_38298) *
                      wave_sizze_38298) & (2 * offset_38318 - 1)) == 0) {
                    // read array element
                    {
                        x_38311 = *(volatile __local
                                    int32_t *) &red_arr_mem_38305[(local_tid_35081 +
                                                                   offset_38318) *
                                                                  4];
                    }
                    // apply reduction operation
                    {
                        int32_t res_38312 = smax32(x_38310, x_38311);
                        
                        x_38310 = res_38312;
                    }
                    // write result of operation
                    {
                        *(volatile __local
                          int32_t *) &red_arr_mem_38305[local_tid_35081 * 4] =
                            x_38310;
                    }
                }
                offset_38318 *= 2;
            }
            skip_waves_38319 = 1;
            while (slt32(skip_waves_38319, squot32(group_sizze_35063 +
                                                   wave_sizze_38298 - 1,
                                                   wave_sizze_38298))) {
                barrier(CLK_LOCAL_MEM_FENCE);
                offset_38318 = skip_waves_38319 * wave_sizze_38298;
                if (slt32(local_tid_35081 + offset_38318, group_sizze_35063) &&
                    ((local_tid_35081 - squot32(local_tid_35081,
                                                wave_sizze_38298) *
                      wave_sizze_38298) == 0 && (squot32(local_tid_35081,
                                                         wave_sizze_38298) &
                                                 (2 * skip_waves_38319 - 1)) ==
                     0)) {
                    // read array element
                    {
                        x_38311 = *(__local
                                    int32_t *) &red_arr_mem_38305[(local_tid_35081 +
                                                                   offset_38318) *
                                                                  4];
                    }
                    // apply reduction operation
                    {
                        int32_t res_38312 = smax32(x_38310, x_38311);
                        
                        x_38310 = res_38312;
                    }
                    // write result of operation
                    {
                        *(__local
                          int32_t *) &red_arr_mem_38305[local_tid_35081 * 4] =
                            x_38310;
                    }
                }
                skip_waves_38319 *= 2;
            }
            // and back to memory with the final result
            {
                if (local_tid_35081 == 0) {
                    *(__global int32_t *) &mem_37651[0] = x_38310;
                }
            }
        }
    }
}
__kernel void segred_small_32307(int32_t sizze_30756, int32_t sizze_30757,
                                 int32_t sizze_30758, int32_t n_30761,
                                 int32_t res_30780, int32_t num_groups_32604,
                                 __global unsigned char *images_mem_37201,
                                 __global unsigned char *arg_mem_37210, __global
                                 unsigned char *mem_37302, __global
                                 unsigned char *mem_37307,
                                 int32_t segment_sizze_nonzzero_37878)
{
    const int32_t group_sizze_32594 = mainzigroup_sizze_32289;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    ALIGNED_LOCAL_MEMORY(red_arr_mem_37879_backing_0, 4 *
                         mainzigroup_sizze_32289);
    
    int32_t global_tid_32307;
    int32_t local_tid_32308;
    int32_t group_sizze_37877;
    int32_t wave_sizze_37876;
    int32_t group_id_32309;
    
    global_tid_32307 = get_global_id(0);
    local_tid_32308 = get_local_id(0);
    group_sizze_37877 = get_local_size(0);
    wave_sizze_37876 = LOCKSTEP_WIDTH;
    group_id_32309 = get_group_id(0);
    
    int32_t gtid_32276;
    int32_t gtid_32277;
    int32_t gtid_32278;
    int32_t gtid_32306;
    __local char *red_arr_mem_37879;
    
    red_arr_mem_37879 = (__local char *) red_arr_mem_37879_backing_0;
    for (int32_t i_37881 = 0; i_37881 < squot32(squot32(sizze_30757 *
                                                        res_30780 * res_30780 +
                                                        squot32(group_sizze_32594,
                                                                segment_sizze_nonzzero_37878) -
                                                        1,
                                                        squot32(group_sizze_32594,
                                                                segment_sizze_nonzzero_37878)) -
                                                group_id_32309 +
                                                num_groups_32604 - 1,
                                                num_groups_32604); i_37881++) {
        gtid_32276 = squot32(squot32(local_tid_32308,
                                     segment_sizze_nonzzero_37878) +
                             (group_id_32309 + i_37881 * num_groups_32604) *
                             squot32(group_sizze_32594,
                                     segment_sizze_nonzzero_37878), res_30780 *
                             res_30780);
        gtid_32277 = squot32(squot32(local_tid_32308,
                                     segment_sizze_nonzzero_37878) +
                             (group_id_32309 + i_37881 * num_groups_32604) *
                             squot32(group_sizze_32594,
                                     segment_sizze_nonzzero_37878) -
                             squot32(squot32(local_tid_32308,
                                             segment_sizze_nonzzero_37878) +
                                     (group_id_32309 + i_37881 *
                                      num_groups_32604) *
                                     squot32(group_sizze_32594,
                                             segment_sizze_nonzzero_37878),
                                     res_30780 * res_30780) * (res_30780 *
                                                               res_30780),
                             res_30780);
        gtid_32278 = squot32(local_tid_32308, segment_sizze_nonzzero_37878) +
            (group_id_32309 + i_37881 * num_groups_32604) *
            squot32(group_sizze_32594, segment_sizze_nonzzero_37878) -
            squot32(squot32(local_tid_32308, segment_sizze_nonzzero_37878) +
                    (group_id_32309 + i_37881 * num_groups_32604) *
                    squot32(group_sizze_32594, segment_sizze_nonzzero_37878),
                    res_30780 * res_30780) * (res_30780 * res_30780) -
            squot32(squot32(local_tid_32308, segment_sizze_nonzzero_37878) +
                    (group_id_32309 + i_37881 * num_groups_32604) *
                    squot32(group_sizze_32594, segment_sizze_nonzzero_37878) -
                    squot32(squot32(local_tid_32308,
                                    segment_sizze_nonzzero_37878) +
                            (group_id_32309 + i_37881 * num_groups_32604) *
                            squot32(group_sizze_32594,
                                    segment_sizze_nonzzero_37878), res_30780 *
                            res_30780) * (res_30780 * res_30780), res_30780) *
            res_30780;
        gtid_32306 = srem32(local_tid_32308, n_30761);
        // apply map function if in bounds
        {
            if (slt32(0, n_30761) && (((slt32(gtid_32276, sizze_30757) &&
                                        slt32(gtid_32277, res_30780)) &&
                                       slt32(gtid_32278, res_30780)) &&
                                      slt32(local_tid_32308, n_30761 *
                                            squot32(group_sizze_32594,
                                                    segment_sizze_nonzzero_37878)))) {
                float x_32616;
                float x_32617;
                float x_32618;
                float x_32619;
                bool res_32620;
                float y_32621;
                float res_32622;
                
                x_32616 = *(__global float *) &images_mem_37201[(gtid_32276 *
                                                                 sizze_30758 +
                                                                 gtid_32306) *
                                                                4];
                x_32617 = *(__global float *) &arg_mem_37210[(gtid_32277 *
                                                              sizze_30756 +
                                                              gtid_32306) * 4];
                x_32618 = *(__global float *) &mem_37302[(gtid_32278 *
                                                          sizze_30756 +
                                                          gtid_32306) * 4];
                x_32619 = x_32617 * x_32618;
                res_32620 = futrts_isnan32(x_32616);
                if (res_32620) {
                    y_32621 = 0.0F;
                } else {
                    y_32621 = 1.0F;
                }
                res_32622 = x_32619 * y_32621;
                // save results to be reduced
                {
                    *(__local float *) &red_arr_mem_37879[local_tid_32308 * 4] =
                        res_32622;
                }
                // save map-out results
                { }
            } else {
                *(__local float *) &red_arr_mem_37879[local_tid_32308 * 4] =
                    0.0F;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (slt32(0, n_30761)) {
            // perform segmented scan to imitate reduction
            {
                float x_32610;
                float x_32611;
                float x_37882;
                float x_37883;
                int32_t skip_threads_37885;
                
                if (slt32(local_tid_32308, n_30761 * squot32(group_sizze_32594,
                                                             segment_sizze_nonzzero_37878))) {
                    x_32611 = *(volatile __local
                                float *) &red_arr_mem_37879[local_tid_32308 *
                                                            sizeof(float)];
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_37885 = 1;
                    while (slt32(skip_threads_37885, 32)) {
                        if (sle32(skip_threads_37885, local_tid_32308 -
                                  squot32(local_tid_32308, 32) * 32) &&
                            slt32(local_tid_32308, n_30761 *
                                  squot32(group_sizze_32594,
                                          segment_sizze_nonzzero_37878))) {
                            // read operands
                            {
                                x_32610 = *(volatile __local
                                            float *) &red_arr_mem_37879[(local_tid_32308 -
                                                                         skip_threads_37885) *
                                                                        sizeof(float)];
                            }
                            // perform operation
                            {
                                if (!slt32(srem32(local_tid_32308, n_30761),
                                           local_tid_32308 - (local_tid_32308 -
                                                              skip_threads_37885))) {
                                    float res_32612 = x_32610 + x_32611;
                                    
                                    x_32611 = res_32612;
                                }
                            }
                        }
                        if (sle32(wave_sizze_37876, skip_threads_37885)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_37885, local_tid_32308 -
                                  squot32(local_tid_32308, 32) * 32) &&
                            slt32(local_tid_32308, n_30761 *
                                  squot32(group_sizze_32594,
                                          segment_sizze_nonzzero_37878))) {
                            // write result
                            {
                                *(volatile __local
                                  float *) &red_arr_mem_37879[local_tid_32308 *
                                                              sizeof(float)] =
                                    x_32611;
                            }
                        }
                        if (sle32(wave_sizze_37876, skip_threads_37885)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_37885 *= 2;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // last thread of block 'i' writes its result to offset 'i'
                {
                    if ((local_tid_32308 - squot32(local_tid_32308, 32) * 32) ==
                        31 && slt32(local_tid_32308, n_30761 *
                                    squot32(group_sizze_32594,
                                            segment_sizze_nonzzero_37878))) {
                        *(volatile __local
                          float *) &red_arr_mem_37879[squot32(local_tid_32308,
                                                              32) *
                                                      sizeof(float)] = x_32611;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'
                {
                    int32_t skip_threads_37886;
                    
                    if (squot32(local_tid_32308, 32) == 0 &&
                        slt32(local_tid_32308, n_30761 *
                              squot32(group_sizze_32594,
                                      segment_sizze_nonzzero_37878))) {
                        x_37883 = *(volatile __local
                                    float *) &red_arr_mem_37879[local_tid_32308 *
                                                                sizeof(float)];
                    }
                    // in-block scan (hopefully no barriers needed)
                    {
                        skip_threads_37886 = 1;
                        while (slt32(skip_threads_37886, 32)) {
                            if (sle32(skip_threads_37886, local_tid_32308 -
                                      squot32(local_tid_32308, 32) * 32) &&
                                (squot32(local_tid_32308, 32) == 0 &&
                                 slt32(local_tid_32308, n_30761 *
                                       squot32(group_sizze_32594,
                                               segment_sizze_nonzzero_37878)))) {
                                // read operands
                                {
                                    x_37882 = *(volatile __local
                                                float *) &red_arr_mem_37879[(local_tid_32308 -
                                                                             skip_threads_37886) *
                                                                            sizeof(float)];
                                }
                                // perform operation
                                {
                                    if (!slt32(srem32(local_tid_32308 * 32 +
                                                      32 - 1, n_30761),
                                               local_tid_32308 * 32 + 32 - 1 -
                                               ((local_tid_32308 -
                                                 skip_threads_37886) * 32 + 32 -
                                                1))) {
                                        float res_37884 = x_37882 + x_37883;
                                        
                                        x_37883 = res_37884;
                                    }
                                }
                            }
                            if (sle32(wave_sizze_37876, skip_threads_37886)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            if (sle32(skip_threads_37886, local_tid_32308 -
                                      squot32(local_tid_32308, 32) * 32) &&
                                (squot32(local_tid_32308, 32) == 0 &&
                                 slt32(local_tid_32308, n_30761 *
                                       squot32(group_sizze_32594,
                                               segment_sizze_nonzzero_37878)))) {
                                // write result
                                {
                                    *(volatile __local
                                      float *) &red_arr_mem_37879[local_tid_32308 *
                                                                  sizeof(float)] =
                                        x_37883;
                                }
                            }
                            if (sle32(wave_sizze_37876, skip_threads_37886)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            skip_threads_37886 *= 2;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // carry-in for every block except the first
                {
                    if (!(squot32(local_tid_32308, 32) == 0 ||
                          !slt32(local_tid_32308, n_30761 *
                                 squot32(group_sizze_32594,
                                         segment_sizze_nonzzero_37878)))) {
                        // read operands
                        {
                            x_32610 = *(volatile __local
                                        float *) &red_arr_mem_37879[(squot32(local_tid_32308,
                                                                             32) -
                                                                     1) *
                                                                    sizeof(float)];
                        }
                        // perform operation
                        {
                            if (!slt32(srem32(local_tid_32308, n_30761),
                                       local_tid_32308 -
                                       (squot32(local_tid_32308, 32) * 32 -
                                        1))) {
                                float res_32612 = x_32610 + x_32611;
                                
                                x_32611 = res_32612;
                            }
                        }
                        // write final result
                        {
                            *(volatile __local
                              float *) &red_arr_mem_37879[local_tid_32308 *
                                                          sizeof(float)] =
                                x_32611;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // restore correct values for first block
                {
                    if (squot32(local_tid_32308, 32) == 0) {
                        *(volatile __local
                          float *) &red_arr_mem_37879[local_tid_32308 *
                                                      sizeof(float)] = x_32611;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // save final values of segments
        {
            if (slt32((group_id_32309 + i_37881 * num_groups_32604) *
                      squot32(group_sizze_32594, segment_sizze_nonzzero_37878) +
                      local_tid_32308, sizze_30757 * res_30780 * res_30780) &&
                slt32(local_tid_32308, squot32(group_sizze_32594,
                                               segment_sizze_nonzzero_37878))) {
                *(__global float *) &mem_37307[(squot32((group_id_32309 +
                                                         i_37881 *
                                                         num_groups_32604) *
                                                        squot32(group_sizze_32594,
                                                                segment_sizze_nonzzero_37878) +
                                                        local_tid_32308,
                                                        res_30780 * res_30780) *
                                                (res_30780 * res_30780) +
                                                squot32((group_id_32309 +
                                                         i_37881 *
                                                         num_groups_32604) *
                                                        squot32(group_sizze_32594,
                                                                segment_sizze_nonzzero_37878) +
                                                        local_tid_32308 -
                                                        squot32((group_id_32309 +
                                                                 i_37881 *
                                                                 num_groups_32604) *
                                                                squot32(group_sizze_32594,
                                                                        segment_sizze_nonzzero_37878) +
                                                                local_tid_32308,
                                                                res_30780 *
                                                                res_30780) *
                                                        (res_30780 * res_30780),
                                                        res_30780) * res_30780 +
                                                ((group_id_32309 + i_37881 *
                                                  num_groups_32604) *
                                                 squot32(group_sizze_32594,
                                                         segment_sizze_nonzzero_37878) +
                                                 local_tid_32308 -
                                                 squot32((group_id_32309 +
                                                          i_37881 *
                                                          num_groups_32604) *
                                                         squot32(group_sizze_32594,
                                                                 segment_sizze_nonzzero_37878) +
                                                         local_tid_32308,
                                                         res_30780 *
                                                         res_30780) *
                                                 (res_30780 * res_30780) -
                                                 squot32((group_id_32309 +
                                                          i_37881 *
                                                          num_groups_32604) *
                                                         squot32(group_sizze_32594,
                                                                 segment_sizze_nonzzero_37878) +
                                                         local_tid_32308 -
                                                         squot32((group_id_32309 +
                                                                  i_37881 *
                                                                  num_groups_32604) *
                                                                 squot32(group_sizze_32594,
                                                                         segment_sizze_nonzzero_37878) +
                                                                 local_tid_32308,
                                                                 res_30780 *
                                                                 res_30780) *
                                                         (res_30780 *
                                                          res_30780),
                                                         res_30780) *
                                                 res_30780)) * 4] = *(__local
                                                                      float *) &red_arr_mem_37879[((local_tid_32308 +
                                                                                                    1) *
                                                                                                   segment_sizze_nonzzero_37878 -
                                                                                                   1) *
                                                                                                  4];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
__kernel void segred_small_33523(int32_t sizze_30756, int32_t sizze_30757,
                                 int32_t sizze_30758, int32_t n_30761,
                                 int32_t res_30780, int32_t num_groups_33630,
                                 __global unsigned char *images_mem_37201,
                                 __global unsigned char *arg_mem_37210, __global
                                 unsigned char *mem_37389,
                                 int32_t segment_sizze_nonzzero_37963)
{
    const int32_t group_sizze_33620 = mainzigroup_sizze_33505;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    ALIGNED_LOCAL_MEMORY(red_arr_mem_37964_backing_0, 4 *
                         mainzigroup_sizze_33505);
    
    int32_t global_tid_33523;
    int32_t local_tid_33524;
    int32_t group_sizze_37962;
    int32_t wave_sizze_37961;
    int32_t group_id_33525;
    
    global_tid_33523 = get_global_id(0);
    local_tid_33524 = get_local_id(0);
    group_sizze_37962 = get_local_size(0);
    wave_sizze_37961 = LOCKSTEP_WIDTH;
    group_id_33525 = get_group_id(0);
    
    int32_t gtid_33496;
    int32_t gtid_33497;
    int32_t gtid_33522;
    __local char *red_arr_mem_37964;
    
    red_arr_mem_37964 = (__local char *) red_arr_mem_37964_backing_0;
    for (int32_t i_37966 = 0; i_37966 < squot32(squot32(sizze_30757 *
                                                        res_30780 +
                                                        squot32(group_sizze_33620,
                                                                segment_sizze_nonzzero_37963) -
                                                        1,
                                                        squot32(group_sizze_33620,
                                                                segment_sizze_nonzzero_37963)) -
                                                group_id_33525 +
                                                num_groups_33630 - 1,
                                                num_groups_33630); i_37966++) {
        gtid_33496 = squot32(squot32(local_tid_33524,
                                     segment_sizze_nonzzero_37963) +
                             (group_id_33525 + i_37966 * num_groups_33630) *
                             squot32(group_sizze_33620,
                                     segment_sizze_nonzzero_37963), res_30780);
        gtid_33497 = squot32(local_tid_33524, segment_sizze_nonzzero_37963) +
            (group_id_33525 + i_37966 * num_groups_33630) *
            squot32(group_sizze_33620, segment_sizze_nonzzero_37963) -
            squot32(squot32(local_tid_33524, segment_sizze_nonzzero_37963) +
                    (group_id_33525 + i_37966 * num_groups_33630) *
                    squot32(group_sizze_33620, segment_sizze_nonzzero_37963),
                    res_30780) * res_30780;
        gtid_33522 = srem32(local_tid_33524, n_30761);
        // apply map function if in bounds
        {
            if (slt32(0, n_30761) && ((slt32(gtid_33496, sizze_30757) &&
                                       slt32(gtid_33497, res_30780)) &&
                                      slt32(local_tid_33524, n_30761 *
                                            squot32(group_sizze_33620,
                                                    segment_sizze_nonzzero_37963)))) {
                float x_33641;
                float x_33642;
                bool res_33643;
                float res_33644;
                
                x_33641 = *(__global float *) &arg_mem_37210[(gtid_33497 *
                                                              sizze_30756 +
                                                              gtid_33522) * 4];
                x_33642 = *(__global float *) &images_mem_37201[(gtid_33496 *
                                                                 sizze_30758 +
                                                                 gtid_33522) *
                                                                4];
                res_33643 = futrts_isnan32(x_33642);
                if (res_33643) {
                    res_33644 = 0.0F;
                } else {
                    float res_33645 = x_33641 * x_33642;
                    
                    res_33644 = res_33645;
                }
                // save results to be reduced
                {
                    *(__local float *) &red_arr_mem_37964[local_tid_33524 * 4] =
                        res_33644;
                }
                // save map-out results
                { }
            } else {
                *(__local float *) &red_arr_mem_37964[local_tid_33524 * 4] =
                    0.0F;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (slt32(0, n_30761)) {
            // perform segmented scan to imitate reduction
            {
                float x_33636;
                float x_33637;
                float x_37967;
                float x_37968;
                int32_t skip_threads_37970;
                
                if (slt32(local_tid_33524, n_30761 * squot32(group_sizze_33620,
                                                             segment_sizze_nonzzero_37963))) {
                    x_33637 = *(volatile __local
                                float *) &red_arr_mem_37964[local_tid_33524 *
                                                            sizeof(float)];
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_37970 = 1;
                    while (slt32(skip_threads_37970, 32)) {
                        if (sle32(skip_threads_37970, local_tid_33524 -
                                  squot32(local_tid_33524, 32) * 32) &&
                            slt32(local_tid_33524, n_30761 *
                                  squot32(group_sizze_33620,
                                          segment_sizze_nonzzero_37963))) {
                            // read operands
                            {
                                x_33636 = *(volatile __local
                                            float *) &red_arr_mem_37964[(local_tid_33524 -
                                                                         skip_threads_37970) *
                                                                        sizeof(float)];
                            }
                            // perform operation
                            {
                                if (!slt32(srem32(local_tid_33524, n_30761),
                                           local_tid_33524 - (local_tid_33524 -
                                                              skip_threads_37970))) {
                                    float res_33638 = x_33636 + x_33637;
                                    
                                    x_33637 = res_33638;
                                }
                            }
                        }
                        if (sle32(wave_sizze_37961, skip_threads_37970)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_37970, local_tid_33524 -
                                  squot32(local_tid_33524, 32) * 32) &&
                            slt32(local_tid_33524, n_30761 *
                                  squot32(group_sizze_33620,
                                          segment_sizze_nonzzero_37963))) {
                            // write result
                            {
                                *(volatile __local
                                  float *) &red_arr_mem_37964[local_tid_33524 *
                                                              sizeof(float)] =
                                    x_33637;
                            }
                        }
                        if (sle32(wave_sizze_37961, skip_threads_37970)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_37970 *= 2;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // last thread of block 'i' writes its result to offset 'i'
                {
                    if ((local_tid_33524 - squot32(local_tid_33524, 32) * 32) ==
                        31 && slt32(local_tid_33524, n_30761 *
                                    squot32(group_sizze_33620,
                                            segment_sizze_nonzzero_37963))) {
                        *(volatile __local
                          float *) &red_arr_mem_37964[squot32(local_tid_33524,
                                                              32) *
                                                      sizeof(float)] = x_33637;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'
                {
                    int32_t skip_threads_37971;
                    
                    if (squot32(local_tid_33524, 32) == 0 &&
                        slt32(local_tid_33524, n_30761 *
                              squot32(group_sizze_33620,
                                      segment_sizze_nonzzero_37963))) {
                        x_37968 = *(volatile __local
                                    float *) &red_arr_mem_37964[local_tid_33524 *
                                                                sizeof(float)];
                    }
                    // in-block scan (hopefully no barriers needed)
                    {
                        skip_threads_37971 = 1;
                        while (slt32(skip_threads_37971, 32)) {
                            if (sle32(skip_threads_37971, local_tid_33524 -
                                      squot32(local_tid_33524, 32) * 32) &&
                                (squot32(local_tid_33524, 32) == 0 &&
                                 slt32(local_tid_33524, n_30761 *
                                       squot32(group_sizze_33620,
                                               segment_sizze_nonzzero_37963)))) {
                                // read operands
                                {
                                    x_37967 = *(volatile __local
                                                float *) &red_arr_mem_37964[(local_tid_33524 -
                                                                             skip_threads_37971) *
                                                                            sizeof(float)];
                                }
                                // perform operation
                                {
                                    if (!slt32(srem32(local_tid_33524 * 32 +
                                                      32 - 1, n_30761),
                                               local_tid_33524 * 32 + 32 - 1 -
                                               ((local_tid_33524 -
                                                 skip_threads_37971) * 32 + 32 -
                                                1))) {
                                        float res_37969 = x_37967 + x_37968;
                                        
                                        x_37968 = res_37969;
                                    }
                                }
                            }
                            if (sle32(wave_sizze_37961, skip_threads_37971)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            if (sle32(skip_threads_37971, local_tid_33524 -
                                      squot32(local_tid_33524, 32) * 32) &&
                                (squot32(local_tid_33524, 32) == 0 &&
                                 slt32(local_tid_33524, n_30761 *
                                       squot32(group_sizze_33620,
                                               segment_sizze_nonzzero_37963)))) {
                                // write result
                                {
                                    *(volatile __local
                                      float *) &red_arr_mem_37964[local_tid_33524 *
                                                                  sizeof(float)] =
                                        x_37968;
                                }
                            }
                            if (sle32(wave_sizze_37961, skip_threads_37971)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            skip_threads_37971 *= 2;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // carry-in for every block except the first
                {
                    if (!(squot32(local_tid_33524, 32) == 0 ||
                          !slt32(local_tid_33524, n_30761 *
                                 squot32(group_sizze_33620,
                                         segment_sizze_nonzzero_37963)))) {
                        // read operands
                        {
                            x_33636 = *(volatile __local
                                        float *) &red_arr_mem_37964[(squot32(local_tid_33524,
                                                                             32) -
                                                                     1) *
                                                                    sizeof(float)];
                        }
                        // perform operation
                        {
                            if (!slt32(srem32(local_tid_33524, n_30761),
                                       local_tid_33524 -
                                       (squot32(local_tid_33524, 32) * 32 -
                                        1))) {
                                float res_33638 = x_33636 + x_33637;
                                
                                x_33637 = res_33638;
                            }
                        }
                        // write final result
                        {
                            *(volatile __local
                              float *) &red_arr_mem_37964[local_tid_33524 *
                                                          sizeof(float)] =
                                x_33637;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // restore correct values for first block
                {
                    if (squot32(local_tid_33524, 32) == 0) {
                        *(volatile __local
                          float *) &red_arr_mem_37964[local_tid_33524 *
                                                      sizeof(float)] = x_33637;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // save final values of segments
        {
            if (slt32((group_id_33525 + i_37966 * num_groups_33630) *
                      squot32(group_sizze_33620, segment_sizze_nonzzero_37963) +
                      local_tid_33524, sizze_30757 * res_30780) &&
                slt32(local_tid_33524, squot32(group_sizze_33620,
                                               segment_sizze_nonzzero_37963))) {
                *(__global float *) &mem_37389[(squot32((group_id_33525 +
                                                         i_37966 *
                                                         num_groups_33630) *
                                                        squot32(group_sizze_33620,
                                                                segment_sizze_nonzzero_37963) +
                                                        local_tid_33524,
                                                        res_30780) * res_30780 +
                                                ((group_id_33525 + i_37966 *
                                                  num_groups_33630) *
                                                 squot32(group_sizze_33620,
                                                         segment_sizze_nonzzero_37963) +
                                                 local_tid_33524 -
                                                 squot32((group_id_33525 +
                                                          i_37966 *
                                                          num_groups_33630) *
                                                         squot32(group_sizze_33620,
                                                                 segment_sizze_nonzzero_37963) +
                                                         local_tid_33524,
                                                         res_30780) *
                                                 res_30780)) * 4] = *(__local
                                                                      float *) &red_arr_mem_37964[((local_tid_33524 +
                                                                                                    1) *
                                                                                                   segment_sizze_nonzzero_37963 -
                                                                                                   1) *
                                                                                                  4];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
__kernel void segred_small_33855(int32_t sizze_30757, int32_t res_30780,
                                 int32_t j_m_i_30913, int32_t num_groups_33956,
                                 __global unsigned char *res_mem_37344, __global
                                 unsigned char *res_mem_37393, __global
                                 unsigned char *mem_37445,
                                 int32_t segment_sizze_nonzzero_38024)
{
    const int32_t group_sizze_33946 = mainzigroup_sizze_33837;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    ALIGNED_LOCAL_MEMORY(red_arr_mem_38025_backing_0, 4 *
                         mainzigroup_sizze_33837);
    
    int32_t global_tid_33855;
    int32_t local_tid_33856;
    int32_t group_sizze_38023;
    int32_t wave_sizze_38022;
    int32_t group_id_33857;
    
    global_tid_33855 = get_global_id(0);
    local_tid_33856 = get_local_id(0);
    group_sizze_38023 = get_local_size(0);
    wave_sizze_38022 = LOCKSTEP_WIDTH;
    group_id_33857 = get_group_id(0);
    
    int32_t gtid_33829;
    int32_t gtid_33830;
    int32_t gtid_33854;
    __local char *red_arr_mem_38025;
    
    red_arr_mem_38025 = (__local char *) red_arr_mem_38025_backing_0;
    for (int32_t i_38027 = 0; i_38027 < squot32(squot32(sizze_30757 *
                                                        res_30780 +
                                                        squot32(group_sizze_33946,
                                                                segment_sizze_nonzzero_38024) -
                                                        1,
                                                        squot32(group_sizze_33946,
                                                                segment_sizze_nonzzero_38024)) -
                                                group_id_33857 +
                                                num_groups_33956 - 1,
                                                num_groups_33956); i_38027++) {
        gtid_33829 = squot32(squot32(local_tid_33856,
                                     segment_sizze_nonzzero_38024) +
                             (group_id_33857 + i_38027 * num_groups_33956) *
                             squot32(group_sizze_33946,
                                     segment_sizze_nonzzero_38024), res_30780);
        gtid_33830 = squot32(local_tid_33856, segment_sizze_nonzzero_38024) +
            (group_id_33857 + i_38027 * num_groups_33956) *
            squot32(group_sizze_33946, segment_sizze_nonzzero_38024) -
            squot32(squot32(local_tid_33856, segment_sizze_nonzzero_38024) +
                    (group_id_33857 + i_38027 * num_groups_33956) *
                    squot32(group_sizze_33946, segment_sizze_nonzzero_38024),
                    res_30780) * res_30780;
        gtid_33854 = srem32(local_tid_33856, j_m_i_30913);
        // apply map function if in bounds
        {
            if (slt32(0, j_m_i_30913) && ((slt32(gtid_33829, sizze_30757) &&
                                           slt32(gtid_33830, res_30780)) &&
                                          slt32(local_tid_33856, j_m_i_30913 *
                                                squot32(group_sizze_33946,
                                                        segment_sizze_nonzzero_38024)))) {
                int32_t binop_x_36468;
                int32_t binop_x_36469;
                int32_t new_index_36470;
                int32_t binop_y_36476;
                int32_t new_index_36477;
                float x_33968;
                float x_33969;
                float res_33970;
                
                binop_x_36468 = j_m_i_30913 * gtid_33829;
                binop_x_36469 = gtid_33854 + binop_x_36468;
                new_index_36470 = squot32(binop_x_36469, res_30780);
                binop_y_36476 = res_30780 * new_index_36470;
                new_index_36477 = binop_x_36469 - binop_y_36476;
                x_33968 = *(__global float *) &res_mem_37393[(new_index_36470 *
                                                              res_30780 +
                                                              new_index_36477) *
                                                             4];
                x_33969 = *(__global float *) &res_mem_37344[(gtid_33829 *
                                                              (j_m_i_30913 *
                                                               res_30780) +
                                                              gtid_33830 *
                                                              j_m_i_30913 +
                                                              gtid_33854) * 4];
                res_33970 = x_33968 * x_33969;
                // save results to be reduced
                {
                    *(__local float *) &red_arr_mem_38025[local_tid_33856 * 4] =
                        res_33970;
                }
                // save map-out results
                { }
            } else {
                *(__local float *) &red_arr_mem_38025[local_tid_33856 * 4] =
                    0.0F;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (slt32(0, j_m_i_30913)) {
            // perform segmented scan to imitate reduction
            {
                float x_33962;
                float x_33963;
                float x_38028;
                float x_38029;
                int32_t skip_threads_38031;
                
                if (slt32(local_tid_33856, j_m_i_30913 *
                          squot32(group_sizze_33946,
                                  segment_sizze_nonzzero_38024))) {
                    x_33963 = *(volatile __local
                                float *) &red_arr_mem_38025[local_tid_33856 *
                                                            sizeof(float)];
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_38031 = 1;
                    while (slt32(skip_threads_38031, 32)) {
                        if (sle32(skip_threads_38031, local_tid_33856 -
                                  squot32(local_tid_33856, 32) * 32) &&
                            slt32(local_tid_33856, j_m_i_30913 *
                                  squot32(group_sizze_33946,
                                          segment_sizze_nonzzero_38024))) {
                            // read operands
                            {
                                x_33962 = *(volatile __local
                                            float *) &red_arr_mem_38025[(local_tid_33856 -
                                                                         skip_threads_38031) *
                                                                        sizeof(float)];
                            }
                            // perform operation
                            {
                                if (!slt32(srem32(local_tid_33856, j_m_i_30913),
                                           local_tid_33856 - (local_tid_33856 -
                                                              skip_threads_38031))) {
                                    float res_33964 = x_33962 + x_33963;
                                    
                                    x_33963 = res_33964;
                                }
                            }
                        }
                        if (sle32(wave_sizze_38022, skip_threads_38031)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_38031, local_tid_33856 -
                                  squot32(local_tid_33856, 32) * 32) &&
                            slt32(local_tid_33856, j_m_i_30913 *
                                  squot32(group_sizze_33946,
                                          segment_sizze_nonzzero_38024))) {
                            // write result
                            {
                                *(volatile __local
                                  float *) &red_arr_mem_38025[local_tid_33856 *
                                                              sizeof(float)] =
                                    x_33963;
                            }
                        }
                        if (sle32(wave_sizze_38022, skip_threads_38031)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_38031 *= 2;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // last thread of block 'i' writes its result to offset 'i'
                {
                    if ((local_tid_33856 - squot32(local_tid_33856, 32) * 32) ==
                        31 && slt32(local_tid_33856, j_m_i_30913 *
                                    squot32(group_sizze_33946,
                                            segment_sizze_nonzzero_38024))) {
                        *(volatile __local
                          float *) &red_arr_mem_38025[squot32(local_tid_33856,
                                                              32) *
                                                      sizeof(float)] = x_33963;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'
                {
                    int32_t skip_threads_38032;
                    
                    if (squot32(local_tid_33856, 32) == 0 &&
                        slt32(local_tid_33856, j_m_i_30913 *
                              squot32(group_sizze_33946,
                                      segment_sizze_nonzzero_38024))) {
                        x_38029 = *(volatile __local
                                    float *) &red_arr_mem_38025[local_tid_33856 *
                                                                sizeof(float)];
                    }
                    // in-block scan (hopefully no barriers needed)
                    {
                        skip_threads_38032 = 1;
                        while (slt32(skip_threads_38032, 32)) {
                            if (sle32(skip_threads_38032, local_tid_33856 -
                                      squot32(local_tid_33856, 32) * 32) &&
                                (squot32(local_tid_33856, 32) == 0 &&
                                 slt32(local_tid_33856, j_m_i_30913 *
                                       squot32(group_sizze_33946,
                                               segment_sizze_nonzzero_38024)))) {
                                // read operands
                                {
                                    x_38028 = *(volatile __local
                                                float *) &red_arr_mem_38025[(local_tid_33856 -
                                                                             skip_threads_38032) *
                                                                            sizeof(float)];
                                }
                                // perform operation
                                {
                                    if (!slt32(srem32(local_tid_33856 * 32 +
                                                      32 - 1, j_m_i_30913),
                                               local_tid_33856 * 32 + 32 - 1 -
                                               ((local_tid_33856 -
                                                 skip_threads_38032) * 32 + 32 -
                                                1))) {
                                        float res_38030 = x_38028 + x_38029;
                                        
                                        x_38029 = res_38030;
                                    }
                                }
                            }
                            if (sle32(wave_sizze_38022, skip_threads_38032)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            if (sle32(skip_threads_38032, local_tid_33856 -
                                      squot32(local_tid_33856, 32) * 32) &&
                                (squot32(local_tid_33856, 32) == 0 &&
                                 slt32(local_tid_33856, j_m_i_30913 *
                                       squot32(group_sizze_33946,
                                               segment_sizze_nonzzero_38024)))) {
                                // write result
                                {
                                    *(volatile __local
                                      float *) &red_arr_mem_38025[local_tid_33856 *
                                                                  sizeof(float)] =
                                        x_38029;
                                }
                            }
                            if (sle32(wave_sizze_38022, skip_threads_38032)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            skip_threads_38032 *= 2;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // carry-in for every block except the first
                {
                    if (!(squot32(local_tid_33856, 32) == 0 ||
                          !slt32(local_tid_33856, j_m_i_30913 *
                                 squot32(group_sizze_33946,
                                         segment_sizze_nonzzero_38024)))) {
                        // read operands
                        {
                            x_33962 = *(volatile __local
                                        float *) &red_arr_mem_38025[(squot32(local_tid_33856,
                                                                             32) -
                                                                     1) *
                                                                    sizeof(float)];
                        }
                        // perform operation
                        {
                            if (!slt32(srem32(local_tid_33856, j_m_i_30913),
                                       local_tid_33856 -
                                       (squot32(local_tid_33856, 32) * 32 -
                                        1))) {
                                float res_33964 = x_33962 + x_33963;
                                
                                x_33963 = res_33964;
                            }
                        }
                        // write final result
                        {
                            *(volatile __local
                              float *) &red_arr_mem_38025[local_tid_33856 *
                                                          sizeof(float)] =
                                x_33963;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // restore correct values for first block
                {
                    if (squot32(local_tid_33856, 32) == 0) {
                        *(volatile __local
                          float *) &red_arr_mem_38025[local_tid_33856 *
                                                      sizeof(float)] = x_33963;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // save final values of segments
        {
            if (slt32((group_id_33857 + i_38027 * num_groups_33956) *
                      squot32(group_sizze_33946, segment_sizze_nonzzero_38024) +
                      local_tid_33856, sizze_30757 * res_30780) &&
                slt32(local_tid_33856, squot32(group_sizze_33946,
                                               segment_sizze_nonzzero_38024))) {
                *(__global float *) &mem_37445[(squot32((group_id_33857 +
                                                         i_38027 *
                                                         num_groups_33956) *
                                                        squot32(group_sizze_33946,
                                                                segment_sizze_nonzzero_38024) +
                                                        local_tid_33856,
                                                        res_30780) * res_30780 +
                                                ((group_id_33857 + i_38027 *
                                                  num_groups_33956) *
                                                 squot32(group_sizze_33946,
                                                         segment_sizze_nonzzero_38024) +
                                                 local_tid_33856 -
                                                 squot32((group_id_33857 +
                                                          i_38027 *
                                                          num_groups_33956) *
                                                         squot32(group_sizze_33946,
                                                                 segment_sizze_nonzzero_38024) +
                                                         local_tid_33856,
                                                         res_30780) *
                                                 res_30780)) * 4] = *(__local
                                                                      float *) &red_arr_mem_38025[((local_tid_33856 +
                                                                                                    1) *
                                                                                                   segment_sizze_nonzzero_38024 -
                                                                                                   1) *
                                                                                                  4];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
__kernel void segred_small_34174(int32_t sizze_30756, int32_t sizze_30757,
                                 int32_t res_30780, int32_t num_groups_34275,
                                 __global unsigned char *mem_37218, __global
                                 unsigned char *res_mem_37449, __global
                                 unsigned char *mem_37502,
                                 int32_t segment_sizze_nonzzero_38085)
{
    const int32_t group_sizze_34265 = mainzigroup_sizze_34156;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    ALIGNED_LOCAL_MEMORY(red_arr_mem_38086_backing_0, 4 *
                         mainzigroup_sizze_34156);
    
    int32_t global_tid_34174;
    int32_t local_tid_34175;
    int32_t group_sizze_38084;
    int32_t wave_sizze_38083;
    int32_t group_id_34176;
    
    global_tid_34174 = get_global_id(0);
    local_tid_34175 = get_local_id(0);
    group_sizze_38084 = get_local_size(0);
    wave_sizze_38083 = LOCKSTEP_WIDTH;
    group_id_34176 = get_group_id(0);
    
    int32_t gtid_34147;
    int32_t gtid_34148;
    int32_t gtid_34173;
    __local char *red_arr_mem_38086;
    
    red_arr_mem_38086 = (__local char *) red_arr_mem_38086_backing_0;
    for (int32_t i_38088 = 0; i_38088 < squot32(squot32(sizze_30757 *
                                                        sizze_30756 +
                                                        squot32(group_sizze_34265,
                                                                segment_sizze_nonzzero_38085) -
                                                        1,
                                                        squot32(group_sizze_34265,
                                                                segment_sizze_nonzzero_38085)) -
                                                group_id_34176 +
                                                num_groups_34275 - 1,
                                                num_groups_34275); i_38088++) {
        gtid_34147 = squot32(squot32(local_tid_34175,
                                     segment_sizze_nonzzero_38085) +
                             (group_id_34176 + i_38088 * num_groups_34275) *
                             squot32(group_sizze_34265,
                                     segment_sizze_nonzzero_38085),
                             sizze_30756);
        gtid_34148 = squot32(local_tid_34175, segment_sizze_nonzzero_38085) +
            (group_id_34176 + i_38088 * num_groups_34275) *
            squot32(group_sizze_34265, segment_sizze_nonzzero_38085) -
            squot32(squot32(local_tid_34175, segment_sizze_nonzzero_38085) +
                    (group_id_34176 + i_38088 * num_groups_34275) *
                    squot32(group_sizze_34265, segment_sizze_nonzzero_38085),
                    sizze_30756) * sizze_30756;
        gtid_34173 = srem32(local_tid_34175, res_30780);
        // apply map function if in bounds
        {
            if (slt32(0, res_30780) && ((slt32(gtid_34147, sizze_30757) &&
                                         slt32(gtid_34148, sizze_30756)) &&
                                        slt32(local_tid_34175, res_30780 *
                                              squot32(group_sizze_34265,
                                                      segment_sizze_nonzzero_38085)))) {
                float x_34286;
                float x_34287;
                float res_34288;
                
                x_34286 = *(__global float *) &res_mem_37449[(gtid_34147 *
                                                              res_30780 +
                                                              gtid_34173) * 4];
                x_34287 = *(__global float *) &mem_37218[(gtid_34148 *
                                                          res_30780 +
                                                          gtid_34173) * 4];
                res_34288 = x_34286 * x_34287;
                // save results to be reduced
                {
                    *(__local float *) &red_arr_mem_38086[local_tid_34175 * 4] =
                        res_34288;
                }
                // save map-out results
                { }
            } else {
                *(__local float *) &red_arr_mem_38086[local_tid_34175 * 4] =
                    0.0F;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (slt32(0, res_30780)) {
            // perform segmented scan to imitate reduction
            {
                float x_34281;
                float x_34282;
                float x_38089;
                float x_38090;
                int32_t skip_threads_38092;
                
                if (slt32(local_tid_34175, res_30780 *
                          squot32(group_sizze_34265,
                                  segment_sizze_nonzzero_38085))) {
                    x_34282 = *(volatile __local
                                float *) &red_arr_mem_38086[local_tid_34175 *
                                                            sizeof(float)];
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_38092 = 1;
                    while (slt32(skip_threads_38092, 32)) {
                        if (sle32(skip_threads_38092, local_tid_34175 -
                                  squot32(local_tid_34175, 32) * 32) &&
                            slt32(local_tid_34175, res_30780 *
                                  squot32(group_sizze_34265,
                                          segment_sizze_nonzzero_38085))) {
                            // read operands
                            {
                                x_34281 = *(volatile __local
                                            float *) &red_arr_mem_38086[(local_tid_34175 -
                                                                         skip_threads_38092) *
                                                                        sizeof(float)];
                            }
                            // perform operation
                            {
                                if (!slt32(srem32(local_tid_34175, res_30780),
                                           local_tid_34175 - (local_tid_34175 -
                                                              skip_threads_38092))) {
                                    float res_34283 = x_34281 + x_34282;
                                    
                                    x_34282 = res_34283;
                                }
                            }
                        }
                        if (sle32(wave_sizze_38083, skip_threads_38092)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_38092, local_tid_34175 -
                                  squot32(local_tid_34175, 32) * 32) &&
                            slt32(local_tid_34175, res_30780 *
                                  squot32(group_sizze_34265,
                                          segment_sizze_nonzzero_38085))) {
                            // write result
                            {
                                *(volatile __local
                                  float *) &red_arr_mem_38086[local_tid_34175 *
                                                              sizeof(float)] =
                                    x_34282;
                            }
                        }
                        if (sle32(wave_sizze_38083, skip_threads_38092)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_38092 *= 2;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // last thread of block 'i' writes its result to offset 'i'
                {
                    if ((local_tid_34175 - squot32(local_tid_34175, 32) * 32) ==
                        31 && slt32(local_tid_34175, res_30780 *
                                    squot32(group_sizze_34265,
                                            segment_sizze_nonzzero_38085))) {
                        *(volatile __local
                          float *) &red_arr_mem_38086[squot32(local_tid_34175,
                                                              32) *
                                                      sizeof(float)] = x_34282;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'
                {
                    int32_t skip_threads_38093;
                    
                    if (squot32(local_tid_34175, 32) == 0 &&
                        slt32(local_tid_34175, res_30780 *
                              squot32(group_sizze_34265,
                                      segment_sizze_nonzzero_38085))) {
                        x_38090 = *(volatile __local
                                    float *) &red_arr_mem_38086[local_tid_34175 *
                                                                sizeof(float)];
                    }
                    // in-block scan (hopefully no barriers needed)
                    {
                        skip_threads_38093 = 1;
                        while (slt32(skip_threads_38093, 32)) {
                            if (sle32(skip_threads_38093, local_tid_34175 -
                                      squot32(local_tid_34175, 32) * 32) &&
                                (squot32(local_tid_34175, 32) == 0 &&
                                 slt32(local_tid_34175, res_30780 *
                                       squot32(group_sizze_34265,
                                               segment_sizze_nonzzero_38085)))) {
                                // read operands
                                {
                                    x_38089 = *(volatile __local
                                                float *) &red_arr_mem_38086[(local_tid_34175 -
                                                                             skip_threads_38093) *
                                                                            sizeof(float)];
                                }
                                // perform operation
                                {
                                    if (!slt32(srem32(local_tid_34175 * 32 +
                                                      32 - 1, res_30780),
                                               local_tid_34175 * 32 + 32 - 1 -
                                               ((local_tid_34175 -
                                                 skip_threads_38093) * 32 + 32 -
                                                1))) {
                                        float res_38091 = x_38089 + x_38090;
                                        
                                        x_38090 = res_38091;
                                    }
                                }
                            }
                            if (sle32(wave_sizze_38083, skip_threads_38093)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            if (sle32(skip_threads_38093, local_tid_34175 -
                                      squot32(local_tid_34175, 32) * 32) &&
                                (squot32(local_tid_34175, 32) == 0 &&
                                 slt32(local_tid_34175, res_30780 *
                                       squot32(group_sizze_34265,
                                               segment_sizze_nonzzero_38085)))) {
                                // write result
                                {
                                    *(volatile __local
                                      float *) &red_arr_mem_38086[local_tid_34175 *
                                                                  sizeof(float)] =
                                        x_38090;
                                }
                            }
                            if (sle32(wave_sizze_38083, skip_threads_38093)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            skip_threads_38093 *= 2;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // carry-in for every block except the first
                {
                    if (!(squot32(local_tid_34175, 32) == 0 ||
                          !slt32(local_tid_34175, res_30780 *
                                 squot32(group_sizze_34265,
                                         segment_sizze_nonzzero_38085)))) {
                        // read operands
                        {
                            x_34281 = *(volatile __local
                                        float *) &red_arr_mem_38086[(squot32(local_tid_34175,
                                                                             32) -
                                                                     1) *
                                                                    sizeof(float)];
                        }
                        // perform operation
                        {
                            if (!slt32(srem32(local_tid_34175, res_30780),
                                       local_tid_34175 -
                                       (squot32(local_tid_34175, 32) * 32 -
                                        1))) {
                                float res_34283 = x_34281 + x_34282;
                                
                                x_34282 = res_34283;
                            }
                        }
                        // write final result
                        {
                            *(volatile __local
                              float *) &red_arr_mem_38086[local_tid_34175 *
                                                          sizeof(float)] =
                                x_34282;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // restore correct values for first block
                {
                    if (squot32(local_tid_34175, 32) == 0) {
                        *(volatile __local
                          float *) &red_arr_mem_38086[local_tid_34175 *
                                                      sizeof(float)] = x_34282;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // save final values of segments
        {
            if (slt32((group_id_34176 + i_38088 * num_groups_34275) *
                      squot32(group_sizze_34265, segment_sizze_nonzzero_38085) +
                      local_tid_34175, sizze_30757 * sizze_30756) &&
                slt32(local_tid_34175, squot32(group_sizze_34265,
                                               segment_sizze_nonzzero_38085))) {
                *(__global float *) &mem_37502[(squot32((group_id_34176 +
                                                         i_38088 *
                                                         num_groups_34275) *
                                                        squot32(group_sizze_34265,
                                                                segment_sizze_nonzzero_38085) +
                                                        local_tid_34175,
                                                        sizze_30756) *
                                                sizze_30756 + ((group_id_34176 +
                                                                i_38088 *
                                                                num_groups_34275) *
                                                               squot32(group_sizze_34265,
                                                                       segment_sizze_nonzzero_38085) +
                                                               local_tid_34175 -
                                                               squot32((group_id_34176 +
                                                                        i_38088 *
                                                                        num_groups_34275) *
                                                                       squot32(group_sizze_34265,
                                                                               segment_sizze_nonzzero_38085) +
                                                                       local_tid_34175,
                                                                       sizze_30756) *
                                                               sizze_30756)) *
                                               4] = *(__local
                                                      float *) &red_arr_mem_38086[((local_tid_34175 +
                                                                                    1) *
                                                                                   segment_sizze_nonzzero_38085 -
                                                                                   1) *
                                                                                  4];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
__kernel void segred_small_34952(int32_t sizze_30756, int32_t sizze_30757,
                                 int32_t n_30761, int32_t num_groups_35021,
                                 __global unsigned char *res_mem_37597, __global
                                 unsigned char *mem_37633, __global
                                 unsigned char *mem_37636,
                                 int32_t segment_sizze_nonzzero_38263)
{
    const int32_t group_sizze_35011 = mainzigroup_sizze_34934;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    ALIGNED_LOCAL_MEMORY(red_arr_mem_38264_backing_0, 4 *
                         mainzigroup_sizze_34934);
    
    int32_t global_tid_34952;
    int32_t local_tid_34953;
    int32_t group_sizze_38262;
    int32_t wave_sizze_38261;
    int32_t group_id_34954;
    
    global_tid_34952 = get_global_id(0);
    local_tid_34953 = get_local_id(0);
    group_sizze_38262 = get_local_size(0);
    wave_sizze_38261 = LOCKSTEP_WIDTH;
    group_id_34954 = get_group_id(0);
    
    int32_t gtid_34929;
    int32_t gtid_34951;
    __local char *red_arr_mem_38264;
    
    red_arr_mem_38264 = (__local char *) red_arr_mem_38264_backing_0;
    for (int32_t i_38266 = 0; i_38266 < squot32(squot32(sizze_30757 +
                                                        squot32(group_sizze_35011,
                                                                segment_sizze_nonzzero_38263) -
                                                        1,
                                                        squot32(group_sizze_35011,
                                                                segment_sizze_nonzzero_38263)) -
                                                group_id_34954 +
                                                num_groups_35021 - 1,
                                                num_groups_35021); i_38266++) {
        gtid_34929 = squot32(local_tid_34953, segment_sizze_nonzzero_38263) +
            (group_id_34954 + i_38266 * num_groups_35021) *
            squot32(group_sizze_35011, segment_sizze_nonzzero_38263);
        gtid_34951 = srem32(local_tid_34953, n_30761);
        // apply map function if in bounds
        {
            if (slt32(0, n_30761) && (slt32(gtid_34929, sizze_30757) &&
                                      slt32(local_tid_34953, n_30761 *
                                            squot32(group_sizze_35011,
                                                    segment_sizze_nonzzero_38263)))) {
                int32_t res_35031;
                bool cond_35033;
                float res_35034;
                float res_35036;
                
                res_35031 = *(__global int32_t *) &mem_37633[gtid_34929 * 4];
                cond_35033 = slt32(gtid_34951, res_35031);
                if (cond_35033) {
                    float res_35035 = *(__global
                                        float *) &res_mem_37597[(gtid_34929 *
                                                                 sizze_30756 +
                                                                 gtid_34951) *
                                                                4];
                    
                    res_35034 = res_35035;
                } else {
                    res_35034 = 0.0F;
                }
                res_35036 = res_35034 * res_35034;
                // save results to be reduced
                {
                    *(__local float *) &red_arr_mem_38264[local_tid_34953 * 4] =
                        res_35036;
                }
                // save map-out results
                { }
            } else {
                *(__local float *) &red_arr_mem_38264[local_tid_34953 * 4] =
                    0.0F;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (slt32(0, n_30761)) {
            // perform segmented scan to imitate reduction
            {
                float x_35027;
                float x_35028;
                float x_38267;
                float x_38268;
                int32_t skip_threads_38270;
                
                if (slt32(local_tid_34953, n_30761 * squot32(group_sizze_35011,
                                                             segment_sizze_nonzzero_38263))) {
                    x_35028 = *(volatile __local
                                float *) &red_arr_mem_38264[local_tid_34953 *
                                                            sizeof(float)];
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_38270 = 1;
                    while (slt32(skip_threads_38270, 32)) {
                        if (sle32(skip_threads_38270, local_tid_34953 -
                                  squot32(local_tid_34953, 32) * 32) &&
                            slt32(local_tid_34953, n_30761 *
                                  squot32(group_sizze_35011,
                                          segment_sizze_nonzzero_38263))) {
                            // read operands
                            {
                                x_35027 = *(volatile __local
                                            float *) &red_arr_mem_38264[(local_tid_34953 -
                                                                         skip_threads_38270) *
                                                                        sizeof(float)];
                            }
                            // perform operation
                            {
                                if (!slt32(srem32(local_tid_34953, n_30761),
                                           local_tid_34953 - (local_tid_34953 -
                                                              skip_threads_38270))) {
                                    float res_35029 = x_35027 + x_35028;
                                    
                                    x_35028 = res_35029;
                                }
                            }
                        }
                        if (sle32(wave_sizze_38261, skip_threads_38270)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_38270, local_tid_34953 -
                                  squot32(local_tid_34953, 32) * 32) &&
                            slt32(local_tid_34953, n_30761 *
                                  squot32(group_sizze_35011,
                                          segment_sizze_nonzzero_38263))) {
                            // write result
                            {
                                *(volatile __local
                                  float *) &red_arr_mem_38264[local_tid_34953 *
                                                              sizeof(float)] =
                                    x_35028;
                            }
                        }
                        if (sle32(wave_sizze_38261, skip_threads_38270)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_38270 *= 2;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // last thread of block 'i' writes its result to offset 'i'
                {
                    if ((local_tid_34953 - squot32(local_tid_34953, 32) * 32) ==
                        31 && slt32(local_tid_34953, n_30761 *
                                    squot32(group_sizze_35011,
                                            segment_sizze_nonzzero_38263))) {
                        *(volatile __local
                          float *) &red_arr_mem_38264[squot32(local_tid_34953,
                                                              32) *
                                                      sizeof(float)] = x_35028;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'
                {
                    int32_t skip_threads_38271;
                    
                    if (squot32(local_tid_34953, 32) == 0 &&
                        slt32(local_tid_34953, n_30761 *
                              squot32(group_sizze_35011,
                                      segment_sizze_nonzzero_38263))) {
                        x_38268 = *(volatile __local
                                    float *) &red_arr_mem_38264[local_tid_34953 *
                                                                sizeof(float)];
                    }
                    // in-block scan (hopefully no barriers needed)
                    {
                        skip_threads_38271 = 1;
                        while (slt32(skip_threads_38271, 32)) {
                            if (sle32(skip_threads_38271, local_tid_34953 -
                                      squot32(local_tid_34953, 32) * 32) &&
                                (squot32(local_tid_34953, 32) == 0 &&
                                 slt32(local_tid_34953, n_30761 *
                                       squot32(group_sizze_35011,
                                               segment_sizze_nonzzero_38263)))) {
                                // read operands
                                {
                                    x_38267 = *(volatile __local
                                                float *) &red_arr_mem_38264[(local_tid_34953 -
                                                                             skip_threads_38271) *
                                                                            sizeof(float)];
                                }
                                // perform operation
                                {
                                    if (!slt32(srem32(local_tid_34953 * 32 +
                                                      32 - 1, n_30761),
                                               local_tid_34953 * 32 + 32 - 1 -
                                               ((local_tid_34953 -
                                                 skip_threads_38271) * 32 + 32 -
                                                1))) {
                                        float res_38269 = x_38267 + x_38268;
                                        
                                        x_38268 = res_38269;
                                    }
                                }
                            }
                            if (sle32(wave_sizze_38261, skip_threads_38271)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            if (sle32(skip_threads_38271, local_tid_34953 -
                                      squot32(local_tid_34953, 32) * 32) &&
                                (squot32(local_tid_34953, 32) == 0 &&
                                 slt32(local_tid_34953, n_30761 *
                                       squot32(group_sizze_35011,
                                               segment_sizze_nonzzero_38263)))) {
                                // write result
                                {
                                    *(volatile __local
                                      float *) &red_arr_mem_38264[local_tid_34953 *
                                                                  sizeof(float)] =
                                        x_38268;
                                }
                            }
                            if (sle32(wave_sizze_38261, skip_threads_38271)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            skip_threads_38271 *= 2;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // carry-in for every block except the first
                {
                    if (!(squot32(local_tid_34953, 32) == 0 ||
                          !slt32(local_tid_34953, n_30761 *
                                 squot32(group_sizze_35011,
                                         segment_sizze_nonzzero_38263)))) {
                        // read operands
                        {
                            x_35027 = *(volatile __local
                                        float *) &red_arr_mem_38264[(squot32(local_tid_34953,
                                                                             32) -
                                                                     1) *
                                                                    sizeof(float)];
                        }
                        // perform operation
                        {
                            if (!slt32(srem32(local_tid_34953, n_30761),
                                       local_tid_34953 -
                                       (squot32(local_tid_34953, 32) * 32 -
                                        1))) {
                                float res_35029 = x_35027 + x_35028;
                                
                                x_35028 = res_35029;
                            }
                        }
                        // write final result
                        {
                            *(volatile __local
                              float *) &red_arr_mem_38264[local_tid_34953 *
                                                          sizeof(float)] =
                                x_35028;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // restore correct values for first block
                {
                    if (squot32(local_tid_34953, 32) == 0) {
                        *(volatile __local
                          float *) &red_arr_mem_38264[local_tid_34953 *
                                                      sizeof(float)] = x_35028;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // save final values of segments
        {
            if (slt32((group_id_34954 + i_38266 * num_groups_35021) *
                      squot32(group_sizze_35011, segment_sizze_nonzzero_38263) +
                      local_tid_34953, sizze_30757) && slt32(local_tid_34953,
                                                             squot32(group_sizze_35011,
                                                                     segment_sizze_nonzzero_38263))) {
                *(__global float *) &mem_37636[((group_id_34954 + i_38266 *
                                                 num_groups_35021) *
                                                squot32(group_sizze_35011,
                                                        segment_sizze_nonzzero_38263) +
                                                local_tid_34953) * 4] =
                    *(__local float *) &red_arr_mem_38264[((local_tid_34953 +
                                                            1) *
                                                           segment_sizze_nonzzero_38263 -
                                                           1) * 4];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
__kernel void segred_small_34977(int32_t sizze_30757, int32_t sizze_30758,
                                 int32_t n_30761, int32_t num_groups_34993,
                                 __global unsigned char *images_mem_37201,
                                 __global unsigned char *mem_37633,
                                 int32_t segment_sizze_nonzzero_38228)
{
    const int32_t group_sizze_34983 = mainzigroup_sizze_34959;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    ALIGNED_LOCAL_MEMORY(red_arr_mem_38229_backing_0, 4 *
                         mainzigroup_sizze_34959);
    
    int32_t global_tid_34977;
    int32_t local_tid_34978;
    int32_t group_sizze_38227;
    int32_t wave_sizze_38226;
    int32_t group_id_34979;
    
    global_tid_34977 = get_global_id(0);
    local_tid_34978 = get_local_id(0);
    group_sizze_38227 = get_local_size(0);
    wave_sizze_38226 = LOCKSTEP_WIDTH;
    group_id_34979 = get_group_id(0);
    
    int32_t gtid_34955;
    int32_t gtid_34976;
    __local char *red_arr_mem_38229;
    
    red_arr_mem_38229 = (__local char *) red_arr_mem_38229_backing_0;
    for (int32_t i_38231 = 0; i_38231 < squot32(squot32(sizze_30757 +
                                                        squot32(group_sizze_34983,
                                                                segment_sizze_nonzzero_38228) -
                                                        1,
                                                        squot32(group_sizze_34983,
                                                                segment_sizze_nonzzero_38228)) -
                                                group_id_34979 +
                                                num_groups_34993 - 1,
                                                num_groups_34993); i_38231++) {
        gtid_34955 = squot32(local_tid_34978, segment_sizze_nonzzero_38228) +
            (group_id_34979 + i_38231 * num_groups_34993) *
            squot32(group_sizze_34983, segment_sizze_nonzzero_38228);
        gtid_34976 = srem32(local_tid_34978, n_30761);
        // apply map function if in bounds
        {
            if (slt32(0, n_30761) && (slt32(gtid_34955, sizze_30757) &&
                                      slt32(local_tid_34978, n_30761 *
                                            squot32(group_sizze_34983,
                                                    segment_sizze_nonzzero_38228)))) {
                float x_35003;
                bool res_35004;
                bool cond_35005;
                int32_t res_35006;
                
                x_35003 = *(__global float *) &images_mem_37201[(gtid_34955 *
                                                                 sizze_30758 +
                                                                 gtid_34976) *
                                                                4];
                res_35004 = futrts_isnan32(x_35003);
                cond_35005 = !res_35004;
                if (cond_35005) {
                    res_35006 = 1;
                } else {
                    res_35006 = 0;
                }
                // save results to be reduced
                {
                    *(__local int32_t *) &red_arr_mem_38229[local_tid_34978 *
                                                            4] = res_35006;
                }
                // save map-out results
                { }
            } else {
                *(__local int32_t *) &red_arr_mem_38229[local_tid_34978 * 4] =
                    0;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (slt32(0, n_30761)) {
            // perform segmented scan to imitate reduction
            {
                int32_t x_34999;
                int32_t x_35000;
                int32_t x_38232;
                int32_t x_38233;
                int32_t skip_threads_38235;
                
                if (slt32(local_tid_34978, n_30761 * squot32(group_sizze_34983,
                                                             segment_sizze_nonzzero_38228))) {
                    x_35000 = *(volatile __local
                                int32_t *) &red_arr_mem_38229[local_tid_34978 *
                                                              sizeof(int32_t)];
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_38235 = 1;
                    while (slt32(skip_threads_38235, 32)) {
                        if (sle32(skip_threads_38235, local_tid_34978 -
                                  squot32(local_tid_34978, 32) * 32) &&
                            slt32(local_tid_34978, n_30761 *
                                  squot32(group_sizze_34983,
                                          segment_sizze_nonzzero_38228))) {
                            // read operands
                            {
                                x_34999 = *(volatile __local
                                            int32_t *) &red_arr_mem_38229[(local_tid_34978 -
                                                                           skip_threads_38235) *
                                                                          sizeof(int32_t)];
                            }
                            // perform operation
                            {
                                if (!slt32(srem32(local_tid_34978, n_30761),
                                           local_tid_34978 - (local_tid_34978 -
                                                              skip_threads_38235))) {
                                    int32_t res_35001 = x_34999 + x_35000;
                                    
                                    x_35000 = res_35001;
                                }
                            }
                        }
                        if (sle32(wave_sizze_38226, skip_threads_38235)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_38235, local_tid_34978 -
                                  squot32(local_tid_34978, 32) * 32) &&
                            slt32(local_tid_34978, n_30761 *
                                  squot32(group_sizze_34983,
                                          segment_sizze_nonzzero_38228))) {
                            // write result
                            {
                                *(volatile __local
                                  int32_t *) &red_arr_mem_38229[local_tid_34978 *
                                                                sizeof(int32_t)] =
                                    x_35000;
                            }
                        }
                        if (sle32(wave_sizze_38226, skip_threads_38235)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_38235 *= 2;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // last thread of block 'i' writes its result to offset 'i'
                {
                    if ((local_tid_34978 - squot32(local_tid_34978, 32) * 32) ==
                        31 && slt32(local_tid_34978, n_30761 *
                                    squot32(group_sizze_34983,
                                            segment_sizze_nonzzero_38228))) {
                        *(volatile __local
                          int32_t *) &red_arr_mem_38229[squot32(local_tid_34978,
                                                                32) *
                                                        sizeof(int32_t)] =
                            x_35000;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'
                {
                    int32_t skip_threads_38236;
                    
                    if (squot32(local_tid_34978, 32) == 0 &&
                        slt32(local_tid_34978, n_30761 *
                              squot32(group_sizze_34983,
                                      segment_sizze_nonzzero_38228))) {
                        x_38233 = *(volatile __local
                                    int32_t *) &red_arr_mem_38229[local_tid_34978 *
                                                                  sizeof(int32_t)];
                    }
                    // in-block scan (hopefully no barriers needed)
                    {
                        skip_threads_38236 = 1;
                        while (slt32(skip_threads_38236, 32)) {
                            if (sle32(skip_threads_38236, local_tid_34978 -
                                      squot32(local_tid_34978, 32) * 32) &&
                                (squot32(local_tid_34978, 32) == 0 &&
                                 slt32(local_tid_34978, n_30761 *
                                       squot32(group_sizze_34983,
                                               segment_sizze_nonzzero_38228)))) {
                                // read operands
                                {
                                    x_38232 = *(volatile __local
                                                int32_t *) &red_arr_mem_38229[(local_tid_34978 -
                                                                               skip_threads_38236) *
                                                                              sizeof(int32_t)];
                                }
                                // perform operation
                                {
                                    if (!slt32(srem32(local_tid_34978 * 32 +
                                                      32 - 1, n_30761),
                                               local_tid_34978 * 32 + 32 - 1 -
                                               ((local_tid_34978 -
                                                 skip_threads_38236) * 32 + 32 -
                                                1))) {
                                        int32_t res_38234 = x_38232 + x_38233;
                                        
                                        x_38233 = res_38234;
                                    }
                                }
                            }
                            if (sle32(wave_sizze_38226, skip_threads_38236)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            if (sle32(skip_threads_38236, local_tid_34978 -
                                      squot32(local_tid_34978, 32) * 32) &&
                                (squot32(local_tid_34978, 32) == 0 &&
                                 slt32(local_tid_34978, n_30761 *
                                       squot32(group_sizze_34983,
                                               segment_sizze_nonzzero_38228)))) {
                                // write result
                                {
                                    *(volatile __local
                                      int32_t *) &red_arr_mem_38229[local_tid_34978 *
                                                                    sizeof(int32_t)] =
                                        x_38233;
                                }
                            }
                            if (sle32(wave_sizze_38226, skip_threads_38236)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            skip_threads_38236 *= 2;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // carry-in for every block except the first
                {
                    if (!(squot32(local_tid_34978, 32) == 0 ||
                          !slt32(local_tid_34978, n_30761 *
                                 squot32(group_sizze_34983,
                                         segment_sizze_nonzzero_38228)))) {
                        // read operands
                        {
                            x_34999 = *(volatile __local
                                        int32_t *) &red_arr_mem_38229[(squot32(local_tid_34978,
                                                                               32) -
                                                                       1) *
                                                                      sizeof(int32_t)];
                        }
                        // perform operation
                        {
                            if (!slt32(srem32(local_tid_34978, n_30761),
                                       local_tid_34978 -
                                       (squot32(local_tid_34978, 32) * 32 -
                                        1))) {
                                int32_t res_35001 = x_34999 + x_35000;
                                
                                x_35000 = res_35001;
                            }
                        }
                        // write final result
                        {
                            *(volatile __local
                              int32_t *) &red_arr_mem_38229[local_tid_34978 *
                                                            sizeof(int32_t)] =
                                x_35000;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // restore correct values for first block
                {
                    if (squot32(local_tid_34978, 32) == 0) {
                        *(volatile __local
                          int32_t *) &red_arr_mem_38229[local_tid_34978 *
                                                        sizeof(int32_t)] =
                            x_35000;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // save final values of segments
        {
            if (slt32((group_id_34979 + i_38231 * num_groups_34993) *
                      squot32(group_sizze_34983, segment_sizze_nonzzero_38228) +
                      local_tid_34978, sizze_30757) && slt32(local_tid_34978,
                                                             squot32(group_sizze_34983,
                                                                     segment_sizze_nonzzero_38228))) {
                *(__global int32_t *) &mem_37633[((group_id_34979 + i_38231 *
                                                   num_groups_34993) *
                                                  squot32(group_sizze_34983,
                                                          segment_sizze_nonzzero_38228) +
                                                  local_tid_34978) * 4] =
                    *(__local int32_t *) &red_arr_mem_38229[((local_tid_34978 +
                                                              1) *
                                                             segment_sizze_nonzzero_38228 -
                                                             1) * 4];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
__kernel void segred_small_35208(int32_t sizze_30756, int32_t sizze_30757,
                                 int32_t res_31136, int32_t num_groups_35225,
                                 __global unsigned char *res_mem_37597, __global
                                 unsigned char *res_mem_37646, __global
                                 unsigned char *res_mem_37647, __global
                                 unsigned char *mem_37663,
                                 int32_t segment_sizze_nonzzero_38332)
{
    const int32_t group_sizze_35215 = mainzigroup_sizze_35190;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    ALIGNED_LOCAL_MEMORY(red_arr_mem_38333_backing_0, 4 *
                         mainzigroup_sizze_35190);
    
    int32_t global_tid_35208;
    int32_t local_tid_35209;
    int32_t group_sizze_38331;
    int32_t wave_sizze_38330;
    int32_t group_id_35210;
    
    global_tid_35208 = get_global_id(0);
    local_tid_35209 = get_local_id(0);
    group_sizze_38331 = get_local_size(0);
    wave_sizze_38330 = LOCKSTEP_WIDTH;
    group_id_35210 = get_group_id(0);
    
    int32_t gtid_35185;
    int32_t gtid_35207;
    __local char *red_arr_mem_38333;
    
    red_arr_mem_38333 = (__local char *) red_arr_mem_38333_backing_0;
    for (int32_t i_38335 = 0; i_38335 < squot32(squot32(sizze_30757 +
                                                        squot32(group_sizze_35215,
                                                                segment_sizze_nonzzero_38332) -
                                                        1,
                                                        squot32(group_sizze_35215,
                                                                segment_sizze_nonzzero_38332)) -
                                                group_id_35210 +
                                                num_groups_35225 - 1,
                                                num_groups_35225); i_38335++) {
        gtid_35185 = squot32(local_tid_35209, segment_sizze_nonzzero_38332) +
            (group_id_35210 + i_38335 * num_groups_35225) *
            squot32(group_sizze_35215, segment_sizze_nonzzero_38332);
        gtid_35207 = srem32(local_tid_35209, res_31136);
        // apply map function if in bounds
        {
            if (slt32(0, res_31136) && (slt32(gtid_35185, sizze_30757) &&
                                        slt32(local_tid_35209, res_31136 *
                                              squot32(group_sizze_35215,
                                                      segment_sizze_nonzzero_38332)))) {
                int32_t x_35235;
                int32_t x_35236;
                bool cond_35238;
                float res_35239;
                
                x_35235 = *(__global int32_t *) &res_mem_37647[gtid_35185 * 4];
                x_35236 = *(__global int32_t *) &res_mem_37646[gtid_35185 * 4];
                cond_35238 = slt32(gtid_35207, x_35236);
                if (cond_35238) {
                    int32_t x_35240;
                    int32_t x_35241;
                    int32_t i_35242;
                    float res_35243;
                    
                    x_35240 = gtid_35207 + x_35235;
                    x_35241 = x_35240 - x_35236;
                    i_35242 = 1 + x_35241;
                    res_35243 = *(__global float *) &res_mem_37597[(gtid_35185 *
                                                                    sizze_30756 +
                                                                    i_35242) *
                                                                   4];
                    res_35239 = res_35243;
                } else {
                    res_35239 = 0.0F;
                }
                // save results to be reduced
                {
                    *(__local float *) &red_arr_mem_38333[local_tid_35209 * 4] =
                        res_35239;
                }
                // save map-out results
                { }
            } else {
                *(__local float *) &red_arr_mem_38333[local_tid_35209 * 4] =
                    0.0F;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (slt32(0, res_31136)) {
            // perform segmented scan to imitate reduction
            {
                float x_35231;
                float x_35232;
                float x_38336;
                float x_38337;
                int32_t skip_threads_38339;
                
                if (slt32(local_tid_35209, res_31136 *
                          squot32(group_sizze_35215,
                                  segment_sizze_nonzzero_38332))) {
                    x_35232 = *(volatile __local
                                float *) &red_arr_mem_38333[local_tid_35209 *
                                                            sizeof(float)];
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_38339 = 1;
                    while (slt32(skip_threads_38339, 32)) {
                        if (sle32(skip_threads_38339, local_tid_35209 -
                                  squot32(local_tid_35209, 32) * 32) &&
                            slt32(local_tid_35209, res_31136 *
                                  squot32(group_sizze_35215,
                                          segment_sizze_nonzzero_38332))) {
                            // read operands
                            {
                                x_35231 = *(volatile __local
                                            float *) &red_arr_mem_38333[(local_tid_35209 -
                                                                         skip_threads_38339) *
                                                                        sizeof(float)];
                            }
                            // perform operation
                            {
                                if (!slt32(srem32(local_tid_35209, res_31136),
                                           local_tid_35209 - (local_tid_35209 -
                                                              skip_threads_38339))) {
                                    float res_35233 = x_35231 + x_35232;
                                    
                                    x_35232 = res_35233;
                                }
                            }
                        }
                        if (sle32(wave_sizze_38330, skip_threads_38339)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_38339, local_tid_35209 -
                                  squot32(local_tid_35209, 32) * 32) &&
                            slt32(local_tid_35209, res_31136 *
                                  squot32(group_sizze_35215,
                                          segment_sizze_nonzzero_38332))) {
                            // write result
                            {
                                *(volatile __local
                                  float *) &red_arr_mem_38333[local_tid_35209 *
                                                              sizeof(float)] =
                                    x_35232;
                            }
                        }
                        if (sle32(wave_sizze_38330, skip_threads_38339)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_38339 *= 2;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // last thread of block 'i' writes its result to offset 'i'
                {
                    if ((local_tid_35209 - squot32(local_tid_35209, 32) * 32) ==
                        31 && slt32(local_tid_35209, res_31136 *
                                    squot32(group_sizze_35215,
                                            segment_sizze_nonzzero_38332))) {
                        *(volatile __local
                          float *) &red_arr_mem_38333[squot32(local_tid_35209,
                                                              32) *
                                                      sizeof(float)] = x_35232;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'
                {
                    int32_t skip_threads_38340;
                    
                    if (squot32(local_tid_35209, 32) == 0 &&
                        slt32(local_tid_35209, res_31136 *
                              squot32(group_sizze_35215,
                                      segment_sizze_nonzzero_38332))) {
                        x_38337 = *(volatile __local
                                    float *) &red_arr_mem_38333[local_tid_35209 *
                                                                sizeof(float)];
                    }
                    // in-block scan (hopefully no barriers needed)
                    {
                        skip_threads_38340 = 1;
                        while (slt32(skip_threads_38340, 32)) {
                            if (sle32(skip_threads_38340, local_tid_35209 -
                                      squot32(local_tid_35209, 32) * 32) &&
                                (squot32(local_tid_35209, 32) == 0 &&
                                 slt32(local_tid_35209, res_31136 *
                                       squot32(group_sizze_35215,
                                               segment_sizze_nonzzero_38332)))) {
                                // read operands
                                {
                                    x_38336 = *(volatile __local
                                                float *) &red_arr_mem_38333[(local_tid_35209 -
                                                                             skip_threads_38340) *
                                                                            sizeof(float)];
                                }
                                // perform operation
                                {
                                    if (!slt32(srem32(local_tid_35209 * 32 +
                                                      32 - 1, res_31136),
                                               local_tid_35209 * 32 + 32 - 1 -
                                               ((local_tid_35209 -
                                                 skip_threads_38340) * 32 + 32 -
                                                1))) {
                                        float res_38338 = x_38336 + x_38337;
                                        
                                        x_38337 = res_38338;
                                    }
                                }
                            }
                            if (sle32(wave_sizze_38330, skip_threads_38340)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            if (sle32(skip_threads_38340, local_tid_35209 -
                                      squot32(local_tid_35209, 32) * 32) &&
                                (squot32(local_tid_35209, 32) == 0 &&
                                 slt32(local_tid_35209, res_31136 *
                                       squot32(group_sizze_35215,
                                               segment_sizze_nonzzero_38332)))) {
                                // write result
                                {
                                    *(volatile __local
                                      float *) &red_arr_mem_38333[local_tid_35209 *
                                                                  sizeof(float)] =
                                        x_38337;
                                }
                            }
                            if (sle32(wave_sizze_38330, skip_threads_38340)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            skip_threads_38340 *= 2;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // carry-in for every block except the first
                {
                    if (!(squot32(local_tid_35209, 32) == 0 ||
                          !slt32(local_tid_35209, res_31136 *
                                 squot32(group_sizze_35215,
                                         segment_sizze_nonzzero_38332)))) {
                        // read operands
                        {
                            x_35231 = *(volatile __local
                                        float *) &red_arr_mem_38333[(squot32(local_tid_35209,
                                                                             32) -
                                                                     1) *
                                                                    sizeof(float)];
                        }
                        // perform operation
                        {
                            if (!slt32(srem32(local_tid_35209, res_31136),
                                       local_tid_35209 -
                                       (squot32(local_tid_35209, 32) * 32 -
                                        1))) {
                                float res_35233 = x_35231 + x_35232;
                                
                                x_35232 = res_35233;
                            }
                        }
                        // write final result
                        {
                            *(volatile __local
                              float *) &red_arr_mem_38333[local_tid_35209 *
                                                          sizeof(float)] =
                                x_35232;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // restore correct values for first block
                {
                    if (squot32(local_tid_35209, 32) == 0) {
                        *(volatile __local
                          float *) &red_arr_mem_38333[local_tid_35209 *
                                                      sizeof(float)] = x_35232;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // save final values of segments
        {
            if (slt32((group_id_35210 + i_38335 * num_groups_35225) *
                      squot32(group_sizze_35215, segment_sizze_nonzzero_38332) +
                      local_tid_35209, sizze_30757) && slt32(local_tid_35209,
                                                             squot32(group_sizze_35215,
                                                                     segment_sizze_nonzzero_38332))) {
                *(__global float *) &mem_37663[((group_id_35210 + i_38335 *
                                                 num_groups_35225) *
                                                squot32(group_sizze_35215,
                                                        segment_sizze_nonzzero_38332) +
                                                local_tid_35209) * 4] =
                    *(__local float *) &red_arr_mem_38333[((local_tid_35209 +
                                                            1) *
                                                           segment_sizze_nonzzero_38332 -
                                                           1) * 4];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
__kernel void segred_small_35795(int32_t sizze_30757, int32_t arg_31158,
                                 int32_t num_groups_35964, __global
                                 unsigned char *mem_37668, __global
                                 unsigned char *mem_37700, __global
                                 unsigned char *mem_37703, __global
                                 unsigned char *mem_37707, __global
                                 unsigned char *mem_37709, __global
                                 unsigned char *mem_37712, __global
                                 unsigned char *mem_37715,
                                 int32_t segment_sizze_nonzzero_38439)
{
    const int32_t group_sizze_35954 = mainzigroup_sizze_35777;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    ALIGNED_LOCAL_MEMORY(red_arr_mem_38440_backing_0, mainzigroup_sizze_35777);
    ALIGNED_LOCAL_MEMORY(red_arr_mem_38442_backing_1, 4 *
                         mainzigroup_sizze_35777);
    ALIGNED_LOCAL_MEMORY(red_arr_mem_38444_backing_2, 4 *
                         mainzigroup_sizze_35777);
    
    int32_t global_tid_35795;
    int32_t local_tid_35796;
    int32_t group_sizze_38438;
    int32_t wave_sizze_38437;
    int32_t group_id_35797;
    
    global_tid_35795 = get_global_id(0);
    local_tid_35796 = get_local_id(0);
    group_sizze_38438 = get_local_size(0);
    wave_sizze_38437 = LOCKSTEP_WIDTH;
    group_id_35797 = get_group_id(0);
    
    int32_t gtid_35771;
    int32_t gtid_35794;
    __local char *red_arr_mem_38440;
    
    red_arr_mem_38440 = (__local char *) red_arr_mem_38440_backing_0;
    
    __local char *red_arr_mem_38442;
    
    red_arr_mem_38442 = (__local char *) red_arr_mem_38442_backing_1;
    
    __local char *red_arr_mem_38444;
    
    red_arr_mem_38444 = (__local char *) red_arr_mem_38444_backing_2;
    for (int32_t i_38446 = 0; i_38446 < squot32(squot32(sizze_30757 +
                                                        squot32(group_sizze_35954,
                                                                segment_sizze_nonzzero_38439) -
                                                        1,
                                                        squot32(group_sizze_35954,
                                                                segment_sizze_nonzzero_38439)) -
                                                group_id_35797 +
                                                num_groups_35964 - 1,
                                                num_groups_35964); i_38446++) {
        gtid_35771 = squot32(local_tid_35796, segment_sizze_nonzzero_38439) +
            (group_id_35797 + i_38446 * num_groups_35964) *
            squot32(group_sizze_35954, segment_sizze_nonzzero_38439);
        gtid_35794 = srem32(local_tid_35796, arg_31158);
        // apply map function if in bounds
        {
            if (slt32(0, arg_31158) && (slt32(gtid_35771, sizze_30757) &&
                                        slt32(local_tid_35796, arg_31158 *
                                              squot32(group_sizze_35954,
                                                      segment_sizze_nonzzero_38439)))) {
                int32_t y_35985;
                float y_35986;
                float x_35990;
                float x_35991;
                float res_35994;
                bool cond_35995;
                bool res_35996;
                bool res_35997;
                bool x_35998;
                float res_35999;
                bool res_36000;
                bool x_36001;
                float res_36002;
                
                y_35985 = *(__global int32_t *) &mem_37703[gtid_35771 * 4];
                y_35986 = *(__global float *) &mem_37700[gtid_35771 * 4];
                x_35990 = *(__global float *) &mem_37707[(gtid_35771 *
                                                          arg_31158 +
                                                          gtid_35794) * 4];
                x_35991 = *(__global float *) &mem_37668[gtid_35794 * 4];
                res_35994 = x_35990 / y_35986;
                cond_35995 = slt32(gtid_35794, y_35985);
                res_35996 = futrts_isnan32(res_35994);
                res_35997 = !res_35996;
                x_35998 = cond_35995 && res_35997;
                res_35999 = (float) fabs(res_35994);
                res_36000 = x_35991 < res_35999;
                x_36001 = x_35998 && res_36000;
                if (cond_35995) {
                    res_36002 = res_35994;
                } else {
                    res_36002 = 0.0F;
                }
                // save results to be reduced
                {
                    *(__local bool *) &red_arr_mem_38440[local_tid_35796] =
                        x_36001;
                    *(__local int32_t *) &red_arr_mem_38442[local_tid_35796 *
                                                            4] = gtid_35794;
                    *(__local float *) &red_arr_mem_38444[local_tid_35796 * 4] =
                        res_36002;
                }
                // save map-out results
                { }
            } else {
                *(__local bool *) &red_arr_mem_38440[local_tid_35796] = 0;
                *(__local int32_t *) &red_arr_mem_38442[local_tid_35796 * 4] =
                    -1;
                *(__local float *) &red_arr_mem_38444[local_tid_35796 * 4] =
                    0.0F;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (slt32(0, arg_31158)) {
            // perform segmented scan to imitate reduction
            {
                bool x_35972;
                int32_t x_35973;
                float x_35974;
                bool x_35975;
                int32_t x_35976;
                float x_35977;
                bool x_38447;
                int32_t x_38448;
                float x_38449;
                bool x_38450;
                int32_t x_38451;
                float x_38452;
                int32_t skip_threads_38460;
                
                if (slt32(local_tid_35796, arg_31158 *
                          squot32(group_sizze_35954,
                                  segment_sizze_nonzzero_38439))) {
                    x_35975 = *(volatile __local
                                bool *) &red_arr_mem_38440[local_tid_35796 *
                                                           sizeof(bool)];
                    x_35976 = *(volatile __local
                                int32_t *) &red_arr_mem_38442[local_tid_35796 *
                                                              sizeof(int32_t)];
                    x_35977 = *(volatile __local
                                float *) &red_arr_mem_38444[local_tid_35796 *
                                                            sizeof(float)];
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_38460 = 1;
                    while (slt32(skip_threads_38460, 32)) {
                        if (sle32(skip_threads_38460, local_tid_35796 -
                                  squot32(local_tid_35796, 32) * 32) &&
                            slt32(local_tid_35796, arg_31158 *
                                  squot32(group_sizze_35954,
                                          segment_sizze_nonzzero_38439))) {
                            // read operands
                            {
                                x_35972 = *(volatile __local
                                            bool *) &red_arr_mem_38440[(local_tid_35796 -
                                                                        skip_threads_38460) *
                                                                       sizeof(bool)];
                                x_35973 = *(volatile __local
                                            int32_t *) &red_arr_mem_38442[(local_tid_35796 -
                                                                           skip_threads_38460) *
                                                                          sizeof(int32_t)];
                                x_35974 = *(volatile __local
                                            float *) &red_arr_mem_38444[(local_tid_35796 -
                                                                         skip_threads_38460) *
                                                                        sizeof(float)];
                            }
                            // perform operation
                            {
                                if (!slt32(srem32(local_tid_35796, arg_31158),
                                           local_tid_35796 - (local_tid_35796 -
                                                              skip_threads_38460))) {
                                    bool res_35978;
                                    int32_t res_35979;
                                    float res_35984;
                                    
                                    if (x_35972) {
                                        res_35978 = x_35972;
                                        res_35979 = x_35973;
                                    } else {
                                        bool x_35980;
                                        bool y_35981;
                                        bool res_35982;
                                        int32_t res_35983;
                                        
                                        x_35980 = !x_35975;
                                        y_35981 = x_35972 && x_35980;
                                        res_35982 = x_35975 || y_35981;
                                        if (x_35975) {
                                            res_35983 = x_35976;
                                        } else {
                                            res_35983 = x_35973;
                                        }
                                        res_35978 = res_35982;
                                        res_35979 = res_35983;
                                    }
                                    res_35984 = x_35974 + x_35977;
                                    x_35975 = res_35978;
                                    x_35976 = res_35979;
                                    x_35977 = res_35984;
                                }
                            }
                        }
                        if (sle32(wave_sizze_38437, skip_threads_38460)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_38460, local_tid_35796 -
                                  squot32(local_tid_35796, 32) * 32) &&
                            slt32(local_tid_35796, arg_31158 *
                                  squot32(group_sizze_35954,
                                          segment_sizze_nonzzero_38439))) {
                            // write result
                            {
                                *(volatile __local
                                  bool *) &red_arr_mem_38440[local_tid_35796 *
                                                             sizeof(bool)] =
                                    x_35975;
                                *(volatile __local
                                  int32_t *) &red_arr_mem_38442[local_tid_35796 *
                                                                sizeof(int32_t)] =
                                    x_35976;
                                *(volatile __local
                                  float *) &red_arr_mem_38444[local_tid_35796 *
                                                              sizeof(float)] =
                                    x_35977;
                            }
                        }
                        if (sle32(wave_sizze_38437, skip_threads_38460)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_38460 *= 2;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // last thread of block 'i' writes its result to offset 'i'
                {
                    if ((local_tid_35796 - squot32(local_tid_35796, 32) * 32) ==
                        31 && slt32(local_tid_35796, arg_31158 *
                                    squot32(group_sizze_35954,
                                            segment_sizze_nonzzero_38439))) {
                        *(volatile __local
                          bool *) &red_arr_mem_38440[squot32(local_tid_35796,
                                                             32) *
                                                     sizeof(bool)] = x_35975;
                        *(volatile __local
                          int32_t *) &red_arr_mem_38442[squot32(local_tid_35796,
                                                                32) *
                                                        sizeof(int32_t)] =
                            x_35976;
                        *(volatile __local
                          float *) &red_arr_mem_38444[squot32(local_tid_35796,
                                                              32) *
                                                      sizeof(float)] = x_35977;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'
                {
                    int32_t skip_threads_38461;
                    
                    if (squot32(local_tid_35796, 32) == 0 &&
                        slt32(local_tid_35796, arg_31158 *
                              squot32(group_sizze_35954,
                                      segment_sizze_nonzzero_38439))) {
                        x_38450 = *(volatile __local
                                    bool *) &red_arr_mem_38440[local_tid_35796 *
                                                               sizeof(bool)];
                        x_38451 = *(volatile __local
                                    int32_t *) &red_arr_mem_38442[local_tid_35796 *
                                                                  sizeof(int32_t)];
                        x_38452 = *(volatile __local
                                    float *) &red_arr_mem_38444[local_tid_35796 *
                                                                sizeof(float)];
                    }
                    // in-block scan (hopefully no barriers needed)
                    {
                        skip_threads_38461 = 1;
                        while (slt32(skip_threads_38461, 32)) {
                            if (sle32(skip_threads_38461, local_tid_35796 -
                                      squot32(local_tid_35796, 32) * 32) &&
                                (squot32(local_tid_35796, 32) == 0 &&
                                 slt32(local_tid_35796, arg_31158 *
                                       squot32(group_sizze_35954,
                                               segment_sizze_nonzzero_38439)))) {
                                // read operands
                                {
                                    x_38447 = *(volatile __local
                                                bool *) &red_arr_mem_38440[(local_tid_35796 -
                                                                            skip_threads_38461) *
                                                                           sizeof(bool)];
                                    x_38448 = *(volatile __local
                                                int32_t *) &red_arr_mem_38442[(local_tid_35796 -
                                                                               skip_threads_38461) *
                                                                              sizeof(int32_t)];
                                    x_38449 = *(volatile __local
                                                float *) &red_arr_mem_38444[(local_tid_35796 -
                                                                             skip_threads_38461) *
                                                                            sizeof(float)];
                                }
                                // perform operation
                                {
                                    if (!slt32(srem32(local_tid_35796 * 32 +
                                                      32 - 1, arg_31158),
                                               local_tid_35796 * 32 + 32 - 1 -
                                               ((local_tid_35796 -
                                                 skip_threads_38461) * 32 + 32 -
                                                1))) {
                                        bool res_38453;
                                        int32_t res_38454;
                                        float res_38459;
                                        
                                        if (x_38447) {
                                            res_38453 = x_38447;
                                            res_38454 = x_38448;
                                        } else {
                                            bool x_38455;
                                            bool y_38456;
                                            bool res_38457;
                                            int32_t res_38458;
                                            
                                            x_38455 = !x_38450;
                                            y_38456 = x_38447 && x_38455;
                                            res_38457 = x_38450 || y_38456;
                                            if (x_38450) {
                                                res_38458 = x_38451;
                                            } else {
                                                res_38458 = x_38448;
                                            }
                                            res_38453 = res_38457;
                                            res_38454 = res_38458;
                                        }
                                        res_38459 = x_38449 + x_38452;
                                        x_38450 = res_38453;
                                        x_38451 = res_38454;
                                        x_38452 = res_38459;
                                    }
                                }
                            }
                            if (sle32(wave_sizze_38437, skip_threads_38461)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            if (sle32(skip_threads_38461, local_tid_35796 -
                                      squot32(local_tid_35796, 32) * 32) &&
                                (squot32(local_tid_35796, 32) == 0 &&
                                 slt32(local_tid_35796, arg_31158 *
                                       squot32(group_sizze_35954,
                                               segment_sizze_nonzzero_38439)))) {
                                // write result
                                {
                                    *(volatile __local
                                      bool *) &red_arr_mem_38440[local_tid_35796 *
                                                                 sizeof(bool)] =
                                        x_38450;
                                    *(volatile __local
                                      int32_t *) &red_arr_mem_38442[local_tid_35796 *
                                                                    sizeof(int32_t)] =
                                        x_38451;
                                    *(volatile __local
                                      float *) &red_arr_mem_38444[local_tid_35796 *
                                                                  sizeof(float)] =
                                        x_38452;
                                }
                            }
                            if (sle32(wave_sizze_38437, skip_threads_38461)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            skip_threads_38461 *= 2;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // carry-in for every block except the first
                {
                    if (!(squot32(local_tid_35796, 32) == 0 ||
                          !slt32(local_tid_35796, arg_31158 *
                                 squot32(group_sizze_35954,
                                         segment_sizze_nonzzero_38439)))) {
                        // read operands
                        {
                            x_35972 = *(volatile __local
                                        bool *) &red_arr_mem_38440[(squot32(local_tid_35796,
                                                                            32) -
                                                                    1) *
                                                                   sizeof(bool)];
                            x_35973 = *(volatile __local
                                        int32_t *) &red_arr_mem_38442[(squot32(local_tid_35796,
                                                                               32) -
                                                                       1) *
                                                                      sizeof(int32_t)];
                            x_35974 = *(volatile __local
                                        float *) &red_arr_mem_38444[(squot32(local_tid_35796,
                                                                             32) -
                                                                     1) *
                                                                    sizeof(float)];
                        }
                        // perform operation
                        {
                            if (!slt32(srem32(local_tid_35796, arg_31158),
                                       local_tid_35796 -
                                       (squot32(local_tid_35796, 32) * 32 -
                                        1))) {
                                bool res_35978;
                                int32_t res_35979;
                                float res_35984;
                                
                                if (x_35972) {
                                    res_35978 = x_35972;
                                    res_35979 = x_35973;
                                } else {
                                    bool x_35980;
                                    bool y_35981;
                                    bool res_35982;
                                    int32_t res_35983;
                                    
                                    x_35980 = !x_35975;
                                    y_35981 = x_35972 && x_35980;
                                    res_35982 = x_35975 || y_35981;
                                    if (x_35975) {
                                        res_35983 = x_35976;
                                    } else {
                                        res_35983 = x_35973;
                                    }
                                    res_35978 = res_35982;
                                    res_35979 = res_35983;
                                }
                                res_35984 = x_35974 + x_35977;
                                x_35975 = res_35978;
                                x_35976 = res_35979;
                                x_35977 = res_35984;
                            }
                        }
                        // write final result
                        {
                            *(volatile __local
                              bool *) &red_arr_mem_38440[local_tid_35796 *
                                                         sizeof(bool)] =
                                x_35975;
                            *(volatile __local
                              int32_t *) &red_arr_mem_38442[local_tid_35796 *
                                                            sizeof(int32_t)] =
                                x_35976;
                            *(volatile __local
                              float *) &red_arr_mem_38444[local_tid_35796 *
                                                          sizeof(float)] =
                                x_35977;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // restore correct values for first block
                {
                    if (squot32(local_tid_35796, 32) == 0) {
                        *(volatile __local
                          bool *) &red_arr_mem_38440[local_tid_35796 *
                                                     sizeof(bool)] = x_35975;
                        *(volatile __local
                          int32_t *) &red_arr_mem_38442[local_tid_35796 *
                                                        sizeof(int32_t)] =
                            x_35976;
                        *(volatile __local
                          float *) &red_arr_mem_38444[local_tid_35796 *
                                                      sizeof(float)] = x_35977;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // save final values of segments
        {
            if (slt32((group_id_35797 + i_38446 * num_groups_35964) *
                      squot32(group_sizze_35954, segment_sizze_nonzzero_38439) +
                      local_tid_35796, sizze_30757) && slt32(local_tid_35796,
                                                             squot32(group_sizze_35954,
                                                                     segment_sizze_nonzzero_38439))) {
                *(__global bool *) &mem_37709[(group_id_35797 + i_38446 *
                                               num_groups_35964) *
                                              squot32(group_sizze_35954,
                                                      segment_sizze_nonzzero_38439) +
                                              local_tid_35796] = *(__local
                                                                   bool *) &red_arr_mem_38440[(local_tid_35796 +
                                                                                               1) *
                                                                                              segment_sizze_nonzzero_38439 -
                                                                                              1];
                *(__global int32_t *) &mem_37712[((group_id_35797 + i_38446 *
                                                   num_groups_35964) *
                                                  squot32(group_sizze_35954,
                                                          segment_sizze_nonzzero_38439) +
                                                  local_tid_35796) * 4] =
                    *(__local int32_t *) &red_arr_mem_38442[((local_tid_35796 +
                                                              1) *
                                                             segment_sizze_nonzzero_38439 -
                                                             1) * 4];
                *(__global float *) &mem_37715[((group_id_35797 + i_38446 *
                                                 num_groups_35964) *
                                                squot32(group_sizze_35954,
                                                        segment_sizze_nonzzero_38439) +
                                                local_tid_35796) * 4] =
                    *(__local float *) &red_arr_mem_38444[((local_tid_35796 +
                                                            1) *
                                                           segment_sizze_nonzzero_38439 -
                                                           1) * 4];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
"""
# Hacky parser/reader/writer for values written in Futhark syntax.
# Used for reading stdin when compiling standalone programs with the
# Python code generator.

import numpy as np
import string
import struct
import sys

class ReaderInput:
    def __init__(self, f):
        self.f = f
        self.lookahead_buffer = []

    def get_char(self):
        if len(self.lookahead_buffer) == 0:
            return self.f.read(1)
        else:
            c = self.lookahead_buffer[0]
            self.lookahead_buffer = self.lookahead_buffer[1:]
            return c

    def unget_char(self, c):
        self.lookahead_buffer = [c] + self.lookahead_buffer

    def get_chars(self, n):
        n1 = min(n, len(self.lookahead_buffer))
        s = b''.join(self.lookahead_buffer[:n1])
        self.lookahead_buffer = self.lookahead_buffer[n1:]
        n2 = n - n1
        if n2 > 0:
            s += self.f.read(n2)
        return s

    def peek_char(self):
        c = self.get_char()
        if c:
            self.unget_char(c)
        return c

def skip_spaces(f):
    c = f.get_char()
    while c != None:
        if c.isspace():
            c = f.get_char()
        elif c == b'-':
          # May be line comment.
          if f.peek_char() == b'-':
            # Yes, line comment. Skip to end of line.
            while (c != b'\n' and c != None):
              c = f.get_char()
          else:
            break
        else:
          break
    if c:
        f.unget_char(c)

def parse_specific_char(f, expected):
    got = f.get_char()
    if got != expected:
        f.unget_char(got)
        raise ValueError
    return True

def parse_specific_string(f, s):
    # This funky mess is intended, and is caused by the fact that if `type(b) ==
    # bytes` then `type(b[0]) == int`, but we need to match each element with a
    # `bytes`, so therefore we make each character an array element
    b = s.encode('utf8')
    bs = [b[i:i+1] for i in range(len(b))]
    read = []
    try:
        for c in bs:
            parse_specific_char(f, c)
            read.append(c)
        return True
    except ValueError:
        for c in read[::-1]:
            f.unget_char(c)
        raise

def optional(p, *args):
    try:
        return p(*args)
    except ValueError:
        return None

def optional_specific_string(f, s):
    c = f.peek_char()
    # This funky mess is intended, and is caused by the fact that if `type(b) ==
    # bytes` then `type(b[0]) == int`, but we need to match each element with a
    # `bytes`, so therefore we make each character an array element
    b = s.encode('utf8')
    bs = [b[i:i+1] for i in range(len(b))]
    if c == bs[0]:
        return parse_specific_string(f, s)
    else:
        return False

def sepBy(p, sep, *args):
    elems = []
    x = optional(p, *args)
    if x != None:
        elems += [x]
        while optional(sep, *args) != None:
            x = p(*args)
            elems += [x]
    return elems

# Assumes '0x' has already been read
def parse_hex_int(f):
    s = b''
    c = f.get_char()
    while c != None:
        if c in b'01234556789ABCDEFabcdef':
            s += c
            c = f.get_char()
        elif c == b'_':
            c = f.get_char() # skip _
        else:
            f.unget_char(c)
            break
    return str(int(s, 16)).encode('utf8') # ugh

def parse_int(f):
    s = b''
    c = f.get_char()
    if c == b'0' and f.peek_char() in b'xX':
        c = f.get_char() # skip X
        return parse_hex_int(f)
    else:
        while c != None:
            if c.isdigit():
                s += c
                c = f.get_char()
            elif c == b'_':
                c = f.get_char() # skip _
            else:
                f.unget_char(c)
                break
        if len(s) == 0:
            raise ValueError
        return s

def parse_int_signed(f):
    s = b''
    c = f.get_char()

    if c == b'-' and f.peek_char().isdigit():
      return c + parse_int(f)
    else:
      if c != b'+':
          f.unget_char(c)
      return parse_int(f)

def read_str_comma(f):
    skip_spaces(f)
    parse_specific_char(f, b',')
    return b','

def read_str_int(f, s):
    skip_spaces(f)
    x = int(parse_int_signed(f))
    optional_specific_string(f, s)
    return x

def read_str_uint(f, s):
    skip_spaces(f)
    x = int(parse_int(f))
    optional_specific_string(f, s)
    return x

def read_str_i8(f):
    return np.int8(read_str_int(f, 'i8'))
def read_str_i16(f):
    return np.int16(read_str_int(f, 'i16'))
def read_str_i32(f):
    return np.int32(read_str_int(f, 'i32'))
def read_str_i64(f):
    return np.int64(read_str_int(f, 'i64'))

def read_str_u8(f):
    return np.uint8(read_str_int(f, 'u8'))
def read_str_u16(f):
    return np.uint16(read_str_int(f, 'u16'))
def read_str_u32(f):
    return np.uint32(read_str_int(f, 'u32'))
def read_str_u64(f):
    return np.uint64(read_str_int(f, 'u64'))

def read_char(f):
    skip_spaces(f)
    parse_specific_char(f, b'\'')
    c = f.get_char()
    parse_specific_char(f, b'\'')
    return c

def read_str_hex_float(f, sign):
    int_part = parse_hex_int(f)
    parse_specific_char(f, b'.')
    frac_part = parse_hex_int(f)
    parse_specific_char(f, b'p')
    exponent = parse_int(f)

    int_val = int(int_part, 16)
    frac_val = float(int(frac_part, 16)) / (16 ** len(frac_part))
    exp_val = int(exponent)

    total_val = (int_val + frac_val) * (2.0 ** exp_val)
    if sign == b'-':
        total_val = -1 * total_val

    return float(total_val)


def read_str_decimal(f):
    skip_spaces(f)
    c = f.get_char()
    if (c == b'-'):
      sign = b'-'
    else:
      f.unget_char(c)
      sign = b''

    # Check for hexadecimal float
    c = f.get_char()
    if (c == '0' and (f.peek_char() in ['x', 'X'])):
        f.get_char()
        return read_str_hex_float(f, sign)
    else:
        f.unget_char(c)

    bef = optional(parse_int, f)
    if bef == None:
        bef = b'0'
        parse_specific_char(f, b'.')
        aft = parse_int(f)
    elif optional(parse_specific_char, f, b'.'):
        aft = parse_int(f)
    else:
        aft = b'0'
    if (optional(parse_specific_char, f, b'E') or
        optional(parse_specific_char, f, b'e')):
        expt = parse_int_signed(f)
    else:
        expt = b'0'
    return float(sign + bef + b'.' + aft + b'E' + expt)

def read_str_f32(f):
    skip_spaces(f)
    try:
        parse_specific_string(f, 'f32.nan')
        return np.float32(np.nan)
    except ValueError:
        try:
            parse_specific_string(f, 'f32.inf')
            return np.float32(np.inf)
        except ValueError:
            try:
               parse_specific_string(f, '-f32.inf')
               return np.float32(-np.inf)
            except ValueError:
               x = read_str_decimal(f)
               optional_specific_string(f, 'f32')
               return x

def read_str_f64(f):
    skip_spaces(f)
    try:
        parse_specific_string(f, 'f64.nan')
        return np.float64(np.nan)
    except ValueError:
        try:
            parse_specific_string(f, 'f64.inf')
            return np.float64(np.inf)
        except ValueError:
            try:
               parse_specific_string(f, '-f64.inf')
               return np.float64(-np.inf)
            except ValueError:
               x = read_str_decimal(f)
               optional_specific_string(f, 'f64')
               return x

def read_str_bool(f):
    skip_spaces(f)
    if f.peek_char() == b't':
        parse_specific_string(f, 'true')
        return True
    elif f.peek_char() == b'f':
        parse_specific_string(f, 'false')
        return False
    else:
        raise ValueError

def read_str_empty_array(f, type_name, rank):
    parse_specific_string(f, 'empty')
    parse_specific_char(f, b'(')
    for i in range(rank):
        parse_specific_string(f, '[]')
    parse_specific_string(f, type_name)
    parse_specific_char(f, b')')

    return None

def read_str_array_elems(f, elem_reader, type_name, rank):
    skip_spaces(f)
    try:
        parse_specific_char(f, b'[')
    except ValueError:
        return read_str_empty_array(f, type_name, rank)
    else:
        xs = sepBy(elem_reader, read_str_comma, f)
        skip_spaces(f)
        parse_specific_char(f, b']')
        return xs

def read_str_array_helper(f, elem_reader, type_name, rank):
    def nested_row_reader(_):
        return read_str_array_helper(f, elem_reader, type_name, rank-1)
    if rank == 1:
        row_reader = elem_reader
    else:
        row_reader = nested_row_reader
    return read_str_array_elems(f, row_reader, type_name, rank-1)

def expected_array_dims(l, rank):
  if rank > 1:
      n = len(l)
      if n == 0:
          elem = []
      else:
          elem = l[0]
      return [n] + expected_array_dims(elem, rank-1)
  else:
      return [len(l)]

def verify_array_dims(l, dims):
    if dims[0] != len(l):
        raise ValueError
    if len(dims) > 1:
        for x in l:
            verify_array_dims(x, dims[1:])

def read_str_array(f, elem_reader, type_name, rank, bt):
    elems = read_str_array_helper(f, elem_reader, type_name, rank)
    if elems == None:
        # Empty array
        return np.empty([0]*rank, dtype=bt)
    else:
        dims = expected_array_dims(elems, rank)
        verify_array_dims(elems, dims)
        return np.array(elems, dtype=bt)

################################################################################

READ_BINARY_VERSION = 2

# struct format specified at
# https://docs.python.org/2/library/struct.html#format-characters

def mk_bin_scalar_reader(t):
    def bin_reader(f):
        fmt = FUTHARK_PRIMTYPES[t]['bin_format']
        size = FUTHARK_PRIMTYPES[t]['size']
        return struct.unpack('<' + fmt, f.get_chars(size))[0]
    return bin_reader

read_bin_i8 = mk_bin_scalar_reader('i8')
read_bin_i16 = mk_bin_scalar_reader('i16')
read_bin_i32 = mk_bin_scalar_reader('i32')
read_bin_i64 = mk_bin_scalar_reader('i64')

read_bin_u8 = mk_bin_scalar_reader('u8')
read_bin_u16 = mk_bin_scalar_reader('u16')
read_bin_u32 = mk_bin_scalar_reader('u32')
read_bin_u64 = mk_bin_scalar_reader('u64')

read_bin_f32 = mk_bin_scalar_reader('f32')
read_bin_f64 = mk_bin_scalar_reader('f64')

read_bin_bool = mk_bin_scalar_reader('bool')

def read_is_binary(f):
    skip_spaces(f)
    c = f.get_char()
    if c == b'b':
        bin_version = read_bin_u8(f)
        if bin_version != READ_BINARY_VERSION:
            panic(1, "binary-input: File uses version %i, but I only understand version %i.\n",
                  bin_version, READ_BINARY_VERSION)
        return True
    else:
        f.unget_char(c)
        return False

FUTHARK_PRIMTYPES = {
    'i8':  {'binname' : b"  i8",
            'size' : 1,
            'bin_reader': read_bin_i8,
            'str_reader': read_str_i8,
            'bin_format': 'b',
            'numpy_type': np.int8 },

    'i16': {'binname' : b" i16",
            'size' : 2,
            'bin_reader': read_bin_i16,
            'str_reader': read_str_i16,
            'bin_format': 'h',
            'numpy_type': np.int16 },

    'i32': {'binname' : b" i32",
            'size' : 4,
            'bin_reader': read_bin_i32,
            'str_reader': read_str_i32,
            'bin_format': 'i',
            'numpy_type': np.int32 },

    'i64': {'binname' : b" i64",
            'size' : 8,
            'bin_reader': read_bin_i64,
            'str_reader': read_str_i64,
            'bin_format': 'q',
            'numpy_type': np.int64},

    'u8':  {'binname' : b"  u8",
            'size' : 1,
            'bin_reader': read_bin_u8,
            'str_reader': read_str_u8,
            'bin_format': 'B',
            'numpy_type': np.uint8 },

    'u16': {'binname' : b" u16",
            'size' : 2,
            'bin_reader': read_bin_u16,
            'str_reader': read_str_u16,
            'bin_format': 'H',
            'numpy_type': np.uint16 },

    'u32': {'binname' : b" u32",
            'size' : 4,
            'bin_reader': read_bin_u32,
            'str_reader': read_str_u32,
            'bin_format': 'I',
            'numpy_type': np.uint32 },

    'u64': {'binname' : b" u64",
            'size' : 8,
            'bin_reader': read_bin_u64,
            'str_reader': read_str_u64,
            'bin_format': 'Q',
            'numpy_type': np.uint64 },

    'f32': {'binname' : b" f32",
            'size' : 4,
            'bin_reader': read_bin_f32,
            'str_reader': read_str_f32,
            'bin_format': 'f',
            'numpy_type': np.float32 },

    'f64': {'binname' : b" f64",
            'size' : 8,
            'bin_reader': read_bin_f64,
            'str_reader': read_str_f64,
            'bin_format': 'd',
            'numpy_type': np.float64 },

    'bool': {'binname' : b"bool",
             'size' : 1,
             'bin_reader': read_bin_bool,
             'str_reader': read_str_bool,
             'bin_format': 'b',
             'numpy_type': np.bool }
}

def read_bin_read_type(f):
    read_binname = f.get_chars(4)

    for (k,v) in FUTHARK_PRIMTYPES.items():
        if v['binname'] == read_binname:
            return k
    panic(1, "binary-input: Did not recognize the type '%s'.\n", read_binname)

def numpy_type_to_type_name(t):
    for (k,v) in FUTHARK_PRIMTYPES.items():
        if v['numpy_type'] == t:
            return k
    raise Exception('Unknown Numpy type: {}'.format(t))

def read_bin_ensure_scalar(f, expected_type):
  dims = read_bin_i8(f)

  if dims != 0:
      panic(1, "binary-input: Expected scalar (0 dimensions), but got array with %i dimensions.\n", dims)

  bin_type = read_bin_read_type(f)
  if bin_type != expected_type:
      panic(1, "binary-input: Expected scalar of type %s but got scalar of type %s.\n",
            expected_type, bin_type)

# ------------------------------------------------------------------------------
# General interface for reading Primitive Futhark Values
# ------------------------------------------------------------------------------

def read_scalar(f, ty):
    if read_is_binary(f):
        read_bin_ensure_scalar(f, ty)
        return FUTHARK_PRIMTYPES[ty]['bin_reader'](f)
    return FUTHARK_PRIMTYPES[ty]['str_reader'](f)

def read_array(f, expected_type, rank):
    if not read_is_binary(f):
        str_reader = FUTHARK_PRIMTYPES[expected_type]['str_reader']
        return read_str_array(f, str_reader, expected_type, rank,
                              FUTHARK_PRIMTYPES[expected_type]['numpy_type'])

    bin_rank = read_bin_u8(f)

    if bin_rank != rank:
        panic(1, "binary-input: Expected %i dimensions, but got array with %i dimensions.\n",
              rank, bin_rank)

    bin_type_enum = read_bin_read_type(f)
    if expected_type != bin_type_enum:
        panic(1, "binary-input: Expected %iD-array with element type '%s' but got %iD-array with element type '%s'.\n",
              rank, expected_type, bin_rank, bin_type_enum)

    shape = []
    elem_count = 1
    for i in range(rank):
        bin_size = read_bin_u64(f)
        elem_count *= bin_size
        shape.append(bin_size)

    bin_fmt = FUTHARK_PRIMTYPES[bin_type_enum]['bin_format']

    # We first read the expected number of types into a bytestring,
    # then use np.fromstring.  This is because np.fromfile does not
    # work on things that are insufficiently file-like, like a network
    # stream.
    bytes = f.get_chars(elem_count * FUTHARK_PRIMTYPES[expected_type]['size'])
    arr = np.fromstring(bytes, dtype='<'+bin_fmt)
    arr.shape = shape

    return arr

if sys.version_info >= (3,0):
    input_reader = ReaderInput(sys.stdin.buffer)
else:
    input_reader = ReaderInput(sys.stdin)

import re

def read_value(type_desc, reader=input_reader):
    """Read a value of the given type.  The type is a string
representation of the Futhark type."""
    m = re.match(r'((?:\[\])*)([a-z0-9]+)$', type_desc)
    if m:
        dims = int(len(m.group(1))/2)
        basetype = m.group(2)
        assert basetype in FUTHARK_PRIMTYPES, "Unknown type: {}".format(type_desc)
        if dims > 0:
            return read_array(reader, basetype, dims)
        else:
            return read_scalar(reader, basetype)
        return (dims, basetype)

def write_value_text(v, out=sys.stdout):
    if type(v) == np.uint8:
        out.write("%uu8" % v)
    elif type(v) == np.uint16:
        out.write("%uu16" % v)
    elif type(v) == np.uint32:
        out.write("%uu32" % v)
    elif type(v) == np.uint64:
        out.write("%uu64" % v)
    elif type(v) == np.int8:
        out.write("%di8" % v)
    elif type(v) == np.int16:
        out.write("%di16" % v)
    elif type(v) == np.int32:
        out.write("%di32" % v)
    elif type(v) == np.int64:
        out.write("%di64" % v)
    elif type(v) in [np.bool, np.bool_]:
        if v:
            out.write("true")
        else:
            out.write("false")
    elif type(v) == np.float32:
        if np.isnan(v):
            out.write('f32.nan')
        elif np.isinf(v):
            if v >= 0:
                out.write('f32.inf')
            else:
                out.write('-f32.inf')
        else:
            out.write("%.6ff32" % v)
    elif type(v) == np.float64:
        if np.isnan(v):
            out.write('f64.nan')
        elif np.isinf(v):
            if v >= 0:
                out.write('f64.inf')
            else:
                out.write('-f64.inf')
        else:
            out.write("%.6ff64" % v)
    elif type(v) == np.ndarray:
        if np.product(v.shape) == 0:
            tname = numpy_type_to_type_name(v.dtype)
            out.write('empty({}{})'.format(''.join(['[]' for _ in v.shape[1:]]), tname))
        else:
            first = True
            out.write('[')
            for x in v:
                if not first: out.write(', ')
                first = False
                write_value(x, out=out)
            out.write(']')
    else:
        raise Exception("Cannot print value of type {}: {}".format(type(v), v))

type_strs = { np.dtype('int8'): b'  i8',
              np.dtype('int16'): b' i16',
              np.dtype('int32'): b' i32',
              np.dtype('int64'): b' i64',
              np.dtype('uint8'): b'  u8',
              np.dtype('uint16'): b' u16',
              np.dtype('uint32'): b' u32',
              np.dtype('uint64'): b' u64',
              np.dtype('float32'): b' f32',
              np.dtype('float64'): b' f64',
              np.dtype('bool'): b'bool'}

def construct_binary_value(v):
    t = v.dtype
    shape = v.shape

    elems = 1
    for d in shape:
        elems *= d

    num_bytes = 1 + 1 + 1 + 4 + len(shape) * 8 + elems * t.itemsize
    bytes = bytearray(num_bytes)
    bytes[0] = np.int8(ord('b'))
    bytes[1] = 2
    bytes[2] = np.int8(len(shape))
    bytes[3:7] = type_strs[t]

    for i in range(len(shape)):
        bytes[7+i*8:7+(i+1)*8] = np.int64(shape[i]).tostring()

    bytes[7+len(shape)*8:] = np.ascontiguousarray(v).tostring()

    return bytes

def write_value_binary(v, out=sys.stdout):
    if sys.version_info >= (3,0):
        out = out.buffer
    out.write(construct_binary_value(v))

def write_value(v, out=sys.stdout, binary=False):
    if binary:
        return write_value_binary(v, out=out)
    else:
        return write_value_text(v, out=out)

################################################################################
### end of values.py
################################################################################
# Helper functions dealing with memory blocks.

import ctypes as ct

def addressOffset(x, offset, bt):
  return ct.cast(ct.addressof(x.contents)+int(offset), ct.POINTER(bt))

def allocateMem(size):
  return ct.cast((ct.c_byte * max(0,size))(), ct.POINTER(ct.c_byte))

# Copy an array if its is not-None.  This is important for treating
# Numpy arrays as flat memory, but has some overhead.
def normaliseArray(x):
  if (x.base is x) or (x.base is None):
    return x
  else:
    return x.copy()

def unwrapArray(x):
  return normaliseArray(x).ctypes.data_as(ct.POINTER(ct.c_byte))

def createArray(x, dim):
  return np.ctypeslib.as_array(x, shape=dim)

def indexArray(x, offset, bt, nptype):
  return nptype(addressOffset(x, offset, bt)[0])

def writeScalarArray(x, offset, v):
  ct.memmove(ct.addressof(x.contents)+int(offset), ct.addressof(v), ct.sizeof(v))

# An opaque Futhark value.
class opaque(object):
  def __init__(self, desc, *payload):
    self.data = payload
    self.desc = desc

  def __repr__(self):
    return "<opaque Futhark value of type {}>".format(self.desc)
def panic(exitcode, fmt, *args):
    sys.stderr.write('%s: ' % sys.argv[0])
    sys.stderr.write(fmt % args)
    sys.exit(exitcode)
### start of tuning.py
###
### Reading the .tuning file.

def read_tuning_file(kvs, f):
    for line in f.read().splitlines():
        size, value = line.split('=')
        kvs[size] = int(value)
    return kvs

### end of tuning.py
# Scalar functions.

import numpy as np
import struct

def signed(x):
  if type(x) == np.uint8:
    return np.int8(x)
  elif type(x) == np.uint16:
    return np.int16(x)
  elif type(x) == np.uint32:
    return np.int32(x)
  else:
    return np.int64(x)

def unsigned(x):
  if type(x) == np.int8:
    return np.uint8(x)
  elif type(x) == np.int16:
    return np.uint16(x)
  elif type(x) == np.int32:
    return np.uint32(x)
  else:
    return np.uint64(x)

def shlN(x,y):
  return x << y

def ashrN(x,y):
  return x >> y

def sdivN(x,y):
  return x // y

def smodN(x,y):
  return x % y

def udivN(x,y):
  return signed(unsigned(x) // unsigned(y))

def umodN(x,y):
  return signed(unsigned(x) % unsigned(y))

def squotN(x,y):
  return np.floor_divide(np.abs(x), np.abs(y)) * np.sign(x) * np.sign(y)

def sremN(x,y):
  return np.remainder(np.abs(x), np.abs(y)) * np.sign(x)

def sminN(x,y):
  return min(x,y)

def smaxN(x,y):
  return max(x,y)

def uminN(x,y):
  return signed(min(unsigned(x),unsigned(y)))

def umaxN(x,y):
  return signed(max(unsigned(x),unsigned(y)))

def fminN(x,y):
  return min(x,y)

def fmaxN(x,y):
  return max(x,y)

def powN(x,y):
  return x ** y

def fpowN(x,y):
  return x ** y

def sleN(x,y):
  return x <= y

def sltN(x,y):
  return x < y

def uleN(x,y):
  return unsigned(x) <= unsigned(y)

def ultN(x,y):
  return unsigned(x) < unsigned(y)

def lshr8(x,y):
  return np.int8(np.uint8(x) >> np.uint8(y))

def lshr16(x,y):
  return np.int16(np.uint16(x) >> np.uint16(y))

def lshr32(x,y):
  return np.int32(np.uint32(x) >> np.uint32(y))

def lshr64(x,y):
  return np.int64(np.uint64(x) >> np.uint64(y))

def sext_T_i8(x):
  return np.int8(x)

def sext_T_i16(x):
  return np.int16(x)

def sext_T_i32(x):
  return np.int32(x)

def sext_T_i64(x):
  return np.int64(x)

def itob_T_bool(x):
  return np.bool(x)

def btoi_bool_i8(x):
  return np.int8(x)

def btoi_bool_i16(x):
  return np.int8(x)

def btoi_bool_i32(x):
  return np.int8(x)

def btoi_bool_i64(x):
  return np.int8(x)

def zext_i8_i8(x):
  return np.int8(np.uint8(x))

def zext_i8_i16(x):
  return np.int16(np.uint8(x))

def zext_i8_i32(x):
  return np.int32(np.uint8(x))

def zext_i8_i64(x):
  return np.int64(np.uint8(x))

def zext_i16_i8(x):
  return np.int8(np.uint16(x))

def zext_i16_i16(x):
  return np.int16(np.uint16(x))

def zext_i16_i32(x):
  return np.int32(np.uint16(x))

def zext_i16_i64(x):
  return np.int64(np.uint16(x))

def zext_i32_i8(x):
  return np.int8(np.uint32(x))

def zext_i32_i16(x):
  return np.int16(np.uint32(x))

def zext_i32_i32(x):
  return np.int32(np.uint32(x))

def zext_i32_i64(x):
  return np.int64(np.uint32(x))

def zext_i64_i8(x):
  return np.int8(np.uint64(x))

def zext_i64_i16(x):
  return np.int16(np.uint64(x))

def zext_i64_i32(x):
  return np.int32(np.uint64(x))

def zext_i64_i64(x):
  return np.int64(np.uint64(x))

shl8 = shl16 = shl32 = shl64 = shlN
ashr8 = ashr16 = ashr32 = ashr64 = ashrN
sdiv8 = sdiv16 = sdiv32 = sdiv64 = sdivN
smod8 = smod16 = smod32 = smod64 = smodN
udiv8 = udiv16 = udiv32 = udiv64 = udivN
umod8 = umod16 = umod32 = umod64 = umodN
squot8 = squot16 = squot32 = squot64 = squotN
srem8 = srem16 = srem32 = srem64 = sremN
smax8 = smax16 = smax32 = smax64 = smaxN
smin8 = smin16 = smin32 = smin64 = sminN
umax8 = umax16 = umax32 = umax64 = umaxN
umin8 = umin16 = umin32 = umin64 = uminN
pow8 = pow16 = pow32 = pow64 = powN
fpow32 = fpow64 = fpowN
fmax32 = fmax64 = fmaxN
fmin32 = fmin64 = fminN
sle8 = sle16 = sle32 = sle64 = sleN
slt8 = slt16 = slt32 = slt64 = sltN
ule8 = ule16 = ule32 = ule64 = uleN
ult8 = ult16 = ult32 = ult64 = ultN
sext_i8_i8 = sext_i16_i8 = sext_i32_i8 = sext_i64_i8 = sext_T_i8
sext_i8_i16 = sext_i16_i16 = sext_i32_i16 = sext_i64_i16 = sext_T_i16
sext_i8_i32 = sext_i16_i32 = sext_i32_i32 = sext_i64_i32 = sext_T_i32
sext_i8_i64 = sext_i16_i64 = sext_i32_i64 = sext_i64_i64 = sext_T_i64
itob_i8_bool = itob_i16_bool = itob_i32_bool = itob_i64_bool = itob_T_bool

def ssignum(x):
  return np.sign(x)

def usignum(x):
  if x < 0:
    return ssignum(-x)
  else:
    return ssignum(x)

def sitofp_T_f32(x):
  return np.float32(x)
sitofp_i8_f32 = sitofp_i16_f32 = sitofp_i32_f32 = sitofp_i64_f32 = sitofp_T_f32

def sitofp_T_f64(x):
  return np.float64(x)
sitofp_i8_f64 = sitofp_i16_f64 = sitofp_i32_f64 = sitofp_i64_f64 = sitofp_T_f64

def uitofp_T_f32(x):
  return np.float32(unsigned(x))
uitofp_i8_f32 = uitofp_i16_f32 = uitofp_i32_f32 = uitofp_i64_f32 = uitofp_T_f32

def uitofp_T_f64(x):
  return np.float64(unsigned(x))
uitofp_i8_f64 = uitofp_i16_f64 = uitofp_i32_f64 = uitofp_i64_f64 = uitofp_T_f64

def fptosi_T_i8(x):
  return np.int8(np.trunc(x))
fptosi_f32_i8 = fptosi_f64_i8 = fptosi_T_i8

def fptosi_T_i16(x):
  return np.int16(np.trunc(x))
fptosi_f32_i16 = fptosi_f64_i16 = fptosi_T_i16

def fptosi_T_i32(x):
  return np.int32(np.trunc(x))
fptosi_f32_i32 = fptosi_f64_i32 = fptosi_T_i32

def fptosi_T_i64(x):
  return np.int64(np.trunc(x))
fptosi_f32_i64 = fptosi_f64_i64 = fptosi_T_i64

def fptoui_T_i8(x):
  return np.uint8(np.trunc(x))
fptoui_f32_i8 = fptoui_f64_i8 = fptoui_T_i8

def fptoui_T_i16(x):
  return np.uint16(np.trunc(x))
fptoui_f32_i16 = fptoui_f64_i16 = fptoui_T_i16

def fptoui_T_i32(x):
  return np.uint32(np.trunc(x))
fptoui_f32_i32 = fptoui_f64_i32 = fptoui_T_i32

def fptoui_T_i64(x):
  return np.uint64(np.trunc(x))
fptoui_f32_i64 = fptoui_f64_i64 = fptoui_T_i64

def fpconv_f32_f64(x):
  return np.float64(x)

def fpconv_f64_f32(x):
  return np.float32(x)

def futhark_log64(x):
  return np.float64(np.log(x))

def futhark_log2_64(x):
  return np.float64(np.log2(x))

def futhark_log10_64(x):
  return np.float64(np.log10(x))

def futhark_sqrt64(x):
  return np.sqrt(x)

def futhark_exp64(x):
  return np.exp(x)

def futhark_cos64(x):
  return np.cos(x)

def futhark_sin64(x):
  return np.sin(x)

def futhark_tan64(x):
  return np.tan(x)

def futhark_acos64(x):
  return np.arccos(x)

def futhark_asin64(x):
  return np.arcsin(x)

def futhark_atan64(x):
  return np.arctan(x)

def futhark_atan2_64(x, y):
  return np.arctan2(x, y)

def futhark_round64(x):
  return np.round(x)

def futhark_isnan64(x):
  return np.isnan(x)

def futhark_isinf64(x):
  return np.isinf(x)

def futhark_to_bits64(x):
  s = struct.pack('>d', x)
  return np.int64(struct.unpack('>q', s)[0])

def futhark_from_bits64(x):
  s = struct.pack('>q', x)
  return np.float64(struct.unpack('>d', s)[0])

def futhark_log32(x):
  return np.float32(np.log(x))

def futhark_log2_32(x):
  return np.float32(np.log2(x))

def futhark_log10_32(x):
  return np.float32(np.log10(x))

def futhark_sqrt32(x):
  return np.float32(np.sqrt(x))

def futhark_exp32(x):
  return np.exp(x)

def futhark_cos32(x):
  return np.cos(x)

def futhark_sin32(x):
  return np.sin(x)

def futhark_tan32(x):
  return np.tan(x)

def futhark_acos32(x):
  return np.arccos(x)

def futhark_asin32(x):
  return np.arcsin(x)

def futhark_atan32(x):
  return np.arctan(x)

def futhark_atan2_32(x, y):
  return np.arctan2(x, y)

def futhark_round32(x):
  return np.round(x)

def futhark_isnan32(x):
  return np.isnan(x)

def futhark_isinf32(x):
  return np.isinf(x)

def futhark_to_bits32(x):
  s = struct.pack('>f', x)
  return np.int32(struct.unpack('>l', s)[0])

def futhark_from_bits32(x):
  s = struct.pack('>l', x)
  return np.float32(struct.unpack('>f', s)[0])
class bfastfinal:
  entry_points = {"main": (["i32", "i32", "i32", "f32", "f32", "f32", "[]i32",
                            "[][]f32"], ["[]i32", "[]f32"]),
                  "remove_nans": (["i16", "[][][]i16"], ["[][][]f32"]),
                  "reshapeTransp": (["[][][]f32"], ["[][]f32"])}
  def __init__(self, command_queue=None, interactive=False,
               platform_pref=preferred_platform, device_pref=preferred_device,
               default_group_size=default_group_size,
               default_num_groups=default_num_groups,
               default_tile_size=default_tile_size,
               default_threshold=default_threshold, sizes=sizes):
    size_heuristics=[("NVIDIA CUDA", cl.device_type.GPU, "lockstep_width", 32),
     ("AMD Accelerated Parallel Processing", cl.device_type.GPU, "lockstep_width",
      64), ("", cl.device_type.GPU, "lockstep_width", 1), ("", cl.device_type.GPU,
                                                           "num_groups", 256), ("",
                                                                                cl.device_type.GPU,
                                                                                "group_size",
                                                                                256),
     ("", cl.device_type.GPU, "tile_size", 32), ("", cl.device_type.GPU,
                                                 "threshold", 32768), ("",
                                                                       cl.device_type.CPU,
                                                                       "lockstep_width",
                                                                       1), ("",
                                                                            cl.device_type.CPU,
                                                                            "num_groups",
                                                                            "MAX_COMPUTE_UNITS"),
     ("", cl.device_type.CPU, "group_size", 32), ("", cl.device_type.CPU,
                                                  "tile_size", 4), ("",
                                                                    cl.device_type.CPU,
                                                                    "threshold",
                                                                    "MAX_COMPUTE_UNITS")]
    program = initialise_opencl_object(self,
                                       program_src=fut_opencl_src,
                                       command_queue=command_queue,
                                       interactive=interactive,
                                       platform_pref=platform_pref,
                                       device_pref=device_pref,
                                       default_group_size=default_group_size,
                                       default_num_groups=default_num_groups,
                                       default_tile_size=default_tile_size,
                                       default_threshold=default_threshold,
                                       size_heuristics=size_heuristics,
                                       required_types=["i16", "i32", "f32", "bool"],
                                       user_sizes=sizes,
                                       all_sizes={"main.group_size_31570": {"class": "group_size", "value": None},
                                        "main.group_size_31772": {"class": "group_size", "value": None},
                                        "main.group_size_31926": {"class": "group_size", "value": None},
                                        "main.group_size_31993": {"class": "group_size", "value": None},
                                        "main.group_size_32097": {"class": "group_size", "value": None},
                                        "main.group_size_32289": {"class": "group_size", "value": None},
                                        "main.group_size_32971": {"class": "group_size", "value": None},
                                        "main.group_size_33021": {"class": "group_size", "value": None},
                                        "main.group_size_33082": {"class": "group_size", "value": None},
                                        "main.group_size_33179": {"class": "group_size", "value": None},
                                        "main.group_size_33340": {"class": "group_size", "value": None},
                                        "main.group_size_33505": {"class": "group_size", "value": None},
                                        "main.group_size_33676": {"class": "group_size", "value": None},
                                        "main.group_size_33771": {"class": "group_size", "value": None},
                                        "main.group_size_33837": {"class": "group_size", "value": None},
                                        "main.group_size_34001": {"class": "group_size", "value": None},
                                        "main.group_size_34156": {"class": "group_size", "value": None},
                                        "main.group_size_34337": {"class": "group_size", "value": None},
                                        "main.group_size_34512": {"class": "group_size", "value": None},
                                        "main.group_size_34612": {"class": "group_size", "value": None},
                                        "main.group_size_34766": {"class": "group_size", "value": None},
                                        "main.group_size_34911": {"class": "group_size", "value": None},
                                        "main.group_size_34934": {"class": "group_size", "value": None},
                                        "main.group_size_34959": {"class": "group_size", "value": None},
                                        "main.group_size_35062": {"class": "group_size", "value": None},
                                        "main.group_size_35103": {"class": "group_size", "value": None},
                                        "main.group_size_35190": {"class": "group_size", "value": None},
                                        "main.group_size_35286": {"class": "group_size", "value": None},
                                        "main.group_size_35365": {"class": "group_size", "value": None},
                                        "main.group_size_35727": {"class": "group_size", "value": None},
                                        "main.group_size_35777": {"class": "group_size", "value": None},
                                        "main.group_size_35818": {"class": "group_size", "value": None},
                                        "main.group_size_35851": {"class": "group_size", "value": None},
                                        "main.group_size_37933": {"class": "group_size", "value": None},
                                        "main.group_size_37999": {"class": "group_size", "value": None},
                                        "main.group_size_38188": {"class": "group_size", "value": None},
                                        "main.group_size_38195": {"class": "group_size", "value": None},
                                        "main.group_size_38200": {"class": "group_size", "value": None},
                                        "main.group_size_38205": {"class": "group_size", "value": None},
                                        "main.group_size_38433": {"class": "group_size", "value": None},
                                        "main.max_num_groups_32291": {"class": "num_groups", "value": None},
                                        "main.max_num_groups_33507": {"class": "num_groups", "value": None},
                                        "main.max_num_groups_33839": {"class": "num_groups", "value": None},
                                        "main.max_num_groups_34158": {"class": "num_groups", "value": None},
                                        "main.max_num_groups_34614": {"class": "num_groups", "value": None},
                                        "main.max_num_groups_34936": {"class": "num_groups", "value": None},
                                        "main.max_num_groups_34961": {"class": "num_groups", "value": None},
                                        "main.max_num_groups_35064": {"class": "num_groups", "value": None},
                                        "main.max_num_groups_35192": {"class": "num_groups", "value": None},
                                        "main.max_num_groups_35779": {"class": "num_groups", "value": None},
                                        "main.max_num_groups_35820": {"class": "num_groups", "value": None},
                                        "main.suff_intra_par_11": {"class": "threshold (!main.suff_outer_par_10 !main.suff_outer_par_8 !main.suff_intra_par_9 !main.suff_outer_par_6 !main.suff_intra_par_7)",
                                                                   "value": None},
                                        "main.suff_intra_par_13": {"class": "threshold (!main.suff_outer_par_12)",
                                                                   "value": None},
                                        "main.suff_intra_par_18": {"class": "threshold (!main.suff_outer_par_17)",
                                                                   "value": None},
                                        "main.suff_intra_par_20": {"class": "threshold (!main.suff_outer_par_19 !main.suff_outer_par_17 !main.suff_intra_par_18)",
                                                                   "value": None},
                                        "main.suff_intra_par_22": {"class": "threshold (!main.suff_outer_par_21)",
                                                                   "value": None},
                                        "main.suff_intra_par_24": {"class": "threshold (!main.suff_outer_par_23 !main.suff_outer_par_21 !main.suff_intra_par_22)",
                                                                   "value": None},
                                        "main.suff_intra_par_26": {"class": "threshold (!main.suff_outer_par_25)",
                                                                   "value": None},
                                        "main.suff_intra_par_28": {"class": "threshold (!main.suff_outer_par_27 !main.suff_outer_par_25 !main.suff_intra_par_26)",
                                                                   "value": None},
                                        "main.suff_intra_par_30": {"class": "threshold (!main.suff_outer_par_29)",
                                                                   "value": None},
                                        "main.suff_intra_par_34": {"class": "threshold (!main.suff_outer_par_33)",
                                                                   "value": None},
                                        "main.suff_intra_par_36": {"class": "threshold (!main.suff_outer_par_35)",
                                                                   "value": None},
                                        "main.suff_intra_par_39": {"class": "threshold (!main.suff_outer_par_38)",
                                                                   "value": None},
                                        "main.suff_intra_par_7": {"class": "threshold (!main.suff_outer_par_6)",
                                                                  "value": None},
                                        "main.suff_intra_par_9": {"class": "threshold (!main.suff_outer_par_8 !main.suff_outer_par_6 !main.suff_intra_par_7)",
                                                                  "value": None},
                                        "main.suff_outer_par_10": {"class": "threshold (!main.suff_outer_par_8 !main.suff_intra_par_9 !main.suff_outer_par_6 !main.suff_intra_par_7)",
                                                                   "value": None},
                                        "main.suff_outer_par_17": {"class": "threshold ()", "value": None},
                                        "main.suff_outer_par_19": {"class": "threshold (!main.suff_outer_par_17 !main.suff_intra_par_18)",
                                                                   "value": None},
                                        "main.suff_outer_par_21": {"class": "threshold ()", "value": None},
                                        "main.suff_outer_par_23": {"class": "threshold (!main.suff_outer_par_21 !main.suff_intra_par_22)",
                                                                   "value": None},
                                        "main.suff_outer_par_25": {"class": "threshold ()", "value": None},
                                        "main.suff_outer_par_27": {"class": "threshold (!main.suff_outer_par_25 !main.suff_intra_par_26)",
                                                                   "value": None},
                                        "main.suff_outer_par_29": {"class": "threshold ()", "value": None},
                                        "main.suff_outer_par_33": {"class": "threshold ()", "value": None},
                                        "main.suff_outer_par_35": {"class": "threshold ()", "value": None},
                                        "main.suff_outer_par_38": {"class": "threshold ()", "value": None},
                                        "main.suff_outer_par_6": {"class": "threshold ()", "value": None},
                                        "main.suff_outer_par_8": {"class": "threshold (!main.suff_outer_par_6 !main.suff_intra_par_7)",
                                                                  "value": None},
                                        "main.tile_size_36510": {"class": "tile_size", "value": None},
                                        "main.tile_size_37017": {"class": "tile_size", "value": None},
                                        "main.tile_size_37067": {"class": "tile_size", "value": None},
                                        "remove_nans.group_size_31393": {"class": "group_size", "value": None}})
    self.copy_37930_var = program.copy_37930
    self.copy_37996_var = program.copy_37996
    self.copy_38192_var = program.copy_38192
    self.map_31399_var = program.map_31399
    self.map_31576_var = program.map_31576
    self.map_31778_var = program.map_31778
    self.map_31932_var = program.map_31932
    self.map_31999_var = program.map_31999
    self.map_32103_var = program.map_32103
    self.map_32210_var = program.map_32210
    self.map_32977_var = program.map_32977
    self.map_33027_var = program.map_33027
    self.map_33088_var = program.map_33088
    self.map_33185_var = program.map_33185
    self.map_33346_var = program.map_33346
    self.map_33440_var = program.map_33440
    self.map_33682_var = program.map_33682
    self.map_33777_var = program.map_33777
    self.map_34007_var = program.map_34007
    self.map_34095_var = program.map_34095
    self.map_34343_var = program.map_34343
    self.map_34518_var = program.map_34518
    self.map_34772_var = program.map_34772
    self.map_34917_var = program.map_34917
    self.map_35109_var = program.map_35109
    self.map_35292_var = program.map_35292
    self.map_35371_var = program.map_35371
    self.map_35733_var = program.map_35733
    self.map_35857_var = program.map_35857
    self.map_intra_group_31978_var = program.map_intra_group_31978
    self.map_intra_group_32126_var = program.map_intra_group_32126
    self.map_intra_group_32223_var = program.map_intra_group_32223
    self.map_intra_group_32634_var = program.map_intra_group_32634
    self.map_intra_group_33329_var = program.map_intra_group_33329
    self.map_intra_group_33451_var = program.map_intra_group_33451
    self.map_intra_group_33665_var = program.map_intra_group_33665
    self.map_intra_group_33788_var = program.map_intra_group_33788
    self.map_intra_group_33990_var = program.map_intra_group_33990
    self.map_intra_group_34106_var = program.map_intra_group_34106
    self.map_intra_group_34303_var = program.map_intra_group_34303
    self.map_intra_group_34741_var = program.map_intra_group_34741
    self.map_intra_group_35089_var = program.map_intra_group_35089
    self.map_intra_group_35333_var = program.map_intra_group_35333
    self.map_transpose_f32_var = program.map_transpose_f32
    self.map_transpose_f32_low_height_var = program.map_transpose_f32_low_height
    self.map_transpose_f32_low_width_var = program.map_transpose_f32_low_width
    self.map_transpose_f32_small_var = program.map_transpose_f32_small
    self.map_transpose_i32_var = program.map_transpose_i32
    self.map_transpose_i32_low_height_var = program.map_transpose_i32_low_height
    self.map_transpose_i32_low_width_var = program.map_transpose_i32_low_width
    self.map_transpose_i32_small_var = program.map_transpose_i32_small
    self.replicate_38197_var = program.replicate_38197
    self.replicate_38202_var = program.replicate_38202
    self.scan_stage1_34630_var = program.scan_stage1_34630
    self.scan_stage1_35836_var = program.scan_stage1_35836
    self.scan_stage2_38172_var = program.scan_stage2_38172
    self.scan_stage2_38417_var = program.scan_stage2_38417
    self.scan_stage3_38185_var = program.scan_stage3_38185
    self.scan_stage3_38430_var = program.scan_stage3_38430
    self.segred_large_32307_var = program.segred_large_32307
    self.segred_large_33523_var = program.segred_large_33523
    self.segred_large_33855_var = program.segred_large_33855
    self.segred_large_34174_var = program.segred_large_34174
    self.segred_large_34952_var = program.segred_large_34952
    self.segred_large_34977_var = program.segred_large_34977
    self.segred_large_35208_var = program.segred_large_35208
    self.segred_large_35795_var = program.segred_large_35795
    self.segred_nonseg_35080_var = program.segred_nonseg_35080
    self.segred_small_32307_var = program.segred_small_32307
    self.segred_small_33523_var = program.segred_small_33523
    self.segred_small_33855_var = program.segred_small_33855
    self.segred_small_34174_var = program.segred_small_34174
    self.segred_small_34952_var = program.segred_small_34952
    self.segred_small_34977_var = program.segred_small_34977
    self.segred_small_35208_var = program.segred_small_35208
    self.segred_small_35795_var = program.segred_small_35795
    counter_mem_37894 = np.zeros(1024, dtype=np.int32)
    static_mem_38508 = opencl_alloc(self, 4096, "static_mem_38508")
    if (4096 != 0):
      cl.enqueue_copy(self.queue, static_mem_38508,
                      normaliseArray(counter_mem_37894),
                      is_blocking=synchronous)
    self.counter_mem_37894 = static_mem_38508
    counter_mem_37979 = np.zeros(1024, dtype=np.int32)
    static_mem_38511 = opencl_alloc(self, 4096, "static_mem_38511")
    if (4096 != 0):
      cl.enqueue_copy(self.queue, static_mem_38511,
                      normaliseArray(counter_mem_37979),
                      is_blocking=synchronous)
    self.counter_mem_37979 = static_mem_38511
    counter_mem_38040 = np.zeros(1024, dtype=np.int32)
    static_mem_38512 = opencl_alloc(self, 4096, "static_mem_38512")
    if (4096 != 0):
      cl.enqueue_copy(self.queue, static_mem_38512,
                      normaliseArray(counter_mem_38040),
                      is_blocking=synchronous)
    self.counter_mem_38040 = static_mem_38512
    counter_mem_38101 = np.zeros(1024, dtype=np.int32)
    static_mem_38513 = opencl_alloc(self, 4096, "static_mem_38513")
    if (4096 != 0):
      cl.enqueue_copy(self.queue, static_mem_38513,
                      normaliseArray(counter_mem_38101),
                      is_blocking=synchronous)
    self.counter_mem_38101 = static_mem_38513
    counter_mem_38244 = np.zeros(1024, dtype=np.int32)
    static_mem_38514 = opencl_alloc(self, 4096, "static_mem_38514")
    if (4096 != 0):
      cl.enqueue_copy(self.queue, static_mem_38514,
                      normaliseArray(counter_mem_38244),
                      is_blocking=synchronous)
    self.counter_mem_38244 = static_mem_38514
    counter_mem_38279 = np.zeros(1024, dtype=np.int32)
    static_mem_38515 = opencl_alloc(self, 4096, "static_mem_38515")
    if (4096 != 0):
      cl.enqueue_copy(self.queue, static_mem_38515,
                      normaliseArray(counter_mem_38279),
                      is_blocking=synchronous)
    self.counter_mem_38279 = static_mem_38515
    counter_mem_38300 = np.array([np.int32(0)], dtype=np.int32)
    static_mem_38516 = opencl_alloc(self, 4, "static_mem_38516")
    if (4 != 0):
      cl.enqueue_copy(self.queue, static_mem_38516,
                      normaliseArray(counter_mem_38300),
                      is_blocking=synchronous)
    self.counter_mem_38300 = static_mem_38516
    counter_mem_38348 = np.zeros(1024, dtype=np.int32)
    static_mem_38518 = opencl_alloc(self, 4096, "static_mem_38518")
    if (4096 != 0):
      cl.enqueue_copy(self.queue, static_mem_38518,
                      normaliseArray(counter_mem_38348),
                      is_blocking=synchronous)
    self.counter_mem_38348 = static_mem_38518
    counter_mem_38473 = np.zeros(1024, dtype=np.int32)
    static_mem_38520 = opencl_alloc(self, 4096, "static_mem_38520")
    if (4096 != 0):
      cl.enqueue_copy(self.queue, static_mem_38520,
                      normaliseArray(counter_mem_38473),
                      is_blocking=synchronous)
    self.counter_mem_38473 = static_mem_38520
  def futhark_main(self, mappingindices_mem_37200, images_mem_37201,
                   sizze_30756, sizze_30757, sizze_30758, trend_30759, k_30760,
                   n_30761, freq_30762, hfrac_30763, lam_30764):
    dim_zzero_30767 = (np.int32(0) == sizze_30757)
    dim_zzero_30768 = (np.int32(0) == sizze_30758)
    old_empty_30769 = (dim_zzero_30767 or dim_zzero_30768)
    dim_zzero_30770 = (np.int32(0) == sizze_30756)
    new_empty_30771 = (dim_zzero_30767 or dim_zzero_30770)
    both_empty_30772 = (old_empty_30769 and new_empty_30771)
    dim_match_30773 = (sizze_30756 == sizze_30758)
    empty_or_match_30774 = (both_empty_30772 or dim_match_30773)
    empty_or_match_cert_30775 = True
    assert empty_or_match_30774, ("Error at bfastfinal.fut:112:1-241:20: %s" % ("function arguments of wrong shape",))
    x_30777 = (np.int32(2) * k_30760)
    res_30778 = (np.int32(2) + x_30777)
    cond_30779 = slt32(np.int32(0), trend_30759)
    if cond_30779:
      res_30780 = res_30778
    else:
      res_30781 = (res_30778 - np.int32(1))
      res_30780 = res_30781
    bounds_invalid_upwards_30782 = slt32(res_30780, np.int32(0))
    convop_x_37203 = (sizze_30756 * res_30780)
    binop_x_37204 = sext_i32_i64(convop_x_37203)
    bytes_37202 = (np.int64(4) * binop_x_37204)
    if cond_30779:
      eq_x_zz_30784 = (np.int32(0) == res_30780)
      not_p_30785 = not(bounds_invalid_upwards_30782)
      p_and_eq_x_y_30786 = (eq_x_zz_30784 and not_p_30785)
      dim_zzero_30787 = (bounds_invalid_upwards_30782 or p_and_eq_x_y_30786)
      both_empty_30788 = (eq_x_zz_30784 and dim_zzero_30787)
      empty_or_match_30792 = (not_p_30785 or both_empty_30788)
      empty_or_match_cert_30793 = True
      assert empty_or_match_30792, ("Error at bfastfinal.fut:112:1-241:20 -> bfastfinal.fut:123:16-55 -> bfastfinal.fut:45:10-18 -> /futlib/array.fut:61:1-62:12: %s%s%s%d%s%s" % ("Function return value does not match shape of type ",
                                                                                                                                                                                   "*",
                                                                                                                                                                                   "[",
                                                                                                                                                                                   res_30780,
                                                                                                                                                                                   "]",
                                                                                                                                                                                   "intrinsics.i32"))
      group_sizze_31657 = self.sizes["main.group_size_31570"]
      y_31658 = (group_sizze_31657 - np.int32(1))
      x_31659 = (y_31658 + convop_x_37203)
      num_groups_31660 = squot32(x_31659, group_sizze_31657)
      num_threads_31661 = (group_sizze_31657 * num_groups_31660)
      mem_37205 = opencl_alloc(self, bytes_37202, "mem_37205")
      if ((1 * (np.long(num_groups_31660) * np.long(group_sizze_31657))) != 0):
        self.map_31576_var.set_args(np.int32(sizze_30756),
                                    np.float32(freq_30762), np.int32(res_30780),
                                    mappingindices_mem_37200, mem_37205)
        cl.enqueue_nd_range_kernel(self.queue, self.map_31576_var,
                                   ((np.long(num_groups_31660) * np.long(group_sizze_31657)),),
                                   (np.long(group_sizze_31657),))
        if synchronous:
          self.queue.finish()
      arg_mem_37210 = mem_37205
    else:
      eq_x_zz_30815 = (np.int32(0) == res_30780)
      not_p_30816 = not(bounds_invalid_upwards_30782)
      p_and_eq_x_y_30817 = (eq_x_zz_30815 and not_p_30816)
      dim_zzero_30818 = (bounds_invalid_upwards_30782 or p_and_eq_x_y_30817)
      both_empty_30819 = (eq_x_zz_30815 and dim_zzero_30818)
      empty_or_match_30823 = (not_p_30816 or both_empty_30819)
      empty_or_match_cert_30824 = True
      assert empty_or_match_30823, ("Error at bfastfinal.fut:112:1-241:20 -> bfastfinal.fut:124:16-55 -> bfastfinal.fut:57:10-20 -> /futlib/array.fut:61:1-62:12: %s%s%s%d%s%s" % ("Function return value does not match shape of type ",
                                                                                                                                                                                   "*",
                                                                                                                                                                                   "[",
                                                                                                                                                                                   res_30780,
                                                                                                                                                                                   "]",
                                                                                                                                                                                   "intrinsics.i32"))
      group_sizze_31852 = self.sizes["main.group_size_31772"]
      y_31853 = (group_sizze_31852 - np.int32(1))
      x_31854 = (y_31853 + convop_x_37203)
      num_groups_31855 = squot32(x_31854, group_sizze_31852)
      num_threads_31856 = (group_sizze_31852 * num_groups_31855)
      mem_37209 = opencl_alloc(self, bytes_37202, "mem_37209")
      if ((1 * (np.long(num_groups_31855) * np.long(group_sizze_31852))) != 0):
        self.map_31778_var.set_args(np.int32(sizze_30756),
                                    np.float32(freq_30762), np.int32(res_30780),
                                    mappingindices_mem_37200, mem_37209)
        cl.enqueue_nd_range_kernel(self.queue, self.map_31778_var,
                                   ((np.long(num_groups_31855) * np.long(group_sizze_31852)),),
                                   (np.long(group_sizze_31852),))
        if synchronous:
          self.queue.finish()
      arg_mem_37210 = mem_37209
    x_30845 = (sizze_30756 * sizze_30756)
    y_30846 = (np.int32(2) * sizze_30756)
    x_30847 = (x_30845 + y_30846)
    x_30848 = (np.int32(1) + x_30847)
    y_30849 = (np.int32(1) + sizze_30756)
    x_30850 = sdiv32(x_30848, y_30849)
    x_30851 = (x_30850 - sizze_30756)
    arg_30852 = (x_30851 - np.int32(1))
    res_30853 = sitofp_i32_f32(arg_30852)
    group_sizze_31954 = self.sizes["main.group_size_31926"]
    y_31955 = (group_sizze_31954 - np.int32(1))
    x_31956 = (y_31955 + convop_x_37203)
    num_groups_31957 = squot32(x_31956, group_sizze_31954)
    num_threads_31958 = (group_sizze_31954 * num_groups_31957)
    mem_37214 = opencl_alloc(self, bytes_37202, "mem_37214")
    self.futhark__map_transpose_f32(mem_37214, np.int32(0), arg_mem_37210,
                                    np.int32(0), np.int32(1), sizze_30756,
                                    res_30780, (res_30780 * sizze_30756),
                                    (res_30780 * sizze_30756))
    mem_37218 = opencl_alloc(self, bytes_37202, "mem_37218")
    if ((1 * (np.long(num_groups_31957) * np.long(group_sizze_31954))) != 0):
      self.map_31932_var.set_args(np.int32(sizze_30756), np.int32(res_30780),
                                  np.float32(res_30853), mem_37214, mem_37218)
      cl.enqueue_nd_range_kernel(self.queue, self.map_31932_var,
                                 ((np.long(num_groups_31957) * np.long(group_sizze_31954)),),
                                 (np.long(group_sizze_31954),))
      if synchronous:
        self.queue.finish()
    m_30862 = (res_30780 - np.int32(1))
    empty_slice_30869 = (n_30761 == np.int32(0))
    m_30870 = (n_30761 - np.int32(1))
    zzero_leq_i_p_m_t_s_30871 = sle32(np.int32(0), m_30870)
    i_p_m_t_s_leq_w_30872 = slt32(m_30870, sizze_30756)
    i_lte_j_30873 = sle32(np.int32(0), n_30761)
    y_30874 = (zzero_leq_i_p_m_t_s_30871 and i_p_m_t_s_leq_w_30872)
    y_30875 = (i_lte_j_30873 and y_30874)
    ok_or_empty_30876 = (empty_slice_30869 or y_30875)
    index_certs_30878 = True
    assert ok_or_empty_30876, ("Error at bfastfinal.fut:112:1-241:20 -> bfastfinal.fut:133:15-21: %s%d%s%s%s%d%s%d%s%d%s" % ("Index [",
                                                                                                                             np.int32(0),
                                                                                                                             ", ",
                                                                                                                             "",
                                                                                                                             ":",
                                                                                                                             n_30761,
                                                                                                                             "] out of bounds for array of shape [",
                                                                                                                             res_30780,
                                                                                                                             "][",
                                                                                                                             sizze_30756,
                                                                                                                             "]."))
    index_certs_30880 = True
    assert ok_or_empty_30876, ("Error at bfastfinal.fut:112:1-241:20 -> bfastfinal.fut:134:15-22: %s%s%s%d%s%d%s%d%s%d%s" % ("Index [",
                                                                                                                             "",
                                                                                                                             ":",
                                                                                                                             n_30761,
                                                                                                                             ", ",
                                                                                                                             np.int32(0),
                                                                                                                             "] out of bounds for array of shape [",
                                                                                                                             sizze_30756,
                                                                                                                             "][",
                                                                                                                             res_30780,
                                                                                                                             "]."))
    index_certs_30891 = True
    assert ok_or_empty_30876, ("Error at bfastfinal.fut:112:1-241:20 -> bfastfinal.fut:135:15-26: %s%d%s%s%s%d%s%d%s%d%s" % ("Index [",
                                                                                                                             np.int32(0),
                                                                                                                             ", ",
                                                                                                                             "",
                                                                                                                             ":",
                                                                                                                             n_30761,
                                                                                                                             "] out of bounds for array of shape [",
                                                                                                                             sizze_30757,
                                                                                                                             "][",
                                                                                                                             sizze_30756,
                                                                                                                             "]."))
    suff_outer_par_31963 = (self.sizes["main.suff_outer_par_6"] <= sizze_30757)
    one_intra_par_min_31973 = (res_30780 * res_30780)
    intra_avail_par_31975 = smin32(res_30780, one_intra_par_min_31973)
    computed_group_sizze_31976 = smax32(res_30780, one_intra_par_min_31973)
    num_threads_31977 = (sizze_30757 * computed_group_sizze_31976)
    max_group_sizze_32061 = self.max_group_size
    fits_32062 = sle32(computed_group_sizze_31976, max_group_sizze_32061)
    suff_intra_par_32060 = (self.sizes["main.suff_intra_par_7"] <= intra_avail_par_31975)
    intra_suff_and_fits_32063 = (suff_intra_par_32060 and fits_32062)
    convop_x_37220 = (sizze_30757 * sizze_30758)
    binop_x_37221 = sext_i32_i64(convop_x_37220)
    bytes_37219 = (np.int64(4) * binop_x_37221)
    convop_x_37235 = (sizze_30757 * one_intra_par_min_31973)
    binop_x_37236 = sext_i32_i64(convop_x_37235)
    bytes_37233 = (np.int64(4) * binop_x_37236)
    binop_x_37239 = (sizze_30757 * res_30780)
    convop_x_37240 = (res_30780 * binop_x_37239)
    binop_x_37241 = sext_i32_i64(convop_x_37240)
    bytes_37238 = (np.int64(4) * binop_x_37241)
    group_sizze_32456 = self.sizes["main.group_size_32097"]
    y_32457 = (group_sizze_32456 - np.int32(1))
    x_32458 = (y_32457 + binop_x_37239)
    suff_outer_par_32461 = (self.sizes["main.suff_outer_par_8"] <= binop_x_37239)
    fits_32465 = sle32(res_30780, max_group_sizze_32061)
    suff_intra_par_32466 = (self.sizes["main.suff_intra_par_9"] <= res_30780)
    intra_suff_and_fits_32467 = (fits_32465 and suff_intra_par_32466)
    suff_outer_par_32529 = (self.sizes["main.suff_outer_par_10"] <= convop_x_37235)
    num_threads_32532 = (n_30761 * convop_x_37235)
    fits_32534 = sle32(n_30761, max_group_sizze_32061)
    suff_intra_par_32535 = (self.sizes["main.suff_intra_par_11"] <= n_30761)
    intra_suff_and_fits_32536 = (fits_32534 and suff_intra_par_32535)
    if suff_outer_par_31963:
      group_sizze_32028 = self.sizes["main.group_size_31993"]
      y_32029 = (group_sizze_32028 - np.int32(1))
      x_32030 = (sizze_30757 + y_32029)
      num_groups_32031 = squot32(x_32030, group_sizze_32028)
      num_threads_32032 = (group_sizze_32028 * num_groups_32031)
      mem_37222 = opencl_alloc(self, bytes_37219, "mem_37222")
      self.futhark__map_transpose_f32(mem_37222, np.int32(0), images_mem_37201,
                                      np.int32(0), np.int32(1), sizze_30758,
                                      sizze_30757, (sizze_30757 * sizze_30758),
                                      (sizze_30757 * sizze_30758))
      mem_37237 = opencl_alloc(self, bytes_37233, "mem_37237")
      binop_x_37225 = sext_i32_i64(one_intra_par_min_31973)
      bytes_37223 = (np.int64(4) * binop_x_37225)
      num_threads64_37732 = sext_i32_i64(num_threads_32032)
      total_sizze_37733 = (bytes_37223 * num_threads64_37732)
      mem_37226 = opencl_alloc(self, total_sizze_37733, "mem_37226")
      if ((1 * (np.long(num_groups_32031) * np.long(group_sizze_32028))) != 0):
        self.map_31999_var.set_args(np.int32(sizze_30756),
                                    np.int32(sizze_30757), np.int32(n_30761),
                                    np.int32(res_30780), arg_mem_37210,
                                    mem_37218, mem_37222, mem_37226, mem_37237)
        cl.enqueue_nd_range_kernel(self.queue, self.map_31999_var,
                                   ((np.long(num_groups_32031) * np.long(group_sizze_32028)),),
                                   (np.long(group_sizze_32028),))
        if synchronous:
          self.queue.finish()
      mem_37222 = None
      mem_37226 = None
      mem_37242 = opencl_alloc(self, bytes_37238, "mem_37242")
      self.futhark__map_transpose_f32(mem_37242, np.int32(0), mem_37237,
                                      np.int32(0), np.int32(1), sizze_30757,
                                      (res_30780 * res_30780),
                                      ((sizze_30757 * res_30780) * res_30780),
                                      ((sizze_30757 * res_30780) * res_30780))
      mem_37237 = None
      res_mem_37313 = mem_37242
    else:
      if intra_suff_and_fits_32063:
        mem_37255 = opencl_alloc(self, bytes_37238, "mem_37255")
        binop_x_37245 = sext_i32_i64(res_30780)
        bytes_37244 = (np.int64(4) * binop_x_37245)
        binop_x_37249 = sext_i32_i64(one_intra_par_min_31973)
        bytes_37247 = (np.int64(4) * binop_x_37249)
        num_threads64_37734 = sext_i32_i64(num_threads_31977)
        total_sizze_37735 = (bytes_37244 * num_threads64_37734)
        mem_37246 = opencl_alloc(self, total_sizze_37735, "mem_37246")
        if ((1 * (np.long(sizze_30757) * np.long(computed_group_sizze_31976))) != 0):
          self.map_intra_group_31978_var.set_args(cl.LocalMemory(np.long(bytes_37247)),
                                                  np.int32(sizze_30756),
                                                  np.int32(sizze_30757),
                                                  np.int32(sizze_30758),
                                                  np.int32(n_30761),
                                                  np.int32(res_30780),
                                                  np.int32(computed_group_sizze_31976),
                                                  images_mem_37201,
                                                  arg_mem_37210, mem_37218,
                                                  mem_37246, mem_37255)
          cl.enqueue_nd_range_kernel(self.queue, self.map_intra_group_31978_var,
                                     ((np.long(sizze_30757) * np.long(computed_group_sizze_31976)),),
                                     (np.long(computed_group_sizze_31976),))
          if synchronous:
            self.queue.finish()
        mem_37246 = None
        res_mem_37312 = mem_37255
      else:
        num_groups_32459 = squot32(x_32458, group_sizze_32456)
        num_threads_32460 = (group_sizze_32456 * num_groups_32459)
        if suff_outer_par_32461:
          mem_37263 = opencl_alloc(self, bytes_37238, "mem_37263")
          binop_x_37257 = sext_i32_i64(res_30780)
          bytes_37256 = (np.int64(4) * binop_x_37257)
          num_threads64_37736 = sext_i32_i64(num_threads_32460)
          total_sizze_37737 = (bytes_37256 * num_threads64_37736)
          mem_37258 = opencl_alloc(self, total_sizze_37737, "mem_37258")
          if ((1 * (np.long(num_groups_32459) * np.long(group_sizze_32456))) != 0):
            self.map_32103_var.set_args(np.int32(sizze_30757),
                                        np.int32(sizze_30758),
                                        np.int32(n_30761), np.int32(res_30780),
                                        images_mem_37201, mem_37214, mem_37218,
                                        mem_37258, mem_37263)
            cl.enqueue_nd_range_kernel(self.queue, self.map_32103_var,
                                       ((np.long(num_groups_32459) * np.long(group_sizze_32456)),),
                                       (np.long(group_sizze_32456),))
            if synchronous:
              self.queue.finish()
          mem_37258 = None
          mem_37268 = opencl_alloc(self, bytes_37238, "mem_37268")
          self.futhark__map_transpose_f32(mem_37268, np.int32(0), mem_37263,
                                          np.int32(0), np.int32(1),
                                          (sizze_30757 * res_30780), res_30780,
                                          ((sizze_30757 * res_30780) * res_30780),
                                          ((sizze_30757 * res_30780) * res_30780))
          mem_37263 = None
          res_mem_37311 = mem_37268
        else:
          if intra_suff_and_fits_32467:
            mem_37276 = opencl_alloc(self, bytes_37238, "mem_37276")
            binop_x_37271 = sext_i32_i64(res_30780)
            bytes_37270 = (np.int64(4) * binop_x_37271)
            if ((1 * (np.long(binop_x_37239) * np.long(res_30780))) != 0):
              self.map_intra_group_32126_var.set_args(cl.LocalMemory(np.long(bytes_37270)),
                                                      np.int32(sizze_30757),
                                                      np.int32(sizze_30758),
                                                      np.int32(n_30761),
                                                      np.int32(res_30780),
                                                      images_mem_37201,
                                                      mem_37214, mem_37218,
                                                      mem_37276)
              cl.enqueue_nd_range_kernel(self.queue,
                                         self.map_intra_group_32126_var,
                                         ((np.long(binop_x_37239) * np.long(res_30780)),),
                                         (np.long(res_30780),))
              if synchronous:
                self.queue.finish()
            res_mem_37310 = mem_37276
          else:
            if suff_outer_par_32529:
              tmp_36509 = (np.int32(29) + sizze_30757)
              gidzz_range_36508 = squot32(tmp_36509, np.int32(30))
              tile_sizze_36511 = self.sizes["main.tile_size_36510"]
              tile_sizze_x_36512 = smin32(res_30780, tile_sizze_36511)
              tiled_group_sizze_36514 = (tile_sizze_x_36512 * tile_sizze_x_36512)
              y_36521 = (tile_sizze_x_36512 - np.int32(1))
              x_36522 = (res_30780 + y_36521)
              groups_in_dim_36523 = squot32(x_36522, tile_sizze_x_36512)
              y_36528 = (groups_in_dim_36523 * groups_in_dim_36523)
              num_groups_36529 = (gidzz_range_36508 * y_36528)
              num_threads_36530 = (tiled_group_sizze_36514 * num_groups_36529)
              mem_37281 = opencl_alloc(self, bytes_37238, "mem_37281")
              mem_37285 = opencl_alloc(self, bytes_37219, "mem_37285")
              self.futhark__map_transpose_f32(mem_37285, np.int32(0),
                                              images_mem_37201, np.int32(0),
                                              np.int32(1), sizze_30758,
                                              sizze_30757,
                                              (sizze_30757 * sizze_30758),
                                              (sizze_30757 * sizze_30758))
              binop_x_37287 = sext_i32_i64(tiled_group_sizze_36514)
              bytes_37286 = (np.int64(4) * binop_x_37287)
              if ((1 * (np.long(num_groups_36529) * np.long(tiled_group_sizze_36514))) != 0):
                self.map_32210_var.set_args(cl.LocalMemory(np.long(bytes_37286)),
                                            np.int32(sizze_30757),
                                            np.int32(n_30761),
                                            np.int32(res_30780),
                                            np.int32(gidzz_range_36508),
                                            np.int32(tile_sizze_x_36512),
                                            np.int32(tiled_group_sizze_36514),
                                            mem_37214, mem_37218, mem_37281,
                                            mem_37285)
                cl.enqueue_nd_range_kernel(self.queue, self.map_32210_var,
                                           ((np.long(num_groups_36529) * np.long(tiled_group_sizze_36514)),),
                                           (np.long(tiled_group_sizze_36514),))
                if synchronous:
                  self.queue.finish()
              mem_37285 = None
              res_mem_37309 = mem_37281
            else:
              if intra_suff_and_fits_32536:
                mem_37292 = opencl_alloc(self, bytes_37202, "mem_37292")
                self.futhark__map_transpose_f32(mem_37292, np.int32(0),
                                                mem_37218, np.int32(0),
                                                np.int32(1), res_30780,
                                                sizze_30756,
                                                (sizze_30756 * res_30780),
                                                (sizze_30756 * res_30780))
                mem_37298 = opencl_alloc(self, bytes_37233, "mem_37298")
                binop_x_37294 = sext_i32_i64(n_30761)
                bytes_37293 = (np.int64(4) * binop_x_37294)
                if ((1 * (np.long(convop_x_37235) * np.long(n_30761))) != 0):
                  self.map_intra_group_32223_var.set_args(cl.LocalMemory(np.long(bytes_37293)),
                                                          np.int32(sizze_30756),
                                                          np.int32(sizze_30757),
                                                          np.int32(sizze_30758),
                                                          np.int32(n_30761),
                                                          np.int32(res_30780),
                                                          images_mem_37201,
                                                          arg_mem_37210,
                                                          mem_37292, mem_37298)
                  cl.enqueue_nd_range_kernel(self.queue,
                                             self.map_intra_group_32223_var,
                                             ((np.long(convop_x_37235) * np.long(n_30761)),),
                                             (np.long(n_30761),))
                  if synchronous:
                    self.queue.finish()
                mem_37292 = None
                res_mem_37308 = mem_37298
              else:
                total_num_elements_32593 = sext_i32_i64(num_threads_32532)
                group_sizze_32594 = self.sizes["main.group_size_32289"]
                max_num_groups_32595 = self.sizes["main.max_num_groups_32291"]
                group_sizze_32596 = sext_i32_i64(group_sizze_32594)
                max_num_groups_32597 = sext_i32_i64(max_num_groups_32595)
                y_32598 = (group_sizze_32596 - np.int64(1))
                x_32599 = (total_num_elements_32593 + y_32598)
                w_div_group_sizze_32600 = squot64(x_32599, group_sizze_32596)
                num_groups_maybe_zzero_32601 = smin64(max_num_groups_32597,
                                                      w_div_group_sizze_32600)
                num_groups_32602 = smax64(np.int64(1),
                                          num_groups_maybe_zzero_32601)
                num_threads_32603 = (group_sizze_32596 * num_groups_32602)
                num_groups_32604 = sext_i64_i32(num_groups_32602)
                num_threads_32605 = sext_i64_i32(num_threads_32603)
                mem_37302 = opencl_alloc(self, bytes_37202, "mem_37302")
                self.futhark__map_transpose_f32(mem_37302, np.int32(0),
                                                mem_37218, np.int32(0),
                                                np.int32(1), res_30780,
                                                sizze_30756,
                                                (sizze_30756 * res_30780),
                                                (sizze_30756 * res_30780))
                mem_37307 = opencl_alloc(self, bytes_37238, "mem_37307")
                if slt32((n_30761 * np.int32(2)), group_sizze_32594):
                  segment_sizze_nonzzero_37878 = smax32(np.int32(1), n_30761)
                  if ((1 * (np.long(num_groups_32604) * np.long(group_sizze_32594))) != 0):
                    self.segred_small_32307_var.set_args(np.int32(sizze_30756),
                                                         np.int32(sizze_30757),
                                                         np.int32(sizze_30758),
                                                         np.int32(n_30761),
                                                         np.int32(res_30780),
                                                         np.int32(num_groups_32604),
                                                         images_mem_37201,
                                                         arg_mem_37210,
                                                         mem_37302, mem_37307,
                                                         np.int32(segment_sizze_nonzzero_37878))
                    cl.enqueue_nd_range_kernel(self.queue,
                                               self.segred_small_32307_var,
                                               ((np.long(num_groups_32604) * np.long(group_sizze_32594)),),
                                               (np.long(group_sizze_32594),))
                    if synchronous:
                      self.queue.finish()
                else:
                  num_groups_37889 = (squot32(((num_groups_32604 + smax32(np.int32(1),
                                                                          ((sizze_30757 * res_30780) * res_30780))) - np.int32(1)),
                                              smax32(np.int32(1),
                                                     ((sizze_30757 * res_30780) * res_30780))) * ((sizze_30757 * res_30780) * res_30780))
                  num_threads_37890 = (num_groups_37889 * group_sizze_32594)
                  thread_per_segment_37891 = (squot32(((num_groups_32604 + smax32(np.int32(1),
                                                                                  ((sizze_30757 * res_30780) * res_30780))) - np.int32(1)),
                                                      smax32(np.int32(1),
                                                             ((sizze_30757 * res_30780) * res_30780))) * group_sizze_32594)
                  group_res_arr_mem_37892 = opencl_alloc(self,
                                                         (np.int32(4) * num_groups_37889),
                                                         "group_res_arr_mem_37892")
                  counter_mem_37894 = self.counter_mem_37894
                  if ((1 * (np.long(num_groups_37889) * np.long(group_sizze_32594))) != 0):
                    self.segred_large_32307_var.set_args(np.int32(sizze_30756),
                                                         np.int32(sizze_30757),
                                                         np.int32(sizze_30758),
                                                         np.int32(n_30761),
                                                         np.int32(res_30780),
                                                         np.int32(num_groups_32604),
                                                         images_mem_37201,
                                                         arg_mem_37210,
                                                         mem_37302, mem_37307,
                                                         np.int32(thread_per_segment_37891),
                                                         group_res_arr_mem_37892,
                                                         counter_mem_37894)
                    cl.enqueue_nd_range_kernel(self.queue,
                                               self.segred_large_32307_var,
                                               ((np.long(num_groups_37889) * np.long(group_sizze_32594)),),
                                               (np.long(group_sizze_32594),))
                    if synchronous:
                      self.queue.finish()
                mem_37302 = None
                res_mem_37308 = mem_37307
              res_mem_37309 = res_mem_37308
            res_mem_37310 = res_mem_37309
          res_mem_37311 = res_mem_37310
        res_mem_37312 = res_mem_37311
      res_mem_37313 = res_mem_37312
    j_30912 = (np.int32(2) * res_30780)
    j_m_i_30913 = (j_30912 - res_30780)
    res_30916 = (res_30780 * j_30912)
    empty_slice_30929 = (j_m_i_30913 == np.int32(0))
    m_30930 = (j_m_i_30913 - np.int32(1))
    i_p_m_t_s_30931 = (res_30780 + m_30930)
    zzero_leq_i_p_m_t_s_30932 = sle32(np.int32(0), i_p_m_t_s_30931)
    ok_or_empty_30939 = (empty_slice_30929 or zzero_leq_i_p_m_t_s_30932)
    index_certs_30941 = True
    assert ok_or_empty_30939, ("Error at bfastfinal.fut:112:1-241:20 -> bfastfinal.fut:147:14-29 -> bfastfinal.fut:88:8-37: %s%d%s%d%s%d%s%d%s%d%s%d%s" % ("Index [",
                                                                                                                                                           np.int32(0),
                                                                                                                                                           ":",
                                                                                                                                                           res_30780,
                                                                                                                                                           ", ",
                                                                                                                                                           res_30780,
                                                                                                                                                           ":",
                                                                                                                                                           j_30912,
                                                                                                                                                           "] out of bounds for array of shape [",
                                                                                                                                                           res_30780,
                                                                                                                                                           "][",
                                                                                                                                                           j_30912,
                                                                                                                                                           "]."))
    num_threads_32633 = (sizze_30757 * res_30916)
    fits_32795 = sle32(res_30916, max_group_sizze_32061)
    suff_intra_par_32793 = (self.sizes["main.suff_intra_par_13"] <= res_30916)
    intra_suff_and_fits_32796 = (suff_intra_par_32793 and fits_32795)
    convop_x_37324 = (j_m_i_30913 * binop_x_37239)
    binop_x_37325 = sext_i32_i64(convop_x_37324)
    bytes_37322 = (np.int64(4) * binop_x_37325)
    binop_x_37329 = sext_i32_i64(num_threads_32633)
    bytes_37327 = (np.int64(4) * binop_x_37329)
    if intra_suff_and_fits_32796:
      mem_37326 = opencl_alloc(self, bytes_37322, "mem_37326")
      binop_x_37315 = sext_i32_i64(res_30916)
      bytes_37314 = (np.int64(4) * binop_x_37315)
      if ((1 * (np.long(sizze_30757) * np.long(res_30916))) != 0):
        self.map_intra_group_32634_var.set_args(cl.LocalMemory(np.long(bytes_37314)),
                                                cl.LocalMemory(np.long(bytes_37314)),
                                                np.int32(sizze_30757),
                                                np.int32(res_30780),
                                                np.int32(m_30862),
                                                np.int32(j_30912),
                                                np.int32(j_m_i_30913),
                                                np.int32(res_30916),
                                                res_mem_37313, mem_37326)
        cl.enqueue_nd_range_kernel(self.queue, self.map_intra_group_32634_var,
                                   ((np.long(sizze_30757) * np.long(res_30916)),),
                                   (np.long(res_30916),))
        if synchronous:
          self.queue.finish()
      res_mem_37344 = mem_37326
    else:
      group_sizze_33223 = self.sizes["main.group_size_33179"]
      y_33224 = (group_sizze_33223 - np.int32(1))
      x_33225 = (num_threads_32633 + y_33224)
      num_groups_33226 = squot32(x_33225, group_sizze_33223)
      num_threads_33227 = (group_sizze_33223 * num_groups_33226)
      mem_37330 = opencl_alloc(self, bytes_37327, "mem_37330")
      if ((1 * (np.long(num_groups_33226) * np.long(group_sizze_33223))) != 0):
        self.map_33185_var.set_args(np.int32(sizze_30757), np.int32(res_30780),
                                    np.int32(j_30912), np.int32(res_30916),
                                    res_mem_37313, mem_37330)
        cl.enqueue_nd_range_kernel(self.queue, self.map_33185_var,
                                   ((np.long(num_groups_33226) * np.long(group_sizze_33223)),),
                                   (np.long(group_sizze_33223),))
        if synchronous:
          self.queue.finish()
      loop_nonempty_36206 = slt32(np.int32(0), res_30780)
      group_sizze_33244 = self.sizes["main.group_size_33082"]
      y_33245 = (group_sizze_33244 - np.int32(1))
      x_33246 = (sizze_30757 + y_33245)
      if loop_nonempty_36206:
        x_36207 = squot32(x_33246, group_sizze_33244)
        num_groups_33247 = x_36207
      else:
        num_groups_33247 = np.int32(0)
      num_threads_33248 = (group_sizze_33244 * num_groups_33247)
      group_sizze_33262 = self.sizes["main.group_size_33021"]
      y_33263 = (group_sizze_33262 - np.int32(1))
      x_33264 = (num_threads_32633 + y_33263)
      if loop_nonempty_36206:
        x_36209 = squot32(x_33264, group_sizze_33262)
        num_groups_33265 = x_36209
      else:
        num_groups_33265 = np.int32(0)
      num_threads_33266 = (group_sizze_33262 * num_groups_33265)
      group_sizze_33292 = self.sizes["main.group_size_32971"]
      y_33293 = (group_sizze_33292 - np.int32(1))
      x_33294 = (num_threads_32633 + y_33293)
      if loop_nonempty_36206:
        x_36211 = squot32(x_33294, group_sizze_33292)
        num_groups_33295 = x_36211
      else:
        num_groups_33295 = np.int32(0)
      num_threads_33296 = (group_sizze_33292 * num_groups_33295)
      bytes_37332 = sext_i32_i64(sizze_30757)
      mem_37333 = opencl_alloc(self, bytes_37332, "mem_37333")
      mem_37337 = opencl_alloc(self, bytes_37327, "mem_37337")
      i_33242 = np.int32(0)
      one_38510 = np.int32(1)
      for counter_38509 in range(res_30780):
        if ((1 * (np.long(num_groups_33247) * np.long(group_sizze_33244))) != 0):
          self.map_33088_var.set_args(np.int32(sizze_30757),
                                      np.int32(res_30916), np.int32(i_33242),
                                      mem_37330, mem_37333)
          cl.enqueue_nd_range_kernel(self.queue, self.map_33088_var,
                                     ((np.long(num_groups_33247) * np.long(group_sizze_33244)),),
                                     (np.long(group_sizze_33244),))
          if synchronous:
            self.queue.finish()
        if ((1 * (np.long(num_groups_33265) * np.long(group_sizze_33262))) != 0):
          self.map_33027_var.set_args(np.int32(sizze_30757), np.int32(m_30862),
                                      np.int32(j_30912), np.int32(res_30916),
                                      np.int32(i_33242), mem_37330, mem_37333,
                                      mem_37337)
          cl.enqueue_nd_range_kernel(self.queue, self.map_33027_var,
                                     ((np.long(num_groups_33265) * np.long(group_sizze_33262)),),
                                     (np.long(group_sizze_33262),))
          if synchronous:
            self.queue.finish()
        if ((1 * (np.long(num_groups_33295) * np.long(group_sizze_33292))) != 0):
          self.map_32977_var.set_args(np.int32(sizze_30757),
                                      np.int32(res_30916), mem_37330, mem_37337)
          cl.enqueue_nd_range_kernel(self.queue, self.map_32977_var,
                                     ((np.long(num_groups_33295) * np.long(group_sizze_33292)),),
                                     (np.long(group_sizze_33292),))
          if synchronous:
            self.queue.finish()
        i_33242 += one_38510
      mem_37333 = None
      mem_37337 = None
      mem_37343 = opencl_alloc(self, bytes_37322, "mem_37343")
      group_sizze_37933 = self.sizes["main.group_size_37933"]
      num_groups_37934 = squot32((((sizze_30757 * (res_30780 * j_m_i_30913)) + sext_i32_i32(group_sizze_37933)) - np.int32(1)),
                                 sext_i32_i32(group_sizze_37933))
      if ((1 * (np.long(num_groups_37934) * np.long(group_sizze_37933))) != 0):
        self.copy_37930_var.set_args(np.int32(sizze_30757), np.int32(res_30780),
                                     np.int32(j_30912), np.int32(j_m_i_30913),
                                     mem_37330, mem_37343)
        cl.enqueue_nd_range_kernel(self.queue, self.copy_37930_var,
                                   ((np.long(num_groups_37934) * np.long(group_sizze_37933)),),
                                   (np.long(group_sizze_37933),))
        if synchronous:
          self.queue.finish()
      mem_37330 = None
      res_mem_37344 = mem_37343
    res_mem_37313 = None
    suff_outer_par_33313 = (self.sizes["main.suff_outer_par_17"] <= sizze_30757)
    suff_intra_par_33391 = (self.sizes["main.suff_intra_par_18"] <= res_30780)
    intra_suff_and_fits_33394 = (fits_32465 and suff_intra_par_33391)
    binop_x_37354 = sext_i32_i64(binop_x_37239)
    bytes_37352 = (np.int64(4) * binop_x_37354)
    suff_outer_par_33567 = (self.sizes["main.suff_outer_par_19"] <= binop_x_37239)
    num_threads_33569 = (n_30761 * binop_x_37239)
    suff_intra_par_33572 = (self.sizes["main.suff_intra_par_20"] <= n_30761)
    intra_suff_and_fits_33573 = (fits_32534 and suff_intra_par_33572)
    if suff_outer_par_33313:
      group_sizze_33367 = self.sizes["main.group_size_33340"]
      y_33368 = (group_sizze_33367 - np.int32(1))
      x_33369 = (sizze_30757 + y_33368)
      num_groups_33370 = squot32(x_33369, group_sizze_33367)
      num_threads_33371 = (group_sizze_33367 * num_groups_33370)
      mem_37348 = opencl_alloc(self, bytes_37219, "mem_37348")
      self.futhark__map_transpose_f32(mem_37348, np.int32(0), images_mem_37201,
                                      np.int32(0), np.int32(1), sizze_30758,
                                      sizze_30757, (sizze_30757 * sizze_30758),
                                      (sizze_30757 * sizze_30758))
      mem_37355 = opencl_alloc(self, bytes_37352, "mem_37355")
      binop_x_37350 = sext_i32_i64(res_30780)
      bytes_37349 = (np.int64(4) * binop_x_37350)
      num_threads64_37747 = sext_i32_i64(num_threads_33371)
      total_sizze_37748 = (bytes_37349 * num_threads64_37747)
      mem_37351 = opencl_alloc(self, total_sizze_37748, "mem_37351")
      if ((1 * (np.long(num_groups_33370) * np.long(group_sizze_33367))) != 0):
        self.map_33346_var.set_args(np.int32(sizze_30756),
                                    np.int32(sizze_30757), np.int32(n_30761),
                                    np.int32(res_30780), arg_mem_37210,
                                    mem_37348, mem_37351, mem_37355)
        cl.enqueue_nd_range_kernel(self.queue, self.map_33346_var,
                                   ((np.long(num_groups_33370) * np.long(group_sizze_33367)),),
                                   (np.long(group_sizze_33367),))
        if synchronous:
          self.queue.finish()
      mem_37348 = None
      mem_37351 = None
      mem_37359 = opencl_alloc(self, bytes_37352, "mem_37359")
      self.futhark__map_transpose_f32(mem_37359, np.int32(0), mem_37355,
                                      np.int32(0), np.int32(1), sizze_30757,
                                      res_30780, (sizze_30757 * res_30780),
                                      (sizze_30757 * res_30780))
      mem_37355 = None
      res_mem_37393 = mem_37359
    else:
      if intra_suff_and_fits_33394:
        mem_37367 = opencl_alloc(self, bytes_37352, "mem_37367")
        binop_x_37362 = sext_i32_i64(res_30780)
        bytes_37361 = (np.int64(4) * binop_x_37362)
        if ((1 * (np.long(sizze_30757) * np.long(res_30780))) != 0):
          self.map_intra_group_33329_var.set_args(cl.LocalMemory(np.long(bytes_37361)),
                                                  np.int32(sizze_30757),
                                                  np.int32(sizze_30758),
                                                  np.int32(n_30761),
                                                  np.int32(res_30780),
                                                  images_mem_37201, mem_37214,
                                                  mem_37367)
          cl.enqueue_nd_range_kernel(self.queue, self.map_intra_group_33329_var,
                                     ((np.long(sizze_30757) * np.long(res_30780)),),
                                     (np.long(res_30780),))
          if synchronous:
            self.queue.finish()
        res_mem_37392 = mem_37367
      else:
        if suff_outer_par_33567:
          tile_sizze_37018 = self.sizes["main.tile_size_37017"]
          tiled_group_sizze_37019 = (tile_sizze_37018 * tile_sizze_37018)
          y_37022 = (tile_sizze_37018 - np.int32(1))
          x_37023 = (sizze_30757 + y_37022)
          groups_in_dim_37024 = squot32(x_37023, tile_sizze_37018)
          x_37026 = (res_30780 + y_37022)
          groups_in_dim_37027 = squot32(x_37026, tile_sizze_37018)
          num_groups_37029 = (groups_in_dim_37024 * groups_in_dim_37027)
          num_threads_37030 = (tiled_group_sizze_37019 * num_groups_37029)
          mem_37379 = opencl_alloc(self, bytes_37352, "mem_37379")
          binop_x_37370 = sext_i32_i64(tiled_group_sizze_37019)
          bytes_37368 = (np.int64(4) * binop_x_37370)
          if ((1 * (np.long(num_groups_37029) * np.long(tiled_group_sizze_37019))) != 0):
            self.map_33440_var.set_args(np.int32(sizze_30757),
                                        np.int32(sizze_30758),
                                        np.int32(n_30761), np.int32(res_30780),
                                        images_mem_37201, mem_37214, mem_37379)
            cl.enqueue_nd_range_kernel(self.queue, self.map_33440_var,
                                       ((np.long(num_groups_37029) * np.long(tiled_group_sizze_37019)),),
                                       (np.long(tiled_group_sizze_37019),))
            if synchronous:
              self.queue.finish()
          res_mem_37391 = mem_37379
        else:
          if intra_suff_and_fits_33573:
            mem_37385 = opencl_alloc(self, bytes_37352, "mem_37385")
            binop_x_37381 = sext_i32_i64(n_30761)
            bytes_37380 = (np.int64(4) * binop_x_37381)
            if ((1 * (np.long(binop_x_37239) * np.long(n_30761))) != 0):
              self.map_intra_group_33451_var.set_args(cl.LocalMemory(np.long(bytes_37380)),
                                                      np.int32(sizze_30756),
                                                      np.int32(sizze_30757),
                                                      np.int32(sizze_30758),
                                                      np.int32(n_30761),
                                                      np.int32(res_30780),
                                                      images_mem_37201,
                                                      arg_mem_37210, mem_37385)
              cl.enqueue_nd_range_kernel(self.queue,
                                         self.map_intra_group_33451_var,
                                         ((np.long(binop_x_37239) * np.long(n_30761)),),
                                         (np.long(n_30761),))
              if synchronous:
                self.queue.finish()
            res_mem_37390 = mem_37385
          else:
            total_num_elements_33619 = sext_i32_i64(num_threads_33569)
            group_sizze_33620 = self.sizes["main.group_size_33505"]
            max_num_groups_33621 = self.sizes["main.max_num_groups_33507"]
            group_sizze_33622 = sext_i32_i64(group_sizze_33620)
            max_num_groups_33623 = sext_i32_i64(max_num_groups_33621)
            y_33624 = (group_sizze_33622 - np.int64(1))
            x_33625 = (total_num_elements_33619 + y_33624)
            w_div_group_sizze_33626 = squot64(x_33625, group_sizze_33622)
            num_groups_maybe_zzero_33627 = smin64(max_num_groups_33623,
                                                  w_div_group_sizze_33626)
            num_groups_33628 = smax64(np.int64(1), num_groups_maybe_zzero_33627)
            num_threads_33629 = (group_sizze_33622 * num_groups_33628)
            num_groups_33630 = sext_i64_i32(num_groups_33628)
            num_threads_33631 = sext_i64_i32(num_threads_33629)
            mem_37389 = opencl_alloc(self, bytes_37352, "mem_37389")
            if slt32((n_30761 * np.int32(2)), group_sizze_33620):
              segment_sizze_nonzzero_37963 = smax32(np.int32(1), n_30761)
              if ((1 * (np.long(num_groups_33630) * np.long(group_sizze_33620))) != 0):
                self.segred_small_33523_var.set_args(np.int32(sizze_30756),
                                                     np.int32(sizze_30757),
                                                     np.int32(sizze_30758),
                                                     np.int32(n_30761),
                                                     np.int32(res_30780),
                                                     np.int32(num_groups_33630),
                                                     images_mem_37201,
                                                     arg_mem_37210, mem_37389,
                                                     np.int32(segment_sizze_nonzzero_37963))
                cl.enqueue_nd_range_kernel(self.queue,
                                           self.segred_small_33523_var,
                                           ((np.long(num_groups_33630) * np.long(group_sizze_33620)),),
                                           (np.long(group_sizze_33620),))
                if synchronous:
                  self.queue.finish()
            else:
              num_groups_37974 = (squot32(((num_groups_33630 + smax32(np.int32(1),
                                                                      (sizze_30757 * res_30780))) - np.int32(1)),
                                          smax32(np.int32(1),
                                                 (sizze_30757 * res_30780))) * (sizze_30757 * res_30780))
              num_threads_37975 = (num_groups_37974 * group_sizze_33620)
              thread_per_segment_37976 = (squot32(((num_groups_33630 + smax32(np.int32(1),
                                                                              (sizze_30757 * res_30780))) - np.int32(1)),
                                                  smax32(np.int32(1),
                                                         (sizze_30757 * res_30780))) * group_sizze_33620)
              group_res_arr_mem_37977 = opencl_alloc(self,
                                                     (np.int32(4) * num_groups_37974),
                                                     "group_res_arr_mem_37977")
              counter_mem_37979 = self.counter_mem_37979
              if ((1 * (np.long(num_groups_37974) * np.long(group_sizze_33620))) != 0):
                self.segred_large_33523_var.set_args(np.int32(sizze_30756),
                                                     np.int32(sizze_30757),
                                                     np.int32(sizze_30758),
                                                     np.int32(n_30761),
                                                     np.int32(res_30780),
                                                     np.int32(num_groups_33630),
                                                     images_mem_37201,
                                                     arg_mem_37210, mem_37389,
                                                     np.int32(thread_per_segment_37976),
                                                     group_res_arr_mem_37977,
                                                     counter_mem_37979)
                cl.enqueue_nd_range_kernel(self.queue,
                                           self.segred_large_33523_var,
                                           ((np.long(num_groups_37974) * np.long(group_sizze_33620)),),
                                           (np.long(group_sizze_33620),))
                if synchronous:
                  self.queue.finish()
            res_mem_37390 = mem_37389
          res_mem_37391 = res_mem_37390
        res_mem_37392 = res_mem_37391
      res_mem_37393 = res_mem_37392
    arg_mem_37210 = None
    mem_37214 = None
    suff_outer_par_33649 = (self.sizes["main.suff_outer_par_21"] <= sizze_30757)
    suff_intra_par_33727 = (self.sizes["main.suff_intra_par_22"] <= res_30780)
    intra_suff_and_fits_33730 = (fits_32465 and suff_intra_par_33727)
    binop_x_37399 = (res_30780 * j_m_i_30913)
    convop_x_37400 = (sizze_30757 * binop_x_37399)
    binop_x_37401 = sext_i32_i64(convop_x_37400)
    bytes_37398 = (np.int64(4) * binop_x_37401)
    binop_x_37416 = (sizze_30757 * j_m_i_30913)
    convop_x_37417 = (res_30780 * binop_x_37416)
    binop_x_37418 = sext_i32_i64(convop_x_37417)
    bytes_37415 = (np.int64(4) * binop_x_37418)
    group_sizze_33893 = self.sizes["main.group_size_33771"]
    y_33894 = (group_sizze_33893 - np.int32(1))
    x_33895 = (y_33894 + binop_x_37239)
    suff_outer_par_33898 = (self.sizes["main.suff_outer_par_23"] <= binop_x_37239)
    fits_33902 = sle32(j_m_i_30913, max_group_sizze_32061)
    suff_intra_par_33903 = (self.sizes["main.suff_intra_par_24"] <= j_m_i_30913)
    intra_suff_and_fits_33904 = (fits_33902 and suff_intra_par_33903)
    if suff_outer_par_33649:
      group_sizze_33703 = self.sizes["main.group_size_33676"]
      y_33704 = (group_sizze_33703 - np.int32(1))
      x_33705 = (sizze_30757 + y_33704)
      num_groups_33706 = squot32(x_33705, group_sizze_33703)
      num_threads_33707 = (group_sizze_33703 * num_groups_33706)
      mem_37397 = opencl_alloc(self, bytes_37352, "mem_37397")
      self.futhark__map_transpose_f32(mem_37397, np.int32(0), res_mem_37393,
                                      np.int32(0), np.int32(1), res_30780,
                                      sizze_30757, (sizze_30757 * res_30780),
                                      (sizze_30757 * res_30780))
      mem_37402 = opencl_alloc(self, bytes_37398, "mem_37402")
      group_sizze_37999 = self.sizes["main.group_size_37999"]
      num_groups_38000 = squot32((((sizze_30757 * (res_30780 * j_m_i_30913)) + sext_i32_i32(group_sizze_37999)) - np.int32(1)),
                                 sext_i32_i32(group_sizze_37999))
      if ((1 * (np.long(num_groups_38000) * np.long(group_sizze_37999))) != 0):
        self.copy_37996_var.set_args(np.int32(sizze_30757), np.int32(res_30780),
                                     np.int32(j_m_i_30913), res_mem_37344,
                                     mem_37402)
        cl.enqueue_nd_range_kernel(self.queue, self.copy_37996_var,
                                   ((np.long(num_groups_38000) * np.long(group_sizze_37999)),),
                                   (np.long(group_sizze_37999),))
        if synchronous:
          self.queue.finish()
      mem_37409 = opencl_alloc(self, bytes_37352, "mem_37409")
      binop_x_37404 = sext_i32_i64(res_30780)
      bytes_37403 = (np.int64(4) * binop_x_37404)
      num_threads64_37753 = sext_i32_i64(num_threads_33707)
      total_sizze_37754 = (bytes_37403 * num_threads64_37753)
      mem_37405 = opencl_alloc(self, total_sizze_37754, "mem_37405")
      if ((1 * (np.long(num_groups_33706) * np.long(group_sizze_33703))) != 0):
        self.map_33682_var.set_args(np.int32(sizze_30757), np.int32(res_30780),
                                    np.int32(j_m_i_30913), mem_37397, mem_37402,
                                    mem_37405, mem_37409)
        cl.enqueue_nd_range_kernel(self.queue, self.map_33682_var,
                                   ((np.long(num_groups_33706) * np.long(group_sizze_33703)),),
                                   (np.long(group_sizze_33703),))
        if synchronous:
          self.queue.finish()
      mem_37397 = None
      mem_37402 = None
      mem_37405 = None
      mem_37413 = opencl_alloc(self, bytes_37352, "mem_37413")
      self.futhark__map_transpose_f32(mem_37413, np.int32(0), mem_37409,
                                      np.int32(0), np.int32(1), sizze_30757,
                                      res_30780, (sizze_30757 * res_30780),
                                      (sizze_30757 * res_30780))
      mem_37409 = None
      res_mem_37449 = mem_37413
    else:
      if intra_suff_and_fits_33730:
        mem_37419 = opencl_alloc(self, bytes_37415, "mem_37419")
        self.futhark__map_transpose_f32(mem_37419, np.int32(0), res_mem_37344,
                                        np.int32(0), np.int32(1), j_m_i_30913,
                                        (sizze_30757 * res_30780),
                                        ((sizze_30757 * res_30780) * j_m_i_30913),
                                        ((sizze_30757 * res_30780) * j_m_i_30913))
        mem_37426 = opencl_alloc(self, bytes_37352, "mem_37426")
        binop_x_37421 = sext_i32_i64(res_30780)
        bytes_37420 = (np.int64(4) * binop_x_37421)
        if ((1 * (np.long(sizze_30757) * np.long(res_30780))) != 0):
          self.map_intra_group_33665_var.set_args(cl.LocalMemory(np.long(bytes_37420)),
                                                  np.int32(sizze_30757),
                                                  np.int32(res_30780),
                                                  np.int32(j_m_i_30913),
                                                  res_mem_37393, mem_37419,
                                                  mem_37426)
          cl.enqueue_nd_range_kernel(self.queue, self.map_intra_group_33665_var,
                                     ((np.long(sizze_30757) * np.long(res_30780)),),
                                     (np.long(res_30780),))
          if synchronous:
            self.queue.finish()
        mem_37419 = None
        res_mem_37448 = mem_37426
      else:
        num_groups_33896 = squot32(x_33895, group_sizze_33893)
        num_threads_33897 = (group_sizze_33893 * num_groups_33896)
        if suff_outer_par_33898:
          mem_37431 = opencl_alloc(self, bytes_37415, "mem_37431")
          self.futhark__map_transpose_f32(mem_37431, np.int32(0), res_mem_37344,
                                          np.int32(0), np.int32(1), j_m_i_30913,
                                          (sizze_30757 * res_30780),
                                          ((sizze_30757 * res_30780) * j_m_i_30913),
                                          ((sizze_30757 * res_30780) * j_m_i_30913))
          mem_37435 = opencl_alloc(self, bytes_37352, "mem_37435")
          if ((1 * (np.long(num_groups_33896) * np.long(group_sizze_33893))) != 0):
            self.map_33777_var.set_args(np.int32(sizze_30757),
                                        np.int32(res_30780),
                                        np.int32(j_m_i_30913), res_mem_37393,
                                        mem_37431, mem_37435)
            cl.enqueue_nd_range_kernel(self.queue, self.map_33777_var,
                                       ((np.long(num_groups_33896) * np.long(group_sizze_33893)),),
                                       (np.long(group_sizze_33893),))
            if synchronous:
              self.queue.finish()
          mem_37431 = None
          res_mem_37447 = mem_37435
        else:
          if intra_suff_and_fits_33904:
            mem_37441 = opencl_alloc(self, bytes_37352, "mem_37441")
            binop_x_37437 = sext_i32_i64(j_m_i_30913)
            bytes_37436 = (np.int64(4) * binop_x_37437)
            if ((1 * (np.long(binop_x_37239) * np.long(j_m_i_30913))) != 0):
              self.map_intra_group_33788_var.set_args(cl.LocalMemory(np.long(bytes_37436)),
                                                      np.int32(sizze_30757),
                                                      np.int32(res_30780),
                                                      np.int32(j_m_i_30913),
                                                      res_mem_37344,
                                                      res_mem_37393, mem_37441)
              cl.enqueue_nd_range_kernel(self.queue,
                                         self.map_intra_group_33788_var,
                                         ((np.long(binop_x_37239) * np.long(j_m_i_30913)),),
                                         (np.long(j_m_i_30913),))
              if synchronous:
                self.queue.finish()
            res_mem_37446 = mem_37441
          else:
            group_sizze_33946 = self.sizes["main.group_size_33837"]
            max_num_groups_33947 = self.sizes["main.max_num_groups_33839"]
            group_sizze_33948 = sext_i32_i64(group_sizze_33946)
            max_num_groups_33949 = sext_i32_i64(max_num_groups_33947)
            y_33950 = (group_sizze_33948 - np.int64(1))
            x_33951 = (y_33950 + binop_x_37325)
            w_div_group_sizze_33952 = squot64(x_33951, group_sizze_33948)
            num_groups_maybe_zzero_33953 = smin64(max_num_groups_33949,
                                                  w_div_group_sizze_33952)
            num_groups_33954 = smax64(np.int64(1), num_groups_maybe_zzero_33953)
            num_threads_33955 = (group_sizze_33948 * num_groups_33954)
            num_groups_33956 = sext_i64_i32(num_groups_33954)
            num_threads_33957 = sext_i64_i32(num_threads_33955)
            mem_37445 = opencl_alloc(self, bytes_37352, "mem_37445")
            if slt32((j_m_i_30913 * np.int32(2)), group_sizze_33946):
              segment_sizze_nonzzero_38024 = smax32(np.int32(1), j_m_i_30913)
              if ((1 * (np.long(num_groups_33956) * np.long(group_sizze_33946))) != 0):
                self.segred_small_33855_var.set_args(np.int32(sizze_30757),
                                                     np.int32(res_30780),
                                                     np.int32(j_m_i_30913),
                                                     np.int32(num_groups_33956),
                                                     res_mem_37344,
                                                     res_mem_37393, mem_37445,
                                                     np.int32(segment_sizze_nonzzero_38024))
                cl.enqueue_nd_range_kernel(self.queue,
                                           self.segred_small_33855_var,
                                           ((np.long(num_groups_33956) * np.long(group_sizze_33946)),),
                                           (np.long(group_sizze_33946),))
                if synchronous:
                  self.queue.finish()
            else:
              num_groups_38035 = (squot32(((num_groups_33956 + smax32(np.int32(1),
                                                                      (sizze_30757 * res_30780))) - np.int32(1)),
                                          smax32(np.int32(1),
                                                 (sizze_30757 * res_30780))) * (sizze_30757 * res_30780))
              num_threads_38036 = (num_groups_38035 * group_sizze_33946)
              thread_per_segment_38037 = (squot32(((num_groups_33956 + smax32(np.int32(1),
                                                                              (sizze_30757 * res_30780))) - np.int32(1)),
                                                  smax32(np.int32(1),
                                                         (sizze_30757 * res_30780))) * group_sizze_33946)
              group_res_arr_mem_38038 = opencl_alloc(self,
                                                     (np.int32(4) * num_groups_38035),
                                                     "group_res_arr_mem_38038")
              counter_mem_38040 = self.counter_mem_38040
              if ((1 * (np.long(num_groups_38035) * np.long(group_sizze_33946))) != 0):
                self.segred_large_33855_var.set_args(np.int32(sizze_30757),
                                                     np.int32(res_30780),
                                                     np.int32(j_m_i_30913),
                                                     np.int32(num_groups_33956),
                                                     res_mem_37344,
                                                     res_mem_37393, mem_37445,
                                                     np.int32(thread_per_segment_38037),
                                                     group_res_arr_mem_38038,
                                                     counter_mem_38040)
                cl.enqueue_nd_range_kernel(self.queue,
                                           self.segred_large_33855_var,
                                           ((np.long(num_groups_38035) * np.long(group_sizze_33946)),),
                                           (np.long(group_sizze_33946),))
                if synchronous:
                  self.queue.finish()
            res_mem_37446 = mem_37445
          res_mem_37447 = res_mem_37446
        res_mem_37448 = res_mem_37447
      res_mem_37449 = res_mem_37448
    res_mem_37344 = None
    res_mem_37393 = None
    suff_outer_par_33974 = (self.sizes["main.suff_outer_par_25"] <= sizze_30757)
    num_threads_33989 = (sizze_30756 * sizze_30757)
    fits_34050 = sle32(sizze_30756, max_group_sizze_32061)
    suff_intra_par_34048 = (self.sizes["main.suff_intra_par_26"] <= sizze_30756)
    intra_suff_and_fits_34051 = (suff_intra_par_34048 and fits_34050)
    binop_x_37459 = sext_i32_i64(num_threads_33989)
    bytes_37457 = (np.int64(4) * binop_x_37459)
    suff_outer_par_34216 = (self.sizes["main.suff_outer_par_27"] <= num_threads_33989)
    num_threads_34218 = (res_30780 * num_threads_33989)
    suff_intra_par_34221 = (self.sizes["main.suff_intra_par_28"] <= res_30780)
    intra_suff_and_fits_34222 = (fits_32465 and suff_intra_par_34221)
    if suff_outer_par_33974:
      group_sizze_34026 = self.sizes["main.group_size_34001"]
      y_34027 = (group_sizze_34026 - np.int32(1))
      x_34028 = (sizze_30757 + y_34027)
      num_groups_34029 = squot32(x_34028, group_sizze_34026)
      num_threads_34030 = (group_sizze_34026 * num_groups_34029)
      mem_37453 = opencl_alloc(self, bytes_37352, "mem_37453")
      self.futhark__map_transpose_f32(mem_37453, np.int32(0), res_mem_37449,
                                      np.int32(0), np.int32(1), res_30780,
                                      sizze_30757, (sizze_30757 * res_30780),
                                      (sizze_30757 * res_30780))
      mem_37460 = opencl_alloc(self, bytes_37457, "mem_37460")
      binop_x_37455 = sext_i32_i64(sizze_30756)
      bytes_37454 = (np.int64(4) * binop_x_37455)
      num_threads64_37759 = sext_i32_i64(num_threads_34030)
      total_sizze_37760 = (bytes_37454 * num_threads64_37759)
      mem_37456 = opencl_alloc(self, total_sizze_37760, "mem_37456")
      if ((1 * (np.long(num_groups_34029) * np.long(group_sizze_34026))) != 0):
        self.map_34007_var.set_args(np.int32(sizze_30756),
                                    np.int32(sizze_30757), np.int32(res_30780),
                                    mem_37218, mem_37453, mem_37456, mem_37460)
        cl.enqueue_nd_range_kernel(self.queue, self.map_34007_var,
                                   ((np.long(num_groups_34029) * np.long(group_sizze_34026)),),
                                   (np.long(group_sizze_34026),))
        if synchronous:
          self.queue.finish()
      mem_37453 = None
      mem_37456 = None
      mem_37464 = opencl_alloc(self, bytes_37457, "mem_37464")
      self.futhark__map_transpose_f32(mem_37464, np.int32(0), mem_37460,
                                      np.int32(0), np.int32(1), sizze_30757,
                                      sizze_30756, (sizze_30757 * sizze_30756),
                                      (sizze_30757 * sizze_30756))
      mem_37460 = None
      res_mem_37506 = mem_37464
    else:
      if intra_suff_and_fits_34051:
        mem_37469 = opencl_alloc(self, bytes_37202, "mem_37469")
        self.futhark__map_transpose_f32(mem_37469, np.int32(0), mem_37218,
                                        np.int32(0), np.int32(1), res_30780,
                                        sizze_30756, (sizze_30756 * res_30780),
                                        (sizze_30756 * res_30780))
        mem_37476 = opencl_alloc(self, bytes_37457, "mem_37476")
        binop_x_37471 = sext_i32_i64(sizze_30756)
        bytes_37470 = (np.int64(4) * binop_x_37471)
        if ((1 * (np.long(sizze_30757) * np.long(sizze_30756))) != 0):
          self.map_intra_group_33990_var.set_args(cl.LocalMemory(np.long(bytes_37470)),
                                                  np.int32(sizze_30756),
                                                  np.int32(sizze_30757),
                                                  np.int32(res_30780),
                                                  res_mem_37449, mem_37469,
                                                  mem_37476)
          cl.enqueue_nd_range_kernel(self.queue, self.map_intra_group_33990_var,
                                     ((np.long(sizze_30757) * np.long(sizze_30756)),),
                                     (np.long(sizze_30756),))
          if synchronous:
            self.queue.finish()
        mem_37469 = None
        res_mem_37505 = mem_37476
      else:
        if suff_outer_par_34216:
          mem_37480 = opencl_alloc(self, bytes_37202, "mem_37480")
          self.futhark__map_transpose_f32(mem_37480, np.int32(0), mem_37218,
                                          np.int32(0), np.int32(1), res_30780,
                                          sizze_30756,
                                          (sizze_30756 * res_30780),
                                          (sizze_30756 * res_30780))
          tile_sizze_37068 = self.sizes["main.tile_size_37067"]
          tiled_group_sizze_37069 = (tile_sizze_37068 * tile_sizze_37068)
          y_37072 = (tile_sizze_37068 - np.int32(1))
          x_37073 = (sizze_30757 + y_37072)
          groups_in_dim_37074 = squot32(x_37073, tile_sizze_37068)
          x_37076 = (sizze_30756 + y_37072)
          groups_in_dim_37077 = squot32(x_37076, tile_sizze_37068)
          num_groups_37079 = (groups_in_dim_37074 * groups_in_dim_37077)
          num_threads_37080 = (tiled_group_sizze_37069 * num_groups_37079)
          mem_37492 = opencl_alloc(self, bytes_37457, "mem_37492")
          binop_x_37483 = sext_i32_i64(tiled_group_sizze_37069)
          bytes_37481 = (np.int64(4) * binop_x_37483)
          if ((1 * (np.long(num_groups_37079) * np.long(tiled_group_sizze_37069))) != 0):
            self.map_34095_var.set_args(np.int32(sizze_30756),
                                        np.int32(sizze_30757),
                                        np.int32(res_30780), res_mem_37449,
                                        mem_37480, mem_37492)
            cl.enqueue_nd_range_kernel(self.queue, self.map_34095_var,
                                       ((np.long(num_groups_37079) * np.long(tiled_group_sizze_37069)),),
                                       (np.long(tiled_group_sizze_37069),))
            if synchronous:
              self.queue.finish()
          mem_37480 = None
          res_mem_37504 = mem_37492
        else:
          if intra_suff_and_fits_34222:
            mem_37498 = opencl_alloc(self, bytes_37457, "mem_37498")
            binop_x_37494 = sext_i32_i64(res_30780)
            bytes_37493 = (np.int64(4) * binop_x_37494)
            if ((1 * (np.long(num_threads_33989) * np.long(res_30780))) != 0):
              self.map_intra_group_34106_var.set_args(cl.LocalMemory(np.long(bytes_37493)),
                                                      np.int32(sizze_30756),
                                                      np.int32(sizze_30757),
                                                      np.int32(res_30780),
                                                      mem_37218, res_mem_37449,
                                                      mem_37498)
              cl.enqueue_nd_range_kernel(self.queue,
                                         self.map_intra_group_34106_var,
                                         ((np.long(num_threads_33989) * np.long(res_30780)),),
                                         (np.long(res_30780),))
              if synchronous:
                self.queue.finish()
            res_mem_37503 = mem_37498
          else:
            total_num_elements_34264 = sext_i32_i64(num_threads_34218)
            group_sizze_34265 = self.sizes["main.group_size_34156"]
            max_num_groups_34266 = self.sizes["main.max_num_groups_34158"]
            group_sizze_34267 = sext_i32_i64(group_sizze_34265)
            max_num_groups_34268 = sext_i32_i64(max_num_groups_34266)
            y_34269 = (group_sizze_34267 - np.int64(1))
            x_34270 = (total_num_elements_34264 + y_34269)
            w_div_group_sizze_34271 = squot64(x_34270, group_sizze_34267)
            num_groups_maybe_zzero_34272 = smin64(max_num_groups_34268,
                                                  w_div_group_sizze_34271)
            num_groups_34273 = smax64(np.int64(1), num_groups_maybe_zzero_34272)
            num_threads_34274 = (group_sizze_34267 * num_groups_34273)
            num_groups_34275 = sext_i64_i32(num_groups_34273)
            num_threads_34276 = sext_i64_i32(num_threads_34274)
            mem_37502 = opencl_alloc(self, bytes_37457, "mem_37502")
            if slt32((res_30780 * np.int32(2)), group_sizze_34265):
              segment_sizze_nonzzero_38085 = smax32(np.int32(1), res_30780)
              if ((1 * (np.long(num_groups_34275) * np.long(group_sizze_34265))) != 0):
                self.segred_small_34174_var.set_args(np.int32(sizze_30756),
                                                     np.int32(sizze_30757),
                                                     np.int32(res_30780),
                                                     np.int32(num_groups_34275),
                                                     mem_37218, res_mem_37449,
                                                     mem_37502,
                                                     np.int32(segment_sizze_nonzzero_38085))
                cl.enqueue_nd_range_kernel(self.queue,
                                           self.segred_small_34174_var,
                                           ((np.long(num_groups_34275) * np.long(group_sizze_34265)),),
                                           (np.long(group_sizze_34265),))
                if synchronous:
                  self.queue.finish()
            else:
              num_groups_38096 = (squot32(((num_groups_34275 + smax32(np.int32(1),
                                                                      (sizze_30757 * sizze_30756))) - np.int32(1)),
                                          smax32(np.int32(1),
                                                 (sizze_30757 * sizze_30756))) * (sizze_30757 * sizze_30756))
              num_threads_38097 = (num_groups_38096 * group_sizze_34265)
              thread_per_segment_38098 = (squot32(((num_groups_34275 + smax32(np.int32(1),
                                                                              (sizze_30757 * sizze_30756))) - np.int32(1)),
                                                  smax32(np.int32(1),
                                                         (sizze_30757 * sizze_30756))) * group_sizze_34265)
              group_res_arr_mem_38099 = opencl_alloc(self,
                                                     (np.int32(4) * num_groups_38096),
                                                     "group_res_arr_mem_38099")
              counter_mem_38101 = self.counter_mem_38101
              if ((1 * (np.long(num_groups_38096) * np.long(group_sizze_34265))) != 0):
                self.segred_large_34174_var.set_args(np.int32(sizze_30756),
                                                     np.int32(sizze_30757),
                                                     np.int32(res_30780),
                                                     np.int32(num_groups_34275),
                                                     mem_37218, res_mem_37449,
                                                     mem_37502,
                                                     np.int32(thread_per_segment_38098),
                                                     group_res_arr_mem_38099,
                                                     counter_mem_38101)
                cl.enqueue_nd_range_kernel(self.queue,
                                           self.segred_large_34174_var,
                                           ((np.long(num_groups_38096) * np.long(group_sizze_34265)),),
                                           (np.long(group_sizze_34265),))
                if synchronous:
                  self.queue.finish()
            res_mem_37503 = mem_37502
          res_mem_37504 = res_mem_37503
        res_mem_37505 = res_mem_37504
      res_mem_37506 = res_mem_37505
    mem_37218 = None
    res_mem_37449 = None
    i_31033 = (sizze_30756 - np.int32(1))
    x_31034 = sle32(np.int32(0), i_31033)
    index_certs_31037 = True
    assert x_31034, ("Error at bfastfinal.fut:112:1-241:20 -> bfastfinal.fut:167:5-174:25 -> /futlib/soacs.fut:51:3-37 -> /futlib/soacs.fut:51:19-23 -> bfastfinal.fut:171:30-91 -> bfastfinal.fut:28:13-20 -> /futlib/array.fut:18:29-34: %s%d%s%d%s" % ("Index [",
                                                                                                                                                                                                                                                          i_31033,
                                                                                                                                                                                                                                                          "] out of bounds for array of shape [",
                                                                                                                                                                                                                                                          sizze_30756,
                                                                                                                                                                                                                                                          "]."))
    suff_outer_par_34292 = (self.sizes["main.suff_outer_par_29"] <= sizze_30757)
    suff_intra_par_34464 = (self.sizes["main.suff_intra_par_30"] <= sizze_30756)
    intra_suff_and_fits_34467 = (fits_34050 and suff_intra_par_34464)
    binop_x_37531 = sext_i32_i64(sizze_30757)
    bytes_37530 = (np.int64(4) * binop_x_37531)
    if suff_outer_par_34292:
      group_sizze_34401 = self.sizes["main.group_size_34337"]
      y_34402 = (group_sizze_34401 - np.int32(1))
      x_34403 = (sizze_30757 + y_34402)
      num_groups_34404 = squot32(x_34403, group_sizze_34401)
      num_threads_34405 = (group_sizze_34401 * num_groups_34404)
      mem_37510 = opencl_alloc(self, bytes_37219, "mem_37510")
      self.futhark__map_transpose_f32(mem_37510, np.int32(0), images_mem_37201,
                                      np.int32(0), np.int32(1), sizze_30758,
                                      sizze_30757, (sizze_30757 * sizze_30758),
                                      (sizze_30757 * sizze_30758))
      mem_37514 = opencl_alloc(self, bytes_37457, "mem_37514")
      self.futhark__map_transpose_f32(mem_37514, np.int32(0), res_mem_37506,
                                      np.int32(0), np.int32(1), sizze_30756,
                                      sizze_30757, (sizze_30757 * sizze_30756),
                                      (sizze_30757 * sizze_30756))
      mem_37532 = opencl_alloc(self, bytes_37530, "mem_37532")
      mem_37536 = opencl_alloc(self, bytes_37457, "mem_37536")
      mem_37540 = opencl_alloc(self, bytes_37457, "mem_37540")
      binop_x_37516 = sext_i32_i64(sizze_30756)
      bytes_37515 = (np.int64(4) * binop_x_37516)
      num_threads64_37765 = sext_i32_i64(num_threads_34405)
      total_sizze_37766 = (bytes_37515 * num_threads64_37765)
      mem_37517 = opencl_alloc(self, total_sizze_37766, "mem_37517")
      total_sizze_37767 = (bytes_37515 * num_threads64_37765)
      mem_37520 = opencl_alloc(self, total_sizze_37767, "mem_37520")
      total_sizze_37768 = (bytes_37515 * num_threads64_37765)
      mem_37523 = opencl_alloc(self, total_sizze_37768, "mem_37523")
      total_sizze_37769 = (bytes_37515 * num_threads64_37765)
      mem_37526 = opencl_alloc(self, total_sizze_37769, "mem_37526")
      if ((1 * (np.long(num_groups_34404) * np.long(group_sizze_34401))) != 0):
        self.map_34343_var.set_args(np.int32(sizze_30756),
                                    np.int32(sizze_30757), np.int32(i_31033),
                                    mem_37510, mem_37514, mem_37517, mem_37520,
                                    mem_37523, mem_37526, mem_37532, mem_37536,
                                    mem_37540)
        cl.enqueue_nd_range_kernel(self.queue, self.map_34343_var,
                                   ((np.long(num_groups_34404) * np.long(group_sizze_34401)),),
                                   (np.long(group_sizze_34401),))
        if synchronous:
          self.queue.finish()
      mem_37510 = None
      mem_37514 = None
      mem_37517 = None
      mem_37520 = None
      mem_37523 = None
      mem_37526 = None
      mem_37544 = opencl_alloc(self, bytes_37457, "mem_37544")
      self.futhark__map_transpose_f32(mem_37544, np.int32(0), mem_37536,
                                      np.int32(0), np.int32(1), sizze_30757,
                                      sizze_30756, (sizze_30757 * sizze_30756),
                                      (sizze_30757 * sizze_30756))
      mem_37536 = None
      mem_37549 = opencl_alloc(self, bytes_37457, "mem_37549")
      self.futhark__map_transpose_i32(mem_37549, np.int32(0), mem_37540,
                                      np.int32(0), np.int32(1), sizze_30757,
                                      sizze_30756, (sizze_30757 * sizze_30756),
                                      (sizze_30757 * sizze_30756))
      mem_37540 = None
      res_mem_37596 = mem_37532
      res_mem_37597 = mem_37544
      res_mem_37598 = mem_37549
    else:
      if intra_suff_and_fits_34467:
        mem_37565 = opencl_alloc(self, bytes_37530, "mem_37565")
        mem_37569 = opencl_alloc(self, bytes_37457, "mem_37569")
        mem_37573 = opencl_alloc(self, bytes_37457, "mem_37573")
        binop_x_37552 = sext_i32_i64(sizze_30756)
        bytes_37551 = (np.int64(4) * binop_x_37552)
        if ((1 * (np.long(sizze_30757) * np.long(sizze_30756))) != 0):
          self.map_intra_group_34303_var.set_args(cl.LocalMemory(np.long(bytes_37551)),
                                                  cl.LocalMemory(np.long(bytes_37551)),
                                                  cl.LocalMemory(np.long(bytes_37551)),
                                                  cl.LocalMemory(np.long(bytes_37551)),
                                                  np.int32(sizze_30756),
                                                  np.int32(sizze_30757),
                                                  np.int32(sizze_30758),
                                                  np.int32(i_31033),
                                                  images_mem_37201,
                                                  res_mem_37506, mem_37565,
                                                  mem_37569, mem_37573)
          cl.enqueue_nd_range_kernel(self.queue, self.map_intra_group_34303_var,
                                     ((np.long(sizze_30757) * np.long(sizze_30756)),),
                                     (np.long(sizze_30756),))
          if synchronous:
            self.queue.finish()
        res_mem_37593 = mem_37565
        res_mem_37594 = mem_37569
        res_mem_37595 = mem_37573
      else:
        group_sizze_34637 = self.sizes["main.group_size_34612"]
        max_num_groups_34638 = self.sizes["main.max_num_groups_34614"]
        group_sizze_34639 = sext_i32_i64(group_sizze_34637)
        max_num_groups_34640 = sext_i32_i64(max_num_groups_34638)
        y_34641 = (group_sizze_34639 - np.int64(1))
        x_34642 = (y_34641 + binop_x_37459)
        w_div_group_sizze_34643 = squot64(x_34642, group_sizze_34639)
        num_groups_maybe_zzero_34644 = smin64(max_num_groups_34640,
                                              w_div_group_sizze_34643)
        num_groups_34645 = smax64(np.int64(1), num_groups_maybe_zzero_34644)
        num_threads_34646 = (group_sizze_34639 * num_groups_34645)
        num_groups_34647 = sext_i64_i32(num_groups_34645)
        num_threads_34648 = sext_i64_i32(num_threads_34646)
        mem_37577 = opencl_alloc(self, bytes_37457, "mem_37577")
        mem_37581 = opencl_alloc(self, bytes_37457, "mem_37581")
        if ((1 * (np.long(num_groups_34647) * np.long(group_sizze_34637))) != 0):
          self.scan_stage1_34630_var.set_args(np.int32(sizze_30756),
                                              np.int32(sizze_30757),
                                              np.int32(sizze_30758),
                                              np.int32(num_groups_34647),
                                              images_mem_37201, res_mem_37506,
                                              mem_37577, mem_37581)
          cl.enqueue_nd_range_kernel(self.queue, self.scan_stage1_34630_var,
                                     ((np.long(num_groups_34647) * np.long(group_sizze_34637)),),
                                     (np.long(group_sizze_34637),))
          if synchronous:
            self.queue.finish()
        if ((1 * (np.long(np.int32(1)) * np.long(num_groups_34647))) != 0):
          self.scan_stage2_38172_var.set_args(cl.LocalMemory(np.long((np.int32(4) * num_groups_34647))),
                                              np.int32(sizze_30756),
                                              np.int32(sizze_30757),
                                              np.int32(num_groups_34647),
                                              mem_37577)
          cl.enqueue_nd_range_kernel(self.queue, self.scan_stage2_38172_var,
                                     ((np.long(np.int32(1)) * np.long(num_groups_34647)),),
                                     (np.long(num_groups_34647),))
          if synchronous:
            self.queue.finish()
        group_sizze_38188 = self.sizes["main.group_size_38188"]
        num_groups_38189 = squot32((((sizze_30757 * sizze_30756) + sext_i32_i32(group_sizze_38188)) - np.int32(1)),
                                   sext_i32_i32(group_sizze_38188))
        if ((1 * (np.long(num_groups_38189) * np.long(group_sizze_38188))) != 0):
          self.scan_stage3_38185_var.set_args(np.int32(sizze_30756),
                                              np.int32(sizze_30757),
                                              np.int32(num_groups_34647),
                                              mem_37577)
          cl.enqueue_nd_range_kernel(self.queue, self.scan_stage3_38185_var,
                                     ((np.long(num_groups_38189) * np.long(group_sizze_38188)),),
                                     (np.long(group_sizze_38188),))
          if synchronous:
            self.queue.finish()
        mem_37584 = opencl_alloc(self, bytes_37530, "mem_37584")
        group_sizze_38195 = self.sizes["main.group_size_38195"]
        num_groups_38196 = squot32(((sizze_30757 + sext_i32_i32(group_sizze_38195)) - np.int32(1)),
                                   sext_i32_i32(group_sizze_38195))
        if ((1 * (np.long(num_groups_38196) * np.long(group_sizze_38195))) != 0):
          self.copy_38192_var.set_args(np.int32(sizze_30756),
                                       np.int32(sizze_30757), np.int32(i_31033),
                                       mem_37577, mem_37584)
          cl.enqueue_nd_range_kernel(self.queue, self.copy_38192_var,
                                     ((np.long(num_groups_38196) * np.long(group_sizze_38195)),),
                                     (np.long(group_sizze_38195),))
          if synchronous:
            self.queue.finish()
        mem_37588 = opencl_alloc(self, bytes_37457, "mem_37588")
        group_sizze_38200 = self.sizes["main.group_size_38200"]
        num_groups_38201 = squot32((((sizze_30757 * sizze_30756) + sext_i32_i32(group_sizze_38200)) - np.int32(1)),
                                   sext_i32_i32(group_sizze_38200))
        if ((1 * (np.long(num_groups_38201) * np.long(group_sizze_38200))) != 0):
          self.replicate_38197_var.set_args(np.int32(sizze_30756),
                                            np.int32(sizze_30757), mem_37588)
          cl.enqueue_nd_range_kernel(self.queue, self.replicate_38197_var,
                                     ((np.long(num_groups_38201) * np.long(group_sizze_38200)),),
                                     (np.long(group_sizze_38200),))
          if synchronous:
            self.queue.finish()
        mem_37592 = opencl_alloc(self, bytes_37457, "mem_37592")
        group_sizze_38205 = self.sizes["main.group_size_38205"]
        num_groups_38206 = squot32((((sizze_30757 * sizze_30756) + sext_i32_i32(group_sizze_38205)) - np.int32(1)),
                                   sext_i32_i32(group_sizze_38205))
        if ((1 * (np.long(num_groups_38206) * np.long(group_sizze_38205))) != 0):
          self.replicate_38202_var.set_args(np.int32(sizze_30756),
                                            np.int32(sizze_30757), mem_37592)
          cl.enqueue_nd_range_kernel(self.queue, self.replicate_38202_var,
                                     ((np.long(num_groups_38206) * np.long(group_sizze_38205)),),
                                     (np.long(group_sizze_38205),))
          if synchronous:
            self.queue.finish()
        group_sizze_34709 = self.sizes["main.group_size_34512"]
        y_34710 = (group_sizze_34709 - np.int32(1))
        x_34711 = (num_threads_33989 + y_34710)
        num_groups_34712 = squot32(x_34711, group_sizze_34709)
        num_threads_34713 = (group_sizze_34709 * num_groups_34712)
        if ((1 * (np.long(num_groups_34712) * np.long(group_sizze_34709))) != 0):
          self.map_34518_var.set_args(np.int32(sizze_30756),
                                      np.int32(sizze_30757), mem_37577,
                                      mem_37581, mem_37588, mem_37592)
          cl.enqueue_nd_range_kernel(self.queue, self.map_34518_var,
                                     ((np.long(num_groups_34712) * np.long(group_sizze_34709)),),
                                     (np.long(group_sizze_34709),))
          if synchronous:
            self.queue.finish()
        mem_37577 = None
        mem_37581 = None
        res_mem_37593 = mem_37584
        res_mem_37594 = mem_37588
        res_mem_37595 = mem_37592
      res_mem_37596 = res_mem_37593
      res_mem_37597 = res_mem_37594
      res_mem_37598 = res_mem_37595
    res_mem_37506 = None
    suff_outer_par_34733 = (self.sizes["main.suff_outer_par_33"] <= sizze_30757)
    num_threads_34740 = (sizze_30757 * n_30761)
    suff_intra_par_34871 = (self.sizes["main.suff_intra_par_34"] <= n_30761)
    intra_suff_and_fits_34874 = (fits_32534 and suff_intra_par_34871)
    if suff_outer_par_34733:
      group_sizze_34819 = self.sizes["main.group_size_34766"]
      y_34820 = (group_sizze_34819 - np.int32(1))
      x_34821 = (sizze_30757 + y_34820)
      num_groups_34822 = squot32(x_34821, group_sizze_34819)
      num_threads_34823 = (group_sizze_34819 * num_groups_34822)
      mem_37602 = opencl_alloc(self, bytes_37219, "mem_37602")
      self.futhark__map_transpose_f32(mem_37602, np.int32(0), images_mem_37201,
                                      np.int32(0), np.int32(1), sizze_30758,
                                      sizze_30757, (sizze_30757 * sizze_30758),
                                      (sizze_30757 * sizze_30758))
      mem_37606 = opencl_alloc(self, bytes_37457, "mem_37606")
      self.futhark__map_transpose_f32(mem_37606, np.int32(0), res_mem_37597,
                                      np.int32(0), np.int32(1), sizze_30756,
                                      sizze_30757, (sizze_30757 * sizze_30756),
                                      (sizze_30757 * sizze_30756))
      mem_37609 = opencl_alloc(self, bytes_37530, "mem_37609")
      mem_37612 = opencl_alloc(self, bytes_37530, "mem_37612")
      mem_37615 = opencl_alloc(self, bytes_37530, "mem_37615")
      if ((1 * (np.long(num_groups_34822) * np.long(group_sizze_34819))) != 0):
        self.map_34772_var.set_args(np.int32(sizze_30757), np.int32(n_30761),
                                    np.float32(hfrac_30763),
                                    np.int32(res_30778), mem_37602, mem_37606,
                                    mem_37609, mem_37612, mem_37615)
        cl.enqueue_nd_range_kernel(self.queue, self.map_34772_var,
                                   ((np.long(num_groups_34822) * np.long(group_sizze_34819)),),
                                   (np.long(group_sizze_34819),))
        if synchronous:
          self.queue.finish()
      mem_37602 = None
      mem_37606 = None
      res_mem_37646 = mem_37609
      res_mem_37647 = mem_37612
      res_mem_37648 = mem_37615
    else:
      if intra_suff_and_fits_34874:
        mem_37624 = opencl_alloc(self, bytes_37530, "mem_37624")
        mem_37627 = opencl_alloc(self, bytes_37530, "mem_37627")
        mem_37630 = opencl_alloc(self, bytes_37530, "mem_37630")
        binop_x_37617 = sext_i32_i64(n_30761)
        bytes_37616 = (np.int64(4) * binop_x_37617)
        if ((1 * (np.long(sizze_30757) * np.long(n_30761))) != 0):
          self.map_intra_group_34741_var.set_args(cl.LocalMemory(np.long(bytes_37616)),
                                                  cl.LocalMemory(np.long(bytes_37616)),
                                                  np.int32(sizze_30756),
                                                  np.int32(sizze_30757),
                                                  np.int32(sizze_30758),
                                                  np.int32(n_30761),
                                                  np.float32(hfrac_30763),
                                                  np.int32(res_30778),
                                                  images_mem_37201,
                                                  res_mem_37597, mem_37624,
                                                  mem_37627, mem_37630)
          cl.enqueue_nd_range_kernel(self.queue, self.map_intra_group_34741_var,
                                     ((np.long(sizze_30757) * np.long(n_30761)),),
                                     (np.long(n_30761),))
          if synchronous:
            self.queue.finish()
        res_mem_37643 = mem_37624
        res_mem_37644 = mem_37627
        res_mem_37645 = mem_37630
      else:
        total_num_elements_34982 = sext_i32_i64(num_threads_34740)
        group_sizze_34983 = self.sizes["main.group_size_34959"]
        max_num_groups_34984 = self.sizes["main.max_num_groups_34961"]
        group_sizze_34985 = sext_i32_i64(group_sizze_34983)
        max_num_groups_34986 = sext_i32_i64(max_num_groups_34984)
        y_34987 = (group_sizze_34985 - np.int64(1))
        x_34988 = (total_num_elements_34982 + y_34987)
        w_div_group_sizze_34989 = squot64(x_34988, group_sizze_34985)
        num_groups_maybe_zzero_34990 = smin64(max_num_groups_34986,
                                              w_div_group_sizze_34989)
        num_groups_34991 = smax64(np.int64(1), num_groups_maybe_zzero_34990)
        num_threads_34992 = (group_sizze_34985 * num_groups_34991)
        num_groups_34993 = sext_i64_i32(num_groups_34991)
        num_threads_34994 = sext_i64_i32(num_threads_34992)
        mem_37633 = opencl_alloc(self, bytes_37530, "mem_37633")
        if slt32((n_30761 * np.int32(2)), group_sizze_34983):
          segment_sizze_nonzzero_38228 = smax32(np.int32(1), n_30761)
          if ((1 * (np.long(num_groups_34993) * np.long(group_sizze_34983))) != 0):
            self.segred_small_34977_var.set_args(np.int32(sizze_30757),
                                                 np.int32(sizze_30758),
                                                 np.int32(n_30761),
                                                 np.int32(num_groups_34993),
                                                 images_mem_37201, mem_37633,
                                                 np.int32(segment_sizze_nonzzero_38228))
            cl.enqueue_nd_range_kernel(self.queue, self.segred_small_34977_var,
                                       ((np.long(num_groups_34993) * np.long(group_sizze_34983)),),
                                       (np.long(group_sizze_34983),))
            if synchronous:
              self.queue.finish()
        else:
          num_groups_38239 = (squot32(((num_groups_34993 + smax32(np.int32(1),
                                                                  sizze_30757)) - np.int32(1)),
                                      smax32(np.int32(1),
                                             sizze_30757)) * sizze_30757)
          num_threads_38240 = (num_groups_38239 * group_sizze_34983)
          thread_per_segment_38241 = (squot32(((num_groups_34993 + smax32(np.int32(1),
                                                                          sizze_30757)) - np.int32(1)),
                                              smax32(np.int32(1),
                                                     sizze_30757)) * group_sizze_34983)
          group_res_arr_mem_38242 = opencl_alloc(self,
                                                 (np.int32(4) * num_groups_38239),
                                                 "group_res_arr_mem_38242")
          counter_mem_38244 = self.counter_mem_38244
          if ((1 * (np.long(num_groups_38239) * np.long(group_sizze_34983))) != 0):
            self.segred_large_34977_var.set_args(np.int32(sizze_30757),
                                                 np.int32(sizze_30758),
                                                 np.int32(n_30761),
                                                 np.int32(num_groups_34993),
                                                 images_mem_37201, mem_37633,
                                                 np.int32(thread_per_segment_38241),
                                                 group_res_arr_mem_38242,
                                                 counter_mem_38244)
            cl.enqueue_nd_range_kernel(self.queue, self.segred_large_34977_var,
                                       ((np.long(num_groups_38239) * np.long(group_sizze_34983)),),
                                       (np.long(group_sizze_34983),))
            if synchronous:
              self.queue.finish()
        group_sizze_35011 = self.sizes["main.group_size_34934"]
        max_num_groups_35012 = self.sizes["main.max_num_groups_34936"]
        group_sizze_35013 = sext_i32_i64(group_sizze_35011)
        max_num_groups_35014 = sext_i32_i64(max_num_groups_35012)
        y_35015 = (group_sizze_35013 - np.int64(1))
        x_35016 = (total_num_elements_34982 + y_35015)
        w_div_group_sizze_35017 = squot64(x_35016, group_sizze_35013)
        num_groups_maybe_zzero_35018 = smin64(max_num_groups_35014,
                                              w_div_group_sizze_35017)
        num_groups_35019 = smax64(np.int64(1), num_groups_maybe_zzero_35018)
        num_threads_35020 = (group_sizze_35013 * num_groups_35019)
        num_groups_35021 = sext_i64_i32(num_groups_35019)
        num_threads_35022 = sext_i64_i32(num_threads_35020)
        mem_37636 = opencl_alloc(self, bytes_37530, "mem_37636")
        if slt32((n_30761 * np.int32(2)), group_sizze_35011):
          segment_sizze_nonzzero_38263 = smax32(np.int32(1), n_30761)
          if ((1 * (np.long(num_groups_35021) * np.long(group_sizze_35011))) != 0):
            self.segred_small_34952_var.set_args(np.int32(sizze_30756),
                                                 np.int32(sizze_30757),
                                                 np.int32(n_30761),
                                                 np.int32(num_groups_35021),
                                                 res_mem_37597, mem_37633,
                                                 mem_37636,
                                                 np.int32(segment_sizze_nonzzero_38263))
            cl.enqueue_nd_range_kernel(self.queue, self.segred_small_34952_var,
                                       ((np.long(num_groups_35021) * np.long(group_sizze_35011)),),
                                       (np.long(group_sizze_35011),))
            if synchronous:
              self.queue.finish()
        else:
          num_groups_38274 = (squot32(((num_groups_35021 + smax32(np.int32(1),
                                                                  sizze_30757)) - np.int32(1)),
                                      smax32(np.int32(1),
                                             sizze_30757)) * sizze_30757)
          num_threads_38275 = (num_groups_38274 * group_sizze_35011)
          thread_per_segment_38276 = (squot32(((num_groups_35021 + smax32(np.int32(1),
                                                                          sizze_30757)) - np.int32(1)),
                                              smax32(np.int32(1),
                                                     sizze_30757)) * group_sizze_35011)
          group_res_arr_mem_38277 = opencl_alloc(self,
                                                 (np.int32(4) * num_groups_38274),
                                                 "group_res_arr_mem_38277")
          counter_mem_38279 = self.counter_mem_38279
          if ((1 * (np.long(num_groups_38274) * np.long(group_sizze_35011))) != 0):
            self.segred_large_34952_var.set_args(np.int32(sizze_30756),
                                                 np.int32(sizze_30757),
                                                 np.int32(n_30761),
                                                 np.int32(num_groups_35021),
                                                 res_mem_37597, mem_37633,
                                                 mem_37636,
                                                 np.int32(thread_per_segment_38276),
                                                 group_res_arr_mem_38277,
                                                 counter_mem_38279)
            cl.enqueue_nd_range_kernel(self.queue, self.segred_large_34952_var,
                                       ((np.long(num_groups_38274) * np.long(group_sizze_35011)),),
                                       (np.long(group_sizze_35011),))
            if synchronous:
              self.queue.finish()
        group_sizze_35037 = self.sizes["main.group_size_34911"]
        y_35038 = (group_sizze_35037 - np.int32(1))
        x_35039 = (sizze_30757 + y_35038)
        num_groups_35040 = squot32(x_35039, group_sizze_35037)
        num_threads_35041 = (group_sizze_35037 * num_groups_35040)
        mem_37639 = opencl_alloc(self, bytes_37530, "mem_37639")
        mem_37642 = opencl_alloc(self, bytes_37530, "mem_37642")
        if ((1 * (np.long(num_groups_35040) * np.long(group_sizze_35037))) != 0):
          self.map_34917_var.set_args(np.int32(sizze_30757),
                                      np.float32(hfrac_30763),
                                      np.int32(res_30778), mem_37633, mem_37636,
                                      mem_37639, mem_37642)
          cl.enqueue_nd_range_kernel(self.queue, self.map_34917_var,
                                     ((np.long(num_groups_35040) * np.long(group_sizze_35037)),),
                                     (np.long(group_sizze_35037),))
          if synchronous:
            self.queue.finish()
        mem_37636 = None
        res_mem_37643 = mem_37639
        res_mem_37644 = mem_37633
        res_mem_37645 = mem_37642
      res_mem_37646 = res_mem_37643
      res_mem_37647 = res_mem_37644
      res_mem_37648 = res_mem_37645
    group_sizze_35063 = self.sizes["main.group_size_35062"]
    max_num_groups_35065 = self.sizes["main.max_num_groups_35064"]
    group_sizze_35066 = sext_i32_i64(group_sizze_35063)
    max_num_groups_35067 = sext_i32_i64(max_num_groups_35065)
    y_35068 = (group_sizze_35066 - np.int64(1))
    x_35069 = (y_35068 + binop_x_37531)
    w_div_group_sizze_35070 = squot64(x_35069, group_sizze_35066)
    num_groups_maybe_zzero_35071 = smin64(max_num_groups_35067,
                                          w_div_group_sizze_35070)
    num_groups_35072 = smax64(np.int64(1), num_groups_maybe_zzero_35071)
    num_threads_35073 = (group_sizze_35066 * num_groups_35072)
    num_groups_35074 = sext_i64_i32(num_groups_35072)
    num_threads_35075 = sext_i64_i32(num_threads_35073)
    mem_37651 = opencl_alloc(self, np.int64(4), "mem_37651")
    counter_mem_38300 = self.counter_mem_38300
    group_res_arr_mem_38302 = opencl_alloc(self,
                                           (np.int32(4) * num_groups_35074),
                                           "group_res_arr_mem_38302")
    num_threads_38304 = (group_sizze_35063 * num_groups_35074)
    if ((1 * (np.long(num_groups_35074) * np.long(group_sizze_35063))) != 0):
      self.segred_nonseg_35080_var.set_args(np.int32(sizze_30757),
                                            np.int32(num_groups_35074),
                                            res_mem_37646, mem_37651,
                                            counter_mem_38300,
                                            group_res_arr_mem_38302,
                                            np.int32(num_threads_38304))
      cl.enqueue_nd_range_kernel(self.queue, self.segred_nonseg_35080_var,
                                 ((np.long(num_groups_35074) * np.long(group_sizze_35063)),),
                                 (np.long(group_sizze_35063),))
      if synchronous:
        self.queue.finish()
    read_res_38517 = np.empty(1, dtype=ct.c_int32)
    cl.enqueue_copy(self.queue, read_res_38517, mem_37651,
                    device_offset=np.long(np.int32(0)), is_blocking=True)
    res_31136 = read_res_38517[0]
    mem_37651 = None
    suff_outer_par_35083 = (self.sizes["main.suff_outer_par_35"] <= sizze_30757)
    num_threads_35088 = (sizze_30757 * res_31136)
    fits_35166 = sle32(res_31136, max_group_sizze_32061)
    suff_intra_par_35164 = (self.sizes["main.suff_intra_par_36"] <= res_31136)
    intra_suff_and_fits_35167 = (suff_intra_par_35164 and fits_35166)
    if suff_outer_par_35083:
      group_sizze_35135 = self.sizes["main.group_size_35103"]
      y_35136 = (group_sizze_35135 - np.int32(1))
      x_35137 = (sizze_30757 + y_35136)
      num_groups_35138 = squot32(x_35137, group_sizze_35135)
      num_threads_35139 = (group_sizze_35135 * num_groups_35138)
      mem_37654 = opencl_alloc(self, bytes_37530, "mem_37654")
      if ((1 * (np.long(num_groups_35138) * np.long(group_sizze_35135))) != 0):
        self.map_35109_var.set_args(np.int32(sizze_30756),
                                    np.int32(sizze_30757), np.int32(res_31136),
                                    res_mem_37597, res_mem_37646, res_mem_37647,
                                    mem_37654)
        cl.enqueue_nd_range_kernel(self.queue, self.map_35109_var,
                                   ((np.long(num_groups_35138) * np.long(group_sizze_35135)),),
                                   (np.long(group_sizze_35135),))
        if synchronous:
          self.queue.finish()
      res_mem_37665 = mem_37654
    else:
      if intra_suff_and_fits_35167:
        mem_37660 = opencl_alloc(self, bytes_37530, "mem_37660")
        binop_x_37656 = sext_i32_i64(res_31136)
        bytes_37655 = (np.int64(4) * binop_x_37656)
        if ((1 * (np.long(sizze_30757) * np.long(res_31136))) != 0):
          self.map_intra_group_35089_var.set_args(cl.LocalMemory(np.long(bytes_37655)),
                                                  np.int32(sizze_30756),
                                                  np.int32(sizze_30757),
                                                  np.int32(res_31136),
                                                  res_mem_37597, res_mem_37646,
                                                  res_mem_37647, mem_37660)
          cl.enqueue_nd_range_kernel(self.queue, self.map_intra_group_35089_var,
                                     ((np.long(sizze_30757) * np.long(res_31136)),),
                                     (np.long(res_31136),))
          if synchronous:
            self.queue.finish()
        res_mem_37664 = mem_37660
      else:
        total_num_elements_35214 = sext_i32_i64(num_threads_35088)
        group_sizze_35215 = self.sizes["main.group_size_35190"]
        max_num_groups_35216 = self.sizes["main.max_num_groups_35192"]
        group_sizze_35217 = sext_i32_i64(group_sizze_35215)
        max_num_groups_35218 = sext_i32_i64(max_num_groups_35216)
        y_35219 = (group_sizze_35217 - np.int64(1))
        x_35220 = (total_num_elements_35214 + y_35219)
        w_div_group_sizze_35221 = squot64(x_35220, group_sizze_35217)
        num_groups_maybe_zzero_35222 = smin64(max_num_groups_35218,
                                              w_div_group_sizze_35221)
        num_groups_35223 = smax64(np.int64(1), num_groups_maybe_zzero_35222)
        num_threads_35224 = (group_sizze_35217 * num_groups_35223)
        num_groups_35225 = sext_i64_i32(num_groups_35223)
        num_threads_35226 = sext_i64_i32(num_threads_35224)
        mem_37663 = opencl_alloc(self, bytes_37530, "mem_37663")
        if slt32((res_31136 * np.int32(2)), group_sizze_35215):
          segment_sizze_nonzzero_38332 = smax32(np.int32(1), res_31136)
          if ((1 * (np.long(num_groups_35225) * np.long(group_sizze_35215))) != 0):
            self.segred_small_35208_var.set_args(np.int32(sizze_30756),
                                                 np.int32(sizze_30757),
                                                 np.int32(res_31136),
                                                 np.int32(num_groups_35225),
                                                 res_mem_37597, res_mem_37646,
                                                 res_mem_37647, mem_37663,
                                                 np.int32(segment_sizze_nonzzero_38332))
            cl.enqueue_nd_range_kernel(self.queue, self.segred_small_35208_var,
                                       ((np.long(num_groups_35225) * np.long(group_sizze_35215)),),
                                       (np.long(group_sizze_35215),))
            if synchronous:
              self.queue.finish()
        else:
          num_groups_38343 = (squot32(((num_groups_35225 + smax32(np.int32(1),
                                                                  sizze_30757)) - np.int32(1)),
                                      smax32(np.int32(1),
                                             sizze_30757)) * sizze_30757)
          num_threads_38344 = (num_groups_38343 * group_sizze_35215)
          thread_per_segment_38345 = (squot32(((num_groups_35225 + smax32(np.int32(1),
                                                                          sizze_30757)) - np.int32(1)),
                                              smax32(np.int32(1),
                                                     sizze_30757)) * group_sizze_35215)
          group_res_arr_mem_38346 = opencl_alloc(self,
                                                 (np.int32(4) * num_groups_38343),
                                                 "group_res_arr_mem_38346")
          counter_mem_38348 = self.counter_mem_38348
          if ((1 * (np.long(num_groups_38343) * np.long(group_sizze_35215))) != 0):
            self.segred_large_35208_var.set_args(np.int32(sizze_30756),
                                                 np.int32(sizze_30757),
                                                 np.int32(res_31136),
                                                 np.int32(num_groups_35225),
                                                 res_mem_37597, res_mem_37646,
                                                 res_mem_37647, mem_37663,
                                                 np.int32(thread_per_segment_38345),
                                                 group_res_arr_mem_38346,
                                                 counter_mem_38348)
            cl.enqueue_nd_range_kernel(self.queue, self.segred_large_35208_var,
                                       ((np.long(num_groups_38343) * np.long(group_sizze_35215)),),
                                       (np.long(group_sizze_35215),))
            if synchronous:
              self.queue.finish()
        res_mem_37664 = mem_37663
      res_mem_37665 = res_mem_37664
    arg_31158 = (sizze_30756 - n_30761)
    bounds_invalid_upwards_31159 = slt32(arg_31158, np.int32(0))
    eq_x_zz_31160 = (np.int32(0) == arg_31158)
    not_p_31161 = not(bounds_invalid_upwards_31159)
    p_and_eq_x_y_31162 = (eq_x_zz_31160 and not_p_31161)
    dim_zzero_31163 = (bounds_invalid_upwards_31159 or p_and_eq_x_y_31162)
    both_empty_31164 = (eq_x_zz_31160 and dim_zzero_31163)
    eq_x_y_31165 = (arg_31158 == np.int32(0))
    p_and_eq_x_y_31166 = (bounds_invalid_upwards_31159 and eq_x_y_31165)
    dim_match_31167 = (not_p_31161 or p_and_eq_x_y_31166)
    empty_or_match_31168 = (both_empty_31164 or dim_match_31167)
    empty_or_match_cert_31169 = True
    assert empty_or_match_31168, ("Error at bfastfinal.fut:112:1-241:20 -> bfastfinal.fut:204:22-31 -> /futlib/array.fut:61:1-62:12: %s%s%s%d%s%s" % ("Function return value does not match shape of type ",
                                                                                                                                                      "*",
                                                                                                                                                      "[",
                                                                                                                                                      arg_31158,
                                                                                                                                                      "]",
                                                                                                                                                      "intrinsics.i32"))
    x_31171 = (np.int32(1) + n_30761)
    index_certs_31172 = True
    assert x_31034, ("Error at bfastfinal.fut:112:1-241:20 -> bfastfinal.fut:200:15-204:32 -> bfastfinal.fut:202:63-81: %s%d%s%d%s" % ("Index [",
                                                                                                                                       i_31033,
                                                                                                                                       "] out of bounds for array of shape [",
                                                                                                                                       sizze_30756,
                                                                                                                                       "]."))
    read_res_38519 = np.empty(1, dtype=ct.c_int32)
    cl.enqueue_copy(self.queue, read_res_38519, mappingindices_mem_37200,
                    device_offset=np.long((i_31033 * np.int32(4))),
                    is_blocking=True)
    arg_31173 = read_res_38519[0]
    res_31174 = sitofp_i32_f32(arg_31173)
    group_sizze_35306 = self.sizes["main.group_size_35286"]
    y_35307 = (group_sizze_35306 - np.int32(1))
    x_35308 = (arg_31158 + y_35307)
    num_groups_35309 = squot32(x_35308, group_sizze_35306)
    num_threads_35310 = (group_sizze_35306 * num_groups_35309)
    binop_x_37667 = sext_i32_i64(arg_31158)
    bytes_37666 = (np.int64(4) * binop_x_37667)
    mem_37668 = opencl_alloc(self, bytes_37666, "mem_37668")
    if ((1 * (np.long(num_groups_35309) * np.long(group_sizze_35306))) != 0):
      self.map_35292_var.set_args(np.float32(lam_30764), np.int32(arg_31158),
                                  np.int32(x_31171), np.float32(res_31174),
                                  mappingindices_mem_37200, mem_37668)
      cl.enqueue_nd_range_kernel(self.queue, self.map_35292_var,
                                 ((np.long(num_groups_35309) * np.long(group_sizze_35306)),),
                                 (np.long(group_sizze_35306),))
      if synchronous:
        self.queue.finish()
    suff_outer_par_35323 = (self.sizes["main.suff_outer_par_38"] <= sizze_30757)
    num_threads_35332 = (sizze_30757 * arg_31158)
    fits_35619 = sle32(arg_31158, max_group_sizze_32061)
    suff_intra_par_35617 = (self.sizes["main.suff_intra_par_39"] <= arg_31158)
    intra_suff_and_fits_35620 = (suff_intra_par_35617 and fits_35619)
    binop_x_37706 = sext_i32_i64(num_threads_35332)
    bytes_37704 = (np.int64(4) * binop_x_37706)
    if suff_outer_par_35323:
      group_sizze_35492 = self.sizes["main.group_size_35365"]
      y_35493 = (group_sizze_35492 - np.int32(1))
      x_35494 = (sizze_30757 + y_35493)
      num_groups_35495 = squot32(x_35494, group_sizze_35492)
      num_threads_35496 = (group_sizze_35492 * num_groups_35495)
      mem_37677 = opencl_alloc(self, bytes_37530, "mem_37677")
      mem_37680 = opencl_alloc(self, bytes_37530, "mem_37680")
      binop_x_37670 = sext_i32_i64(group_sizze_35492)
      bytes_37669 = (np.int64(4) * binop_x_37670)
      num_threads64_37783 = sext_i32_i64(num_threads_35496)
      total_sizze_37784 = (bytes_37669 * num_threads64_37783)
      mem_37674 = opencl_alloc(self, total_sizze_37784, "mem_37674")
      if ((1 * (np.long(num_groups_35495) * np.long(group_sizze_35492))) != 0):
        self.map_35371_var.set_args(np.int32(sizze_30756),
                                    np.int32(sizze_30757), np.int32(n_30761),
                                    np.int32(arg_31158), res_mem_37596,
                                    res_mem_37597, res_mem_37598, res_mem_37646,
                                    res_mem_37647, res_mem_37648, res_mem_37665,
                                    mem_37668, mem_37674, mem_37677, mem_37680)
        cl.enqueue_nd_range_kernel(self.queue, self.map_35371_var,
                                   ((np.long(num_groups_35495) * np.long(group_sizze_35492)),),
                                   (np.long(group_sizze_35492),))
        if synchronous:
          self.queue.finish()
      mem_37674 = None
      res_mem_37724 = mem_37677
      res_mem_37725 = mem_37680
    else:
      if intra_suff_and_fits_35620:
        mem_37694 = opencl_alloc(self, bytes_37530, "mem_37694")
        mem_37697 = opencl_alloc(self, bytes_37530, "mem_37697")
        if ((1 * (np.long(sizze_30757) * np.long(arg_31158))) != 0):
          self.map_intra_group_35333_var.set_args(cl.LocalMemory(np.long(bytes_37666)),
                                                  cl.LocalMemory(np.long(binop_x_37667)),
                                                  cl.LocalMemory(np.long(bytes_37666)),
                                                  cl.LocalMemory(np.long(bytes_37666)),
                                                  np.int32(sizze_30756),
                                                  np.int32(sizze_30757),
                                                  np.int32(n_30761),
                                                  np.int32(arg_31158),
                                                  res_mem_37596, res_mem_37597,
                                                  res_mem_37598, res_mem_37646,
                                                  res_mem_37647, res_mem_37648,
                                                  res_mem_37665, mem_37668,
                                                  mem_37694, mem_37697)
          cl.enqueue_nd_range_kernel(self.queue, self.map_intra_group_35333_var,
                                     ((np.long(sizze_30757) * np.long(arg_31158)),),
                                     (np.long(arg_31158),))
          if synchronous:
            self.queue.finish()
        res_mem_37722 = mem_37694
        res_mem_37723 = mem_37697
      else:
        group_sizze_35874 = self.sizes["main.group_size_35851"]
        y_35875 = (group_sizze_35874 - np.int32(1))
        x_35876 = (sizze_30757 + y_35875)
        num_groups_35877 = squot32(x_35876, group_sizze_35874)
        num_threads_35878 = (group_sizze_35874 * num_groups_35877)
        mem_37700 = opencl_alloc(self, bytes_37530, "mem_37700")
        mem_37703 = opencl_alloc(self, bytes_37530, "mem_37703")
        if ((1 * (np.long(num_groups_35877) * np.long(group_sizze_35874))) != 0):
          self.map_35857_var.set_args(np.int32(sizze_30757), res_mem_37596,
                                      res_mem_37647, res_mem_37648, mem_37700,
                                      mem_37703)
          cl.enqueue_nd_range_kernel(self.queue, self.map_35857_var,
                                     ((np.long(num_groups_35877) * np.long(group_sizze_35874)),),
                                     (np.long(group_sizze_35874),))
          if synchronous:
            self.queue.finish()
        group_sizze_35900 = self.sizes["main.group_size_35818"]
        max_num_groups_35901 = self.sizes["main.max_num_groups_35820"]
        group_sizze_35902 = sext_i32_i64(group_sizze_35900)
        max_num_groups_35903 = sext_i32_i64(max_num_groups_35901)
        y_35904 = (group_sizze_35902 - np.int64(1))
        x_35905 = (y_35904 + binop_x_37706)
        w_div_group_sizze_35906 = squot64(x_35905, group_sizze_35902)
        num_groups_maybe_zzero_35907 = smin64(max_num_groups_35903,
                                              w_div_group_sizze_35906)
        num_groups_35908 = smax64(np.int64(1), num_groups_maybe_zzero_35907)
        num_threads_35909 = (group_sizze_35902 * num_groups_35908)
        num_groups_35910 = sext_i64_i32(num_groups_35908)
        num_threads_35911 = sext_i64_i32(num_threads_35909)
        mem_37707 = opencl_alloc(self, bytes_37704, "mem_37707")
        if ((1 * (np.long(num_groups_35910) * np.long(group_sizze_35900))) != 0):
          self.scan_stage1_35836_var.set_args(np.int32(sizze_30756),
                                              np.int32(sizze_30757),
                                              np.int32(arg_31158),
                                              np.int32(num_groups_35910),
                                              res_mem_37597, res_mem_37646,
                                              res_mem_37647, res_mem_37665,
                                              mem_37703, mem_37707)
          cl.enqueue_nd_range_kernel(self.queue, self.scan_stage1_35836_var,
                                     ((np.long(num_groups_35910) * np.long(group_sizze_35900)),),
                                     (np.long(group_sizze_35900),))
          if synchronous:
            self.queue.finish()
        if ((1 * (np.long(np.int32(1)) * np.long(num_groups_35910))) != 0):
          self.scan_stage2_38417_var.set_args(cl.LocalMemory(np.long((np.int32(4) * num_groups_35910))),
                                              np.int32(sizze_30757),
                                              np.int32(arg_31158),
                                              np.int32(num_groups_35910),
                                              mem_37707)
          cl.enqueue_nd_range_kernel(self.queue, self.scan_stage2_38417_var,
                                     ((np.long(np.int32(1)) * np.long(num_groups_35910)),),
                                     (np.long(num_groups_35910),))
          if synchronous:
            self.queue.finish()
        group_sizze_38433 = self.sizes["main.group_size_38433"]
        num_groups_38434 = squot32((((sizze_30757 * arg_31158) + sext_i32_i32(group_sizze_38433)) - np.int32(1)),
                                   sext_i32_i32(group_sizze_38433))
        if ((1 * (np.long(num_groups_38434) * np.long(group_sizze_38433))) != 0):
          self.scan_stage3_38430_var.set_args(np.int32(sizze_30757),
                                              np.int32(arg_31158),
                                              np.int32(num_groups_35910),
                                              mem_37707)
          cl.enqueue_nd_range_kernel(self.queue, self.scan_stage3_38430_var,
                                     ((np.long(num_groups_38434) * np.long(group_sizze_38433)),),
                                     (np.long(group_sizze_38433),))
          if synchronous:
            self.queue.finish()
        group_sizze_35954 = self.sizes["main.group_size_35777"]
        max_num_groups_35955 = self.sizes["main.max_num_groups_35779"]
        group_sizze_35956 = sext_i32_i64(group_sizze_35954)
        max_num_groups_35957 = sext_i32_i64(max_num_groups_35955)
        y_35958 = (group_sizze_35956 - np.int64(1))
        x_35959 = (y_35958 + binop_x_37706)
        w_div_group_sizze_35960 = squot64(x_35959, group_sizze_35956)
        num_groups_maybe_zzero_35961 = smin64(max_num_groups_35957,
                                              w_div_group_sizze_35960)
        num_groups_35962 = smax64(np.int64(1), num_groups_maybe_zzero_35961)
        num_threads_35963 = (group_sizze_35956 * num_groups_35962)
        num_groups_35964 = sext_i64_i32(num_groups_35962)
        num_threads_35965 = sext_i64_i32(num_threads_35963)
        mem_37709 = opencl_alloc(self, binop_x_37531, "mem_37709")
        mem_37712 = opencl_alloc(self, bytes_37530, "mem_37712")
        mem_37715 = opencl_alloc(self, bytes_37530, "mem_37715")
        if slt32((arg_31158 * np.int32(2)), group_sizze_35954):
          segment_sizze_nonzzero_38439 = smax32(np.int32(1), arg_31158)
          if ((1 * (np.long(num_groups_35964) * np.long(group_sizze_35954))) != 0):
            self.segred_small_35795_var.set_args(np.int32(sizze_30757),
                                                 np.int32(arg_31158),
                                                 np.int32(num_groups_35964),
                                                 mem_37668, mem_37700,
                                                 mem_37703, mem_37707,
                                                 mem_37709, mem_37712,
                                                 mem_37715,
                                                 np.int32(segment_sizze_nonzzero_38439))
            cl.enqueue_nd_range_kernel(self.queue, self.segred_small_35795_var,
                                       ((np.long(num_groups_35964) * np.long(group_sizze_35954)),),
                                       (np.long(group_sizze_35954),))
            if synchronous:
              self.queue.finish()
        else:
          num_groups_38464 = (squot32(((num_groups_35964 + smax32(np.int32(1),
                                                                  sizze_30757)) - np.int32(1)),
                                      smax32(np.int32(1),
                                             sizze_30757)) * sizze_30757)
          num_threads_38465 = (num_groups_38464 * group_sizze_35954)
          thread_per_segment_38466 = (squot32(((num_groups_35964 + smax32(np.int32(1),
                                                                          sizze_30757)) - np.int32(1)),
                                              smax32(np.int32(1),
                                                     sizze_30757)) * group_sizze_35954)
          group_res_arr_mem_38467 = opencl_alloc(self,
                                                 (np.int32(1) * num_groups_38464),
                                                 "group_res_arr_mem_38467")
          group_res_arr_mem_38469 = opencl_alloc(self,
                                                 (np.int32(4) * num_groups_38464),
                                                 "group_res_arr_mem_38469")
          group_res_arr_mem_38471 = opencl_alloc(self,
                                                 (np.int32(4) * num_groups_38464),
                                                 "group_res_arr_mem_38471")
          counter_mem_38473 = self.counter_mem_38473
          if ((1 * (np.long(num_groups_38464) * np.long(group_sizze_35954))) != 0):
            self.segred_large_35795_var.set_args(np.int32(sizze_30757),
                                                 np.int32(arg_31158),
                                                 np.int32(num_groups_35964),
                                                 mem_37668, mem_37700,
                                                 mem_37703, mem_37707,
                                                 mem_37709, mem_37712,
                                                 mem_37715,
                                                 group_res_arr_mem_38467,
                                                 group_res_arr_mem_38469,
                                                 group_res_arr_mem_38471,
                                                 counter_mem_38473)
            cl.enqueue_nd_range_kernel(self.queue, self.segred_large_35795_var,
                                       ((np.long(num_groups_38464) * np.long(group_sizze_35954)),),
                                       (np.long(group_sizze_35954),))
            if synchronous:
              self.queue.finish()
        mem_37700 = None
        mem_37707 = None
        group_sizze_36003 = self.sizes["main.group_size_35727"]
        y_36004 = (group_sizze_36003 - np.int32(1))
        x_36005 = (sizze_30757 + y_36004)
        num_groups_36006 = squot32(x_36005, group_sizze_36003)
        num_threads_36007 = (group_sizze_36003 * num_groups_36006)
        mem_37718 = opencl_alloc(self, bytes_37530, "mem_37718")
        if ((sizze_30757 * np.int32(4)) != 0):
          cl.enqueue_copy(self.queue, mem_37718, mem_37715,
                          dest_offset=np.long(np.int32(0)),
                          src_offset=np.long(np.int32(0)),
                          byte_count=np.long((sizze_30757 * np.int32(4))))
        if synchronous:
          self.queue.finish()
        mem_37715 = None
        mem_37721 = opencl_alloc(self, bytes_37530, "mem_37721")
        if ((1 * (np.long(num_groups_36006) * np.long(group_sizze_36003))) != 0):
          self.map_35733_var.set_args(np.int32(sizze_30756),
                                      np.int32(sizze_30757), np.int32(n_30761),
                                      res_mem_37598, res_mem_37647, mem_37703,
                                      mem_37709, mem_37712, mem_37721)
          cl.enqueue_nd_range_kernel(self.queue, self.map_35733_var,
                                     ((np.long(num_groups_36006) * np.long(group_sizze_36003)),),
                                     (np.long(group_sizze_36003),))
          if synchronous:
            self.queue.finish()
        mem_37703 = None
        mem_37709 = None
        mem_37712 = None
        res_mem_37722 = mem_37721
        res_mem_37723 = mem_37718
      res_mem_37724 = res_mem_37722
      res_mem_37725 = res_mem_37723
    res_mem_37596 = None
    res_mem_37597 = None
    res_mem_37598 = None
    res_mem_37646 = None
    res_mem_37647 = None
    res_mem_37648 = None
    res_mem_37665 = None
    mem_37668 = None
    out_arrsizze_37800 = sizze_30757
    out_arrsizze_37802 = sizze_30757
    out_mem_37799 = res_mem_37724
    out_mem_37801 = res_mem_37725
    return (out_mem_37799, out_arrsizze_37800, out_mem_37801,
            out_arrsizze_37802)
  def futhark__map_transpose_i32(self, destmem_0, destoffset_1, srcmem_2,
                                 srcoffset_3, num_arrays_4, x_elems_5,
                                 y_elems_6, in_elems_7, out_elems_8):
    if ((num_arrays_4 == np.int32(0)) or ((x_elems_5 == np.int32(0)) or (y_elems_6 == np.int32(0)))):
      pass
    else:
      muly_10 = squot32(np.int32(16), x_elems_5)
      mulx_9 = squot32(np.int32(16), y_elems_6)
      if ((in_elems_7 == out_elems_8) and (((num_arrays_4 == np.int32(1)) or ((x_elems_5 * y_elems_6) == in_elems_7)) and ((x_elems_5 == np.int32(1)) or (y_elems_6 == np.int32(1))))):
        if ((in_elems_7 * np.int32(4)) != 0):
          cl.enqueue_copy(self.queue, destmem_0, srcmem_2,
                          dest_offset=np.long(destoffset_1),
                          src_offset=np.long(srcoffset_3),
                          byte_count=np.long((in_elems_7 * np.int32(4))))
        if synchronous:
          self.queue.finish()
      else:
        if (sle32(x_elems_5, np.int32(8)) and slt32(np.int32(16), y_elems_6)):
          if ((((1 * (np.long(squot32(((x_elems_5 + np.int32(16)) - np.int32(1)),
                                      np.int32(16))) * np.long(np.int32(16)))) * (np.long(squot32(((squot32(((y_elems_6 + muly_10) - np.int32(1)),
                                                                                                            muly_10) + np.int32(16)) - np.int32(1)),
                                                                                                  np.int32(16))) * np.long(np.int32(16)))) * (np.long(num_arrays_4) * np.long(np.int32(1)))) != 0):
            self.map_transpose_i32_low_width_var.set_args(np.int32(destoffset_1),
                                                          np.int32(srcoffset_3),
                                                          np.int32(num_arrays_4),
                                                          np.int32(x_elems_5),
                                                          np.int32(y_elems_6),
                                                          np.int32(in_elems_7),
                                                          np.int32(out_elems_8),
                                                          np.int32(mulx_9),
                                                          np.int32(muly_10),
                                                          destmem_0, srcmem_2)
            cl.enqueue_nd_range_kernel(self.queue,
                                       self.map_transpose_i32_low_width_var,
                                       ((np.long(squot32(((x_elems_5 + np.int32(16)) - np.int32(1)),
                                                         np.int32(16))) * np.long(np.int32(16))),
                                        (np.long(squot32(((squot32(((y_elems_6 + muly_10) - np.int32(1)),
                                                                   muly_10) + np.int32(16)) - np.int32(1)),
                                                         np.int32(16))) * np.long(np.int32(16))),
                                        (np.long(num_arrays_4) * np.long(np.int32(1)))),
                                       (np.long(np.int32(16)),
                                        np.long(np.int32(16)),
                                        np.long(np.int32(1))))
            if synchronous:
              self.queue.finish()
        else:
          if (sle32(y_elems_6, np.int32(8)) and slt32(np.int32(16), x_elems_5)):
            if ((((1 * (np.long(squot32(((squot32(((x_elems_5 + mulx_9) - np.int32(1)),
                                                  mulx_9) + np.int32(16)) - np.int32(1)),
                                        np.int32(16))) * np.long(np.int32(16)))) * (np.long(squot32(((y_elems_6 + np.int32(16)) - np.int32(1)),
                                                                                                    np.int32(16))) * np.long(np.int32(16)))) * (np.long(num_arrays_4) * np.long(np.int32(1)))) != 0):
              self.map_transpose_i32_low_height_var.set_args(np.int32(destoffset_1),
                                                             np.int32(srcoffset_3),
                                                             np.int32(num_arrays_4),
                                                             np.int32(x_elems_5),
                                                             np.int32(y_elems_6),
                                                             np.int32(in_elems_7),
                                                             np.int32(out_elems_8),
                                                             np.int32(mulx_9),
                                                             np.int32(muly_10),
                                                             destmem_0,
                                                             srcmem_2)
              cl.enqueue_nd_range_kernel(self.queue,
                                         self.map_transpose_i32_low_height_var,
                                         ((np.long(squot32(((squot32(((x_elems_5 + mulx_9) - np.int32(1)),
                                                                     mulx_9) + np.int32(16)) - np.int32(1)),
                                                           np.int32(16))) * np.long(np.int32(16))),
                                          (np.long(squot32(((y_elems_6 + np.int32(16)) - np.int32(1)),
                                                           np.int32(16))) * np.long(np.int32(16))),
                                          (np.long(num_arrays_4) * np.long(np.int32(1)))),
                                         (np.long(np.int32(16)),
                                          np.long(np.int32(16)),
                                          np.long(np.int32(1))))
              if synchronous:
                self.queue.finish()
          else:
            if (sle32(x_elems_5, np.int32(8)) and sle32(y_elems_6,
                                                        np.int32(8))):
              if ((1 * (np.long(squot32(((((num_arrays_4 * x_elems_5) * y_elems_6) + np.int32(256)) - np.int32(1)),
                                        np.int32(256))) * np.long(np.int32(256)))) != 0):
                self.map_transpose_i32_small_var.set_args(np.int32(destoffset_1),
                                                          np.int32(srcoffset_3),
                                                          np.int32(num_arrays_4),
                                                          np.int32(x_elems_5),
                                                          np.int32(y_elems_6),
                                                          np.int32(in_elems_7),
                                                          np.int32(out_elems_8),
                                                          np.int32(mulx_9),
                                                          np.int32(muly_10),
                                                          destmem_0, srcmem_2)
                cl.enqueue_nd_range_kernel(self.queue,
                                           self.map_transpose_i32_small_var,
                                           ((np.long(squot32(((((num_arrays_4 * x_elems_5) * y_elems_6) + np.int32(256)) - np.int32(1)),
                                                             np.int32(256))) * np.long(np.int32(256))),),
                                           (np.long(np.int32(256)),))
                if synchronous:
                  self.queue.finish()
            else:
              if ((((1 * (np.long(squot32(((x_elems_5 + np.int32(32)) - np.int32(1)),
                                          np.int32(32))) * np.long(np.int32(32)))) * (np.long(squot32(((y_elems_6 + np.int32(32)) - np.int32(1)),
                                                                                                      np.int32(32))) * np.long(np.int32(8)))) * (np.long(num_arrays_4) * np.long(np.int32(1)))) != 0):
                self.map_transpose_i32_var.set_args(np.int32(destoffset_1),
                                                    np.int32(srcoffset_3),
                                                    np.int32(num_arrays_4),
                                                    np.int32(x_elems_5),
                                                    np.int32(y_elems_6),
                                                    np.int32(in_elems_7),
                                                    np.int32(out_elems_8),
                                                    np.int32(mulx_9),
                                                    np.int32(muly_10),
                                                    destmem_0, srcmem_2)
                cl.enqueue_nd_range_kernel(self.queue,
                                           self.map_transpose_i32_var,
                                           ((np.long(squot32(((x_elems_5 + np.int32(32)) - np.int32(1)),
                                                             np.int32(32))) * np.long(np.int32(32))),
                                            (np.long(squot32(((y_elems_6 + np.int32(32)) - np.int32(1)),
                                                             np.int32(32))) * np.long(np.int32(8))),
                                            (np.long(num_arrays_4) * np.long(np.int32(1)))),
                                           (np.long(np.int32(32)),
                                            np.long(np.int32(8)),
                                            np.long(np.int32(1))))
                if synchronous:
                  self.queue.finish()
    return ()
  def futhark_remove_nans(self, images_mem_37200, sizze_30742, sizze_30743,
                          sizze_30744, nan_value_30745):
    nesting_sizze_31454 = (sizze_30743 * sizze_30744)
    nesting_sizze_31455 = (sizze_30742 * nesting_sizze_31454)
    group_sizze_31456 = self.sizes["remove_nans.group_size_31393"]
    y_31457 = (group_sizze_31456 - np.int32(1))
    x_31458 = (nesting_sizze_31455 + y_31457)
    num_groups_31459 = squot32(x_31458, group_sizze_31456)
    num_threads_31460 = (group_sizze_31456 * num_groups_31459)
    binop_x_37202 = (sizze_30742 * sizze_30743)
    convop_x_37203 = (sizze_30744 * binop_x_37202)
    binop_x_37204 = sext_i32_i64(convop_x_37203)
    bytes_37201 = (np.int64(4) * binop_x_37204)
    mem_37205 = opencl_alloc(self, bytes_37201, "mem_37205")
    if ((1 * (np.long(num_groups_31459) * np.long(group_sizze_31456))) != 0):
      self.map_31399_var.set_args(np.int32(sizze_30742), np.int32(sizze_30743),
                                  np.int32(sizze_30744),
                                  np.int16(nan_value_30745), images_mem_37200,
                                  mem_37205)
      cl.enqueue_nd_range_kernel(self.queue, self.map_31399_var,
                                 ((np.long(num_groups_31459) * np.long(group_sizze_31456)),),
                                 (np.long(group_sizze_31456),))
      if synchronous:
        self.queue.finish()
    out_arrsizze_37794 = sizze_30742
    out_arrsizze_37795 = sizze_30743
    out_arrsizze_37796 = sizze_30744
    out_mem_37793 = mem_37205
    return (out_mem_37793, out_arrsizze_37794, out_arrsizze_37795,
            out_arrsizze_37796)
  def futhark_reshapeTransp(self, images_mem_37200, sizze_30735, sizze_30736,
                            sizze_30737):
    flat_dim_30739 = (sizze_30736 * sizze_30737)
    convop_x_37202 = (sizze_30735 * flat_dim_30739)
    binop_x_37203 = sext_i32_i64(convop_x_37202)
    bytes_37201 = (np.int64(4) * binop_x_37203)
    mem_37204 = opencl_alloc(self, bytes_37201, "mem_37204")
    self.futhark__map_transpose_f32(mem_37204, np.int32(0), images_mem_37200,
                                    np.int32(0), np.int32(1), flat_dim_30739,
                                    sizze_30735, (flat_dim_30739 * sizze_30735),
                                    (flat_dim_30739 * sizze_30735))
    out_arrsizze_37791 = flat_dim_30739
    out_arrsizze_37792 = sizze_30735
    out_mem_37790 = mem_37204
    return (out_mem_37790, out_arrsizze_37791, out_arrsizze_37792)
  def futhark__map_transpose_f32(self, destmem_0, destoffset_1, srcmem_2,
                                 srcoffset_3, num_arrays_4, x_elems_5,
                                 y_elems_6, in_elems_7, out_elems_8):
    if ((num_arrays_4 == np.int32(0)) or ((x_elems_5 == np.int32(0)) or (y_elems_6 == np.int32(0)))):
      pass
    else:
      muly_10 = squot32(np.int32(16), x_elems_5)
      mulx_9 = squot32(np.int32(16), y_elems_6)
      if ((in_elems_7 == out_elems_8) and (((num_arrays_4 == np.int32(1)) or ((x_elems_5 * y_elems_6) == in_elems_7)) and ((x_elems_5 == np.int32(1)) or (y_elems_6 == np.int32(1))))):
        if ((in_elems_7 * np.int32(4)) != 0):
          cl.enqueue_copy(self.queue, destmem_0, srcmem_2,
                          dest_offset=np.long(destoffset_1),
                          src_offset=np.long(srcoffset_3),
                          byte_count=np.long((in_elems_7 * np.int32(4))))
        if synchronous:
          self.queue.finish()
      else:
        if (sle32(x_elems_5, np.int32(8)) and slt32(np.int32(16), y_elems_6)):
          if ((((1 * (np.long(squot32(((x_elems_5 + np.int32(16)) - np.int32(1)),
                                      np.int32(16))) * np.long(np.int32(16)))) * (np.long(squot32(((squot32(((y_elems_6 + muly_10) - np.int32(1)),
                                                                                                            muly_10) + np.int32(16)) - np.int32(1)),
                                                                                                  np.int32(16))) * np.long(np.int32(16)))) * (np.long(num_arrays_4) * np.long(np.int32(1)))) != 0):
            self.map_transpose_f32_low_width_var.set_args(np.int32(destoffset_1),
                                                          np.int32(srcoffset_3),
                                                          np.int32(num_arrays_4),
                                                          np.int32(x_elems_5),
                                                          np.int32(y_elems_6),
                                                          np.int32(in_elems_7),
                                                          np.int32(out_elems_8),
                                                          np.int32(mulx_9),
                                                          np.int32(muly_10),
                                                          destmem_0, srcmem_2)
            cl.enqueue_nd_range_kernel(self.queue,
                                       self.map_transpose_f32_low_width_var,
                                       ((np.long(squot32(((x_elems_5 + np.int32(16)) - np.int32(1)),
                                                         np.int32(16))) * np.long(np.int32(16))),
                                        (np.long(squot32(((squot32(((y_elems_6 + muly_10) - np.int32(1)),
                                                                   muly_10) + np.int32(16)) - np.int32(1)),
                                                         np.int32(16))) * np.long(np.int32(16))),
                                        (np.long(num_arrays_4) * np.long(np.int32(1)))),
                                       (np.long(np.int32(16)),
                                        np.long(np.int32(16)),
                                        np.long(np.int32(1))))
            if synchronous:
              self.queue.finish()
        else:
          if (sle32(y_elems_6, np.int32(8)) and slt32(np.int32(16), x_elems_5)):
            if ((((1 * (np.long(squot32(((squot32(((x_elems_5 + mulx_9) - np.int32(1)),
                                                  mulx_9) + np.int32(16)) - np.int32(1)),
                                        np.int32(16))) * np.long(np.int32(16)))) * (np.long(squot32(((y_elems_6 + np.int32(16)) - np.int32(1)),
                                                                                                    np.int32(16))) * np.long(np.int32(16)))) * (np.long(num_arrays_4) * np.long(np.int32(1)))) != 0):
              self.map_transpose_f32_low_height_var.set_args(np.int32(destoffset_1),
                                                             np.int32(srcoffset_3),
                                                             np.int32(num_arrays_4),
                                                             np.int32(x_elems_5),
                                                             np.int32(y_elems_6),
                                                             np.int32(in_elems_7),
                                                             np.int32(out_elems_8),
                                                             np.int32(mulx_9),
                                                             np.int32(muly_10),
                                                             destmem_0,
                                                             srcmem_2)
              cl.enqueue_nd_range_kernel(self.queue,
                                         self.map_transpose_f32_low_height_var,
                                         ((np.long(squot32(((squot32(((x_elems_5 + mulx_9) - np.int32(1)),
                                                                     mulx_9) + np.int32(16)) - np.int32(1)),
                                                           np.int32(16))) * np.long(np.int32(16))),
                                          (np.long(squot32(((y_elems_6 + np.int32(16)) - np.int32(1)),
                                                           np.int32(16))) * np.long(np.int32(16))),
                                          (np.long(num_arrays_4) * np.long(np.int32(1)))),
                                         (np.long(np.int32(16)),
                                          np.long(np.int32(16)),
                                          np.long(np.int32(1))))
              if synchronous:
                self.queue.finish()
          else:
            if (sle32(x_elems_5, np.int32(8)) and sle32(y_elems_6,
                                                        np.int32(8))):
              if ((1 * (np.long(squot32(((((num_arrays_4 * x_elems_5) * y_elems_6) + np.int32(256)) - np.int32(1)),
                                        np.int32(256))) * np.long(np.int32(256)))) != 0):
                self.map_transpose_f32_small_var.set_args(np.int32(destoffset_1),
                                                          np.int32(srcoffset_3),
                                                          np.int32(num_arrays_4),
                                                          np.int32(x_elems_5),
                                                          np.int32(y_elems_6),
                                                          np.int32(in_elems_7),
                                                          np.int32(out_elems_8),
                                                          np.int32(mulx_9),
                                                          np.int32(muly_10),
                                                          destmem_0, srcmem_2)
                cl.enqueue_nd_range_kernel(self.queue,
                                           self.map_transpose_f32_small_var,
                                           ((np.long(squot32(((((num_arrays_4 * x_elems_5) * y_elems_6) + np.int32(256)) - np.int32(1)),
                                                             np.int32(256))) * np.long(np.int32(256))),),
                                           (np.long(np.int32(256)),))
                if synchronous:
                  self.queue.finish()
            else:
              if ((((1 * (np.long(squot32(((x_elems_5 + np.int32(32)) - np.int32(1)),
                                          np.int32(32))) * np.long(np.int32(32)))) * (np.long(squot32(((y_elems_6 + np.int32(32)) - np.int32(1)),
                                                                                                      np.int32(32))) * np.long(np.int32(8)))) * (np.long(num_arrays_4) * np.long(np.int32(1)))) != 0):
                self.map_transpose_f32_var.set_args(np.int32(destoffset_1),
                                                    np.int32(srcoffset_3),
                                                    np.int32(num_arrays_4),
                                                    np.int32(x_elems_5),
                                                    np.int32(y_elems_6),
                                                    np.int32(in_elems_7),
                                                    np.int32(out_elems_8),
                                                    np.int32(mulx_9),
                                                    np.int32(muly_10),
                                                    destmem_0, srcmem_2)
                cl.enqueue_nd_range_kernel(self.queue,
                                           self.map_transpose_f32_var,
                                           ((np.long(squot32(((x_elems_5 + np.int32(32)) - np.int32(1)),
                                                             np.int32(32))) * np.long(np.int32(32))),
                                            (np.long(squot32(((y_elems_6 + np.int32(32)) - np.int32(1)),
                                                             np.int32(32))) * np.long(np.int32(8))),
                                            (np.long(num_arrays_4) * np.long(np.int32(1)))),
                                           (np.long(np.int32(32)),
                                            np.long(np.int32(8)),
                                            np.long(np.int32(1))))
                if synchronous:
                  self.queue.finish()
    return ()
  def main(self, trend_30759_ext, k_30760_ext, n_30761_ext, freq_30762_ext,
           hfrac_30763_ext, lam_30764_ext, mappingindices_mem_37200_ext,
           images_mem_37201_ext):
    try:
      trend_30759 = np.int32(ct.c_int32(trend_30759_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("i32",
                                                                                                                            type(trend_30759_ext),
                                                                                                                            trend_30759_ext))
    try:
      k_30760 = np.int32(ct.c_int32(k_30760_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("i32",
                                                                                                                            type(k_30760_ext),
                                                                                                                            k_30760_ext))
    try:
      n_30761 = np.int32(ct.c_int32(n_30761_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #2 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("i32",
                                                                                                                            type(n_30761_ext),
                                                                                                                            n_30761_ext))
    try:
      freq_30762 = np.float32(ct.c_float(freq_30762_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #3 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(freq_30762_ext),
                                                                                                                            freq_30762_ext))
    try:
      hfrac_30763 = np.float32(ct.c_float(hfrac_30763_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #4 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(hfrac_30763_ext),
                                                                                                                            hfrac_30763_ext))
    try:
      lam_30764 = np.float32(ct.c_float(lam_30764_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #5 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(lam_30764_ext),
                                                                                                                            lam_30764_ext))
    try:
      assert ((type(mappingindices_mem_37200_ext) in [np.ndarray,
                                                      cl.array.Array]) and (mappingindices_mem_37200_ext.dtype == np.int32)), "Parameter has unexpected type"
      sizze_30756 = np.int32(mappingindices_mem_37200_ext.shape[0])
      if (type(mappingindices_mem_37200_ext) == cl.array.Array):
        mappingindices_mem_37200 = mappingindices_mem_37200_ext.data
      else:
        mappingindices_mem_37200 = opencl_alloc(self,
                                                np.int64(mappingindices_mem_37200_ext.nbytes),
                                                "mappingindices_mem_37200")
        if (np.int64(mappingindices_mem_37200_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, mappingindices_mem_37200,
                          normaliseArray(mappingindices_mem_37200_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #6 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]i32",
                                                                                                                            type(mappingindices_mem_37200_ext),
                                                                                                                            mappingindices_mem_37200_ext))
    try:
      assert ((type(images_mem_37201_ext) in [np.ndarray,
                                              cl.array.Array]) and (images_mem_37201_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_30757 = np.int32(images_mem_37201_ext.shape[0])
      sizze_30758 = np.int32(images_mem_37201_ext.shape[1])
      if (type(images_mem_37201_ext) == cl.array.Array):
        images_mem_37201 = images_mem_37201_ext.data
      else:
        images_mem_37201 = opencl_alloc(self,
                                        np.int64(images_mem_37201_ext.nbytes),
                                        "images_mem_37201")
        if (np.int64(images_mem_37201_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, images_mem_37201,
                          normaliseArray(images_mem_37201_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #7 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f32",
                                                                                                                            type(images_mem_37201_ext),
                                                                                                                            images_mem_37201_ext))
    (out_mem_37799, out_arrsizze_37800, out_mem_37801,
     out_arrsizze_37802) = self.futhark_main(mappingindices_mem_37200,
                                             images_mem_37201, sizze_30756,
                                             sizze_30757, sizze_30758,
                                             trend_30759, k_30760, n_30761,
                                             freq_30762, hfrac_30763, lam_30764)
    return (cl.array.Array(self.queue, (out_arrsizze_37800,), ct.c_int32,
                           data=out_mem_37799), cl.array.Array(self.queue,
                                                               (out_arrsizze_37802,),
                                                               ct.c_float,
                                                               data=out_mem_37801))
  def remove_nans(self, nan_value_30745_ext, images_mem_37200_ext):
    try:
      nan_value_30745 = np.int16(ct.c_int16(nan_value_30745_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("i16",
                                                                                                                            type(nan_value_30745_ext),
                                                                                                                            nan_value_30745_ext))
    try:
      assert ((type(images_mem_37200_ext) in [np.ndarray,
                                              cl.array.Array]) and (images_mem_37200_ext.dtype == np.int16)), "Parameter has unexpected type"
      sizze_30742 = np.int32(images_mem_37200_ext.shape[0])
      sizze_30743 = np.int32(images_mem_37200_ext.shape[1])
      sizze_30744 = np.int32(images_mem_37200_ext.shape[2])
      if (type(images_mem_37200_ext) == cl.array.Array):
        images_mem_37200 = images_mem_37200_ext.data
      else:
        images_mem_37200 = opencl_alloc(self,
                                        np.int64(images_mem_37200_ext.nbytes),
                                        "images_mem_37200")
        if (np.int64(images_mem_37200_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, images_mem_37200,
                          normaliseArray(images_mem_37200_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][][]i16",
                                                                                                                            type(images_mem_37200_ext),
                                                                                                                            images_mem_37200_ext))
    (out_mem_37793, out_arrsizze_37794, out_arrsizze_37795,
     out_arrsizze_37796) = self.futhark_remove_nans(images_mem_37200,
                                                    sizze_30742, sizze_30743,
                                                    sizze_30744,
                                                    nan_value_30745)
    return cl.array.Array(self.queue, (out_arrsizze_37794, out_arrsizze_37795,
                                       out_arrsizze_37796), ct.c_float,
                          data=out_mem_37793)
  def reshapeTransp(self, images_mem_37200_ext):
    try:
      assert ((type(images_mem_37200_ext) in [np.ndarray,
                                              cl.array.Array]) and (images_mem_37200_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_30735 = np.int32(images_mem_37200_ext.shape[0])
      sizze_30736 = np.int32(images_mem_37200_ext.shape[1])
      sizze_30737 = np.int32(images_mem_37200_ext.shape[2])
      if (type(images_mem_37200_ext) == cl.array.Array):
        images_mem_37200 = images_mem_37200_ext.data
      else:
        images_mem_37200 = opencl_alloc(self,
                                        np.int64(images_mem_37200_ext.nbytes),
                                        "images_mem_37200")
        if (np.int64(images_mem_37200_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, images_mem_37200,
                          normaliseArray(images_mem_37200_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][][]f32",
                                                                                                                            type(images_mem_37200_ext),
                                                                                                                            images_mem_37200_ext))
    (out_mem_37790, out_arrsizze_37791,
     out_arrsizze_37792) = self.futhark_reshapeTransp(images_mem_37200,
                                                      sizze_30735, sizze_30736,
                                                      sizze_30737)
    return cl.array.Array(self.queue, (out_arrsizze_37791, out_arrsizze_37792),
                          ct.c_float, data=out_mem_37790)