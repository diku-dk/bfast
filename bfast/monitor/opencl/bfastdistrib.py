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
__kernel void copy_33740(int32_t sizze_31493, int32_t res_31516,
                         int32_t j_31618, int32_t j_m_i_31619, __global
                         unsigned char *mem_33524, __global
                         unsigned char *mem_33549)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t copy_gtid_33740;
    int32_t copy_ltid_33741;
    int32_t copy_gid_33742;
    
    copy_gtid_33740 = get_global_id(0);
    copy_ltid_33741 = get_local_id(0);
    copy_gid_33742 = get_group_id(0);
    if (slt32(copy_gtid_33740, sizze_31493 * (res_31516 * j_m_i_31619))) {
        *(__global float *) &mem_33549[((copy_gtid_33740 -
                                         squot32(copy_gtid_33740, res_31516 *
                                                 j_m_i_31619) * (res_31516 *
                                                                 j_m_i_31619) -
                                         squot32(copy_gtid_33740 -
                                                 squot32(copy_gtid_33740,
                                                         res_31516 *
                                                         j_m_i_31619) *
                                                 (res_31516 * j_m_i_31619),
                                                 j_m_i_31619) * j_m_i_31619) *
                                        (res_31516 * sizze_31493) +
                                        squot32(copy_gtid_33740, res_31516 *
                                                j_m_i_31619) * res_31516 +
                                        squot32(copy_gtid_33740 -
                                                squot32(copy_gtid_33740,
                                                        res_31516 *
                                                        j_m_i_31619) *
                                                (res_31516 * j_m_i_31619),
                                                j_m_i_31619)) * 4] = *(__global
                                                                       float *) &mem_33524[(res_31516 +
                                                                                            (squot32(copy_gtid_33740,
                                                                                                     res_31516 *
                                                                                                     j_m_i_31619) *
                                                                                             (j_31618 *
                                                                                              res_31516) +
                                                                                             squot32(copy_gtid_33740 -
                                                                                                     squot32(copy_gtid_33740,
                                                                                                             res_31516 *
                                                                                                             j_m_i_31619) *
                                                                                                     (res_31516 *
                                                                                                      j_m_i_31619),
                                                                                                     j_m_i_31619) *
                                                                                             j_31618 +
                                                                                             (copy_gtid_33740 -
                                                                                              squot32(copy_gtid_33740,
                                                                                                      res_31516 *
                                                                                                      j_m_i_31619) *
                                                                                              (res_31516 *
                                                                                               j_m_i_31619) -
                                                                                              squot32(copy_gtid_33740 -
                                                                                                      squot32(copy_gtid_33740,
                                                                                                              res_31516 *
                                                                                                              j_m_i_31619) *
                                                                                                      (res_31516 *
                                                                                                       j_m_i_31619),
                                                                                                      j_m_i_31619) *
                                                                                              j_m_i_31619))) *
                                                                                           4];
    }
}
__kernel void map_32011(int32_t sizze_31478, int32_t sizze_31479,
                        int32_t sizze_31480, int16_t nan_value_31481, __global
                        unsigned char *images_mem_33490, __global
                        unsigned char *mem_33495)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_32011;
    int32_t local_tid_32012;
    int32_t group_sizze_33678;
    int32_t wave_sizze_33677;
    int32_t group_id_32013;
    
    global_tid_32011 = get_global_id(0);
    local_tid_32012 = get_local_id(0);
    group_sizze_33678 = get_local_size(0);
    wave_sizze_33677 = LOCKSTEP_WIDTH;
    group_id_32013 = get_group_id(0);
    
    int32_t gtid_32000;
    int32_t gtid_32001;
    int32_t gtid_32002;
    
    gtid_32000 = squot32(global_tid_32011, sizze_31479 * sizze_31480);
    gtid_32001 = squot32(global_tid_32011 - squot32(global_tid_32011,
                                                    sizze_31479 * sizze_31480) *
                         (sizze_31479 * sizze_31480), sizze_31480);
    gtid_32002 = global_tid_32011 - squot32(global_tid_32011, sizze_31479 *
                                            sizze_31480) * (sizze_31479 *
                                                            sizze_31480) -
        squot32(global_tid_32011 - squot32(global_tid_32011, sizze_31479 *
                                           sizze_31480) * (sizze_31479 *
                                                           sizze_31480),
                sizze_31480) * sizze_31480;
    
    int16_t x_32014;
    bool cond_32015;
    float res_32016;
    
    if ((slt32(gtid_32000, sizze_31478) && slt32(gtid_32001, sizze_31479)) &&
        slt32(gtid_32002, sizze_31480)) {
        x_32014 = *(__global int16_t *) &images_mem_33490[(gtid_32000 *
                                                           (sizze_31480 *
                                                            sizze_31479) +
                                                           gtid_32001 *
                                                           sizze_31480 +
                                                           gtid_32002) * 2];
        cond_32015 = x_32014 == nan_value_31481;
        if (cond_32015) {
            res_32016 = NAN;
        } else {
            float res_32017 = sitofp_i16_f32(x_32014);
            
            res_32016 = res_32017;
        }
    }
    if ((slt32(gtid_32000, sizze_31478) && slt32(gtid_32001, sizze_31479)) &&
        slt32(gtid_32002, sizze_31480)) {
        *(__global float *) &mem_33495[(gtid_32000 * (sizze_31480 *
                                                      sizze_31479) +
                                        gtid_32001 * sizze_31480 + gtid_32002) *
                                       4] = res_32016;
    }
}
__kernel void map_32031(int32_t sizze_31492, float freq_31498,
                        int32_t res_31516, __global
                        unsigned char *mappingindices_mem_33490, __global
                        unsigned char *mem_33495)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_32031;
    int32_t local_tid_32032;
    int32_t group_sizze_33684;
    int32_t wave_sizze_33683;
    int32_t group_id_32033;
    
    global_tid_32031 = get_global_id(0);
    local_tid_32032 = get_local_id(0);
    group_sizze_33684 = get_local_size(0);
    wave_sizze_33683 = LOCKSTEP_WIDTH;
    group_id_32033 = get_group_id(0);
    
    int32_t gtid_32022;
    int32_t gtid_32023;
    
    gtid_32022 = squot32(global_tid_32031, sizze_31492);
    gtid_32023 = global_tid_32031 - squot32(global_tid_32031, sizze_31492) *
        sizze_31492;
    
    bool index_primexp_32893;
    bool index_primexp_32892;
    int32_t cmpop_x_32890;
    bool index_primexp_32891;
    int32_t convop_x_32887;
    float binop_y_32888;
    float index_primexp_32889;
    int32_t x_32038;
    float res_32039;
    
    if (slt32(gtid_32022, res_31516) && slt32(gtid_32023, sizze_31492)) {
        index_primexp_32893 = gtid_32022 == 0;
        index_primexp_32892 = gtid_32022 == 1;
        cmpop_x_32890 = smod32(gtid_32022, 2);
        index_primexp_32891 = cmpop_x_32890 == 0;
        convop_x_32887 = sdiv32(gtid_32022, 2);
        binop_y_32888 = sitofp_i32_f32(convop_x_32887);
        index_primexp_32889 = 6.2831855F * binop_y_32888;
        x_32038 = *(__global int32_t *) &mappingindices_mem_33490[gtid_32023 *
                                                                  4];
        if (index_primexp_32893) {
            res_32039 = 1.0F;
        } else {
            float res_32040;
            
            if (index_primexp_32892) {
                float res_32041 = sitofp_i32_f32(x_32038);
                
                res_32040 = res_32041;
            } else {
                float res_32042;
                float x_32043;
                float res_32044;
                float res_32045;
                
                res_32042 = sitofp_i32_f32(x_32038);
                x_32043 = res_32042 * index_primexp_32889;
                res_32044 = x_32043 / freq_31498;
                if (index_primexp_32891) {
                    float res_32046;
                    
                    res_32046 = futrts_sin32(res_32044);
                    res_32045 = res_32046;
                } else {
                    float res_32047;
                    
                    res_32047 = futrts_cos32(res_32044);
                    res_32045 = res_32047;
                }
                res_32040 = res_32045;
            }
            res_32039 = res_32040;
        }
    }
    if (slt32(gtid_32022, res_31516) && slt32(gtid_32023, sizze_31492)) {
        *(__global float *) &mem_33495[(gtid_32022 * sizze_31492 + gtid_32023) *
                                       4] = res_32039;
    }
}
__kernel void map_32078(int32_t sizze_31492, float freq_31498,
                        int32_t res_31516, __global
                        unsigned char *mappingindices_mem_33490, __global
                        unsigned char *mem_33499)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_32078;
    int32_t local_tid_32079;
    int32_t group_sizze_33686;
    int32_t wave_sizze_33685;
    int32_t group_id_32080;
    
    global_tid_32078 = get_global_id(0);
    local_tid_32079 = get_local_id(0);
    group_sizze_33686 = get_local_size(0);
    wave_sizze_33685 = LOCKSTEP_WIDTH;
    group_id_32080 = get_group_id(0);
    
    int32_t gtid_32069;
    int32_t gtid_32070;
    
    gtid_32069 = squot32(global_tid_32078, sizze_31492);
    gtid_32070 = global_tid_32078 - squot32(global_tid_32078, sizze_31492) *
        sizze_31492;
    
    bool index_primexp_32901;
    int32_t binop_x_32898;
    int32_t cmpop_x_32899;
    bool index_primexp_32900;
    int32_t convop_x_32895;
    float binop_y_32896;
    float index_primexp_32897;
    int32_t x_32084;
    float res_32085;
    
    if (slt32(gtid_32069, res_31516) && slt32(gtid_32070, sizze_31492)) {
        index_primexp_32901 = gtid_32069 == 0;
        binop_x_32898 = 1 + gtid_32069;
        cmpop_x_32899 = smod32(binop_x_32898, 2);
        index_primexp_32900 = cmpop_x_32899 == 0;
        convop_x_32895 = sdiv32(binop_x_32898, 2);
        binop_y_32896 = sitofp_i32_f32(convop_x_32895);
        index_primexp_32897 = 6.2831855F * binop_y_32896;
        x_32084 = *(__global int32_t *) &mappingindices_mem_33490[gtid_32070 *
                                                                  4];
        if (index_primexp_32901) {
            res_32085 = 1.0F;
        } else {
            float res_32086;
            float x_32087;
            float res_32088;
            float res_32089;
            
            res_32086 = sitofp_i32_f32(x_32084);
            x_32087 = res_32086 * index_primexp_32897;
            res_32088 = x_32087 / freq_31498;
            if (index_primexp_32900) {
                float res_32090;
                
                res_32090 = futrts_sin32(res_32088);
                res_32089 = res_32090;
            } else {
                float res_32091;
                
                res_32091 = futrts_cos32(res_32088);
                res_32089 = res_32091;
            }
            res_32085 = res_32089;
        }
    }
    if (slt32(gtid_32069, res_31516) && slt32(gtid_32070, sizze_31492)) {
        *(__global float *) &mem_33499[(gtid_32069 * sizze_31492 + gtid_32070) *
                                       4] = res_32085;
    }
}
__kernel void map_32119(int32_t sizze_31492, int32_t res_31516, float res_31589,
                        __global unsigned char *mem_33504, __global
                        unsigned char *mem_33508)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_32119;
    int32_t local_tid_32120;
    int32_t group_sizze_33688;
    int32_t wave_sizze_33687;
    int32_t group_id_32121;
    
    global_tid_32119 = get_global_id(0);
    local_tid_32120 = get_local_id(0);
    group_sizze_33688 = get_local_size(0);
    wave_sizze_33687 = LOCKSTEP_WIDTH;
    group_id_32121 = get_group_id(0);
    
    int32_t gtid_32110;
    int32_t gtid_32111;
    
    gtid_32110 = squot32(global_tid_32119, res_31516);
    gtid_32111 = global_tid_32119 - squot32(global_tid_32119, res_31516) *
        res_31516;
    
    float x_32122;
    float res_32123;
    
    if (slt32(gtid_32110, sizze_31492) && slt32(gtid_32111, res_31516)) {
        x_32122 = *(__global float *) &mem_33504[(gtid_32110 * res_31516 +
                                                  gtid_32111) * 4];
        res_32123 = res_31589 + x_32122;
    }
    if (slt32(gtid_32110, sizze_31492) && slt32(gtid_32111, res_31516)) {
        *(__global float *) &mem_33508[(gtid_32110 * res_31516 + gtid_32111) *
                                       4] = res_32123;
    }
}
__kernel void map_32148(__local volatile int64_t *mem_33520_backing_aligned_0,
                        int32_t sizze_31493, int32_t n_31497, int32_t res_31516,
                        int32_t gidzz_range_32918, int32_t tile_sizze_x_32922,
                        int32_t tiled_group_sizze_32924, __global
                        unsigned char *mem_33504, __global
                        unsigned char *mem_33508, __global
                        unsigned char *mem_33513, __global
                        unsigned char *mem_33517)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict mem_33520_backing_0 =
                          mem_33520_backing_aligned_0;
    int32_t global_tid_32148;
    int32_t local_tid_32149;
    int32_t group_sizze_33690;
    int32_t wave_sizze_33689;
    int32_t group_id_32150;
    
    global_tid_32148 = get_global_id(0);
    local_tid_32149 = get_local_id(0);
    group_sizze_33690 = get_local_size(0);
    wave_sizze_33689 = LOCKSTEP_WIDTH;
    group_id_32150 = get_group_id(0);
    
    int32_t gtid_32137;
    int32_t gtid_32138;
    int32_t gtid_32139;
    int32_t ltid_32925;
    int32_t ltid_32926;
    int32_t ltid_32927;
    
    gtid_32137 = squot32(srem32(global_tid_32148, tile_sizze_x_32922 *
                                tile_sizze_x_32922), tile_sizze_x_32922 *
                         tile_sizze_x_32922) + squot32(squot32(global_tid_32148,
                                                               tile_sizze_x_32922 *
                                                               tile_sizze_x_32922),
                                                       squot32(res_31516 +
                                                               tile_sizze_x_32922 -
                                                               1,
                                                               tile_sizze_x_32922) *
                                                       squot32(res_31516 +
                                                               tile_sizze_x_32922 -
                                                               1,
                                                               tile_sizze_x_32922));
    gtid_32138 = squot32(srem32(global_tid_32148, tile_sizze_x_32922 *
                                tile_sizze_x_32922) -
                         squot32(srem32(global_tid_32148, tile_sizze_x_32922 *
                                        tile_sizze_x_32922),
                                 tile_sizze_x_32922 * tile_sizze_x_32922) *
                         (tile_sizze_x_32922 * tile_sizze_x_32922),
                         tile_sizze_x_32922) + squot32(squot32(global_tid_32148,
                                                               tile_sizze_x_32922 *
                                                               tile_sizze_x_32922) -
                                                       squot32(squot32(global_tid_32148,
                                                                       tile_sizze_x_32922 *
                                                                       tile_sizze_x_32922),
                                                               squot32(res_31516 +
                                                                       tile_sizze_x_32922 -
                                                                       1,
                                                                       tile_sizze_x_32922) *
                                                               squot32(res_31516 +
                                                                       tile_sizze_x_32922 -
                                                                       1,
                                                                       tile_sizze_x_32922)) *
                                                       (squot32(res_31516 +
                                                                tile_sizze_x_32922 -
                                                                1,
                                                                tile_sizze_x_32922) *
                                                        squot32(res_31516 +
                                                                tile_sizze_x_32922 -
                                                                1,
                                                                tile_sizze_x_32922)),
                                                       squot32(res_31516 +
                                                               tile_sizze_x_32922 -
                                                               1,
                                                               tile_sizze_x_32922)) *
        tile_sizze_x_32922;
    gtid_32139 = srem32(global_tid_32148, tile_sizze_x_32922 *
                        tile_sizze_x_32922) - squot32(srem32(global_tid_32148,
                                                             tile_sizze_x_32922 *
                                                             tile_sizze_x_32922),
                                                      tile_sizze_x_32922 *
                                                      tile_sizze_x_32922) *
        (tile_sizze_x_32922 * tile_sizze_x_32922) -
        squot32(srem32(global_tid_32148, tile_sizze_x_32922 *
                       tile_sizze_x_32922) - squot32(srem32(global_tid_32148,
                                                            tile_sizze_x_32922 *
                                                            tile_sizze_x_32922),
                                                     tile_sizze_x_32922 *
                                                     tile_sizze_x_32922) *
                (tile_sizze_x_32922 * tile_sizze_x_32922), tile_sizze_x_32922) *
        tile_sizze_x_32922 + (squot32(global_tid_32148, tile_sizze_x_32922 *
                                      tile_sizze_x_32922) -
                              squot32(squot32(global_tid_32148,
                                              tile_sizze_x_32922 *
                                              tile_sizze_x_32922),
                                      squot32(res_31516 + tile_sizze_x_32922 -
                                              1, tile_sizze_x_32922) *
                                      squot32(res_31516 + tile_sizze_x_32922 -
                                              1, tile_sizze_x_32922)) *
                              (squot32(res_31516 + tile_sizze_x_32922 - 1,
                                       tile_sizze_x_32922) * squot32(res_31516 +
                                                                     tile_sizze_x_32922 -
                                                                     1,
                                                                     tile_sizze_x_32922)) -
                              squot32(squot32(global_tid_32148,
                                              tile_sizze_x_32922 *
                                              tile_sizze_x_32922) -
                                      squot32(squot32(global_tid_32148,
                                                      tile_sizze_x_32922 *
                                                      tile_sizze_x_32922),
                                              squot32(res_31516 +
                                                      tile_sizze_x_32922 - 1,
                                                      tile_sizze_x_32922) *
                                              squot32(res_31516 +
                                                      tile_sizze_x_32922 - 1,
                                                      tile_sizze_x_32922)) *
                                      (squot32(res_31516 + tile_sizze_x_32922 -
                                               1, tile_sizze_x_32922) *
                                       squot32(res_31516 + tile_sizze_x_32922 -
                                               1, tile_sizze_x_32922)),
                                      squot32(res_31516 + tile_sizze_x_32922 -
                                              1, tile_sizze_x_32922)) *
                              squot32(res_31516 + tile_sizze_x_32922 - 1,
                                      tile_sizze_x_32922)) * tile_sizze_x_32922;
    ltid_32925 = squot32(srem32(global_tid_32148, tile_sizze_x_32922 *
                                tile_sizze_x_32922), tile_sizze_x_32922 *
                         tile_sizze_x_32922);
    ltid_32926 = squot32(srem32(global_tid_32148, tile_sizze_x_32922 *
                                tile_sizze_x_32922) -
                         squot32(srem32(global_tid_32148, tile_sizze_x_32922 *
                                        tile_sizze_x_32922),
                                 tile_sizze_x_32922 * tile_sizze_x_32922) *
                         (tile_sizze_x_32922 * tile_sizze_x_32922),
                         tile_sizze_x_32922);
    ltid_32927 = srem32(global_tid_32148, tile_sizze_x_32922 *
                        tile_sizze_x_32922) - squot32(srem32(global_tid_32148,
                                                             tile_sizze_x_32922 *
                                                             tile_sizze_x_32922),
                                                      tile_sizze_x_32922 *
                                                      tile_sizze_x_32922) *
        (tile_sizze_x_32922 * tile_sizze_x_32922) -
        squot32(srem32(global_tid_32148, tile_sizze_x_32922 *
                       tile_sizze_x_32922) - squot32(srem32(global_tid_32148,
                                                            tile_sizze_x_32922 *
                                                            tile_sizze_x_32922),
                                                     tile_sizze_x_32922 *
                                                     tile_sizze_x_32922) *
                (tile_sizze_x_32922 * tile_sizze_x_32922), tile_sizze_x_32922) *
        tile_sizze_x_32922;
    
    int32_t mm_32915;
    int32_t m_32945;
    bool is_active_33452;
    bool is_active_33453;
    bool active_33455;
    
    if ((slt32(gtid_32137, gidzz_range_32918) && slt32(gtid_32138,
                                                       res_31516)) &&
        slt32(gtid_32139, res_31516)) {
        mm_32915 = 30 * gtid_32137;
        m_32945 = local_tid_32149 + mm_32915;
        is_active_33452 = slt32(local_tid_32149, 30);
        is_active_33453 = slt32(m_32945, sizze_31493);
        active_33455 = is_active_33452 && is_active_33453;
    }
    
    __local char *mem_33520;
    
    mem_33520 = (__local char *) mem_33520_backing_0;
    
    float res_33281;
    float res_33282;
    float res_33283;
    float res_33284;
    float res_33285;
    float res_33286;
    float res_33287;
    float res_33288;
    float res_33289;
    float res_33290;
    float res_33291;
    float res_33292;
    float res_33293;
    float res_33294;
    float res_33295;
    float res_33296;
    float res_33297;
    float res_33298;
    float res_33299;
    float res_33300;
    float res_33301;
    float res_33302;
    float res_33303;
    float res_33304;
    float res_33305;
    float res_33306;
    float res_33307;
    float res_33308;
    float res_33309;
    float res_33310;
    int32_t m_33316;
    int32_t m_33319;
    int32_t m_33322;
    int32_t m_33325;
    int32_t m_33328;
    int32_t m_33331;
    int32_t m_33334;
    int32_t m_33337;
    int32_t m_33340;
    int32_t m_33343;
    int32_t m_33346;
    int32_t m_33349;
    int32_t m_33352;
    int32_t m_33355;
    int32_t m_33358;
    int32_t m_33361;
    int32_t m_33364;
    int32_t m_33367;
    int32_t m_33370;
    int32_t m_33373;
    int32_t m_33376;
    int32_t m_33379;
    int32_t m_33382;
    int32_t m_33385;
    int32_t m_33388;
    int32_t m_33391;
    int32_t m_33394;
    int32_t m_33397;
    int32_t m_33400;
    
    if ((slt32(gtid_32137, gidzz_range_32918) && slt32(gtid_32138,
                                                       res_31516)) &&
        slt32(gtid_32139, res_31516)) {
        float acc_clone_32951;
        float acc_clone_32962;
        float acc_clone_32973;
        float acc_clone_32984;
        float acc_clone_32995;
        float acc_clone_33006;
        float acc_clone_33017;
        float acc_clone_33028;
        float acc_clone_33039;
        float acc_clone_33050;
        float acc_clone_33061;
        float acc_clone_33072;
        float acc_clone_33083;
        float acc_clone_33094;
        float acc_clone_33105;
        float acc_clone_33116;
        float acc_clone_33127;
        float acc_clone_33138;
        float acc_clone_33149;
        float acc_clone_33160;
        float acc_clone_33171;
        float acc_clone_33182;
        float acc_clone_33193;
        float acc_clone_33204;
        float acc_clone_33215;
        float acc_clone_33226;
        float acc_clone_33237;
        float acc_clone_33248;
        float acc_clone_33259;
        float acc_clone_33270;
        
        acc_clone_32951 = 0.0F;
        acc_clone_32962 = 0.0F;
        acc_clone_32973 = 0.0F;
        acc_clone_32984 = 0.0F;
        acc_clone_32995 = 0.0F;
        acc_clone_33006 = 0.0F;
        acc_clone_33017 = 0.0F;
        acc_clone_33028 = 0.0F;
        acc_clone_33039 = 0.0F;
        acc_clone_33050 = 0.0F;
        acc_clone_33061 = 0.0F;
        acc_clone_33072 = 0.0F;
        acc_clone_33083 = 0.0F;
        acc_clone_33094 = 0.0F;
        acc_clone_33105 = 0.0F;
        acc_clone_33116 = 0.0F;
        acc_clone_33127 = 0.0F;
        acc_clone_33138 = 0.0F;
        acc_clone_33149 = 0.0F;
        acc_clone_33160 = 0.0F;
        acc_clone_33171 = 0.0F;
        acc_clone_33182 = 0.0F;
        acc_clone_33193 = 0.0F;
        acc_clone_33204 = 0.0F;
        acc_clone_33215 = 0.0F;
        acc_clone_33226 = 0.0F;
        acc_clone_33237 = 0.0F;
        acc_clone_33248 = 0.0F;
        acc_clone_33259 = 0.0F;
        acc_clone_33270 = 0.0F;
        for (int32_t loop_ind_33280 = 0; loop_ind_33280 < n_31497;
             loop_ind_33280++) {
            int32_t i_32163;
            
            i_32163 = loop_ind_33280;
            barrier(CLK_LOCAL_MEM_FENCE);
            
            float x_32169;
            float x_32170;
            float x_32168;
            
            x_32169 = *(__global float *) &mem_33504[(i_32163 * res_31516 +
                                                      gtid_32138) * 4];
            x_32170 = *(__global float *) &mem_33508[(i_32163 * res_31516 +
                                                      gtid_32139) * 4];
            if (active_33455) {
                float x_33456 = *(__global float *) &mem_33517[(i_32163 *
                                                                sizze_31493 +
                                                                m_32945) * 4];
                
                x_32168 = x_33456;
            } else {
                x_32168 = 0.0F;
            }
            for (int32_t comb_iter_33721 = 0; comb_iter_33721 < 1;
                 comb_iter_33721++) {
                int32_t cid_32949;
                int32_t flat_comb_id_33722 = comb_iter_33721 *
                        tiled_group_sizze_32924 + local_tid_32149;
                
                cid_32949 = flat_comb_id_33722;
                if (slt32(cid_32949, tiled_group_sizze_32924) &&
                    (slt32(local_tid_32149, 30) && slt32(m_32945,
                                                         sizze_31493))) {
                    *(__local float *) &mem_33520[cid_32949 * 4] = x_32168;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            float x_32956;
            bool res_32957;
            float res_32958;
            float res_32960;
            float x_32967;
            bool res_32968;
            float res_32969;
            float res_32971;
            float x_32978;
            bool res_32979;
            float res_32980;
            float res_32982;
            float x_32989;
            bool res_32990;
            float res_32991;
            float res_32993;
            float x_33000;
            bool res_33001;
            float res_33002;
            float res_33004;
            float x_33011;
            bool res_33012;
            float res_33013;
            float res_33015;
            float x_33022;
            bool res_33023;
            float res_33024;
            float res_33026;
            float x_33033;
            bool res_33034;
            float res_33035;
            float res_33037;
            float x_33044;
            bool res_33045;
            float res_33046;
            float res_33048;
            float x_33055;
            bool res_33056;
            float res_33057;
            float res_33059;
            float x_33066;
            bool res_33067;
            float res_33068;
            float res_33070;
            float x_33077;
            bool res_33078;
            float res_33079;
            float res_33081;
            float x_33088;
            bool res_33089;
            float res_33090;
            float res_33092;
            float x_33099;
            bool res_33100;
            float res_33101;
            float res_33103;
            float x_33110;
            bool res_33111;
            float res_33112;
            float res_33114;
            float x_33121;
            bool res_33122;
            float res_33123;
            float res_33125;
            float x_33132;
            bool res_33133;
            float res_33134;
            float res_33136;
            float x_33143;
            bool res_33144;
            float res_33145;
            float res_33147;
            float x_33154;
            bool res_33155;
            float res_33156;
            float res_33158;
            float x_33165;
            bool res_33166;
            float res_33167;
            float res_33169;
            float x_33176;
            bool res_33177;
            float res_33178;
            float res_33180;
            float x_33187;
            bool res_33188;
            float res_33189;
            float res_33191;
            float x_33198;
            bool res_33199;
            float res_33200;
            float res_33202;
            float x_33209;
            bool res_33210;
            float res_33211;
            float res_33213;
            float x_33220;
            bool res_33221;
            float res_33222;
            float res_33224;
            float x_33231;
            bool res_33232;
            float res_33233;
            float res_33235;
            float x_33242;
            bool res_33243;
            float res_33244;
            float res_33246;
            float x_33253;
            bool res_33254;
            float res_33255;
            float res_33257;
            float x_33264;
            bool res_33265;
            float res_33266;
            float res_33268;
            float x_33275;
            bool res_33276;
            float res_33277;
            float res_33279;
            
            x_32956 = *(__local float *) &mem_33520[0];
            res_32957 = futrts_isnan32(x_32956);
            if (res_32957) {
                res_32958 = 0.0F;
            } else {
                float res_32959 = x_32169 * x_32170;
                
                res_32958 = res_32959;
            }
            res_32960 = acc_clone_32951 + res_32958;
            x_32967 = *(__local float *) &mem_33520[4];
            res_32968 = futrts_isnan32(x_32967);
            if (res_32968) {
                res_32969 = 0.0F;
            } else {
                float res_32970 = x_32169 * x_32170;
                
                res_32969 = res_32970;
            }
            res_32971 = acc_clone_32962 + res_32969;
            x_32978 = *(__local float *) &mem_33520[8];
            res_32979 = futrts_isnan32(x_32978);
            if (res_32979) {
                res_32980 = 0.0F;
            } else {
                float res_32981 = x_32169 * x_32170;
                
                res_32980 = res_32981;
            }
            res_32982 = acc_clone_32973 + res_32980;
            x_32989 = *(__local float *) &mem_33520[12];
            res_32990 = futrts_isnan32(x_32989);
            if (res_32990) {
                res_32991 = 0.0F;
            } else {
                float res_32992 = x_32169 * x_32170;
                
                res_32991 = res_32992;
            }
            res_32993 = acc_clone_32984 + res_32991;
            x_33000 = *(__local float *) &mem_33520[16];
            res_33001 = futrts_isnan32(x_33000);
            if (res_33001) {
                res_33002 = 0.0F;
            } else {
                float res_33003 = x_32169 * x_32170;
                
                res_33002 = res_33003;
            }
            res_33004 = acc_clone_32995 + res_33002;
            x_33011 = *(__local float *) &mem_33520[20];
            res_33012 = futrts_isnan32(x_33011);
            if (res_33012) {
                res_33013 = 0.0F;
            } else {
                float res_33014 = x_32169 * x_32170;
                
                res_33013 = res_33014;
            }
            res_33015 = acc_clone_33006 + res_33013;
            x_33022 = *(__local float *) &mem_33520[24];
            res_33023 = futrts_isnan32(x_33022);
            if (res_33023) {
                res_33024 = 0.0F;
            } else {
                float res_33025 = x_32169 * x_32170;
                
                res_33024 = res_33025;
            }
            res_33026 = acc_clone_33017 + res_33024;
            x_33033 = *(__local float *) &mem_33520[28];
            res_33034 = futrts_isnan32(x_33033);
            if (res_33034) {
                res_33035 = 0.0F;
            } else {
                float res_33036 = x_32169 * x_32170;
                
                res_33035 = res_33036;
            }
            res_33037 = acc_clone_33028 + res_33035;
            x_33044 = *(__local float *) &mem_33520[32];
            res_33045 = futrts_isnan32(x_33044);
            if (res_33045) {
                res_33046 = 0.0F;
            } else {
                float res_33047 = x_32169 * x_32170;
                
                res_33046 = res_33047;
            }
            res_33048 = acc_clone_33039 + res_33046;
            x_33055 = *(__local float *) &mem_33520[36];
            res_33056 = futrts_isnan32(x_33055);
            if (res_33056) {
                res_33057 = 0.0F;
            } else {
                float res_33058 = x_32169 * x_32170;
                
                res_33057 = res_33058;
            }
            res_33059 = acc_clone_33050 + res_33057;
            x_33066 = *(__local float *) &mem_33520[40];
            res_33067 = futrts_isnan32(x_33066);
            if (res_33067) {
                res_33068 = 0.0F;
            } else {
                float res_33069 = x_32169 * x_32170;
                
                res_33068 = res_33069;
            }
            res_33070 = acc_clone_33061 + res_33068;
            x_33077 = *(__local float *) &mem_33520[44];
            res_33078 = futrts_isnan32(x_33077);
            if (res_33078) {
                res_33079 = 0.0F;
            } else {
                float res_33080 = x_32169 * x_32170;
                
                res_33079 = res_33080;
            }
            res_33081 = acc_clone_33072 + res_33079;
            x_33088 = *(__local float *) &mem_33520[48];
            res_33089 = futrts_isnan32(x_33088);
            if (res_33089) {
                res_33090 = 0.0F;
            } else {
                float res_33091 = x_32169 * x_32170;
                
                res_33090 = res_33091;
            }
            res_33092 = acc_clone_33083 + res_33090;
            x_33099 = *(__local float *) &mem_33520[52];
            res_33100 = futrts_isnan32(x_33099);
            if (res_33100) {
                res_33101 = 0.0F;
            } else {
                float res_33102 = x_32169 * x_32170;
                
                res_33101 = res_33102;
            }
            res_33103 = acc_clone_33094 + res_33101;
            x_33110 = *(__local float *) &mem_33520[56];
            res_33111 = futrts_isnan32(x_33110);
            if (res_33111) {
                res_33112 = 0.0F;
            } else {
                float res_33113 = x_32169 * x_32170;
                
                res_33112 = res_33113;
            }
            res_33114 = acc_clone_33105 + res_33112;
            x_33121 = *(__local float *) &mem_33520[60];
            res_33122 = futrts_isnan32(x_33121);
            if (res_33122) {
                res_33123 = 0.0F;
            } else {
                float res_33124 = x_32169 * x_32170;
                
                res_33123 = res_33124;
            }
            res_33125 = acc_clone_33116 + res_33123;
            x_33132 = *(__local float *) &mem_33520[64];
            res_33133 = futrts_isnan32(x_33132);
            if (res_33133) {
                res_33134 = 0.0F;
            } else {
                float res_33135 = x_32169 * x_32170;
                
                res_33134 = res_33135;
            }
            res_33136 = acc_clone_33127 + res_33134;
            x_33143 = *(__local float *) &mem_33520[68];
            res_33144 = futrts_isnan32(x_33143);
            if (res_33144) {
                res_33145 = 0.0F;
            } else {
                float res_33146 = x_32169 * x_32170;
                
                res_33145 = res_33146;
            }
            res_33147 = acc_clone_33138 + res_33145;
            x_33154 = *(__local float *) &mem_33520[72];
            res_33155 = futrts_isnan32(x_33154);
            if (res_33155) {
                res_33156 = 0.0F;
            } else {
                float res_33157 = x_32169 * x_32170;
                
                res_33156 = res_33157;
            }
            res_33158 = acc_clone_33149 + res_33156;
            x_33165 = *(__local float *) &mem_33520[76];
            res_33166 = futrts_isnan32(x_33165);
            if (res_33166) {
                res_33167 = 0.0F;
            } else {
                float res_33168 = x_32169 * x_32170;
                
                res_33167 = res_33168;
            }
            res_33169 = acc_clone_33160 + res_33167;
            x_33176 = *(__local float *) &mem_33520[80];
            res_33177 = futrts_isnan32(x_33176);
            if (res_33177) {
                res_33178 = 0.0F;
            } else {
                float res_33179 = x_32169 * x_32170;
                
                res_33178 = res_33179;
            }
            res_33180 = acc_clone_33171 + res_33178;
            x_33187 = *(__local float *) &mem_33520[84];
            res_33188 = futrts_isnan32(x_33187);
            if (res_33188) {
                res_33189 = 0.0F;
            } else {
                float res_33190 = x_32169 * x_32170;
                
                res_33189 = res_33190;
            }
            res_33191 = acc_clone_33182 + res_33189;
            x_33198 = *(__local float *) &mem_33520[88];
            res_33199 = futrts_isnan32(x_33198);
            if (res_33199) {
                res_33200 = 0.0F;
            } else {
                float res_33201 = x_32169 * x_32170;
                
                res_33200 = res_33201;
            }
            res_33202 = acc_clone_33193 + res_33200;
            x_33209 = *(__local float *) &mem_33520[92];
            res_33210 = futrts_isnan32(x_33209);
            if (res_33210) {
                res_33211 = 0.0F;
            } else {
                float res_33212 = x_32169 * x_32170;
                
                res_33211 = res_33212;
            }
            res_33213 = acc_clone_33204 + res_33211;
            x_33220 = *(__local float *) &mem_33520[96];
            res_33221 = futrts_isnan32(x_33220);
            if (res_33221) {
                res_33222 = 0.0F;
            } else {
                float res_33223 = x_32169 * x_32170;
                
                res_33222 = res_33223;
            }
            res_33224 = acc_clone_33215 + res_33222;
            x_33231 = *(__local float *) &mem_33520[100];
            res_33232 = futrts_isnan32(x_33231);
            if (res_33232) {
                res_33233 = 0.0F;
            } else {
                float res_33234 = x_32169 * x_32170;
                
                res_33233 = res_33234;
            }
            res_33235 = acc_clone_33226 + res_33233;
            x_33242 = *(__local float *) &mem_33520[104];
            res_33243 = futrts_isnan32(x_33242);
            if (res_33243) {
                res_33244 = 0.0F;
            } else {
                float res_33245 = x_32169 * x_32170;
                
                res_33244 = res_33245;
            }
            res_33246 = acc_clone_33237 + res_33244;
            x_33253 = *(__local float *) &mem_33520[108];
            res_33254 = futrts_isnan32(x_33253);
            if (res_33254) {
                res_33255 = 0.0F;
            } else {
                float res_33256 = x_32169 * x_32170;
                
                res_33255 = res_33256;
            }
            res_33257 = acc_clone_33248 + res_33255;
            x_33264 = *(__local float *) &mem_33520[112];
            res_33265 = futrts_isnan32(x_33264);
            if (res_33265) {
                res_33266 = 0.0F;
            } else {
                float res_33267 = x_32169 * x_32170;
                
                res_33266 = res_33267;
            }
            res_33268 = acc_clone_33259 + res_33266;
            x_33275 = *(__local float *) &mem_33520[116];
            res_33276 = futrts_isnan32(x_33275);
            if (res_33276) {
                res_33277 = 0.0F;
            } else {
                float res_33278 = x_32169 * x_32170;
                
                res_33277 = res_33278;
            }
            res_33279 = acc_clone_33270 + res_33277;
            
            float acc_clone_tmp_33691 = res_32960;
            float acc_clone_tmp_33692 = res_32971;
            float acc_clone_tmp_33693 = res_32982;
            float acc_clone_tmp_33694 = res_32993;
            float acc_clone_tmp_33695 = res_33004;
            float acc_clone_tmp_33696 = res_33015;
            float acc_clone_tmp_33697 = res_33026;
            float acc_clone_tmp_33698 = res_33037;
            float acc_clone_tmp_33699 = res_33048;
            float acc_clone_tmp_33700 = res_33059;
            float acc_clone_tmp_33701 = res_33070;
            float acc_clone_tmp_33702 = res_33081;
            float acc_clone_tmp_33703 = res_33092;
            float acc_clone_tmp_33704 = res_33103;
            float acc_clone_tmp_33705 = res_33114;
            float acc_clone_tmp_33706 = res_33125;
            float acc_clone_tmp_33707 = res_33136;
            float acc_clone_tmp_33708 = res_33147;
            float acc_clone_tmp_33709 = res_33158;
            float acc_clone_tmp_33710 = res_33169;
            float acc_clone_tmp_33711 = res_33180;
            float acc_clone_tmp_33712 = res_33191;
            float acc_clone_tmp_33713 = res_33202;
            float acc_clone_tmp_33714 = res_33213;
            float acc_clone_tmp_33715 = res_33224;
            float acc_clone_tmp_33716 = res_33235;
            float acc_clone_tmp_33717 = res_33246;
            float acc_clone_tmp_33718 = res_33257;
            float acc_clone_tmp_33719 = res_33268;
            float acc_clone_tmp_33720;
            
            acc_clone_tmp_33720 = res_33279;
            acc_clone_32951 = acc_clone_tmp_33691;
            acc_clone_32962 = acc_clone_tmp_33692;
            acc_clone_32973 = acc_clone_tmp_33693;
            acc_clone_32984 = acc_clone_tmp_33694;
            acc_clone_32995 = acc_clone_tmp_33695;
            acc_clone_33006 = acc_clone_tmp_33696;
            acc_clone_33017 = acc_clone_tmp_33697;
            acc_clone_33028 = acc_clone_tmp_33698;
            acc_clone_33039 = acc_clone_tmp_33699;
            acc_clone_33050 = acc_clone_tmp_33700;
            acc_clone_33061 = acc_clone_tmp_33701;
            acc_clone_33072 = acc_clone_tmp_33702;
            acc_clone_33083 = acc_clone_tmp_33703;
            acc_clone_33094 = acc_clone_tmp_33704;
            acc_clone_33105 = acc_clone_tmp_33705;
            acc_clone_33116 = acc_clone_tmp_33706;
            acc_clone_33127 = acc_clone_tmp_33707;
            acc_clone_33138 = acc_clone_tmp_33708;
            acc_clone_33149 = acc_clone_tmp_33709;
            acc_clone_33160 = acc_clone_tmp_33710;
            acc_clone_33171 = acc_clone_tmp_33711;
            acc_clone_33182 = acc_clone_tmp_33712;
            acc_clone_33193 = acc_clone_tmp_33713;
            acc_clone_33204 = acc_clone_tmp_33714;
            acc_clone_33215 = acc_clone_tmp_33715;
            acc_clone_33226 = acc_clone_tmp_33716;
            acc_clone_33237 = acc_clone_tmp_33717;
            acc_clone_33248 = acc_clone_tmp_33718;
            acc_clone_33259 = acc_clone_tmp_33719;
            acc_clone_33270 = acc_clone_tmp_33720;
        }
        res_33281 = acc_clone_32951;
        res_33282 = acc_clone_32962;
        res_33283 = acc_clone_32973;
        res_33284 = acc_clone_32984;
        res_33285 = acc_clone_32995;
        res_33286 = acc_clone_33006;
        res_33287 = acc_clone_33017;
        res_33288 = acc_clone_33028;
        res_33289 = acc_clone_33039;
        res_33290 = acc_clone_33050;
        res_33291 = acc_clone_33061;
        res_33292 = acc_clone_33072;
        res_33293 = acc_clone_33083;
        res_33294 = acc_clone_33094;
        res_33295 = acc_clone_33105;
        res_33296 = acc_clone_33116;
        res_33297 = acc_clone_33127;
        res_33298 = acc_clone_33138;
        res_33299 = acc_clone_33149;
        res_33300 = acc_clone_33160;
        res_33301 = acc_clone_33171;
        res_33302 = acc_clone_33182;
        res_33303 = acc_clone_33193;
        res_33304 = acc_clone_33204;
        res_33305 = acc_clone_33215;
        res_33306 = acc_clone_33226;
        res_33307 = acc_clone_33237;
        res_33308 = acc_clone_33248;
        res_33309 = acc_clone_33259;
        res_33310 = acc_clone_33270;
        m_33316 = 1 + mm_32915;
        m_33319 = 2 + mm_32915;
        m_33322 = 3 + mm_32915;
        m_33325 = 4 + mm_32915;
        m_33328 = 5 + mm_32915;
        m_33331 = 6 + mm_32915;
        m_33334 = 7 + mm_32915;
        m_33337 = 8 + mm_32915;
        m_33340 = 9 + mm_32915;
        m_33343 = 10 + mm_32915;
        m_33346 = 11 + mm_32915;
        m_33349 = 12 + mm_32915;
        m_33352 = 13 + mm_32915;
        m_33355 = 14 + mm_32915;
        m_33358 = 15 + mm_32915;
        m_33361 = 16 + mm_32915;
        m_33364 = 17 + mm_32915;
        m_33367 = 18 + mm_32915;
        m_33370 = 19 + mm_32915;
        m_33373 = 20 + mm_32915;
        m_33376 = 21 + mm_32915;
        m_33379 = 22 + mm_32915;
        m_33382 = 23 + mm_32915;
        m_33385 = 24 + mm_32915;
        m_33388 = 25 + mm_32915;
        m_33391 = 26 + mm_32915;
        m_33394 = 27 + mm_32915;
        m_33397 = 28 + mm_32915;
        m_33400 = 29 + mm_32915;
    }
    if (((((slt32(gtid_32137, gidzz_range_32918) && slt32(gtid_32138,
                                                          res_31516)) &&
           slt32(gtid_32139, res_31516)) && (sle32(0, mm_32915) &&
                                             slt32(mm_32915, sizze_31493))) &&
         (sle32(0, gtid_32138) && slt32(gtid_32138, res_31516))) && (sle32(0,
                                                                           gtid_32139) &&
                                                                     slt32(gtid_32139,
                                                                           res_31516))) {
        *(__global float *) &mem_33513[(mm_32915 * (res_31516 * res_31516) +
                                        gtid_32138 * res_31516 + gtid_32139) *
                                       4] = res_33281;
    }
    if (((((slt32(gtid_32137, gidzz_range_32918) && slt32(gtid_32138,
                                                          res_31516)) &&
           slt32(gtid_32139, res_31516)) && (sle32(0, m_33316) && slt32(m_33316,
                                                                        sizze_31493))) &&
         (sle32(0, gtid_32138) && slt32(gtid_32138, res_31516))) && (sle32(0,
                                                                           gtid_32139) &&
                                                                     slt32(gtid_32139,
                                                                           res_31516))) {
        *(__global float *) &mem_33513[(m_33316 * (res_31516 * res_31516) +
                                        gtid_32138 * res_31516 + gtid_32139) *
                                       4] = res_33282;
    }
    if (((((slt32(gtid_32137, gidzz_range_32918) && slt32(gtid_32138,
                                                          res_31516)) &&
           slt32(gtid_32139, res_31516)) && (sle32(0, m_33319) && slt32(m_33319,
                                                                        sizze_31493))) &&
         (sle32(0, gtid_32138) && slt32(gtid_32138, res_31516))) && (sle32(0,
                                                                           gtid_32139) &&
                                                                     slt32(gtid_32139,
                                                                           res_31516))) {
        *(__global float *) &mem_33513[(m_33319 * (res_31516 * res_31516) +
                                        gtid_32138 * res_31516 + gtid_32139) *
                                       4] = res_33283;
    }
    if (((((slt32(gtid_32137, gidzz_range_32918) && slt32(gtid_32138,
                                                          res_31516)) &&
           slt32(gtid_32139, res_31516)) && (sle32(0, m_33322) && slt32(m_33322,
                                                                        sizze_31493))) &&
         (sle32(0, gtid_32138) && slt32(gtid_32138, res_31516))) && (sle32(0,
                                                                           gtid_32139) &&
                                                                     slt32(gtid_32139,
                                                                           res_31516))) {
        *(__global float *) &mem_33513[(m_33322 * (res_31516 * res_31516) +
                                        gtid_32138 * res_31516 + gtid_32139) *
                                       4] = res_33284;
    }
    if (((((slt32(gtid_32137, gidzz_range_32918) && slt32(gtid_32138,
                                                          res_31516)) &&
           slt32(gtid_32139, res_31516)) && (sle32(0, m_33325) && slt32(m_33325,
                                                                        sizze_31493))) &&
         (sle32(0, gtid_32138) && slt32(gtid_32138, res_31516))) && (sle32(0,
                                                                           gtid_32139) &&
                                                                     slt32(gtid_32139,
                                                                           res_31516))) {
        *(__global float *) &mem_33513[(m_33325 * (res_31516 * res_31516) +
                                        gtid_32138 * res_31516 + gtid_32139) *
                                       4] = res_33285;
    }
    if (((((slt32(gtid_32137, gidzz_range_32918) && slt32(gtid_32138,
                                                          res_31516)) &&
           slt32(gtid_32139, res_31516)) && (sle32(0, m_33328) && slt32(m_33328,
                                                                        sizze_31493))) &&
         (sle32(0, gtid_32138) && slt32(gtid_32138, res_31516))) && (sle32(0,
                                                                           gtid_32139) &&
                                                                     slt32(gtid_32139,
                                                                           res_31516))) {
        *(__global float *) &mem_33513[(m_33328 * (res_31516 * res_31516) +
                                        gtid_32138 * res_31516 + gtid_32139) *
                                       4] = res_33286;
    }
    if (((((slt32(gtid_32137, gidzz_range_32918) && slt32(gtid_32138,
                                                          res_31516)) &&
           slt32(gtid_32139, res_31516)) && (sle32(0, m_33331) && slt32(m_33331,
                                                                        sizze_31493))) &&
         (sle32(0, gtid_32138) && slt32(gtid_32138, res_31516))) && (sle32(0,
                                                                           gtid_32139) &&
                                                                     slt32(gtid_32139,
                                                                           res_31516))) {
        *(__global float *) &mem_33513[(m_33331 * (res_31516 * res_31516) +
                                        gtid_32138 * res_31516 + gtid_32139) *
                                       4] = res_33287;
    }
    if (((((slt32(gtid_32137, gidzz_range_32918) && slt32(gtid_32138,
                                                          res_31516)) &&
           slt32(gtid_32139, res_31516)) && (sle32(0, m_33334) && slt32(m_33334,
                                                                        sizze_31493))) &&
         (sle32(0, gtid_32138) && slt32(gtid_32138, res_31516))) && (sle32(0,
                                                                           gtid_32139) &&
                                                                     slt32(gtid_32139,
                                                                           res_31516))) {
        *(__global float *) &mem_33513[(m_33334 * (res_31516 * res_31516) +
                                        gtid_32138 * res_31516 + gtid_32139) *
                                       4] = res_33288;
    }
    if (((((slt32(gtid_32137, gidzz_range_32918) && slt32(gtid_32138,
                                                          res_31516)) &&
           slt32(gtid_32139, res_31516)) && (sle32(0, m_33337) && slt32(m_33337,
                                                                        sizze_31493))) &&
         (sle32(0, gtid_32138) && slt32(gtid_32138, res_31516))) && (sle32(0,
                                                                           gtid_32139) &&
                                                                     slt32(gtid_32139,
                                                                           res_31516))) {
        *(__global float *) &mem_33513[(m_33337 * (res_31516 * res_31516) +
                                        gtid_32138 * res_31516 + gtid_32139) *
                                       4] = res_33289;
    }
    if (((((slt32(gtid_32137, gidzz_range_32918) && slt32(gtid_32138,
                                                          res_31516)) &&
           slt32(gtid_32139, res_31516)) && (sle32(0, m_33340) && slt32(m_33340,
                                                                        sizze_31493))) &&
         (sle32(0, gtid_32138) && slt32(gtid_32138, res_31516))) && (sle32(0,
                                                                           gtid_32139) &&
                                                                     slt32(gtid_32139,
                                                                           res_31516))) {
        *(__global float *) &mem_33513[(m_33340 * (res_31516 * res_31516) +
                                        gtid_32138 * res_31516 + gtid_32139) *
                                       4] = res_33290;
    }
    if (((((slt32(gtid_32137, gidzz_range_32918) && slt32(gtid_32138,
                                                          res_31516)) &&
           slt32(gtid_32139, res_31516)) && (sle32(0, m_33343) && slt32(m_33343,
                                                                        sizze_31493))) &&
         (sle32(0, gtid_32138) && slt32(gtid_32138, res_31516))) && (sle32(0,
                                                                           gtid_32139) &&
                                                                     slt32(gtid_32139,
                                                                           res_31516))) {
        *(__global float *) &mem_33513[(m_33343 * (res_31516 * res_31516) +
                                        gtid_32138 * res_31516 + gtid_32139) *
                                       4] = res_33291;
    }
    if (((((slt32(gtid_32137, gidzz_range_32918) && slt32(gtid_32138,
                                                          res_31516)) &&
           slt32(gtid_32139, res_31516)) && (sle32(0, m_33346) && slt32(m_33346,
                                                                        sizze_31493))) &&
         (sle32(0, gtid_32138) && slt32(gtid_32138, res_31516))) && (sle32(0,
                                                                           gtid_32139) &&
                                                                     slt32(gtid_32139,
                                                                           res_31516))) {
        *(__global float *) &mem_33513[(m_33346 * (res_31516 * res_31516) +
                                        gtid_32138 * res_31516 + gtid_32139) *
                                       4] = res_33292;
    }
    if (((((slt32(gtid_32137, gidzz_range_32918) && slt32(gtid_32138,
                                                          res_31516)) &&
           slt32(gtid_32139, res_31516)) && (sle32(0, m_33349) && slt32(m_33349,
                                                                        sizze_31493))) &&
         (sle32(0, gtid_32138) && slt32(gtid_32138, res_31516))) && (sle32(0,
                                                                           gtid_32139) &&
                                                                     slt32(gtid_32139,
                                                                           res_31516))) {
        *(__global float *) &mem_33513[(m_33349 * (res_31516 * res_31516) +
                                        gtid_32138 * res_31516 + gtid_32139) *
                                       4] = res_33293;
    }
    if (((((slt32(gtid_32137, gidzz_range_32918) && slt32(gtid_32138,
                                                          res_31516)) &&
           slt32(gtid_32139, res_31516)) && (sle32(0, m_33352) && slt32(m_33352,
                                                                        sizze_31493))) &&
         (sle32(0, gtid_32138) && slt32(gtid_32138, res_31516))) && (sle32(0,
                                                                           gtid_32139) &&
                                                                     slt32(gtid_32139,
                                                                           res_31516))) {
        *(__global float *) &mem_33513[(m_33352 * (res_31516 * res_31516) +
                                        gtid_32138 * res_31516 + gtid_32139) *
                                       4] = res_33294;
    }
    if (((((slt32(gtid_32137, gidzz_range_32918) && slt32(gtid_32138,
                                                          res_31516)) &&
           slt32(gtid_32139, res_31516)) && (sle32(0, m_33355) && slt32(m_33355,
                                                                        sizze_31493))) &&
         (sle32(0, gtid_32138) && slt32(gtid_32138, res_31516))) && (sle32(0,
                                                                           gtid_32139) &&
                                                                     slt32(gtid_32139,
                                                                           res_31516))) {
        *(__global float *) &mem_33513[(m_33355 * (res_31516 * res_31516) +
                                        gtid_32138 * res_31516 + gtid_32139) *
                                       4] = res_33295;
    }
    if (((((slt32(gtid_32137, gidzz_range_32918) && slt32(gtid_32138,
                                                          res_31516)) &&
           slt32(gtid_32139, res_31516)) && (sle32(0, m_33358) && slt32(m_33358,
                                                                        sizze_31493))) &&
         (sle32(0, gtid_32138) && slt32(gtid_32138, res_31516))) && (sle32(0,
                                                                           gtid_32139) &&
                                                                     slt32(gtid_32139,
                                                                           res_31516))) {
        *(__global float *) &mem_33513[(m_33358 * (res_31516 * res_31516) +
                                        gtid_32138 * res_31516 + gtid_32139) *
                                       4] = res_33296;
    }
    if (((((slt32(gtid_32137, gidzz_range_32918) && slt32(gtid_32138,
                                                          res_31516)) &&
           slt32(gtid_32139, res_31516)) && (sle32(0, m_33361) && slt32(m_33361,
                                                                        sizze_31493))) &&
         (sle32(0, gtid_32138) && slt32(gtid_32138, res_31516))) && (sle32(0,
                                                                           gtid_32139) &&
                                                                     slt32(gtid_32139,
                                                                           res_31516))) {
        *(__global float *) &mem_33513[(m_33361 * (res_31516 * res_31516) +
                                        gtid_32138 * res_31516 + gtid_32139) *
                                       4] = res_33297;
    }
    if (((((slt32(gtid_32137, gidzz_range_32918) && slt32(gtid_32138,
                                                          res_31516)) &&
           slt32(gtid_32139, res_31516)) && (sle32(0, m_33364) && slt32(m_33364,
                                                                        sizze_31493))) &&
         (sle32(0, gtid_32138) && slt32(gtid_32138, res_31516))) && (sle32(0,
                                                                           gtid_32139) &&
                                                                     slt32(gtid_32139,
                                                                           res_31516))) {
        *(__global float *) &mem_33513[(m_33364 * (res_31516 * res_31516) +
                                        gtid_32138 * res_31516 + gtid_32139) *
                                       4] = res_33298;
    }
    if (((((slt32(gtid_32137, gidzz_range_32918) && slt32(gtid_32138,
                                                          res_31516)) &&
           slt32(gtid_32139, res_31516)) && (sle32(0, m_33367) && slt32(m_33367,
                                                                        sizze_31493))) &&
         (sle32(0, gtid_32138) && slt32(gtid_32138, res_31516))) && (sle32(0,
                                                                           gtid_32139) &&
                                                                     slt32(gtid_32139,
                                                                           res_31516))) {
        *(__global float *) &mem_33513[(m_33367 * (res_31516 * res_31516) +
                                        gtid_32138 * res_31516 + gtid_32139) *
                                       4] = res_33299;
    }
    if (((((slt32(gtid_32137, gidzz_range_32918) && slt32(gtid_32138,
                                                          res_31516)) &&
           slt32(gtid_32139, res_31516)) && (sle32(0, m_33370) && slt32(m_33370,
                                                                        sizze_31493))) &&
         (sle32(0, gtid_32138) && slt32(gtid_32138, res_31516))) && (sle32(0,
                                                                           gtid_32139) &&
                                                                     slt32(gtid_32139,
                                                                           res_31516))) {
        *(__global float *) &mem_33513[(m_33370 * (res_31516 * res_31516) +
                                        gtid_32138 * res_31516 + gtid_32139) *
                                       4] = res_33300;
    }
    if (((((slt32(gtid_32137, gidzz_range_32918) && slt32(gtid_32138,
                                                          res_31516)) &&
           slt32(gtid_32139, res_31516)) && (sle32(0, m_33373) && slt32(m_33373,
                                                                        sizze_31493))) &&
         (sle32(0, gtid_32138) && slt32(gtid_32138, res_31516))) && (sle32(0,
                                                                           gtid_32139) &&
                                                                     slt32(gtid_32139,
                                                                           res_31516))) {
        *(__global float *) &mem_33513[(m_33373 * (res_31516 * res_31516) +
                                        gtid_32138 * res_31516 + gtid_32139) *
                                       4] = res_33301;
    }
    if (((((slt32(gtid_32137, gidzz_range_32918) && slt32(gtid_32138,
                                                          res_31516)) &&
           slt32(gtid_32139, res_31516)) && (sle32(0, m_33376) && slt32(m_33376,
                                                                        sizze_31493))) &&
         (sle32(0, gtid_32138) && slt32(gtid_32138, res_31516))) && (sle32(0,
                                                                           gtid_32139) &&
                                                                     slt32(gtid_32139,
                                                                           res_31516))) {
        *(__global float *) &mem_33513[(m_33376 * (res_31516 * res_31516) +
                                        gtid_32138 * res_31516 + gtid_32139) *
                                       4] = res_33302;
    }
    if (((((slt32(gtid_32137, gidzz_range_32918) && slt32(gtid_32138,
                                                          res_31516)) &&
           slt32(gtid_32139, res_31516)) && (sle32(0, m_33379) && slt32(m_33379,
                                                                        sizze_31493))) &&
         (sle32(0, gtid_32138) && slt32(gtid_32138, res_31516))) && (sle32(0,
                                                                           gtid_32139) &&
                                                                     slt32(gtid_32139,
                                                                           res_31516))) {
        *(__global float *) &mem_33513[(m_33379 * (res_31516 * res_31516) +
                                        gtid_32138 * res_31516 + gtid_32139) *
                                       4] = res_33303;
    }
    if (((((slt32(gtid_32137, gidzz_range_32918) && slt32(gtid_32138,
                                                          res_31516)) &&
           slt32(gtid_32139, res_31516)) && (sle32(0, m_33382) && slt32(m_33382,
                                                                        sizze_31493))) &&
         (sle32(0, gtid_32138) && slt32(gtid_32138, res_31516))) && (sle32(0,
                                                                           gtid_32139) &&
                                                                     slt32(gtid_32139,
                                                                           res_31516))) {
        *(__global float *) &mem_33513[(m_33382 * (res_31516 * res_31516) +
                                        gtid_32138 * res_31516 + gtid_32139) *
                                       4] = res_33304;
    }
    if (((((slt32(gtid_32137, gidzz_range_32918) && slt32(gtid_32138,
                                                          res_31516)) &&
           slt32(gtid_32139, res_31516)) && (sle32(0, m_33385) && slt32(m_33385,
                                                                        sizze_31493))) &&
         (sle32(0, gtid_32138) && slt32(gtid_32138, res_31516))) && (sle32(0,
                                                                           gtid_32139) &&
                                                                     slt32(gtid_32139,
                                                                           res_31516))) {
        *(__global float *) &mem_33513[(m_33385 * (res_31516 * res_31516) +
                                        gtid_32138 * res_31516 + gtid_32139) *
                                       4] = res_33305;
    }
    if (((((slt32(gtid_32137, gidzz_range_32918) && slt32(gtid_32138,
                                                          res_31516)) &&
           slt32(gtid_32139, res_31516)) && (sle32(0, m_33388) && slt32(m_33388,
                                                                        sizze_31493))) &&
         (sle32(0, gtid_32138) && slt32(gtid_32138, res_31516))) && (sle32(0,
                                                                           gtid_32139) &&
                                                                     slt32(gtid_32139,
                                                                           res_31516))) {
        *(__global float *) &mem_33513[(m_33388 * (res_31516 * res_31516) +
                                        gtid_32138 * res_31516 + gtid_32139) *
                                       4] = res_33306;
    }
    if (((((slt32(gtid_32137, gidzz_range_32918) && slt32(gtid_32138,
                                                          res_31516)) &&
           slt32(gtid_32139, res_31516)) && (sle32(0, m_33391) && slt32(m_33391,
                                                                        sizze_31493))) &&
         (sle32(0, gtid_32138) && slt32(gtid_32138, res_31516))) && (sle32(0,
                                                                           gtid_32139) &&
                                                                     slt32(gtid_32139,
                                                                           res_31516))) {
        *(__global float *) &mem_33513[(m_33391 * (res_31516 * res_31516) +
                                        gtid_32138 * res_31516 + gtid_32139) *
                                       4] = res_33307;
    }
    if (((((slt32(gtid_32137, gidzz_range_32918) && slt32(gtid_32138,
                                                          res_31516)) &&
           slt32(gtid_32139, res_31516)) && (sle32(0, m_33394) && slt32(m_33394,
                                                                        sizze_31493))) &&
         (sle32(0, gtid_32138) && slt32(gtid_32138, res_31516))) && (sle32(0,
                                                                           gtid_32139) &&
                                                                     slt32(gtid_32139,
                                                                           res_31516))) {
        *(__global float *) &mem_33513[(m_33394 * (res_31516 * res_31516) +
                                        gtid_32138 * res_31516 + gtid_32139) *
                                       4] = res_33308;
    }
    if (((((slt32(gtid_32137, gidzz_range_32918) && slt32(gtid_32138,
                                                          res_31516)) &&
           slt32(gtid_32139, res_31516)) && (sle32(0, m_33397) && slt32(m_33397,
                                                                        sizze_31493))) &&
         (sle32(0, gtid_32138) && slt32(gtid_32138, res_31516))) && (sle32(0,
                                                                           gtid_32139) &&
                                                                     slt32(gtid_32139,
                                                                           res_31516))) {
        *(__global float *) &mem_33513[(m_33397 * (res_31516 * res_31516) +
                                        gtid_32138 * res_31516 + gtid_32139) *
                                       4] = res_33309;
    }
    if (((((slt32(gtid_32137, gidzz_range_32918) && slt32(gtid_32138,
                                                          res_31516)) &&
           slt32(gtid_32139, res_31516)) && (sle32(0, m_33400) && slt32(m_33400,
                                                                        sizze_31493))) &&
         (sle32(0, gtid_32138) && slt32(gtid_32138, res_31516))) && (sle32(0,
                                                                           gtid_32139) &&
                                                                     slt32(gtid_32139,
                                                                           res_31516))) {
        *(__global float *) &mem_33513[(m_33400 * (res_31516 * res_31516) +
                                        gtid_32138 * res_31516 + gtid_32139) *
                                       4] = res_33310;
    }
}
__kernel void map_32203(int32_t sizze_31493, int32_t arg_31622,
                        int32_t arg_31636, __global unsigned char *mem_33524,
                        __global unsigned char *mem_33531)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_32203;
    int32_t local_tid_32204;
    int32_t group_sizze_33731;
    int32_t wave_sizze_33730;
    int32_t group_id_32205;
    
    global_tid_32203 = get_global_id(0);
    local_tid_32204 = get_local_id(0);
    group_sizze_33731 = get_local_size(0);
    wave_sizze_33730 = LOCKSTEP_WIDTH;
    group_id_32205 = get_group_id(0);
    
    int32_t gtid_32194;
    int32_t gtid_32195;
    
    gtid_32194 = squot32(global_tid_32203, arg_31636);
    gtid_32195 = global_tid_32203 - squot32(global_tid_32203, arg_31636) *
        arg_31636;
    
    float write_value_31715;
    
    if (slt32(gtid_32194, sizze_31493) && slt32(gtid_32195, arg_31636)) {
        write_value_31715 = *(__global float *) &mem_33531[(gtid_32194 *
                                                            arg_31636 +
                                                            gtid_32195) * 4];
    }
    if (((slt32(gtid_32194, sizze_31493) && slt32(gtid_32195, arg_31636)) &&
         (sle32(0, gtid_32194) && slt32(gtid_32194, sizze_31493))) && (sle32(0,
                                                                             gtid_32195) &&
                                                                       slt32(gtid_32195,
                                                                             arg_31622))) {
        *(__global float *) &mem_33524[(gtid_32194 * arg_31622 + gtid_32195) *
                                       4] = write_value_31715;
    }
}
__kernel void map_32217(int32_t sizze_31493, int32_t arg_31622,
                        int32_t res_31635, int32_t arg_31636, int32_t m_31652,
                        int32_t i_31686, __global unsigned char *mem_33524,
                        __global unsigned char *mem_33527, __global
                        unsigned char *mem_33531)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_32217;
    int32_t local_tid_32218;
    int32_t group_sizze_33729;
    int32_t wave_sizze_33728;
    int32_t group_id_32219;
    
    global_tid_32217 = get_global_id(0);
    local_tid_32218 = get_local_id(0);
    group_sizze_33729 = get_local_size(0);
    wave_sizze_33728 = LOCKSTEP_WIDTH;
    group_id_32219 = get_group_id(0);
    
    int32_t gtid_32208;
    int32_t gtid_32209;
    
    gtid_32208 = squot32(global_tid_32217, arg_31636);
    gtid_32209 = global_tid_32217 - squot32(global_tid_32217, arg_31636) *
        arg_31636;
    
    float res_32221;
    bool cond_32222;
    int32_t res_32224;
    int32_t res_32225;
    float res_32226;
    
    if (slt32(gtid_32208, sizze_31493) && slt32(gtid_32209, arg_31636)) {
        res_32221 = *(__global float *) &mem_33524[(gtid_32208 * arg_31622 +
                                                    i_31686) * 4];
        cond_32222 = *(__global bool *) &mem_33527[gtid_32208];
        res_32224 = sdiv32(gtid_32209, res_31635);
        res_32225 = smod32(gtid_32209, res_31635);
        if (cond_32222) {
            int32_t x_32227;
            int32_t i_32228;
            float res_32229;
            
            x_32227 = res_31635 * res_32224;
            i_32228 = res_32225 + x_32227;
            res_32229 = *(__global float *) &mem_33524[(gtid_32208 * arg_31622 +
                                                        i_32228) * 4];
            res_32226 = res_32229;
        } else {
            float x_32230;
            float res_32231;
            bool cond_32232;
            float res_32233;
            
            x_32230 = *(__global float *) &mem_33524[(gtid_32208 * arg_31622 +
                                                      res_32225) * 4];
            res_32231 = x_32230 / res_32221;
            cond_32232 = slt32(res_32224, m_31652);
            if (cond_32232) {
                int32_t x_32234;
                int32_t x_32235;
                int32_t i_32236;
                float x_32237;
                int32_t i_32238;
                float x_32239;
                float y_32240;
                float res_32241;
                
                x_32234 = 1 + res_32224;
                x_32235 = res_31635 * x_32234;
                i_32236 = res_32225 + x_32235;
                x_32237 = *(__global float *) &mem_33524[(gtid_32208 *
                                                          arg_31622 + i_32236) *
                                                         4];
                i_32238 = i_31686 + x_32235;
                x_32239 = *(__global float *) &mem_33524[(gtid_32208 *
                                                          arg_31622 + i_32238) *
                                                         4];
                y_32240 = res_32231 * x_32239;
                res_32241 = x_32237 - y_32240;
                res_32233 = res_32241;
            } else {
                res_32233 = res_32231;
            }
            res_32226 = res_32233;
        }
    }
    if (slt32(gtid_32208, sizze_31493) && slt32(gtid_32209, arg_31636)) {
        *(__global float *) &mem_33531[(gtid_32208 * arg_31636 + gtid_32209) *
                                       4] = res_32226;
    }
}
__kernel void map_32249(int32_t sizze_31493, int32_t arg_31622, int32_t i_31686,
                        __global unsigned char *mem_33524, __global
                        unsigned char *mem_33527)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_32249;
    int32_t local_tid_32250;
    int32_t group_sizze_33727;
    int32_t wave_sizze_33726;
    int32_t group_id_32251;
    
    global_tid_32249 = get_global_id(0);
    local_tid_32250 = get_local_id(0);
    group_sizze_33727 = get_local_size(0);
    wave_sizze_33726 = LOCKSTEP_WIDTH;
    group_id_32251 = get_group_id(0);
    
    int32_t gtid_32242;
    
    gtid_32242 = global_tid_32249;
    
    float res_32253;
    bool cond_32254;
    
    if (slt32(gtid_32242, sizze_31493)) {
        res_32253 = *(__global float *) &mem_33524[(gtid_32242 * arg_31622 +
                                                    i_31686) * 4];
        cond_32254 = res_32253 == 0.0F;
    }
    if (slt32(gtid_32242, sizze_31493)) {
        *(__global bool *) &mem_33527[gtid_32242] = cond_32254;
    }
}
__kernel void map_32264(int32_t sizze_31493, int32_t res_31516, int32_t j_31618,
                        int32_t arg_31622, __global unsigned char *mem_33513,
                        __global unsigned char *mem_33524)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_32264;
    int32_t local_tid_32265;
    int32_t group_sizze_33724;
    int32_t wave_sizze_33723;
    int32_t group_id_32266;
    
    global_tid_32264 = get_global_id(0);
    local_tid_32265 = get_local_id(0);
    group_sizze_33724 = get_local_size(0);
    wave_sizze_33723 = LOCKSTEP_WIDTH;
    group_id_32266 = get_group_id(0);
    
    int32_t gtid_32255;
    int32_t gtid_32256;
    
    gtid_32255 = squot32(global_tid_32264, arg_31622);
    gtid_32256 = global_tid_32264 - squot32(global_tid_32264, arg_31622) *
        arg_31622;
    
    int32_t res_32269;
    int32_t res_32270;
    bool cond_32271;
    float res_32272;
    
    if (slt32(gtid_32255, sizze_31493) && slt32(gtid_32256, arg_31622)) {
        res_32269 = sdiv32(gtid_32256, j_31618);
        res_32270 = smod32(gtid_32256, j_31618);
        cond_32271 = slt32(res_32270, res_31516);
        if (cond_32271) {
            float res_32273 = *(__global float *) &mem_33513[(gtid_32255 *
                                                              (res_31516 *
                                                               res_31516) +
                                                              res_32269 *
                                                              res_31516 +
                                                              res_32270) * 4];
            
            res_32272 = res_32273;
        } else {
            int32_t y_32274;
            bool cond_32275;
            float res_32276;
            
            y_32274 = res_31516 + res_32269;
            cond_32275 = res_32270 == y_32274;
            if (cond_32275) {
                res_32276 = 1.0F;
            } else {
                res_32276 = 0.0F;
            }
            res_32272 = res_32276;
        }
    }
    if (slt32(gtid_32255, sizze_31493) && slt32(gtid_32256, arg_31622)) {
        *(__global float *) &mem_33524[(gtid_32255 * arg_31622 + gtid_32256) *
                                       4] = res_32272;
    }
}
__kernel void map_32297(int32_t sizze_31493, int32_t sizze_31494,
                        int32_t n_31497, int32_t res_31516, __global
                        unsigned char *images_mem_33491, __global
                        unsigned char *mem_33504, __global
                        unsigned char *mem_33544)
{
    const int32_t tile_sizze_33403 = mainzitile_sizze_33402;
    const int32_t tiled_group_sizze_33404 = mainzitile_sizze_33402 *
                  mainzitile_sizze_33402;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    ALIGNED_LOCAL_MEMORY(mem_33536_backing_0, 4 *
                         sext_i32_i64(mainzitile_sizze_33402 *
                         mainzitile_sizze_33402));
    ALIGNED_LOCAL_MEMORY(mem_33540_backing_1, 4 *
                         sext_i32_i64(mainzitile_sizze_33402 *
                         mainzitile_sizze_33402));
    
    int32_t global_tid_32297;
    int32_t local_tid_32298;
    int32_t group_sizze_33733;
    int32_t wave_sizze_33732;
    int32_t group_id_32299;
    
    global_tid_32297 = get_global_id(0);
    local_tid_32298 = get_local_id(0);
    group_sizze_33733 = get_local_size(0);
    wave_sizze_33732 = LOCKSTEP_WIDTH;
    group_id_32299 = get_group_id(0);
    
    int32_t gtid_32288;
    int32_t gtid_32289;
    int32_t ltid_33405;
    int32_t ltid_33406;
    
    gtid_32288 = squot32(srem32(global_tid_32297, tile_sizze_33403 *
                                tile_sizze_33403), tile_sizze_33403) +
        squot32(squot32(global_tid_32297, tile_sizze_33403 * tile_sizze_33403),
                squot32(res_31516 + tile_sizze_33403 - 1, tile_sizze_33403)) *
        tile_sizze_33403;
    gtid_32289 = srem32(global_tid_32297, tile_sizze_33403 * tile_sizze_33403) -
        squot32(srem32(global_tid_32297, tile_sizze_33403 * tile_sizze_33403),
                tile_sizze_33403) * tile_sizze_33403 +
        (squot32(global_tid_32297, tile_sizze_33403 * tile_sizze_33403) -
         squot32(squot32(global_tid_32297, tile_sizze_33403 * tile_sizze_33403),
                 squot32(res_31516 + tile_sizze_33403 - 1, tile_sizze_33403)) *
         squot32(res_31516 + tile_sizze_33403 - 1, tile_sizze_33403)) *
        tile_sizze_33403;
    ltid_33405 = squot32(srem32(global_tid_32297, tile_sizze_33403 *
                                tile_sizze_33403), tile_sizze_33403);
    ltid_33406 = srem32(global_tid_32297, tile_sizze_33403 * tile_sizze_33403) -
        squot32(srem32(global_tid_32297, tile_sizze_33403 * tile_sizze_33403),
                tile_sizze_33403) * tile_sizze_33403;
    if (slt32(gtid_32288, sizze_31493) && slt32(gtid_32289, res_31516)) { }
    
    __local char *mem_33536;
    __local char *mem_33540;
    float res_32302;
    
    mem_33536 = (__local char *) mem_33536_backing_0;
    mem_33540 = (__local char *) mem_33540_backing_1;
    
    float x_32305 = 0.0F;
    int32_t chunk_sizze_32303;
    int32_t chunk_offset_32304 = 0;
    
    while (slt32(chunk_offset_32304, n_31497)) {
        if (slt32(n_31497 - chunk_offset_32304, tile_sizze_33403)) {
            chunk_sizze_32303 = n_31497 - chunk_offset_32304;
        } else {
            chunk_sizze_32303 = tile_sizze_33403;
        }
        for (int32_t comb_iter_33734 = 0; comb_iter_33734 <
             squot32(tile_sizze_33403 * tile_sizze_33403 +
                     tiled_group_sizze_33404 - 1, tiled_group_sizze_33404);
             comb_iter_33734++) {
            int32_t cid_33418;
            int32_t cid_33419;
            int32_t flat_comb_id_33735 = comb_iter_33734 *
                    tiled_group_sizze_33404 + local_tid_32298;
            
            cid_33418 = squot32(flat_comb_id_33735, tile_sizze_33403);
            cid_33419 = flat_comb_id_33735 - squot32(flat_comb_id_33735,
                                                     tile_sizze_33403) *
                tile_sizze_33403;
            if ((slt32(cid_33418, chunk_sizze_32303) && slt32(cid_33419,
                                                              tile_sizze_33403)) &&
                slt32(gtid_32289, res_31516)) {
                float x_chunk_outer_elem_33417 = *(__global
                                                   float *) &mem_33504[(res_31516 *
                                                                        0 +
                                                                        gtid_32289 +
                                                                        res_31516 *
                                                                        chunk_offset_32304 +
                                                                        ltid_33405 *
                                                                        res_31516) *
                                                                       4];
                
                *(__local float *) &mem_33536[(cid_33418 * tile_sizze_33403 +
                                               cid_33419) * 4] =
                    x_chunk_outer_elem_33417;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (slt32(gtid_32288, sizze_31493) && slt32(gtid_32289, res_31516)) { }
        for (int32_t comb_iter_33736 = 0; comb_iter_33736 <
             squot32(tile_sizze_33403 * tile_sizze_33403 +
                     tiled_group_sizze_33404 - 1, tiled_group_sizze_33404);
             comb_iter_33736++) {
            int32_t cid_33423;
            int32_t cid_33424;
            int32_t flat_comb_id_33737 = comb_iter_33736 *
                    tiled_group_sizze_33404 + local_tid_32298;
            
            cid_33423 = squot32(flat_comb_id_33737, tile_sizze_33403);
            cid_33424 = flat_comb_id_33737 - squot32(flat_comb_id_33737,
                                                     tile_sizze_33403) *
                tile_sizze_33403;
            if ((slt32(cid_33423, tile_sizze_33403) && slt32(cid_33424,
                                                             chunk_sizze_32303)) &&
                slt32(gtid_32288, sizze_31493)) {
                float x_chunk_outer_elem_33422 = *(__global
                                                   float *) &images_mem_33491[(gtid_32288 *
                                                                               sizze_31494 +
                                                                               chunk_offset_32304 +
                                                                               ltid_33406) *
                                                                              4];
                
                *(__local float *) &mem_33540[(cid_33423 * tile_sizze_33403 +
                                               cid_33424) * 4] =
                    x_chunk_outer_elem_33422;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (slt32(gtid_32288, sizze_31493) && slt32(gtid_32289, res_31516)) { }
        
        float res_32308;
        float sync_33426;
        float acc_32311 = x_32305;
        int32_t groupstream_mapaccum_dummy_chunk_sizze_32309;
        
        groupstream_mapaccum_dummy_chunk_sizze_32309 = 1;
        if (slt32(gtid_32288, sizze_31493) && slt32(gtid_32289, res_31516)) {
            if (chunk_sizze_32303 == tile_sizze_33403) {
                for (int32_t i_32310 = 0; i_32310 < tile_sizze_33403;
                     i_32310++) {
                    float x_32314;
                    float x_32315;
                    bool res_32317;
                    float res_32318;
                    float res_32321;
                    
                    x_32314 = *(__local float *) &mem_33536[(tile_sizze_33403 *
                                                             0 + ltid_33406 +
                                                             tile_sizze_33403 *
                                                             i_32310 + 0 *
                                                             tile_sizze_33403) *
                                                            4];
                    x_32315 = *(__local float *) &mem_33540[(ltid_33405 *
                                                             tile_sizze_33403 +
                                                             i_32310) * 4];
                    res_32317 = futrts_isnan32(x_32315);
                    if (res_32317) {
                        res_32318 = 0.0F;
                    } else {
                        float res_32319 = x_32314 * x_32315;
                        
                        res_32318 = res_32319;
                    }
                    res_32321 = acc_32311 + res_32318;
                    
                    float acc_tmp_33738 = res_32321;
                    
                    acc_32311 = acc_tmp_33738;
                }
            } else {
                for (int32_t i_32310 = 0; i_32310 < chunk_sizze_32303;
                     i_32310++) {
                    float x_32314;
                    float x_32315;
                    bool res_32317;
                    float res_32318;
                    float res_32321;
                    
                    x_32314 = *(__local float *) &mem_33536[(tile_sizze_33403 *
                                                             0 + ltid_33406 +
                                                             tile_sizze_33403 *
                                                             i_32310 + 0 *
                                                             tile_sizze_33403) *
                                                            4];
                    x_32315 = *(__local float *) &mem_33540[(ltid_33405 *
                                                             tile_sizze_33403 +
                                                             i_32310) * 4];
                    res_32317 = futrts_isnan32(x_32315);
                    if (res_32317) {
                        res_32318 = 0.0F;
                    } else {
                        float res_32319 = x_32314 * x_32315;
                        
                        res_32318 = res_32319;
                    }
                    res_32321 = acc_32311 + res_32318;
                    
                    float acc_tmp_33739 = res_32321;
                    
                    acc_32311 = acc_tmp_33739;
                }
            }
        }
        res_32308 = acc_32311;
        sync_33426 = res_32308;
        barrier(CLK_LOCAL_MEM_FENCE);
        x_32305 = sync_33426;
        chunk_offset_32304 += tile_sizze_33403;
    }
    res_32302 = x_32305;
    if (slt32(gtid_32288, sizze_31493) && slt32(gtid_32289, res_31516)) {
        *(__global float *) &mem_33544[(gtid_32288 * res_31516 + gtid_32289) *
                                       4] = res_32302;
    }
}
__kernel void map_32343(int32_t sizze_31493, int32_t res_31516,
                        int32_t j_m_i_31619, __global unsigned char *mem_33544,
                        __global unsigned char *mem_33549, __global
                        unsigned char *mem_33553)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_32343;
    int32_t local_tid_32344;
    int32_t group_sizze_33746;
    int32_t wave_sizze_33745;
    int32_t group_id_32345;
    
    global_tid_32343 = get_global_id(0);
    local_tid_32344 = get_local_id(0);
    group_sizze_33746 = get_local_size(0);
    wave_sizze_33745 = LOCKSTEP_WIDTH;
    group_id_32345 = get_group_id(0);
    
    int32_t gtid_32334;
    int32_t gtid_32335;
    
    gtid_32334 = squot32(global_tid_32343, res_31516);
    gtid_32335 = global_tid_32343 - squot32(global_tid_32343, res_31516) *
        res_31516;
    
    int32_t binop_x_33474;
    float res_32348;
    
    if (slt32(gtid_32334, sizze_31493) && slt32(gtid_32335, res_31516)) {
        binop_x_33474 = j_m_i_31619 * gtid_32334;
        
        float x_32351 = 0.0F;
        
        for (int32_t chunk_offset_32350 = 0; chunk_offset_32350 < j_m_i_31619;
             chunk_offset_32350++) {
            int32_t binop_x_33475;
            int32_t new_index_33476;
            int32_t binop_y_33482;
            int32_t new_index_33483;
            float x_32360;
            float x_32361;
            float res_32363;
            float res_32365;
            
            binop_x_33475 = chunk_offset_32350 + binop_x_33474;
            new_index_33476 = squot32(binop_x_33475, res_31516);
            binop_y_33482 = res_31516 * new_index_33476;
            new_index_33483 = binop_x_33475 - binop_y_33482;
            x_32360 = *(__global float *) &mem_33544[(new_index_33476 *
                                                      res_31516 +
                                                      new_index_33483) * 4];
            x_32361 = *(__global float *) &mem_33549[(chunk_offset_32350 *
                                                      (res_31516 *
                                                       sizze_31493) +
                                                      gtid_32334 * res_31516 +
                                                      gtid_32335) * 4];
            res_32363 = x_32360 * x_32361;
            res_32365 = x_32351 + res_32363;
            
            float x_tmp_33747 = res_32365;
            
            x_32351 = x_tmp_33747;
        }
        res_32348 = x_32351;
    }
    if (slt32(gtid_32334, sizze_31493) && slt32(gtid_32335, res_31516)) {
        *(__global float *) &mem_33553[(gtid_32334 * res_31516 + gtid_32335) *
                                       4] = res_32348;
    }
}
__kernel void map_32386(int32_t sizze_31492, int32_t sizze_31493,
                        int32_t res_31516, __global unsigned char *mem_33553,
                        __global unsigned char *mem_33557, __global
                        unsigned char *mem_33569)
{
    const int32_t tile_sizze_33428 = mainzitile_sizze_33427;
    const int32_t tiled_group_sizze_33429 = mainzitile_sizze_33427 *
                  mainzitile_sizze_33427;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    ALIGNED_LOCAL_MEMORY(mem_33561_backing_0, 4 *
                         sext_i32_i64(mainzitile_sizze_33427 *
                         mainzitile_sizze_33427));
    ALIGNED_LOCAL_MEMORY(mem_33565_backing_1, 4 *
                         sext_i32_i64(mainzitile_sizze_33427 *
                         mainzitile_sizze_33427));
    
    int32_t global_tid_32386;
    int32_t local_tid_32387;
    int32_t group_sizze_33749;
    int32_t wave_sizze_33748;
    int32_t group_id_32388;
    
    global_tid_32386 = get_global_id(0);
    local_tid_32387 = get_local_id(0);
    group_sizze_33749 = get_local_size(0);
    wave_sizze_33748 = LOCKSTEP_WIDTH;
    group_id_32388 = get_group_id(0);
    
    int32_t gtid_32377;
    int32_t gtid_32378;
    int32_t ltid_33430;
    int32_t ltid_33431;
    
    gtid_32377 = squot32(srem32(global_tid_32386, tile_sizze_33428 *
                                tile_sizze_33428), tile_sizze_33428) +
        squot32(squot32(global_tid_32386, tile_sizze_33428 * tile_sizze_33428),
                squot32(sizze_31492 + tile_sizze_33428 - 1, tile_sizze_33428)) *
        tile_sizze_33428;
    gtid_32378 = srem32(global_tid_32386, tile_sizze_33428 * tile_sizze_33428) -
        squot32(srem32(global_tid_32386, tile_sizze_33428 * tile_sizze_33428),
                tile_sizze_33428) * tile_sizze_33428 +
        (squot32(global_tid_32386, tile_sizze_33428 * tile_sizze_33428) -
         squot32(squot32(global_tid_32386, tile_sizze_33428 * tile_sizze_33428),
                 squot32(sizze_31492 + tile_sizze_33428 - 1,
                         tile_sizze_33428)) * squot32(sizze_31492 +
                                                      tile_sizze_33428 - 1,
                                                      tile_sizze_33428)) *
        tile_sizze_33428;
    ltid_33430 = squot32(srem32(global_tid_32386, tile_sizze_33428 *
                                tile_sizze_33428), tile_sizze_33428);
    ltid_33431 = srem32(global_tid_32386, tile_sizze_33428 * tile_sizze_33428) -
        squot32(srem32(global_tid_32386, tile_sizze_33428 * tile_sizze_33428),
                tile_sizze_33428) * tile_sizze_33428;
    if (slt32(gtid_32377, sizze_31493) && slt32(gtid_32378, sizze_31492)) { }
    
    __local char *mem_33561;
    __local char *mem_33565;
    float res_32391;
    
    mem_33561 = (__local char *) mem_33561_backing_0;
    mem_33565 = (__local char *) mem_33565_backing_1;
    
    float x_32394 = 0.0F;
    int32_t chunk_sizze_32392;
    int32_t chunk_offset_32393 = 0;
    
    while (slt32(chunk_offset_32393, res_31516)) {
        if (slt32(res_31516 - chunk_offset_32393, tile_sizze_33428)) {
            chunk_sizze_32392 = res_31516 - chunk_offset_32393;
        } else {
            chunk_sizze_32392 = tile_sizze_33428;
        }
        for (int32_t comb_iter_33750 = 0; comb_iter_33750 <
             squot32(tile_sizze_33428 * tile_sizze_33428 +
                     tiled_group_sizze_33429 - 1, tiled_group_sizze_33429);
             comb_iter_33750++) {
            int32_t cid_33443;
            int32_t cid_33444;
            int32_t flat_comb_id_33751 = comb_iter_33750 *
                    tiled_group_sizze_33429 + local_tid_32387;
            
            cid_33443 = squot32(flat_comb_id_33751, tile_sizze_33428);
            cid_33444 = flat_comb_id_33751 - squot32(flat_comb_id_33751,
                                                     tile_sizze_33428) *
                tile_sizze_33428;
            if ((slt32(cid_33443, tile_sizze_33428) && slt32(cid_33444,
                                                             chunk_sizze_32392)) &&
                slt32(gtid_32377, sizze_31493)) {
                float x_chunk_outer_elem_33442 = *(__global
                                                   float *) &mem_33553[(gtid_32377 *
                                                                        res_31516 +
                                                                        chunk_offset_32393 +
                                                                        ltid_33431) *
                                                                       4];
                
                *(__local float *) &mem_33561[(cid_33443 * tile_sizze_33428 +
                                               cid_33444) * 4] =
                    x_chunk_outer_elem_33442;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (slt32(gtid_32377, sizze_31493) && slt32(gtid_32378,
                                                    sizze_31492)) { }
        for (int32_t comb_iter_33752 = 0; comb_iter_33752 <
             squot32(tile_sizze_33428 * tile_sizze_33428 +
                     tiled_group_sizze_33429 - 1, tiled_group_sizze_33429);
             comb_iter_33752++) {
            int32_t cid_33448;
            int32_t cid_33449;
            int32_t flat_comb_id_33753 = comb_iter_33752 *
                    tiled_group_sizze_33429 + local_tid_32387;
            
            cid_33448 = squot32(flat_comb_id_33753, tile_sizze_33428);
            cid_33449 = flat_comb_id_33753 - squot32(flat_comb_id_33753,
                                                     tile_sizze_33428) *
                tile_sizze_33428;
            if ((slt32(cid_33448, chunk_sizze_32392) && slt32(cid_33449,
                                                              tile_sizze_33428)) &&
                slt32(gtid_32378, sizze_31492)) {
                float x_chunk_outer_elem_33447 = *(__global
                                                   float *) &mem_33557[(gtid_32378 +
                                                                        sizze_31492 *
                                                                        chunk_offset_32393 +
                                                                        ltid_33430 *
                                                                        sizze_31492) *
                                                                       4];
                
                *(__local float *) &mem_33565[(cid_33448 * tile_sizze_33428 +
                                               cid_33449) * 4] =
                    x_chunk_outer_elem_33447;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (slt32(gtid_32377, sizze_31493) && slt32(gtid_32378,
                                                    sizze_31492)) { }
        
        float res_32397;
        float sync_33451;
        float acc_32400 = x_32394;
        int32_t groupstream_mapaccum_dummy_chunk_sizze_32398;
        
        groupstream_mapaccum_dummy_chunk_sizze_32398 = 1;
        if (slt32(gtid_32377, sizze_31493) && slt32(gtid_32378, sizze_31492)) {
            if (chunk_sizze_32392 == tile_sizze_33428) {
                for (int32_t i_32399 = 0; i_32399 < tile_sizze_33428;
                     i_32399++) {
                    float x_32403;
                    float x_32404;
                    float res_32406;
                    float res_32408;
                    
                    x_32403 = *(__local float *) &mem_33561[(ltid_33430 *
                                                             tile_sizze_33428 +
                                                             i_32399) * 4];
                    x_32404 = *(__local float *) &mem_33565[(tile_sizze_33428 *
                                                             0 + ltid_33431 +
                                                             tile_sizze_33428 *
                                                             i_32399 + 0 *
                                                             tile_sizze_33428) *
                                                            4];
                    res_32406 = x_32403 * x_32404;
                    res_32408 = acc_32400 + res_32406;
                    
                    float acc_tmp_33754 = res_32408;
                    
                    acc_32400 = acc_tmp_33754;
                }
            } else {
                for (int32_t i_32399 = 0; i_32399 < chunk_sizze_32392;
                     i_32399++) {
                    float x_32403;
                    float x_32404;
                    float res_32406;
                    float res_32408;
                    
                    x_32403 = *(__local float *) &mem_33561[(ltid_33430 *
                                                             tile_sizze_33428 +
                                                             i_32399) * 4];
                    x_32404 = *(__local float *) &mem_33565[(tile_sizze_33428 *
                                                             0 + ltid_33431 +
                                                             tile_sizze_33428 *
                                                             i_32399 + 0 *
                                                             tile_sizze_33428) *
                                                            4];
                    res_32406 = x_32403 * x_32404;
                    res_32408 = acc_32400 + res_32406;
                    
                    float acc_tmp_33755 = res_32408;
                    
                    acc_32400 = acc_tmp_33755;
                }
            }
        }
        res_32397 = acc_32400;
        sync_33451 = res_32397;
        barrier(CLK_LOCAL_MEM_FENCE);
        x_32394 = sync_33451;
        chunk_offset_32393 += tile_sizze_33428;
    }
    res_32391 = x_32394;
    if (slt32(gtid_32377, sizze_31493) && slt32(gtid_32378, sizze_31492)) {
        *(__global float *) &mem_33569[(gtid_32377 * sizze_31492 + gtid_32378) *
                                       4] = res_32391;
    }
}
__kernel void map_32424(int32_t sizze_31492, int32_t sizze_31493,
                        int32_t i_31764, __global unsigned char *mem_33573,
                        __global unsigned char *mem_33577, __global
                        unsigned char *mem_33580, __global
                        unsigned char *mem_33584, __global
                        unsigned char *mem_33588, __global
                        unsigned char *mem_33592)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_32424;
    int32_t local_tid_32425;
    int32_t group_sizze_33827;
    int32_t wave_sizze_33826;
    int32_t group_id_32426;
    
    global_tid_32424 = get_global_id(0);
    local_tid_32425 = get_local_id(0);
    group_sizze_33827 = get_local_size(0);
    wave_sizze_33826 = LOCKSTEP_WIDTH;
    group_id_32426 = get_group_id(0);
    
    int32_t gtid_32415;
    int32_t gtid_32416;
    
    gtid_32415 = squot32(global_tid_32424, sizze_31492);
    gtid_32416 = global_tid_32424 - squot32(global_tid_32424, sizze_31492) *
        sizze_31492;
    
    int32_t res_31804;
    int32_t x_31809;
    bool x_31810;
    int32_t x_31811;
    float write_value_31812;
    int32_t res_31814;
    int32_t res_31815;
    
    if (slt32(gtid_32415, sizze_31493) && slt32(gtid_32416, sizze_31492)) {
        res_31804 = *(__global int32_t *) &mem_33573[(gtid_32415 * sizze_31492 +
                                                      i_31764) * 4];
        x_31809 = *(__global int32_t *) &mem_33577[(gtid_32415 * sizze_31492 +
                                                    gtid_32416) * 4];
        x_31810 = *(__global bool *) &mem_33580[gtid_32415 * sizze_31492 +
                                                gtid_32416];
        x_31811 = *(__global int32_t *) &mem_33573[(gtid_32415 * sizze_31492 +
                                                    gtid_32416) * 4];
        write_value_31812 = *(__global float *) &mem_33584[(gtid_32415 *
                                                            sizze_31492 +
                                                            gtid_32416) * 4];
        res_31814 = res_31804 + x_31809;
        if (x_31810) {
            int32_t res_31816 = x_31811 - 1;
            
            res_31815 = res_31816;
        } else {
            int32_t res_31817 = res_31814 - 1;
            
            res_31815 = res_31817;
        }
    }
    if (((slt32(gtid_32415, sizze_31493) && slt32(gtid_32416, sizze_31492)) &&
         (sle32(0, gtid_32415) && slt32(gtid_32415, sizze_31493))) && (sle32(0,
                                                                             res_31815) &&
                                                                       slt32(res_31815,
                                                                             sizze_31492))) {
        *(__global float *) &mem_33588[(gtid_32415 * sizze_31492 + res_31815) *
                                       4] = write_value_31812;
    }
    if (((slt32(gtid_32415, sizze_31493) && slt32(gtid_32416, sizze_31492)) &&
         (sle32(0, gtid_32415) && slt32(gtid_32415, sizze_31493))) && (sle32(0,
                                                                             res_31815) &&
                                                                       slt32(res_31815,
                                                                             sizze_31492))) {
        *(__global int32_t *) &mem_33592[(gtid_32415 * sizze_31492 +
                                          res_31815) * 4] = gtid_32416;
    }
}
__kernel void map_32518(int32_t sizze_31493, int32_t n_31497, int32_t res_31514,
                        __global unsigned char *mem_33517, __global
                        unsigned char *mem_33596, __global
                        unsigned char *mem_33599, __global
                        unsigned char *mem_33602)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_32518;
    int32_t local_tid_32519;
    int32_t group_sizze_33829;
    int32_t wave_sizze_33828;
    int32_t group_id_32520;
    
    global_tid_32518 = get_global_id(0);
    local_tid_32519 = get_local_id(0);
    group_sizze_33829 = get_local_size(0);
    wave_sizze_33828 = LOCKSTEP_WIDTH;
    group_id_32520 = get_group_id(0);
    
    int32_t gtid_32511;
    
    gtid_32511 = global_tid_32518;
    
    int32_t res_32523;
    float res_32540;
    int32_t arg_32558;
    float res_32559;
    float arg_32560;
    float res_32561;
    
    if (slt32(gtid_32511, sizze_31493)) {
        int32_t x_32526 = 0;
        
        for (int32_t chunk_offset_32525 = 0; chunk_offset_32525 < n_31497;
             chunk_offset_32525++) {
            float x_32533;
            bool res_32535;
            bool cond_32536;
            int32_t res_32537;
            int32_t res_32539;
            
            x_32533 = *(__global float *) &mem_33517[(chunk_offset_32525 *
                                                      sizze_31493 +
                                                      gtid_32511) * 4];
            res_32535 = futrts_isnan32(x_32533);
            cond_32536 = !res_32535;
            if (cond_32536) {
                res_32537 = 1;
            } else {
                res_32537 = 0;
            }
            res_32539 = x_32526 + res_32537;
            
            int32_t x_tmp_33830 = res_32539;
            
            x_32526 = x_tmp_33830;
        }
        res_32523 = x_32526;
        
        float x_32543 = 0.0F;
        
        for (int32_t chunk_offset_32542 = 0; chunk_offset_32542 < n_31497;
             chunk_offset_32542++) {
            bool cond_32552;
            float res_32553;
            float res_32555;
            float res_32557;
            
            cond_32552 = slt32(chunk_offset_32542, res_32523);
            if (cond_32552) {
                float res_32554 = *(__global
                                    float *) &mem_33596[(chunk_offset_32542 *
                                                         sizze_31493 +
                                                         gtid_32511) * 4];
                
                res_32553 = res_32554;
            } else {
                res_32553 = 0.0F;
            }
            res_32555 = res_32553 * res_32553;
            res_32557 = x_32543 + res_32555;
            
            float x_tmp_33831 = res_32557;
            
            x_32543 = x_tmp_33831;
        }
        res_32540 = x_32543;
        arg_32558 = res_32523 - res_31514;
        res_32559 = sitofp_i32_f32(arg_32558);
        arg_32560 = res_32540 / res_32559;
        res_32561 = futrts_sqrt32(arg_32560);
    }
    if (slt32(gtid_32511, sizze_31493)) {
        *(__global int32_t *) &mem_33599[gtid_32511 * 4] = res_32523;
    }
    if (slt32(gtid_32511, sizze_31493)) {
        *(__global float *) &mem_33602[gtid_32511 * 4] = res_32561;
    }
}
__kernel void map_32578(int32_t sizze_31492, int32_t sizze_31493,
                        float hfrac_31499, __global unsigned char *mem_33588,
                        __global unsigned char *mem_33599, __global
                        unsigned char *mem_33605, __global
                        unsigned char *mem_33608)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_32578;
    int32_t local_tid_32579;
    int32_t group_sizze_33833;
    int32_t wave_sizze_33832;
    int32_t group_id_32580;
    
    global_tid_32578 = get_global_id(0);
    local_tid_32579 = get_local_id(0);
    group_sizze_33833 = get_local_size(0);
    wave_sizze_33832 = LOCKSTEP_WIDTH;
    group_id_32580 = get_group_id(0);
    
    int32_t gtid_32571;
    
    gtid_32571 = global_tid_32578;
    
    int32_t x_32581;
    float res_32583;
    float arg_32584;
    int32_t res_32585;
    float res_32587;
    
    if (slt32(gtid_32571, sizze_31493)) {
        x_32581 = *(__global int32_t *) &mem_33599[gtid_32571 * 4];
        res_32583 = sitofp_i32_f32(x_32581);
        arg_32584 = hfrac_31499 * res_32583;
        res_32585 = fptosi_f32_i32(arg_32584);
        
        float x_32590 = 0.0F;
        
        for (int32_t chunk_offset_32589 = 0; chunk_offset_32589 < res_32585;
             chunk_offset_32589++) {
            int32_t x_32599;
            int32_t x_32600;
            int32_t i_32601;
            float res_32602;
            float res_32604;
            
            x_32599 = x_32581 + chunk_offset_32589;
            x_32600 = x_32599 - res_32585;
            i_32601 = 1 + x_32600;
            res_32602 = *(__global float *) &mem_33588[(gtid_32571 *
                                                        sizze_31492 + i_32601) *
                                                       4];
            res_32604 = x_32590 + res_32602;
            
            float x_tmp_33834 = res_32604;
            
            x_32590 = x_tmp_33834;
        }
        res_32587 = x_32590;
    }
    if (slt32(gtid_32571, sizze_31493)) {
        *(__global float *) &mem_33605[gtid_32571 * 4] = res_32587;
    }
    if (slt32(gtid_32571, sizze_31493)) {
        *(__global int32_t *) &mem_33608[gtid_32571 * 4] = res_32585;
    }
}
__kernel void map_32612(float lam_31500, int32_t num_elems_31882,
                        int32_t x_31884, float res_31890, __global
                        unsigned char *mappingindices_mem_33490, __global
                        unsigned char *mem_33611)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_32612;
    int32_t local_tid_32613;
    int32_t group_sizze_33836;
    int32_t wave_sizze_33835;
    int32_t group_id_32614;
    
    global_tid_32612 = get_global_id(0);
    local_tid_32613 = get_local_id(0);
    group_sizze_33836 = get_local_size(0);
    wave_sizze_33835 = LOCKSTEP_WIDTH;
    group_id_32614 = get_group_id(0);
    
    int32_t gtid_32605;
    
    gtid_32605 = global_tid_32612;
    
    int32_t res_32616;
    int32_t i_32617;
    int32_t res_32618;
    float res_32619;
    float arg_32620;
    bool cond_32621;
    float res_32622;
    float res_32624;
    float res_32625;
    
    if (slt32(gtid_32605, num_elems_31882)) {
        res_32616 = x_31884 + gtid_32605;
        i_32617 = res_32616 - 1;
        res_32618 = *(__global int32_t *) &mappingindices_mem_33490[i_32617 *
                                                                    4];
        res_32619 = sitofp_i32_f32(res_32618);
        arg_32620 = res_32619 / res_31890;
        cond_32621 = 2.7182817F < arg_32620;
        if (cond_32621) {
            float res_32623;
            
            res_32623 = futrts_log32(arg_32620);
            res_32622 = res_32623;
        } else {
            res_32622 = 1.0F;
        }
        res_32624 = futrts_sqrt32(res_32622);
        res_32625 = lam_31500 * res_32624;
    }
    if (slt32(gtid_32605, num_elems_31882)) {
        *(__global float *) &mem_33611[gtid_32605 * 4] = res_32625;
    }
}
__kernel void map_32637(int32_t sizze_31493, int32_t num_elems_31882, __global
                        unsigned char *mem_33599, __global
                        unsigned char *mem_33617, __global
                        unsigned char *mem_33634, __global
                        unsigned char *mem_33637, __global
                        unsigned char *mem_33644, __global
                        unsigned char *mem_33647)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_32637;
    int32_t local_tid_32638;
    int32_t group_sizze_33888;
    int32_t wave_sizze_33887;
    int32_t group_id_32639;
    
    global_tid_32637 = get_global_id(0);
    local_tid_32638 = get_local_id(0);
    group_sizze_33888 = get_local_size(0);
    wave_sizze_33887 = LOCKSTEP_WIDTH;
    group_id_32639 = get_group_id(0);
    
    int32_t gtid_32630;
    
    gtid_32630 = global_tid_32637;
    
    int32_t x_32640;
    int32_t y_32641;
    bool res_32642;
    int32_t res_32643;
    int32_t res_32645;
    bool cond_32647;
    bool res_32648;
    bool x_32649;
    bool y_32650;
    bool cond_32651;
    int32_t res_32652;
    
    if (slt32(gtid_32630, sizze_31493)) {
        x_32640 = *(__global int32_t *) &mem_33599[gtid_32630 * 4];
        y_32641 = *(__global int32_t *) &mem_33617[gtid_32630 * 4];
        res_32642 = *(__global bool *) &mem_33634[gtid_32630];
        res_32643 = *(__global int32_t *) &mem_33637[gtid_32630 * 4];
        if (res_32642) {
            int32_t res_32646 = *(__global int32_t *) &mem_33644[(gtid_32630 *
                                                                  num_elems_31882 +
                                                                  res_32643) *
                                                                 4];
            
            res_32645 = res_32646;
        } else {
            res_32645 = -1;
        }
        cond_32647 = sle32(x_32640, 5);
        res_32648 = sle32(y_32641, 5);
        x_32649 = !cond_32647;
        y_32650 = res_32648 && x_32649;
        cond_32651 = cond_32647 || y_32650;
        if (cond_32651) {
            res_32652 = -2;
        } else {
            res_32652 = res_32645;
        }
    }
    if (slt32(gtid_32630, sizze_31493)) {
        *(__global int32_t *) &mem_33647[gtid_32630 * 4] = res_32652;
    }
}
__kernel void map_32672(int32_t sizze_31492, int32_t sizze_31493,
                        int32_t n_31497, int32_t num_elems_31882, __global
                        unsigned char *mem_33592, __global
                        unsigned char *mem_33599, __global
                        unsigned char *mem_33611, __global
                        unsigned char *mem_33614, __global
                        unsigned char *mem_33617, __global
                        unsigned char *mem_33625, __global
                        unsigned char *mem_33628, __global
                        unsigned char *mem_33632, __global
                        unsigned char *mem_33634, __global
                        unsigned char *mem_33637, __global
                        unsigned char *mem_33640)
{
    const int32_t group_sizze_32667 = mainzigroup_sizze_32666;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_32672;
    int32_t local_tid_32673;
    int32_t group_sizze_33881;
    int32_t wave_sizze_33880;
    int32_t group_id_32674;
    
    global_tid_32672 = get_global_id(0);
    local_tid_32673 = get_local_id(0);
    group_sizze_33881 = get_local_size(0);
    wave_sizze_33880 = LOCKSTEP_WIDTH;
    group_id_32674 = get_group_id(0);
    
    int32_t gtid_32665;
    
    gtid_32665 = global_tid_32672;
    
    int32_t x_32675;
    int32_t y_32677;
    float y_32678;
    bool acc0_32688;
    int32_t acc0_32689;
    float acc0_32690;
    int32_t res_32733;
    
    if (slt32(gtid_32665, sizze_31493)) {
        x_32675 = *(__global int32_t *) &mem_33599[gtid_32665 * 4];
        y_32677 = *(__global int32_t *) &mem_33617[gtid_32665 * 4];
        y_32678 = *(__global float *) &mem_33614[gtid_32665 * 4];
        
        bool redout_32692;
        int32_t redout_32693;
        float redout_32694;
        
        redout_32692 = 0;
        redout_32693 = -1;
        redout_32694 = 0.0F;
        for (int32_t i_32696 = 0; i_32696 < num_elems_31882; i_32696++) {
            float x_32697;
            float x_32699;
            float res_32701;
            bool cond_32702;
            int32_t res_32703;
            bool res_32707;
            bool res_32708;
            bool x_32709;
            float res_32710;
            bool res_32711;
            bool x_32712;
            float res_32713;
            bool res_32720;
            int32_t res_32721;
            float res_32726;
            
            x_32697 = *(__global float *) &mem_33625[(i_32696 * sizze_31493 +
                                                      gtid_32665) * 4];
            x_32699 = *(__global float *) &mem_33611[i_32696 * 4];
            res_32701 = x_32697 / y_32678;
            cond_32702 = slt32(i_32696, y_32677);
            if (cond_32702) {
                int32_t i_32704;
                int32_t x_32705;
                int32_t res_32706;
                
                i_32704 = x_32675 + i_32696;
                x_32705 = *(__global int32_t *) &mem_33592[(gtid_32665 *
                                                            sizze_31492 +
                                                            i_32704) * 4];
                res_32706 = x_32705 - n_31497;
                res_32703 = res_32706;
            } else {
                res_32703 = -1;
            }
            res_32707 = futrts_isnan32(res_32701);
            res_32708 = !res_32707;
            x_32709 = cond_32702 && res_32708;
            res_32710 = (float) fabs(res_32701);
            res_32711 = x_32699 < res_32710;
            x_32712 = x_32709 && res_32711;
            if (cond_32702) {
                res_32713 = res_32701;
            } else {
                res_32713 = 0.0F;
            }
            if (redout_32692) {
                res_32720 = redout_32692;
                res_32721 = redout_32693;
            } else {
                bool x_32722;
                bool y_32723;
                bool res_32724;
                int32_t res_32725;
                
                x_32722 = !x_32712;
                y_32723 = redout_32692 && x_32722;
                res_32724 = x_32712 || y_32723;
                if (x_32712) {
                    res_32725 = i_32696;
                } else {
                    res_32725 = redout_32693;
                }
                res_32720 = res_32724;
                res_32721 = res_32725;
            }
            res_32726 = redout_32694 + res_32713;
            *(__global int32_t *) &mem_33628[(group_id_32674 *
                                              (group_sizze_32667 *
                                               num_elems_31882) +
                                              local_tid_32673 + i_32696 *
                                              group_sizze_32667) * 4] =
                res_32703;
            
            bool redout_tmp_33882 = res_32720;
            int32_t redout_tmp_33883 = res_32721;
            float redout_tmp_33884;
            
            redout_tmp_33884 = res_32726;
            redout_32692 = redout_tmp_33882;
            redout_32693 = redout_tmp_33883;
            redout_32694 = redout_tmp_33884;
        }
        acc0_32688 = redout_32692;
        acc0_32689 = redout_32693;
        acc0_32690 = redout_32694;
        if (acc0_32688) {
            res_32733 = acc0_32689;
        } else {
            res_32733 = -1;
        }
    }
    if (slt32(gtid_32665, sizze_31493)) {
        for (int32_t i_33886 = 0; i_33886 < num_elems_31882; i_33886++) {
            *(__global int32_t *) &mem_33632[(gtid_32665 + i_33886 *
                                              sizze_31493) * 4] = *(__global
                                                                    int32_t *) &mem_33628[(group_id_32674 *
                                                                                           (group_sizze_32667 *
                                                                                            num_elems_31882) +
                                                                                           local_tid_32673 +
                                                                                           i_33886 *
                                                                                           group_sizze_32667) *
                                                                                          4];
        }
    }
    if (slt32(gtid_32665, sizze_31493)) {
        *(__global bool *) &mem_33634[gtid_32665] = acc0_32688;
    }
    if (slt32(gtid_32665, sizze_31493)) {
        *(__global int32_t *) &mem_33637[gtid_32665 * 4] = res_32733;
    }
    if (slt32(gtid_32665, sizze_31493)) {
        *(__global float *) &mem_33640[gtid_32665 * 4] = acc0_32690;
    }
}
__kernel void map_32782(int32_t sizze_31492, int32_t sizze_31493,
                        int32_t i_31764, __global unsigned char *mem_33573,
                        __global unsigned char *mem_33599, __global
                        unsigned char *mem_33602, __global
                        unsigned char *mem_33614, __global
                        unsigned char *mem_33617)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_32782;
    int32_t local_tid_32783;
    int32_t group_sizze_33838;
    int32_t wave_sizze_33837;
    int32_t group_id_32784;
    
    global_tid_32782 = get_global_id(0);
    local_tid_32783 = get_local_id(0);
    group_sizze_33838 = get_local_size(0);
    wave_sizze_33837 = LOCKSTEP_WIDTH;
    group_id_32784 = get_group_id(0);
    
    int32_t gtid_32775;
    
    gtid_32775 = global_tid_32782;
    
    int32_t x_32785;
    int32_t x_32786;
    float x_32787;
    int32_t y_32788;
    float res_32789;
    float res_32790;
    float y_32791;
    
    if (slt32(gtid_32775, sizze_31493)) {
        x_32785 = *(__global int32_t *) &mem_33573[(i_31764 + gtid_32775 *
                                                    sizze_31492) * 4];
        x_32786 = *(__global int32_t *) &mem_33599[gtid_32775 * 4];
        x_32787 = *(__global float *) &mem_33602[gtid_32775 * 4];
        y_32788 = x_32785 - x_32786;
        res_32789 = sitofp_i32_f32(x_32786);
        res_32790 = futrts_sqrt32(res_32789);
        y_32791 = x_32787 * res_32790;
    }
    if (slt32(gtid_32775, sizze_31493)) {
        *(__global float *) &mem_33614[gtid_32775 * 4] = y_32791;
    }
    if (slt32(gtid_32775, sizze_31493)) {
        *(__global int32_t *) &mem_33617[gtid_32775 * 4] = y_32788;
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
__kernel void replicate_33816(int32_t sizze_31492, int32_t sizze_31493, __global
                              unsigned char *mem_33588)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t replicate_gtid_33816;
    int32_t replicate_ltid_33817;
    int32_t replicate_gid_33818;
    
    replicate_gtid_33816 = get_global_id(0);
    replicate_ltid_33817 = get_local_id(0);
    replicate_gid_33818 = get_group_id(0);
    if (slt32(replicate_gtid_33816, sizze_31493 * sizze_31492)) {
        *(__global float *) &mem_33588[(squot32(replicate_gtid_33816,
                                                sizze_31492) * sizze_31492 +
                                        (replicate_gtid_33816 -
                                         squot32(replicate_gtid_33816,
                                                 sizze_31492) * sizze_31492)) *
                                       4] = 0.0F;
    }
}
__kernel void replicate_33821(int32_t sizze_31492, int32_t sizze_31493, __global
                              unsigned char *mem_33592)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t replicate_gtid_33821;
    int32_t replicate_ltid_33822;
    int32_t replicate_gid_33823;
    
    replicate_gtid_33821 = get_global_id(0);
    replicate_ltid_33822 = get_local_id(0);
    replicate_gid_33823 = get_group_id(0);
    if (slt32(replicate_gtid_33821, sizze_31493 * sizze_31492)) {
        *(__global int32_t *) &mem_33592[(squot32(replicate_gtid_33821,
                                                  sizze_31492) * sizze_31492 +
                                          (replicate_gtid_33821 -
                                           squot32(replicate_gtid_33821,
                                                   sizze_31492) *
                                           sizze_31492)) * 4] = 0;
    }
}
__kernel void scan_stage1_32490(int32_t sizze_31492, int32_t sizze_31493,
                                int32_t sizze_31494, int32_t num_groups_32484,
                                __global unsigned char *images_mem_33491,
                                __global unsigned char *mem_33569, __global
                                unsigned char *mem_33573, __global
                                unsigned char *mem_33577, __global
                                unsigned char *mem_33580, __global
                                unsigned char *mem_33584)
{
    const int32_t group_sizze_32473 = mainzigroup_sizze_32472;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    ALIGNED_LOCAL_MEMORY(scan_arr_mem_33764_backing_0, 4 *
                         mainzigroup_sizze_32472);
    ALIGNED_LOCAL_MEMORY(scan_arr_mem_33766_backing_1, 4 *
                         mainzigroup_sizze_32472);
    
    int32_t global_tid_32490;
    int32_t local_tid_32491;
    int32_t group_sizze_33757;
    int32_t wave_sizze_33756;
    int32_t group_id_32492;
    
    global_tid_32490 = get_global_id(0);
    local_tid_32491 = get_local_id(0);
    group_sizze_33757 = get_local_size(0);
    wave_sizze_33756 = LOCKSTEP_WIDTH;
    group_id_32492 = get_group_id(0);
    
    int32_t gtid_32467;
    int32_t gtid_32489;
    __local char *scan_arr_mem_33764;
    
    scan_arr_mem_33764 = (__local char *) scan_arr_mem_33764_backing_0;
    
    __local char *scan_arr_mem_33766;
    
    scan_arr_mem_33766 = (__local char *) scan_arr_mem_33766_backing_1;
    
    int32_t x_31784;
    int32_t x_31785;
    int32_t x_31786;
    int32_t x_31787;
    
    x_31784 = 0;
    x_31785 = 0;
    for (int32_t j_33768 = 0; j_33768 < squot32(sizze_31493 * sizze_31492 +
                                                group_sizze_32473 *
                                                num_groups_32484 - 1,
                                                group_sizze_32473 *
                                                num_groups_32484); j_33768++) {
        int32_t chunk_offset_33769 = group_sizze_32473 * j_33768 +
                group_id_32492 * (group_sizze_32473 * squot32(sizze_31493 *
                                                              sizze_31492 +
                                                              group_sizze_32473 *
                                                              num_groups_32484 -
                                                              1,
                                                              group_sizze_32473 *
                                                              num_groups_32484));
        int32_t flat_idx_33770 = chunk_offset_33769 + local_tid_32491;
        
        gtid_32467 = squot32(flat_idx_33770, sizze_31492);
        gtid_32489 = flat_idx_33770 - squot32(flat_idx_33770, sizze_31492) *
            sizze_31492;
        // threads in bounds read input; others get neutral element
        {
            if (slt32(gtid_32467, sizze_31493) && slt32(gtid_32489,
                                                        sizze_31492)) {
                float x_31790;
                float x_31791;
                bool res_31792;
                bool cond_31793;
                float res_31794;
                bool res_31796;
                bool res_31797;
                int32_t res_31798;
                int32_t res_31799;
                
                x_31790 = *(__global float *) &images_mem_33491[(gtid_32467 *
                                                                 sizze_31494 +
                                                                 gtid_32489) *
                                                                4];
                x_31791 = *(__global float *) &mem_33569[(gtid_32467 *
                                                          sizze_31492 +
                                                          gtid_32489) * 4];
                res_31792 = futrts_isnan32(x_31790);
                cond_31793 = !res_31792;
                if (cond_31793) {
                    float res_31795 = x_31790 - x_31791;
                    
                    res_31794 = res_31795;
                } else {
                    res_31794 = NAN;
                }
                res_31796 = futrts_isnan32(res_31794);
                res_31797 = !res_31796;
                if (res_31797) {
                    res_31798 = 1;
                } else {
                    res_31798 = 0;
                }
                if (res_31797) {
                    res_31799 = 0;
                } else {
                    res_31799 = 1;
                }
                // write to-scan values to parameters
                {
                    x_31786 = res_31798;
                    x_31787 = res_31799;
                }
                // write mapped values results to global memory
                {
                    *(__global bool *) &mem_33580[gtid_32467 * sizze_31492 +
                                                  gtid_32489] = res_31797;
                    *(__global float *) &mem_33584[(gtid_32467 * sizze_31492 +
                                                    gtid_32489) * 4] =
                        res_31794;
                }
            } else {
                x_31786 = 0;
                x_31787 = 0;
            }
        }
        // combine with carry and write to local memory
        {
            int32_t res_31788;
            int32_t res_31789;
            
            res_31788 = x_31784 + x_31786;
            res_31789 = x_31785 + x_31787;
            *(__local int32_t *) &scan_arr_mem_33764[local_tid_32491 * 4] =
                res_31788;
            *(__local int32_t *) &scan_arr_mem_33766[local_tid_32491 * 4] =
                res_31789;
        }
        
        int32_t x_33758;
        int32_t x_33759;
        int32_t x_33760;
        int32_t x_33761;
        int32_t x_33771;
        int32_t x_33772;
        int32_t x_33773;
        int32_t x_33774;
        int32_t skip_threads_33777;
        
        if (slt32(local_tid_32491, group_sizze_32473)) {
            x_33760 = *(volatile __local
                        int32_t *) &scan_arr_mem_33764[local_tid_32491 *
                                                       sizeof(int32_t)];
            x_33761 = *(volatile __local
                        int32_t *) &scan_arr_mem_33766[local_tid_32491 *
                                                       sizeof(int32_t)];
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_33777 = 1;
            while (slt32(skip_threads_33777, 32)) {
                if (sle32(skip_threads_33777, local_tid_32491 -
                          squot32(local_tid_32491, 32) * 32) &&
                    slt32(local_tid_32491, group_sizze_32473)) {
                    // read operands
                    {
                        x_33758 = *(volatile __local
                                    int32_t *) &scan_arr_mem_33764[(local_tid_32491 -
                                                                    skip_threads_33777) *
                                                                   sizeof(int32_t)];
                        x_33759 = *(volatile __local
                                    int32_t *) &scan_arr_mem_33766[(local_tid_32491 -
                                                                    skip_threads_33777) *
                                                                   sizeof(int32_t)];
                    }
                    // perform operation
                    {
                        if (!slt32(srem32(local_tid_32491 + chunk_offset_33769,
                                          sizze_31492), local_tid_32491 +
                                   chunk_offset_33769 - (local_tid_32491 -
                                                         skip_threads_33777 +
                                                         chunk_offset_33769))) {
                            int32_t res_33762;
                            int32_t res_33763;
                            
                            res_33762 = x_33758 + x_33760;
                            res_33763 = x_33759 + x_33761;
                            x_33760 = res_33762;
                            x_33761 = res_33763;
                        }
                    }
                }
                if (sle32(wave_sizze_33756, skip_threads_33777)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_33777, local_tid_32491 -
                          squot32(local_tid_32491, 32) * 32) &&
                    slt32(local_tid_32491, group_sizze_32473)) {
                    // write result
                    {
                        *(volatile __local
                          int32_t *) &scan_arr_mem_33764[local_tid_32491 *
                                                         sizeof(int32_t)] =
                            x_33760;
                        *(volatile __local
                          int32_t *) &scan_arr_mem_33766[local_tid_32491 *
                                                         sizeof(int32_t)] =
                            x_33761;
                    }
                }
                if (sle32(wave_sizze_33756, skip_threads_33777)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_33777 *= 2;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // last thread of block 'i' writes its result to offset 'i'
        {
            if ((local_tid_32491 - squot32(local_tid_32491, 32) * 32) == 31 &&
                slt32(local_tid_32491, group_sizze_32473)) {
                *(volatile __local
                  int32_t *) &scan_arr_mem_33764[squot32(local_tid_32491, 32) *
                                                 sizeof(int32_t)] = x_33760;
                *(volatile __local
                  int32_t *) &scan_arr_mem_33766[squot32(local_tid_32491, 32) *
                                                 sizeof(int32_t)] = x_33761;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'
        {
            int32_t skip_threads_33778;
            
            if (squot32(local_tid_32491, 32) == 0 && slt32(local_tid_32491,
                                                           group_sizze_32473)) {
                x_33773 = *(volatile __local
                            int32_t *) &scan_arr_mem_33764[local_tid_32491 *
                                                           sizeof(int32_t)];
                x_33774 = *(volatile __local
                            int32_t *) &scan_arr_mem_33766[local_tid_32491 *
                                                           sizeof(int32_t)];
            }
            // in-block scan (hopefully no barriers needed)
            {
                skip_threads_33778 = 1;
                while (slt32(skip_threads_33778, 32)) {
                    if (sle32(skip_threads_33778, local_tid_32491 -
                              squot32(local_tid_32491, 32) * 32) &&
                        (squot32(local_tid_32491, 32) == 0 &&
                         slt32(local_tid_32491, group_sizze_32473))) {
                        // read operands
                        {
                            x_33771 = *(volatile __local
                                        int32_t *) &scan_arr_mem_33764[(local_tid_32491 -
                                                                        skip_threads_33778) *
                                                                       sizeof(int32_t)];
                            x_33772 = *(volatile __local
                                        int32_t *) &scan_arr_mem_33766[(local_tid_32491 -
                                                                        skip_threads_33778) *
                                                                       sizeof(int32_t)];
                        }
                        // perform operation
                        {
                            if (!slt32(srem32(local_tid_32491 * 32 + 32 - 1 +
                                              chunk_offset_33769, sizze_31492),
                                       local_tid_32491 * 32 + 32 - 1 +
                                       chunk_offset_33769 - ((local_tid_32491 -
                                                              skip_threads_33778) *
                                                             32 + 32 - 1 +
                                                             chunk_offset_33769))) {
                                int32_t res_33775;
                                int32_t res_33776;
                                
                                res_33775 = x_33771 + x_33773;
                                res_33776 = x_33772 + x_33774;
                                x_33773 = res_33775;
                                x_33774 = res_33776;
                            }
                        }
                    }
                    if (sle32(wave_sizze_33756, skip_threads_33778)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    if (sle32(skip_threads_33778, local_tid_32491 -
                              squot32(local_tid_32491, 32) * 32) &&
                        (squot32(local_tid_32491, 32) == 0 &&
                         slt32(local_tid_32491, group_sizze_32473))) {
                        // write result
                        {
                            *(volatile __local
                              int32_t *) &scan_arr_mem_33764[local_tid_32491 *
                                                             sizeof(int32_t)] =
                                x_33773;
                            *(volatile __local
                              int32_t *) &scan_arr_mem_33766[local_tid_32491 *
                                                             sizeof(int32_t)] =
                                x_33774;
                        }
                    }
                    if (sle32(wave_sizze_33756, skip_threads_33778)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    skip_threads_33778 *= 2;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // carry-in for every block except the first
        {
            if (!(squot32(local_tid_32491, 32) == 0 || !slt32(local_tid_32491,
                                                              group_sizze_32473))) {
                // read operands
                {
                    x_33758 = *(volatile __local
                                int32_t *) &scan_arr_mem_33764[(squot32(local_tid_32491,
                                                                        32) -
                                                                1) *
                                                               sizeof(int32_t)];
                    x_33759 = *(volatile __local
                                int32_t *) &scan_arr_mem_33766[(squot32(local_tid_32491,
                                                                        32) -
                                                                1) *
                                                               sizeof(int32_t)];
                }
                // perform operation
                {
                    if (!slt32(srem32(local_tid_32491 + chunk_offset_33769,
                                      sizze_31492), local_tid_32491 +
                               chunk_offset_33769 - (squot32(local_tid_32491,
                                                             32) * 32 - 1 +
                                                     chunk_offset_33769))) {
                        int32_t res_33762;
                        int32_t res_33763;
                        
                        res_33762 = x_33758 + x_33760;
                        res_33763 = x_33759 + x_33761;
                        x_33760 = res_33762;
                        x_33761 = res_33763;
                    }
                }
                // write final result
                {
                    *(volatile __local
                      int32_t *) &scan_arr_mem_33764[local_tid_32491 *
                                                     sizeof(int32_t)] = x_33760;
                    *(volatile __local
                      int32_t *) &scan_arr_mem_33766[local_tid_32491 *
                                                     sizeof(int32_t)] = x_33761;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // restore correct values for first block
        {
            if (squot32(local_tid_32491, 32) == 0) {
                *(volatile __local
                  int32_t *) &scan_arr_mem_33764[local_tid_32491 *
                                                 sizeof(int32_t)] = x_33760;
                *(volatile __local
                  int32_t *) &scan_arr_mem_33766[local_tid_32491 *
                                                 sizeof(int32_t)] = x_33761;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // threads in bounds write partial scan result
        {
            if (slt32(gtid_32467, sizze_31493) && slt32(gtid_32489,
                                                        sizze_31492)) {
                *(__global int32_t *) &mem_33573[(gtid_32467 * sizze_31492 +
                                                  gtid_32489) * 4] = *(__local
                                                                       int32_t *) &scan_arr_mem_33764[local_tid_32491 *
                                                                                                      4];
                *(__global int32_t *) &mem_33577[(gtid_32467 * sizze_31492 +
                                                  gtid_32489) * 4] = *(__local
                                                                       int32_t *) &scan_arr_mem_33766[local_tid_32491 *
                                                                                                      4];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // first thread reads last element as carry-in for next iteration
        {
            if (local_tid_32491 == 0) {
                if (slt32(srem32(chunk_offset_33769 + group_sizze_32473,
                                 sizze_31492), chunk_offset_33769 +
                          group_sizze_32473 - (chunk_offset_33769 +
                                               group_sizze_32473 - 1))) {
                    x_31784 = 0;
                    x_31785 = 0;
                } else {
                    x_31784 = *(__local
                                int32_t *) &scan_arr_mem_33764[(group_sizze_32473 -
                                                                1) * 4];
                    x_31785 = *(__local
                                int32_t *) &scan_arr_mem_33766[(group_sizze_32473 -
                                                                1) * 4];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
__kernel void scan_stage1_32761(int32_t sizze_31492, int32_t sizze_31493,
                                int32_t num_elems_31882,
                                int32_t num_groups_32755, __global
                                unsigned char *mem_33588, __global
                                unsigned char *mem_33599, __global
                                unsigned char *mem_33605, __global
                                unsigned char *mem_33608, __global
                                unsigned char *mem_33617, __global
                                unsigned char *mem_33621)
{
    const int32_t group_sizze_32744 = mainzigroup_sizze_32743;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    ALIGNED_LOCAL_MEMORY(scan_arr_mem_33844_backing_0, 4 *
                         mainzigroup_sizze_32743);
    
    int32_t global_tid_32761;
    int32_t local_tid_32762;
    int32_t group_sizze_33840;
    int32_t wave_sizze_33839;
    int32_t group_id_32763;
    
    global_tid_32761 = get_global_id(0);
    local_tid_32762 = get_local_id(0);
    group_sizze_33840 = get_local_size(0);
    wave_sizze_33839 = LOCKSTEP_WIDTH;
    group_id_32763 = get_group_id(0);
    
    int32_t gtid_32739;
    int32_t gtid_32760;
    __local char *scan_arr_mem_33844;
    
    scan_arr_mem_33844 = (__local char *) scan_arr_mem_33844_backing_0;
    
    float x_31929;
    float x_31930;
    
    x_31929 = 0.0F;
    for (int32_t j_33846 = 0; j_33846 < squot32(sizze_31493 * num_elems_31882 +
                                                group_sizze_32744 *
                                                num_groups_32755 - 1,
                                                group_sizze_32744 *
                                                num_groups_32755); j_33846++) {
        int32_t chunk_offset_33847 = group_sizze_32744 * j_33846 +
                group_id_32763 * (group_sizze_32744 * squot32(sizze_31493 *
                                                              num_elems_31882 +
                                                              group_sizze_32744 *
                                                              num_groups_32755 -
                                                              1,
                                                              group_sizze_32744 *
                                                              num_groups_32755));
        int32_t flat_idx_33848 = chunk_offset_33847 + local_tid_32762;
        
        gtid_32739 = squot32(flat_idx_33848, num_elems_31882);
        gtid_32760 = flat_idx_33848 - squot32(flat_idx_33848, num_elems_31882) *
            num_elems_31882;
        // threads in bounds read input; others get neutral element
        {
            if (slt32(gtid_32739, sizze_31493) && slt32(gtid_32760,
                                                        num_elems_31882)) {
                int32_t x_31906;
                int32_t x_31908;
                float x_31909;
                int32_t y_31912;
                bool cond_31933;
                float res_31934;
                
                x_31906 = *(__global int32_t *) &mem_33599[gtid_32739 * 4];
                x_31908 = *(__global int32_t *) &mem_33608[gtid_32739 * 4];
                x_31909 = *(__global float *) &mem_33605[gtid_32739 * 4];
                y_31912 = *(__global int32_t *) &mem_33617[gtid_32739 * 4];
                cond_31933 = sle32(y_31912, gtid_32760);
                if (cond_31933) {
                    res_31934 = 0.0F;
                } else {
                    bool cond_31935;
                    float res_31936;
                    
                    cond_31935 = gtid_32760 == 0;
                    if (cond_31935) {
                        res_31936 = x_31909;
                    } else {
                        int32_t x_31937;
                        int32_t i_31938;
                        float negate_arg_31939;
                        float x_31940;
                        int32_t i_31941;
                        float y_31942;
                        float res_31943;
                        
                        x_31937 = x_31906 - x_31908;
                        i_31938 = x_31937 + gtid_32760;
                        negate_arg_31939 = *(__global
                                             float *) &mem_33588[(gtid_32739 *
                                                                  sizze_31492 +
                                                                  i_31938) * 4];
                        x_31940 = 0.0F - negate_arg_31939;
                        i_31941 = x_31906 + gtid_32760;
                        y_31942 = *(__global float *) &mem_33588[(gtid_32739 *
                                                                  sizze_31492 +
                                                                  i_31941) * 4];
                        res_31943 = x_31940 + y_31942;
                        res_31936 = res_31943;
                    }
                    res_31934 = res_31936;
                }
                // write to-scan values to parameters
                {
                    x_31930 = res_31934;
                }
                // write mapped values results to global memory
                { }
            } else {
                x_31930 = 0.0F;
            }
        }
        // combine with carry and write to local memory
        {
            float res_31931 = x_31929 + x_31930;
            
            *(__local float *) &scan_arr_mem_33844[local_tid_32762 * 4] =
                res_31931;
        }
        
        float x_33841;
        float x_33842;
        float x_33849;
        float x_33850;
        int32_t skip_threads_33852;
        
        if (slt32(local_tid_32762, group_sizze_32744)) {
            x_33842 = *(volatile __local
                        float *) &scan_arr_mem_33844[local_tid_32762 *
                                                     sizeof(float)];
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_33852 = 1;
            while (slt32(skip_threads_33852, 32)) {
                if (sle32(skip_threads_33852, local_tid_32762 -
                          squot32(local_tid_32762, 32) * 32) &&
                    slt32(local_tid_32762, group_sizze_32744)) {
                    // read operands
                    {
                        x_33841 = *(volatile __local
                                    float *) &scan_arr_mem_33844[(local_tid_32762 -
                                                                  skip_threads_33852) *
                                                                 sizeof(float)];
                    }
                    // perform operation
                    {
                        if (!slt32(srem32(local_tid_32762 + chunk_offset_33847,
                                          num_elems_31882), local_tid_32762 +
                                   chunk_offset_33847 - (local_tid_32762 -
                                                         skip_threads_33852 +
                                                         chunk_offset_33847))) {
                            float res_33843 = x_33841 + x_33842;
                            
                            x_33842 = res_33843;
                        }
                    }
                }
                if (sle32(wave_sizze_33839, skip_threads_33852)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_33852, local_tid_32762 -
                          squot32(local_tid_32762, 32) * 32) &&
                    slt32(local_tid_32762, group_sizze_32744)) {
                    // write result
                    {
                        *(volatile __local
                          float *) &scan_arr_mem_33844[local_tid_32762 *
                                                       sizeof(float)] = x_33842;
                    }
                }
                if (sle32(wave_sizze_33839, skip_threads_33852)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_33852 *= 2;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // last thread of block 'i' writes its result to offset 'i'
        {
            if ((local_tid_32762 - squot32(local_tid_32762, 32) * 32) == 31 &&
                slt32(local_tid_32762, group_sizze_32744)) {
                *(volatile __local
                  float *) &scan_arr_mem_33844[squot32(local_tid_32762, 32) *
                                               sizeof(float)] = x_33842;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'
        {
            int32_t skip_threads_33853;
            
            if (squot32(local_tid_32762, 32) == 0 && slt32(local_tid_32762,
                                                           group_sizze_32744)) {
                x_33850 = *(volatile __local
                            float *) &scan_arr_mem_33844[local_tid_32762 *
                                                         sizeof(float)];
            }
            // in-block scan (hopefully no barriers needed)
            {
                skip_threads_33853 = 1;
                while (slt32(skip_threads_33853, 32)) {
                    if (sle32(skip_threads_33853, local_tid_32762 -
                              squot32(local_tid_32762, 32) * 32) &&
                        (squot32(local_tid_32762, 32) == 0 &&
                         slt32(local_tid_32762, group_sizze_32744))) {
                        // read operands
                        {
                            x_33849 = *(volatile __local
                                        float *) &scan_arr_mem_33844[(local_tid_32762 -
                                                                      skip_threads_33853) *
                                                                     sizeof(float)];
                        }
                        // perform operation
                        {
                            if (!slt32(srem32(local_tid_32762 * 32 + 32 - 1 +
                                              chunk_offset_33847,
                                              num_elems_31882),
                                       local_tid_32762 * 32 + 32 - 1 +
                                       chunk_offset_33847 - ((local_tid_32762 -
                                                              skip_threads_33853) *
                                                             32 + 32 - 1 +
                                                             chunk_offset_33847))) {
                                float res_33851 = x_33849 + x_33850;
                                
                                x_33850 = res_33851;
                            }
                        }
                    }
                    if (sle32(wave_sizze_33839, skip_threads_33853)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    if (sle32(skip_threads_33853, local_tid_32762 -
                              squot32(local_tid_32762, 32) * 32) &&
                        (squot32(local_tid_32762, 32) == 0 &&
                         slt32(local_tid_32762, group_sizze_32744))) {
                        // write result
                        {
                            *(volatile __local
                              float *) &scan_arr_mem_33844[local_tid_32762 *
                                                           sizeof(float)] =
                                x_33850;
                        }
                    }
                    if (sle32(wave_sizze_33839, skip_threads_33853)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    skip_threads_33853 *= 2;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // carry-in for every block except the first
        {
            if (!(squot32(local_tid_32762, 32) == 0 || !slt32(local_tid_32762,
                                                              group_sizze_32744))) {
                // read operands
                {
                    x_33841 = *(volatile __local
                                float *) &scan_arr_mem_33844[(squot32(local_tid_32762,
                                                                      32) - 1) *
                                                             sizeof(float)];
                }
                // perform operation
                {
                    if (!slt32(srem32(local_tid_32762 + chunk_offset_33847,
                                      num_elems_31882), local_tid_32762 +
                               chunk_offset_33847 - (squot32(local_tid_32762,
                                                             32) * 32 - 1 +
                                                     chunk_offset_33847))) {
                        float res_33843 = x_33841 + x_33842;
                        
                        x_33842 = res_33843;
                    }
                }
                // write final result
                {
                    *(volatile __local
                      float *) &scan_arr_mem_33844[local_tid_32762 *
                                                   sizeof(float)] = x_33842;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // restore correct values for first block
        {
            if (squot32(local_tid_32762, 32) == 0) {
                *(volatile __local
                  float *) &scan_arr_mem_33844[local_tid_32762 *
                                               sizeof(float)] = x_33842;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // threads in bounds write partial scan result
        {
            if (slt32(gtid_32739, sizze_31493) && slt32(gtid_32760,
                                                        num_elems_31882)) {
                *(__global float *) &mem_33621[(gtid_32739 * num_elems_31882 +
                                                gtid_32760) * 4] = *(__local
                                                                     float *) &scan_arr_mem_33844[local_tid_32762 *
                                                                                                  4];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // first thread reads last element as carry-in for next iteration
        {
            if (local_tid_32762 == 0) {
                if (slt32(srem32(chunk_offset_33847 + group_sizze_32744,
                                 num_elems_31882), chunk_offset_33847 +
                          group_sizze_32744 - (chunk_offset_33847 +
                                               group_sizze_32744 - 1))) {
                    x_31929 = 0.0F;
                } else {
                    x_31929 = *(__local
                                float *) &scan_arr_mem_33844[(group_sizze_32744 -
                                                              1) * 4];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
__kernel void scan_stage2_33791(__local volatile
                                int64_t *scan_arr_mem_33796_backing_aligned_0,
                                __local volatile
                                int64_t *scan_arr_mem_33798_backing_aligned_1,
                                int32_t sizze_31492, int32_t sizze_31493,
                                int32_t num_groups_32484, __global
                                unsigned char *mem_33573, __global
                                unsigned char *mem_33577)
{
    const int32_t group_sizze_32473 = mainzigroup_sizze_32472;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_33796_backing_0 =
                          scan_arr_mem_33796_backing_aligned_0;
    __local volatile char *restrict scan_arr_mem_33798_backing_1 =
                          scan_arr_mem_33798_backing_aligned_1;
    int32_t global_tid_33791;
    int32_t local_tid_33792;
    int32_t group_sizze_33795;
    int32_t wave_sizze_33794;
    int32_t group_id_33793;
    
    global_tid_33791 = get_global_id(0);
    local_tid_33792 = get_local_id(0);
    group_sizze_33795 = get_local_size(0);
    wave_sizze_33794 = LOCKSTEP_WIDTH;
    group_id_33793 = get_group_id(0);
    
    __local char *scan_arr_mem_33796;
    
    scan_arr_mem_33796 = (__local char *) scan_arr_mem_33796_backing_0;
    
    __local char *scan_arr_mem_33798;
    
    scan_arr_mem_33798 = (__local char *) scan_arr_mem_33798_backing_1;
    
    int32_t flat_idx_33800 = (local_tid_33792 + 1) * (group_sizze_32473 *
                                                      squot32(sizze_31493 *
                                                              sizze_31492 +
                                                              group_sizze_32473 *
                                                              num_groups_32484 -
                                                              1,
                                                              group_sizze_32473 *
                                                              num_groups_32484)) -
            1;
    int32_t gtid_32467 = squot32(flat_idx_33800, sizze_31492);
    int32_t gtid_32489;
    
    gtid_32489 = flat_idx_33800 - squot32(flat_idx_33800, sizze_31492) *
        sizze_31492;
    // threads in bound read carries; others get neutral element
    {
        if (slt32(gtid_32467, sizze_31493) && slt32(gtid_32489, sizze_31492)) {
            *(__local int32_t *) &scan_arr_mem_33796[local_tid_33792 * 4] =
                *(__global int32_t *) &mem_33573[(gtid_32467 * sizze_31492 +
                                                  gtid_32489) * 4];
            *(__local int32_t *) &scan_arr_mem_33798[local_tid_33792 * 4] =
                *(__global int32_t *) &mem_33577[(gtid_32467 * sizze_31492 +
                                                  gtid_32489) * 4];
        } else {
            *(__local int32_t *) &scan_arr_mem_33796[local_tid_33792 * 4] = 0;
            *(__local int32_t *) &scan_arr_mem_33798[local_tid_33792 * 4] = 0;
        }
    }
    
    int32_t x_33779;
    int32_t x_33780;
    int32_t x_33781;
    int32_t x_33782;
    int32_t x_33801;
    int32_t x_33802;
    int32_t x_33803;
    int32_t x_33804;
    int32_t skip_threads_33807;
    
    if (slt32(local_tid_33792, num_groups_32484)) {
        x_33781 = *(volatile __local
                    int32_t *) &scan_arr_mem_33796[local_tid_33792 *
                                                   sizeof(int32_t)];
        x_33782 = *(volatile __local
                    int32_t *) &scan_arr_mem_33798[local_tid_33792 *
                                                   sizeof(int32_t)];
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_33807 = 1;
        while (slt32(skip_threads_33807, 32)) {
            if (sle32(skip_threads_33807, local_tid_33792 -
                      squot32(local_tid_33792, 32) * 32) &&
                slt32(local_tid_33792, num_groups_32484)) {
                // read operands
                {
                    x_33779 = *(volatile __local
                                int32_t *) &scan_arr_mem_33796[(local_tid_33792 -
                                                                skip_threads_33807) *
                                                               sizeof(int32_t)];
                    x_33780 = *(volatile __local
                                int32_t *) &scan_arr_mem_33798[(local_tid_33792 -
                                                                skip_threads_33807) *
                                                               sizeof(int32_t)];
                }
                // perform operation
                {
                    if (!slt32(srem32((local_tid_33792 + 1) *
                                      (group_sizze_32473 * squot32(sizze_31493 *
                                                                   sizze_31492 +
                                                                   group_sizze_32473 *
                                                                   num_groups_32484 -
                                                                   1,
                                                                   group_sizze_32473 *
                                                                   num_groups_32484)) -
                                      1, sizze_31492), (local_tid_33792 + 1) *
                               (group_sizze_32473 * squot32(sizze_31493 *
                                                            sizze_31492 +
                                                            group_sizze_32473 *
                                                            num_groups_32484 -
                                                            1,
                                                            group_sizze_32473 *
                                                            num_groups_32484)) -
                               1 - ((local_tid_33792 - skip_threads_33807 + 1) *
                                    (group_sizze_32473 * squot32(sizze_31493 *
                                                                 sizze_31492 +
                                                                 group_sizze_32473 *
                                                                 num_groups_32484 -
                                                                 1,
                                                                 group_sizze_32473 *
                                                                 num_groups_32484)) -
                                    1))) {
                        int32_t res_33783;
                        int32_t res_33784;
                        
                        res_33783 = x_33779 + x_33781;
                        res_33784 = x_33780 + x_33782;
                        x_33781 = res_33783;
                        x_33782 = res_33784;
                    }
                }
            }
            if (sle32(wave_sizze_33794, skip_threads_33807)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_33807, local_tid_33792 -
                      squot32(local_tid_33792, 32) * 32) &&
                slt32(local_tid_33792, num_groups_32484)) {
                // write result
                {
                    *(volatile __local
                      int32_t *) &scan_arr_mem_33796[local_tid_33792 *
                                                     sizeof(int32_t)] = x_33781;
                    *(volatile __local
                      int32_t *) &scan_arr_mem_33798[local_tid_33792 *
                                                     sizeof(int32_t)] = x_33782;
                }
            }
            if (sle32(wave_sizze_33794, skip_threads_33807)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_33807 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_33792 - squot32(local_tid_33792, 32) * 32) == 31 &&
            slt32(local_tid_33792, num_groups_32484)) {
            *(volatile __local
              int32_t *) &scan_arr_mem_33796[squot32(local_tid_33792, 32) *
                                             sizeof(int32_t)] = x_33781;
            *(volatile __local
              int32_t *) &scan_arr_mem_33798[squot32(local_tid_33792, 32) *
                                             sizeof(int32_t)] = x_33782;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'
    {
        int32_t skip_threads_33808;
        
        if (squot32(local_tid_33792, 32) == 0 && slt32(local_tid_33792,
                                                       num_groups_32484)) {
            x_33803 = *(volatile __local
                        int32_t *) &scan_arr_mem_33796[local_tid_33792 *
                                                       sizeof(int32_t)];
            x_33804 = *(volatile __local
                        int32_t *) &scan_arr_mem_33798[local_tid_33792 *
                                                       sizeof(int32_t)];
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_33808 = 1;
            while (slt32(skip_threads_33808, 32)) {
                if (sle32(skip_threads_33808, local_tid_33792 -
                          squot32(local_tid_33792, 32) * 32) &&
                    (squot32(local_tid_33792, 32) == 0 && slt32(local_tid_33792,
                                                                num_groups_32484))) {
                    // read operands
                    {
                        x_33801 = *(volatile __local
                                    int32_t *) &scan_arr_mem_33796[(local_tid_33792 -
                                                                    skip_threads_33808) *
                                                                   sizeof(int32_t)];
                        x_33802 = *(volatile __local
                                    int32_t *) &scan_arr_mem_33798[(local_tid_33792 -
                                                                    skip_threads_33808) *
                                                                   sizeof(int32_t)];
                    }
                    // perform operation
                    {
                        if (!slt32(srem32((local_tid_33792 * 32 + 32 - 1 + 1) *
                                          (group_sizze_32473 *
                                           squot32(sizze_31493 * sizze_31492 +
                                                   group_sizze_32473 *
                                                   num_groups_32484 - 1,
                                                   group_sizze_32473 *
                                                   num_groups_32484)) - 1,
                                          sizze_31492), (local_tid_33792 * 32 +
                                                         32 - 1 + 1) *
                                   (group_sizze_32473 * squot32(sizze_31493 *
                                                                sizze_31492 +
                                                                group_sizze_32473 *
                                                                num_groups_32484 -
                                                                1,
                                                                group_sizze_32473 *
                                                                num_groups_32484)) -
                                   1 - (((local_tid_33792 -
                                          skip_threads_33808) * 32 + 32 - 1 +
                                         1) * (group_sizze_32473 *
                                               squot32(sizze_31493 *
                                                       sizze_31492 +
                                                       group_sizze_32473 *
                                                       num_groups_32484 - 1,
                                                       group_sizze_32473 *
                                                       num_groups_32484)) -
                                        1))) {
                            int32_t res_33805;
                            int32_t res_33806;
                            
                            res_33805 = x_33801 + x_33803;
                            res_33806 = x_33802 + x_33804;
                            x_33803 = res_33805;
                            x_33804 = res_33806;
                        }
                    }
                }
                if (sle32(wave_sizze_33794, skip_threads_33808)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_33808, local_tid_33792 -
                          squot32(local_tid_33792, 32) * 32) &&
                    (squot32(local_tid_33792, 32) == 0 && slt32(local_tid_33792,
                                                                num_groups_32484))) {
                    // write result
                    {
                        *(volatile __local
                          int32_t *) &scan_arr_mem_33796[local_tid_33792 *
                                                         sizeof(int32_t)] =
                            x_33803;
                        *(volatile __local
                          int32_t *) &scan_arr_mem_33798[local_tid_33792 *
                                                         sizeof(int32_t)] =
                            x_33804;
                    }
                }
                if (sle32(wave_sizze_33794, skip_threads_33808)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_33808 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_33792, 32) == 0 || !slt32(local_tid_33792,
                                                          num_groups_32484))) {
            // read operands
            {
                x_33779 = *(volatile __local
                            int32_t *) &scan_arr_mem_33796[(squot32(local_tid_33792,
                                                                    32) - 1) *
                                                           sizeof(int32_t)];
                x_33780 = *(volatile __local
                            int32_t *) &scan_arr_mem_33798[(squot32(local_tid_33792,
                                                                    32) - 1) *
                                                           sizeof(int32_t)];
            }
            // perform operation
            {
                if (!slt32(srem32((local_tid_33792 + 1) * (group_sizze_32473 *
                                                           squot32(sizze_31493 *
                                                                   sizze_31492 +
                                                                   group_sizze_32473 *
                                                                   num_groups_32484 -
                                                                   1,
                                                                   group_sizze_32473 *
                                                                   num_groups_32484)) -
                                  1, sizze_31492), (local_tid_33792 + 1) *
                           (group_sizze_32473 * squot32(sizze_31493 *
                                                        sizze_31492 +
                                                        group_sizze_32473 *
                                                        num_groups_32484 - 1,
                                                        group_sizze_32473 *
                                                        num_groups_32484)) - 1 -
                           ((squot32(local_tid_33792, 32) * 32 - 1 + 1) *
                            (group_sizze_32473 * squot32(sizze_31493 *
                                                         sizze_31492 +
                                                         group_sizze_32473 *
                                                         num_groups_32484 - 1,
                                                         group_sizze_32473 *
                                                         num_groups_32484)) -
                            1))) {
                    int32_t res_33783;
                    int32_t res_33784;
                    
                    res_33783 = x_33779 + x_33781;
                    res_33784 = x_33780 + x_33782;
                    x_33781 = res_33783;
                    x_33782 = res_33784;
                }
            }
            // write final result
            {
                *(volatile __local
                  int32_t *) &scan_arr_mem_33796[local_tid_33792 *
                                                 sizeof(int32_t)] = x_33781;
                *(volatile __local
                  int32_t *) &scan_arr_mem_33798[local_tid_33792 *
                                                 sizeof(int32_t)] = x_33782;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_33792, 32) == 0) {
            *(volatile __local int32_t *) &scan_arr_mem_33796[local_tid_33792 *
                                                              sizeof(int32_t)] =
                x_33781;
            *(volatile __local int32_t *) &scan_arr_mem_33798[local_tid_33792 *
                                                              sizeof(int32_t)] =
                x_33782;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // threads in bounds write scanned carries
    {
        if (slt32(gtid_32467, sizze_31493) && slt32(gtid_32489, sizze_31492)) {
            *(__global int32_t *) &mem_33573[(gtid_32467 * sizze_31492 +
                                              gtid_32489) * 4] = *(__local
                                                                   int32_t *) &scan_arr_mem_33796[local_tid_33792 *
                                                                                                  4];
            *(__global int32_t *) &mem_33577[(gtid_32467 * sizze_31492 +
                                              gtid_32489) * 4] = *(__local
                                                                   int32_t *) &scan_arr_mem_33798[local_tid_33792 *
                                                                                                  4];
        }
    }
}
__kernel void scan_stage2_33860(__local volatile
                                int64_t *scan_arr_mem_33865_backing_aligned_0,
                                int32_t sizze_31493, int32_t num_elems_31882,
                                int32_t num_groups_32755, __global
                                unsigned char *mem_33621)
{
    const int32_t group_sizze_32744 = mainzigroup_sizze_32743;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_33865_backing_0 =
                          scan_arr_mem_33865_backing_aligned_0;
    int32_t global_tid_33860;
    int32_t local_tid_33861;
    int32_t group_sizze_33864;
    int32_t wave_sizze_33863;
    int32_t group_id_33862;
    
    global_tid_33860 = get_global_id(0);
    local_tid_33861 = get_local_id(0);
    group_sizze_33864 = get_local_size(0);
    wave_sizze_33863 = LOCKSTEP_WIDTH;
    group_id_33862 = get_group_id(0);
    
    __local char *scan_arr_mem_33865;
    
    scan_arr_mem_33865 = (__local char *) scan_arr_mem_33865_backing_0;
    
    int32_t flat_idx_33867 = (local_tid_33861 + 1) * (group_sizze_32744 *
                                                      squot32(sizze_31493 *
                                                              num_elems_31882 +
                                                              group_sizze_32744 *
                                                              num_groups_32755 -
                                                              1,
                                                              group_sizze_32744 *
                                                              num_groups_32755)) -
            1;
    int32_t gtid_32739 = squot32(flat_idx_33867, num_elems_31882);
    int32_t gtid_32760;
    
    gtid_32760 = flat_idx_33867 - squot32(flat_idx_33867, num_elems_31882) *
        num_elems_31882;
    // threads in bound read carries; others get neutral element
    {
        if (slt32(gtid_32739, sizze_31493) && slt32(gtid_32760,
                                                    num_elems_31882)) {
            *(__local float *) &scan_arr_mem_33865[local_tid_33861 * 4] =
                *(__global float *) &mem_33621[(gtid_32739 * num_elems_31882 +
                                                gtid_32760) * 4];
        } else {
            *(__local float *) &scan_arr_mem_33865[local_tid_33861 * 4] = 0.0F;
        }
    }
    
    float x_33854;
    float x_33855;
    float x_33868;
    float x_33869;
    int32_t skip_threads_33871;
    
    if (slt32(local_tid_33861, num_groups_32755)) {
        x_33855 = *(volatile __local
                    float *) &scan_arr_mem_33865[local_tid_33861 *
                                                 sizeof(float)];
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_33871 = 1;
        while (slt32(skip_threads_33871, 32)) {
            if (sle32(skip_threads_33871, local_tid_33861 -
                      squot32(local_tid_33861, 32) * 32) &&
                slt32(local_tid_33861, num_groups_32755)) {
                // read operands
                {
                    x_33854 = *(volatile __local
                                float *) &scan_arr_mem_33865[(local_tid_33861 -
                                                              skip_threads_33871) *
                                                             sizeof(float)];
                }
                // perform operation
                {
                    if (!slt32(srem32((local_tid_33861 + 1) *
                                      (group_sizze_32744 * squot32(sizze_31493 *
                                                                   num_elems_31882 +
                                                                   group_sizze_32744 *
                                                                   num_groups_32755 -
                                                                   1,
                                                                   group_sizze_32744 *
                                                                   num_groups_32755)) -
                                      1, num_elems_31882), (local_tid_33861 +
                                                            1) *
                               (group_sizze_32744 * squot32(sizze_31493 *
                                                            num_elems_31882 +
                                                            group_sizze_32744 *
                                                            num_groups_32755 -
                                                            1,
                                                            group_sizze_32744 *
                                                            num_groups_32755)) -
                               1 - ((local_tid_33861 - skip_threads_33871 + 1) *
                                    (group_sizze_32744 * squot32(sizze_31493 *
                                                                 num_elems_31882 +
                                                                 group_sizze_32744 *
                                                                 num_groups_32755 -
                                                                 1,
                                                                 group_sizze_32744 *
                                                                 num_groups_32755)) -
                                    1))) {
                        float res_33856 = x_33854 + x_33855;
                        
                        x_33855 = res_33856;
                    }
                }
            }
            if (sle32(wave_sizze_33863, skip_threads_33871)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_33871, local_tid_33861 -
                      squot32(local_tid_33861, 32) * 32) &&
                slt32(local_tid_33861, num_groups_32755)) {
                // write result
                {
                    *(volatile __local
                      float *) &scan_arr_mem_33865[local_tid_33861 *
                                                   sizeof(float)] = x_33855;
                }
            }
            if (sle32(wave_sizze_33863, skip_threads_33871)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_33871 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_33861 - squot32(local_tid_33861, 32) * 32) == 31 &&
            slt32(local_tid_33861, num_groups_32755)) {
            *(volatile __local
              float *) &scan_arr_mem_33865[squot32(local_tid_33861, 32) *
                                           sizeof(float)] = x_33855;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'
    {
        int32_t skip_threads_33872;
        
        if (squot32(local_tid_33861, 32) == 0 && slt32(local_tid_33861,
                                                       num_groups_32755)) {
            x_33869 = *(volatile __local
                        float *) &scan_arr_mem_33865[local_tid_33861 *
                                                     sizeof(float)];
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_33872 = 1;
            while (slt32(skip_threads_33872, 32)) {
                if (sle32(skip_threads_33872, local_tid_33861 -
                          squot32(local_tid_33861, 32) * 32) &&
                    (squot32(local_tid_33861, 32) == 0 && slt32(local_tid_33861,
                                                                num_groups_32755))) {
                    // read operands
                    {
                        x_33868 = *(volatile __local
                                    float *) &scan_arr_mem_33865[(local_tid_33861 -
                                                                  skip_threads_33872) *
                                                                 sizeof(float)];
                    }
                    // perform operation
                    {
                        if (!slt32(srem32((local_tid_33861 * 32 + 32 - 1 + 1) *
                                          (group_sizze_32744 *
                                           squot32(sizze_31493 *
                                                   num_elems_31882 +
                                                   group_sizze_32744 *
                                                   num_groups_32755 - 1,
                                                   group_sizze_32744 *
                                                   num_groups_32755)) - 1,
                                          num_elems_31882), (local_tid_33861 *
                                                             32 + 32 - 1 + 1) *
                                   (group_sizze_32744 * squot32(sizze_31493 *
                                                                num_elems_31882 +
                                                                group_sizze_32744 *
                                                                num_groups_32755 -
                                                                1,
                                                                group_sizze_32744 *
                                                                num_groups_32755)) -
                                   1 - (((local_tid_33861 -
                                          skip_threads_33872) * 32 + 32 - 1 +
                                         1) * (group_sizze_32744 *
                                               squot32(sizze_31493 *
                                                       num_elems_31882 +
                                                       group_sizze_32744 *
                                                       num_groups_32755 - 1,
                                                       group_sizze_32744 *
                                                       num_groups_32755)) -
                                        1))) {
                            float res_33870 = x_33868 + x_33869;
                            
                            x_33869 = res_33870;
                        }
                    }
                }
                if (sle32(wave_sizze_33863, skip_threads_33872)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_33872, local_tid_33861 -
                          squot32(local_tid_33861, 32) * 32) &&
                    (squot32(local_tid_33861, 32) == 0 && slt32(local_tid_33861,
                                                                num_groups_32755))) {
                    // write result
                    {
                        *(volatile __local
                          float *) &scan_arr_mem_33865[local_tid_33861 *
                                                       sizeof(float)] = x_33869;
                    }
                }
                if (sle32(wave_sizze_33863, skip_threads_33872)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_33872 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_33861, 32) == 0 || !slt32(local_tid_33861,
                                                          num_groups_32755))) {
            // read operands
            {
                x_33854 = *(volatile __local
                            float *) &scan_arr_mem_33865[(squot32(local_tid_33861,
                                                                  32) - 1) *
                                                         sizeof(float)];
            }
            // perform operation
            {
                if (!slt32(srem32((local_tid_33861 + 1) * (group_sizze_32744 *
                                                           squot32(sizze_31493 *
                                                                   num_elems_31882 +
                                                                   group_sizze_32744 *
                                                                   num_groups_32755 -
                                                                   1,
                                                                   group_sizze_32744 *
                                                                   num_groups_32755)) -
                                  1, num_elems_31882), (local_tid_33861 + 1) *
                           (group_sizze_32744 * squot32(sizze_31493 *
                                                        num_elems_31882 +
                                                        group_sizze_32744 *
                                                        num_groups_32755 - 1,
                                                        group_sizze_32744 *
                                                        num_groups_32755)) - 1 -
                           ((squot32(local_tid_33861, 32) * 32 - 1 + 1) *
                            (group_sizze_32744 * squot32(sizze_31493 *
                                                         num_elems_31882 +
                                                         group_sizze_32744 *
                                                         num_groups_32755 - 1,
                                                         group_sizze_32744 *
                                                         num_groups_32755)) -
                            1))) {
                    float res_33856 = x_33854 + x_33855;
                    
                    x_33855 = res_33856;
                }
            }
            // write final result
            {
                *(volatile __local
                  float *) &scan_arr_mem_33865[local_tid_33861 *
                                               sizeof(float)] = x_33855;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_33861, 32) == 0) {
            *(volatile __local float *) &scan_arr_mem_33865[local_tid_33861 *
                                                            sizeof(float)] =
                x_33855;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // threads in bounds write scanned carries
    {
        if (slt32(gtid_32739, sizze_31493) && slt32(gtid_32760,
                                                    num_elems_31882)) {
            *(__global float *) &mem_33621[(gtid_32739 * num_elems_31882 +
                                            gtid_32760) * 4] = *(__local
                                                                 float *) &scan_arr_mem_33865[local_tid_33861 *
                                                                                              4];
        }
    }
}
__kernel void scan_stage3_33809(int32_t sizze_31492, int32_t sizze_31493,
                                int32_t num_groups_32484, __global
                                unsigned char *mem_33573, __global
                                unsigned char *mem_33577)
{
    const int32_t group_sizze_32473 = mainzigroup_sizze_32472;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t scan_gtid_33809;
    int32_t scan_ltid_33810;
    int32_t scan_gid_33811;
    
    scan_gtid_33809 = get_global_id(0);
    scan_ltid_33810 = get_local_id(0);
    scan_gid_33811 = get_group_id(0);
    
    int32_t gtid_32467 = squot32(scan_gtid_33809, sizze_31492);
    int32_t gtid_32489;
    
    gtid_32489 = scan_gtid_33809 - squot32(scan_gtid_33809, sizze_31492) *
        sizze_31492;
    
    int32_t orig_group_33814 = squot32(scan_gtid_33809, group_sizze_32473 *
                                       squot32(sizze_31493 * sizze_31492 +
                                               group_sizze_32473 *
                                               num_groups_32484 - 1,
                                               group_sizze_32473 *
                                               num_groups_32484));
    int32_t carry_in_flat_idx_33815 = orig_group_33814 * (group_sizze_32473 *
                                                          squot32(sizze_31493 *
                                                                  sizze_31492 +
                                                                  group_sizze_32473 *
                                                                  num_groups_32484 -
                                                                  1,
                                                                  group_sizze_32473 *
                                                                  num_groups_32484)) -
            1;
    
    if (slt32(scan_gtid_33809, sizze_31493 * sizze_31492)) {
        if (!(orig_group_33814 == 0 || (scan_gtid_33809 == (orig_group_33814 +
                                                            1) *
                                        (group_sizze_32473 *
                                         squot32(sizze_31493 * sizze_31492 +
                                                 group_sizze_32473 *
                                                 num_groups_32484 - 1,
                                                 group_sizze_32473 *
                                                 num_groups_32484)) - 1 ||
                                        slt32(srem32(scan_gtid_33809,
                                                     sizze_31492),
                                              scan_gtid_33809 -
                                              carry_in_flat_idx_33815)))) {
            int32_t x_33785;
            int32_t x_33786;
            int32_t x_33787;
            int32_t x_33788;
            
            x_33785 = *(__global
                        int32_t *) &mem_33573[(squot32(carry_in_flat_idx_33815,
                                                       sizze_31492) *
                                               sizze_31492 +
                                               (carry_in_flat_idx_33815 -
                                                squot32(carry_in_flat_idx_33815,
                                                        sizze_31492) *
                                                sizze_31492)) * 4];
            x_33786 = *(__global
                        int32_t *) &mem_33577[(squot32(carry_in_flat_idx_33815,
                                                       sizze_31492) *
                                               sizze_31492 +
                                               (carry_in_flat_idx_33815 -
                                                squot32(carry_in_flat_idx_33815,
                                                        sizze_31492) *
                                                sizze_31492)) * 4];
            x_33787 = *(__global int32_t *) &mem_33573[(gtid_32467 *
                                                        sizze_31492 +
                                                        gtid_32489) * 4];
            x_33788 = *(__global int32_t *) &mem_33577[(gtid_32467 *
                                                        sizze_31492 +
                                                        gtid_32489) * 4];
            
            int32_t res_33789;
            int32_t res_33790;
            
            if (slt32(scan_gtid_33809, sizze_31493 * sizze_31492)) {
                res_33789 = x_33785 + x_33787;
                res_33790 = x_33786 + x_33788;
            }
            x_33785 = res_33789;
            x_33786 = res_33790;
            *(__global int32_t *) &mem_33573[(gtid_32467 * sizze_31492 +
                                              gtid_32489) * 4] = x_33785;
            *(__global int32_t *) &mem_33577[(gtid_32467 * sizze_31492 +
                                              gtid_32489) * 4] = x_33786;
        }
    }
}
__kernel void scan_stage3_33873(int32_t sizze_31493, int32_t num_elems_31882,
                                int32_t num_groups_32755, __global
                                unsigned char *mem_33621)
{
    const int32_t group_sizze_32744 = mainzigroup_sizze_32743;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t scan_gtid_33873;
    int32_t scan_ltid_33874;
    int32_t scan_gid_33875;
    
    scan_gtid_33873 = get_global_id(0);
    scan_ltid_33874 = get_local_id(0);
    scan_gid_33875 = get_group_id(0);
    
    int32_t gtid_32739 = squot32(scan_gtid_33873, num_elems_31882);
    int32_t gtid_32760;
    
    gtid_32760 = scan_gtid_33873 - squot32(scan_gtid_33873, num_elems_31882) *
        num_elems_31882;
    
    int32_t orig_group_33878 = squot32(scan_gtid_33873, group_sizze_32744 *
                                       squot32(sizze_31493 * num_elems_31882 +
                                               group_sizze_32744 *
                                               num_groups_32755 - 1,
                                               group_sizze_32744 *
                                               num_groups_32755));
    int32_t carry_in_flat_idx_33879 = orig_group_33878 * (group_sizze_32744 *
                                                          squot32(sizze_31493 *
                                                                  num_elems_31882 +
                                                                  group_sizze_32744 *
                                                                  num_groups_32755 -
                                                                  1,
                                                                  group_sizze_32744 *
                                                                  num_groups_32755)) -
            1;
    
    if (slt32(scan_gtid_33873, sizze_31493 * num_elems_31882)) {
        if (!(orig_group_33878 == 0 || (scan_gtid_33873 == (orig_group_33878 +
                                                            1) *
                                        (group_sizze_32744 *
                                         squot32(sizze_31493 * num_elems_31882 +
                                                 group_sizze_32744 *
                                                 num_groups_32755 - 1,
                                                 group_sizze_32744 *
                                                 num_groups_32755)) - 1 ||
                                        slt32(srem32(scan_gtid_33873,
                                                     num_elems_31882),
                                              scan_gtid_33873 -
                                              carry_in_flat_idx_33879)))) {
            float x_33857;
            float x_33858;
            
            x_33857 = *(__global
                        float *) &mem_33621[(squot32(carry_in_flat_idx_33879,
                                                     num_elems_31882) *
                                             num_elems_31882 +
                                             (carry_in_flat_idx_33879 -
                                              squot32(carry_in_flat_idx_33879,
                                                      num_elems_31882) *
                                              num_elems_31882)) * 4];
            x_33858 = *(__global float *) &mem_33621[(gtid_32739 *
                                                      num_elems_31882 +
                                                      gtid_32760) * 4];
            
            float res_33859;
            
            if (slt32(scan_gtid_33873, sizze_31493 * num_elems_31882)) {
                res_33859 = x_33857 + x_33858;
            }
            x_33857 = res_33859;
            *(__global float *) &mem_33621[(gtid_32739 * num_elems_31882 +
                                            gtid_32760) * 4] = x_33857;
        }
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
class bfastdistrib:
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
                                       all_sizes={"main.group_size_32025": {"class": "group_size", "value": None},
                                        "main.group_size_32072": {"class": "group_size", "value": None},
                                        "main.group_size_32113": {"class": "group_size", "value": None},
                                        "main.group_size_32197": {"class": "group_size", "value": None},
                                        "main.group_size_32211": {"class": "group_size", "value": None},
                                        "main.group_size_32243": {"class": "group_size", "value": None},
                                        "main.group_size_32258": {"class": "group_size", "value": None},
                                        "main.group_size_32337": {"class": "group_size", "value": None},
                                        "main.group_size_32418": {"class": "group_size", "value": None},
                                        "main.group_size_32472": {"class": "group_size", "value": None},
                                        "main.group_size_32512": {"class": "group_size", "value": None},
                                        "main.group_size_32572": {"class": "group_size", "value": None},
                                        "main.group_size_32606": {"class": "group_size", "value": None},
                                        "main.group_size_32631": {"class": "group_size", "value": None},
                                        "main.group_size_32666": {"class": "group_size", "value": None},
                                        "main.group_size_32743": {"class": "group_size", "value": None},
                                        "main.group_size_32776": {"class": "group_size", "value": None},
                                        "main.group_size_33743": {"class": "group_size", "value": None},
                                        "main.group_size_33812": {"class": "group_size", "value": None},
                                        "main.group_size_33819": {"class": "group_size", "value": None},
                                        "main.group_size_33824": {"class": "group_size", "value": None},
                                        "main.group_size_33876": {"class": "group_size", "value": None},
                                        "main.max_num_groups_32474": {"class": "num_groups", "value": None},
                                        "main.max_num_groups_32745": {"class": "num_groups", "value": None},
                                        "main.tile_size_32920": {"class": "tile_size", "value": None},
                                        "main.tile_size_33402": {"class": "tile_size", "value": None},
                                        "main.tile_size_33427": {"class": "tile_size", "value": None},
                                        "remove_nans.group_size_32005": {"class": "group_size", "value": None}})
    self.copy_33740_var = program.copy_33740
    self.map_32011_var = program.map_32011
    self.map_32031_var = program.map_32031
    self.map_32078_var = program.map_32078
    self.map_32119_var = program.map_32119
    self.map_32148_var = program.map_32148
    self.map_32203_var = program.map_32203
    self.map_32217_var = program.map_32217
    self.map_32249_var = program.map_32249
    self.map_32264_var = program.map_32264
    self.map_32297_var = program.map_32297
    self.map_32343_var = program.map_32343
    self.map_32386_var = program.map_32386
    self.map_32424_var = program.map_32424
    self.map_32518_var = program.map_32518
    self.map_32578_var = program.map_32578
    self.map_32612_var = program.map_32612
    self.map_32637_var = program.map_32637
    self.map_32672_var = program.map_32672
    self.map_32782_var = program.map_32782
    self.map_transpose_f32_var = program.map_transpose_f32
    self.map_transpose_f32_low_height_var = program.map_transpose_f32_low_height
    self.map_transpose_f32_low_width_var = program.map_transpose_f32_low_width
    self.map_transpose_f32_small_var = program.map_transpose_f32_small
    self.map_transpose_i32_var = program.map_transpose_i32
    self.map_transpose_i32_low_height_var = program.map_transpose_i32_low_height
    self.map_transpose_i32_low_width_var = program.map_transpose_i32_low_width
    self.map_transpose_i32_small_var = program.map_transpose_i32_small
    self.replicate_33816_var = program.replicate_33816
    self.replicate_33821_var = program.replicate_33821
    self.scan_stage1_32490_var = program.scan_stage1_32490
    self.scan_stage1_32761_var = program.scan_stage1_32761
    self.scan_stage2_33791_var = program.scan_stage2_33791
    self.scan_stage2_33860_var = program.scan_stage2_33860
    self.scan_stage3_33809_var = program.scan_stage3_33809
    self.scan_stage3_33873_var = program.scan_stage3_33873
  def futhark_main(self, mappingindices_mem_33490, images_mem_33491,
                   sizze_31492, sizze_31493, sizze_31494, trend_31495, k_31496,
                   n_31497, freq_31498, hfrac_31499, lam_31500):
    dim_zzero_31503 = (np.int32(0) == sizze_31493)
    dim_zzero_31504 = (np.int32(0) == sizze_31494)
    old_empty_31505 = (dim_zzero_31503 or dim_zzero_31504)
    dim_zzero_31506 = (np.int32(0) == sizze_31492)
    new_empty_31507 = (dim_zzero_31503 or dim_zzero_31506)
    both_empty_31508 = (old_empty_31505 and new_empty_31507)
    dim_match_31509 = (sizze_31492 == sizze_31494)
    empty_or_match_31510 = (both_empty_31508 or dim_match_31509)
    empty_or_match_cert_31511 = True
    assert empty_or_match_31510, ("Error at bfastdistrib.fut:108:1-244:20: %s" % ("function arguments of wrong shape",))
    x_31513 = (np.int32(2) * k_31496)
    res_31514 = (np.int32(2) + x_31513)
    cond_31515 = slt32(np.int32(0), trend_31495)
    if cond_31515:
      res_31516 = res_31514
    else:
      res_31517 = (res_31514 - np.int32(1))
      res_31516 = res_31517
    bounds_invalid_upwards_31518 = slt32(res_31516, np.int32(0))
    convop_x_33493 = (sizze_31492 * res_31516)
    binop_x_33494 = sext_i32_i64(convop_x_33493)
    bytes_33492 = (np.int64(4) * binop_x_33494)
    if cond_31515:
      eq_x_zz_31520 = (np.int32(0) == res_31516)
      not_p_31521 = not(bounds_invalid_upwards_31518)
      p_and_eq_x_y_31522 = (eq_x_zz_31520 and not_p_31521)
      dim_zzero_31523 = (bounds_invalid_upwards_31518 or p_and_eq_x_y_31522)
      both_empty_31524 = (eq_x_zz_31520 and dim_zzero_31523)
      empty_or_match_31528 = (not_p_31521 or both_empty_31524)
      empty_or_match_cert_31529 = True
      assert empty_or_match_31528, ("Error at bfastdistrib.fut:108:1-244:20 -> bfastdistrib.fut:119:16-55 -> bfastdistrib.fut:40:10-18 -> /futlib/array.fut:61:1-62:12: %s%s%s%d%s%s" % ("Function return value does not match shape of type ",
                                                                                                                                                                                         "*",
                                                                                                                                                                                         "[",
                                                                                                                                                                                         res_31516,
                                                                                                                                                                                         "]",
                                                                                                                                                                                         "intrinsics.i32"))
      group_sizze_32026 = self.sizes["main.group_size_32025"]
      y_32027 = (group_sizze_32026 - np.int32(1))
      x_32028 = (y_32027 + convop_x_33493)
      num_groups_32029 = squot32(x_32028, group_sizze_32026)
      num_threads_32030 = (group_sizze_32026 * num_groups_32029)
      mem_33495 = opencl_alloc(self, bytes_33492, "mem_33495")
      if ((1 * (np.long(num_groups_32029) * np.long(group_sizze_32026))) != 0):
        self.map_32031_var.set_args(np.int32(sizze_31492),
                                    np.float32(freq_31498), np.int32(res_31516),
                                    mappingindices_mem_33490, mem_33495)
        cl.enqueue_nd_range_kernel(self.queue, self.map_32031_var,
                                   ((np.long(num_groups_32029) * np.long(group_sizze_32026)),),
                                   (np.long(group_sizze_32026),))
        if synchronous:
          self.queue.finish()
      arg_mem_33500 = mem_33495
    else:
      eq_x_zz_31551 = (np.int32(0) == res_31516)
      not_p_31552 = not(bounds_invalid_upwards_31518)
      p_and_eq_x_y_31553 = (eq_x_zz_31551 and not_p_31552)
      dim_zzero_31554 = (bounds_invalid_upwards_31518 or p_and_eq_x_y_31553)
      both_empty_31555 = (eq_x_zz_31551 and dim_zzero_31554)
      empty_or_match_31559 = (not_p_31552 or both_empty_31555)
      empty_or_match_cert_31560 = True
      assert empty_or_match_31559, ("Error at bfastdistrib.fut:108:1-244:20 -> bfastdistrib.fut:120:16-55 -> bfastdistrib.fut:52:10-20 -> /futlib/array.fut:61:1-62:12: %s%s%s%d%s%s" % ("Function return value does not match shape of type ",
                                                                                                                                                                                         "*",
                                                                                                                                                                                         "[",
                                                                                                                                                                                         res_31516,
                                                                                                                                                                                         "]",
                                                                                                                                                                                         "intrinsics.i32"))
      group_sizze_32073 = self.sizes["main.group_size_32072"]
      y_32074 = (group_sizze_32073 - np.int32(1))
      x_32075 = (y_32074 + convop_x_33493)
      num_groups_32076 = squot32(x_32075, group_sizze_32073)
      num_threads_32077 = (group_sizze_32073 * num_groups_32076)
      mem_33499 = opencl_alloc(self, bytes_33492, "mem_33499")
      if ((1 * (np.long(num_groups_32076) * np.long(group_sizze_32073))) != 0):
        self.map_32078_var.set_args(np.int32(sizze_31492),
                                    np.float32(freq_31498), np.int32(res_31516),
                                    mappingindices_mem_33490, mem_33499)
        cl.enqueue_nd_range_kernel(self.queue, self.map_32078_var,
                                   ((np.long(num_groups_32076) * np.long(group_sizze_32073)),),
                                   (np.long(group_sizze_32073),))
        if synchronous:
          self.queue.finish()
      arg_mem_33500 = mem_33499
    x_31581 = (sizze_31492 * sizze_31492)
    y_31582 = (np.int32(2) * sizze_31492)
    x_31583 = (x_31581 + y_31582)
    x_31584 = (np.int32(1) + x_31583)
    y_31585 = (np.int32(1) + sizze_31492)
    x_31586 = sdiv32(x_31584, y_31585)
    x_31587 = (x_31586 - sizze_31492)
    arg_31588 = (x_31587 - np.int32(1))
    res_31589 = sitofp_i32_f32(arg_31588)
    group_sizze_32114 = self.sizes["main.group_size_32113"]
    y_32115 = (group_sizze_32114 - np.int32(1))
    x_32116 = (y_32115 + convop_x_33493)
    num_groups_32117 = squot32(x_32116, group_sizze_32114)
    num_threads_32118 = (group_sizze_32114 * num_groups_32117)
    mem_33504 = opencl_alloc(self, bytes_33492, "mem_33504")
    self.futhark__map_transpose_f32(mem_33504, np.int32(0), arg_mem_33500,
                                    np.int32(0), np.int32(1), sizze_31492,
                                    res_31516, (res_31516 * sizze_31492),
                                    (res_31516 * sizze_31492))
    arg_mem_33500 = None
    mem_33508 = opencl_alloc(self, bytes_33492, "mem_33508")
    if ((1 * (np.long(num_groups_32117) * np.long(group_sizze_32114))) != 0):
      self.map_32119_var.set_args(np.int32(sizze_31492), np.int32(res_31516),
                                  np.float32(res_31589), mem_33504, mem_33508)
      cl.enqueue_nd_range_kernel(self.queue, self.map_32119_var,
                                 ((np.long(num_groups_32117) * np.long(group_sizze_32114)),),
                                 (np.long(group_sizze_32114),))
      if synchronous:
        self.queue.finish()
    tmp_32919 = (np.int32(29) + sizze_31493)
    gidzz_range_32918 = squot32(tmp_32919, np.int32(30))
    tile_sizze_32921 = self.sizes["main.tile_size_32920"]
    tile_sizze_x_32922 = smin32(res_31516, tile_sizze_32921)
    tiled_group_sizze_32924 = (tile_sizze_x_32922 * tile_sizze_x_32922)
    y_32931 = (tile_sizze_x_32922 - np.int32(1))
    x_32932 = (res_31516 + y_32931)
    groups_in_dim_32933 = squot32(x_32932, tile_sizze_x_32922)
    y_32938 = (groups_in_dim_32933 * groups_in_dim_32933)
    num_groups_32939 = (gidzz_range_32918 * y_32938)
    num_threads_32940 = (tiled_group_sizze_32924 * num_groups_32939)
    binop_x_33510 = (sizze_31493 * res_31516)
    convop_x_33511 = (res_31516 * binop_x_33510)
    binop_x_33512 = sext_i32_i64(convop_x_33511)
    bytes_33509 = (np.int64(4) * binop_x_33512)
    mem_33513 = opencl_alloc(self, bytes_33509, "mem_33513")
    convop_x_33515 = (sizze_31493 * sizze_31494)
    binop_x_33516 = sext_i32_i64(convop_x_33515)
    bytes_33514 = (np.int64(4) * binop_x_33516)
    mem_33517 = opencl_alloc(self, bytes_33514, "mem_33517")
    self.futhark__map_transpose_f32(mem_33517, np.int32(0), images_mem_33491,
                                    np.int32(0), np.int32(1), sizze_31494,
                                    sizze_31493, (sizze_31493 * sizze_31494),
                                    (sizze_31493 * sizze_31494))
    binop_x_33519 = sext_i32_i64(tiled_group_sizze_32924)
    bytes_33518 = (np.int64(4) * binop_x_33519)
    if ((1 * (np.long(num_groups_32939) * np.long(tiled_group_sizze_32924))) != 0):
      self.map_32148_var.set_args(cl.LocalMemory(np.long(bytes_33518)),
                                  np.int32(sizze_31493), np.int32(n_31497),
                                  np.int32(res_31516),
                                  np.int32(gidzz_range_32918),
                                  np.int32(tile_sizze_x_32922),
                                  np.int32(tiled_group_sizze_32924), mem_33504,
                                  mem_33508, mem_33513, mem_33517)
      cl.enqueue_nd_range_kernel(self.queue, self.map_32148_var,
                                 ((np.long(num_groups_32939) * np.long(tiled_group_sizze_32924)),),
                                 (np.long(tiled_group_sizze_32924),))
      if synchronous:
        self.queue.finish()
    j_31618 = (np.int32(2) * res_31516)
    j_m_i_31619 = (j_31618 - res_31516)
    arg_31622 = (res_31516 * j_31618)
    res_31635 = sdiv32(arg_31622, res_31516)
    arg_31636 = (res_31516 * res_31635)
    m_31652 = (res_31516 - np.int32(1))
    nesting_sizze_32257 = (sizze_31493 * arg_31622)
    group_sizze_32259 = self.sizes["main.group_size_32258"]
    y_32260 = (group_sizze_32259 - np.int32(1))
    x_32261 = (nesting_sizze_32257 + y_32260)
    num_groups_32262 = squot32(x_32261, group_sizze_32259)
    num_threads_32263 = (group_sizze_32259 * num_groups_32262)
    binop_x_33523 = sext_i32_i64(nesting_sizze_32257)
    bytes_33521 = (np.int64(4) * binop_x_33523)
    mem_33524 = opencl_alloc(self, bytes_33521, "mem_33524")
    if ((1 * (np.long(num_groups_32262) * np.long(group_sizze_32259))) != 0):
      self.map_32264_var.set_args(np.int32(sizze_31493), np.int32(res_31516),
                                  np.int32(j_31618), np.int32(arg_31622),
                                  mem_33513, mem_33524)
      cl.enqueue_nd_range_kernel(self.queue, self.map_32264_var,
                                 ((np.long(num_groups_32262) * np.long(group_sizze_32259)),),
                                 (np.long(group_sizze_32259),))
      if synchronous:
        self.queue.finish()
    mem_33513 = None
    loop_nonempty_32826 = slt32(np.int32(0), res_31516)
    group_sizze_32244 = self.sizes["main.group_size_32243"]
    y_32245 = (group_sizze_32244 - np.int32(1))
    x_32246 = (sizze_31493 + y_32245)
    if loop_nonempty_32826:
      x_32827 = squot32(x_32246, group_sizze_32244)
      num_groups_32247 = x_32827
    else:
      num_groups_32247 = np.int32(0)
    num_threads_32248 = (group_sizze_32244 * num_groups_32247)
    nesting_sizze_32210 = (sizze_31493 * arg_31636)
    group_sizze_32212 = self.sizes["main.group_size_32211"]
    y_32213 = (group_sizze_32212 - np.int32(1))
    x_32214 = (nesting_sizze_32210 + y_32213)
    if loop_nonempty_32826:
      x_32829 = squot32(x_32214, group_sizze_32212)
      num_groups_32215 = x_32829
    else:
      num_groups_32215 = np.int32(0)
    num_threads_32216 = (group_sizze_32212 * num_groups_32215)
    group_sizze_32198 = self.sizes["main.group_size_32197"]
    y_32199 = (group_sizze_32198 - np.int32(1))
    x_32200 = (y_32199 + nesting_sizze_32210)
    if loop_nonempty_32826:
      x_32831 = squot32(x_32200, group_sizze_32198)
      num_groups_32201 = x_32831
    else:
      num_groups_32201 = np.int32(0)
    num_threads_32202 = (group_sizze_32198 * num_groups_32201)
    bytes_33526 = sext_i32_i64(sizze_31493)
    mem_33527 = opencl_alloc(self, bytes_33526, "mem_33527")
    binop_x_33530 = sext_i32_i64(nesting_sizze_32210)
    bytes_33528 = (np.int64(4) * binop_x_33530)
    mem_33531 = opencl_alloc(self, bytes_33528, "mem_33531")
    i_31686 = np.int32(0)
    one_33890 = np.int32(1)
    for counter_33889 in range(res_31516):
      if ((1 * (np.long(num_groups_32247) * np.long(group_sizze_32244))) != 0):
        self.map_32249_var.set_args(np.int32(sizze_31493), np.int32(arg_31622),
                                    np.int32(i_31686), mem_33524, mem_33527)
        cl.enqueue_nd_range_kernel(self.queue, self.map_32249_var,
                                   ((np.long(num_groups_32247) * np.long(group_sizze_32244)),),
                                   (np.long(group_sizze_32244),))
        if synchronous:
          self.queue.finish()
      if ((1 * (np.long(num_groups_32215) * np.long(group_sizze_32212))) != 0):
        self.map_32217_var.set_args(np.int32(sizze_31493), np.int32(arg_31622),
                                    np.int32(res_31635), np.int32(arg_31636),
                                    np.int32(m_31652), np.int32(i_31686),
                                    mem_33524, mem_33527, mem_33531)
        cl.enqueue_nd_range_kernel(self.queue, self.map_32217_var,
                                   ((np.long(num_groups_32215) * np.long(group_sizze_32212)),),
                                   (np.long(group_sizze_32212),))
        if synchronous:
          self.queue.finish()
      if ((1 * (np.long(num_groups_32201) * np.long(group_sizze_32198))) != 0):
        self.map_32203_var.set_args(np.int32(sizze_31493), np.int32(arg_31622),
                                    np.int32(arg_31636), mem_33524, mem_33531)
        cl.enqueue_nd_range_kernel(self.queue, self.map_32203_var,
                                   ((np.long(num_groups_32201) * np.long(group_sizze_32198)),),
                                   (np.long(group_sizze_32198),))
        if synchronous:
          self.queue.finish()
      i_31686 += one_33890
    mem_33527 = None
    mem_33531 = None
    tile_sizze_33403 = self.sizes["main.tile_size_33402"]
    tiled_group_sizze_33404 = (tile_sizze_33403 * tile_sizze_33403)
    y_33407 = (tile_sizze_33403 - np.int32(1))
    x_33408 = (sizze_31493 + y_33407)
    groups_in_dim_33409 = squot32(x_33408, tile_sizze_33403)
    x_33411 = (res_31516 + y_33407)
    groups_in_dim_33412 = squot32(x_33411, tile_sizze_33403)
    num_groups_33414 = (groups_in_dim_33409 * groups_in_dim_33412)
    num_threads_33415 = (tiled_group_sizze_33404 * num_groups_33414)
    binop_x_33543 = sext_i32_i64(binop_x_33510)
    bytes_33541 = (np.int64(4) * binop_x_33543)
    mem_33544 = opencl_alloc(self, bytes_33541, "mem_33544")
    binop_x_33535 = sext_i32_i64(tiled_group_sizze_33404)
    bytes_33533 = (np.int64(4) * binop_x_33535)
    if ((1 * (np.long(num_groups_33414) * np.long(tiled_group_sizze_33404))) != 0):
      self.map_32297_var.set_args(np.int32(sizze_31493), np.int32(sizze_31494),
                                  np.int32(n_31497), np.int32(res_31516),
                                  images_mem_33491, mem_33504, mem_33544)
      cl.enqueue_nd_range_kernel(self.queue, self.map_32297_var,
                                 ((np.long(num_groups_33414) * np.long(tiled_group_sizze_33404)),),
                                 (np.long(tiled_group_sizze_33404),))
      if synchronous:
        self.queue.finish()
    mem_33504 = None
    group_sizze_32338 = self.sizes["main.group_size_32337"]
    y_32339 = (group_sizze_32338 - np.int32(1))
    x_32340 = (y_32339 + binop_x_33510)
    num_groups_32341 = squot32(x_32340, group_sizze_32338)
    num_threads_32342 = (group_sizze_32338 * num_groups_32341)
    binop_x_33546 = (sizze_31493 * j_m_i_31619)
    convop_x_33547 = (res_31516 * binop_x_33546)
    binop_x_33548 = sext_i32_i64(convop_x_33547)
    bytes_33545 = (np.int64(4) * binop_x_33548)
    mem_33549 = opencl_alloc(self, bytes_33545, "mem_33549")
    group_sizze_33743 = self.sizes["main.group_size_33743"]
    num_groups_33744 = squot32((((sizze_31493 * (res_31516 * j_m_i_31619)) + sext_i32_i32(group_sizze_33743)) - np.int32(1)),
                               sext_i32_i32(group_sizze_33743))
    if ((1 * (np.long(num_groups_33744) * np.long(group_sizze_33743))) != 0):
      self.copy_33740_var.set_args(np.int32(sizze_31493), np.int32(res_31516),
                                   np.int32(j_31618), np.int32(j_m_i_31619),
                                   mem_33524, mem_33549)
      cl.enqueue_nd_range_kernel(self.queue, self.copy_33740_var,
                                 ((np.long(num_groups_33744) * np.long(group_sizze_33743)),),
                                 (np.long(group_sizze_33743),))
      if synchronous:
        self.queue.finish()
    mem_33524 = None
    mem_33553 = opencl_alloc(self, bytes_33541, "mem_33553")
    if ((1 * (np.long(num_groups_32341) * np.long(group_sizze_32338))) != 0):
      self.map_32343_var.set_args(np.int32(sizze_31493), np.int32(res_31516),
                                  np.int32(j_m_i_31619), mem_33544, mem_33549,
                                  mem_33553)
      cl.enqueue_nd_range_kernel(self.queue, self.map_32343_var,
                                 ((np.long(num_groups_32341) * np.long(group_sizze_32338)),),
                                 (np.long(group_sizze_32338),))
      if synchronous:
        self.queue.finish()
    mem_33544 = None
    mem_33549 = None
    mem_33557 = opencl_alloc(self, bytes_33492, "mem_33557")
    self.futhark__map_transpose_f32(mem_33557, np.int32(0), mem_33508,
                                    np.int32(0), np.int32(1), res_31516,
                                    sizze_31492, (sizze_31492 * res_31516),
                                    (sizze_31492 * res_31516))
    mem_33508 = None
    tile_sizze_33428 = self.sizes["main.tile_size_33427"]
    tiled_group_sizze_33429 = (tile_sizze_33428 * tile_sizze_33428)
    y_33432 = (tile_sizze_33428 - np.int32(1))
    x_33433 = (sizze_31493 + y_33432)
    groups_in_dim_33434 = squot32(x_33433, tile_sizze_33428)
    x_33436 = (sizze_31492 + y_33432)
    groups_in_dim_33437 = squot32(x_33436, tile_sizze_33428)
    num_groups_33439 = (groups_in_dim_33434 * groups_in_dim_33437)
    num_threads_33440 = (tiled_group_sizze_33429 * num_groups_33439)
    convop_x_33567 = (sizze_31492 * sizze_31493)
    binop_x_33568 = sext_i32_i64(convop_x_33567)
    bytes_33566 = (np.int64(4) * binop_x_33568)
    mem_33569 = opencl_alloc(self, bytes_33566, "mem_33569")
    binop_x_33560 = sext_i32_i64(tiled_group_sizze_33429)
    bytes_33558 = (np.int64(4) * binop_x_33560)
    if ((1 * (np.long(num_groups_33439) * np.long(tiled_group_sizze_33429))) != 0):
      self.map_32386_var.set_args(np.int32(sizze_31492), np.int32(sizze_31493),
                                  np.int32(res_31516), mem_33553, mem_33557,
                                  mem_33569)
      cl.enqueue_nd_range_kernel(self.queue, self.map_32386_var,
                                 ((np.long(num_groups_33439) * np.long(tiled_group_sizze_33429)),),
                                 (np.long(tiled_group_sizze_33429),))
      if synchronous:
        self.queue.finish()
    mem_33553 = None
    mem_33557 = None
    i_31764 = (sizze_31492 - np.int32(1))
    group_sizze_32473 = self.sizes["main.group_size_32472"]
    max_num_groups_32475 = self.sizes["main.max_num_groups_32474"]
    group_sizze_32476 = sext_i32_i64(group_sizze_32473)
    max_num_groups_32477 = sext_i32_i64(max_num_groups_32475)
    y_32478 = (group_sizze_32476 - np.int64(1))
    x_32479 = (y_32478 + binop_x_33568)
    w_div_group_sizze_32480 = squot64(x_32479, group_sizze_32476)
    num_groups_maybe_zzero_32481 = smin64(max_num_groups_32477,
                                          w_div_group_sizze_32480)
    num_groups_32482 = smax64(np.int64(1), num_groups_maybe_zzero_32481)
    num_threads_32483 = (group_sizze_32476 * num_groups_32482)
    num_groups_32484 = sext_i64_i32(num_groups_32482)
    num_threads_32485 = sext_i64_i32(num_threads_32483)
    mem_33573 = opencl_alloc(self, bytes_33566, "mem_33573")
    mem_33577 = opencl_alloc(self, bytes_33566, "mem_33577")
    mem_33580 = opencl_alloc(self, binop_x_33568, "mem_33580")
    mem_33584 = opencl_alloc(self, bytes_33566, "mem_33584")
    if ((1 * (np.long(num_groups_32484) * np.long(group_sizze_32473))) != 0):
      self.scan_stage1_32490_var.set_args(np.int32(sizze_31492),
                                          np.int32(sizze_31493),
                                          np.int32(sizze_31494),
                                          np.int32(num_groups_32484),
                                          images_mem_33491, mem_33569,
                                          mem_33573, mem_33577, mem_33580,
                                          mem_33584)
      cl.enqueue_nd_range_kernel(self.queue, self.scan_stage1_32490_var,
                                 ((np.long(num_groups_32484) * np.long(group_sizze_32473)),),
                                 (np.long(group_sizze_32473),))
      if synchronous:
        self.queue.finish()
    if ((1 * (np.long(np.int32(1)) * np.long(num_groups_32484))) != 0):
      self.scan_stage2_33791_var.set_args(cl.LocalMemory(np.long((np.int32(4) * num_groups_32484))),
                                          cl.LocalMemory(np.long((np.int32(4) * num_groups_32484))),
                                          np.int32(sizze_31492),
                                          np.int32(sizze_31493),
                                          np.int32(num_groups_32484), mem_33573,
                                          mem_33577)
      cl.enqueue_nd_range_kernel(self.queue, self.scan_stage2_33791_var,
                                 ((np.long(np.int32(1)) * np.long(num_groups_32484)),),
                                 (np.long(num_groups_32484),))
      if synchronous:
        self.queue.finish()
    group_sizze_33812 = self.sizes["main.group_size_33812"]
    num_groups_33813 = squot32((((sizze_31493 * sizze_31492) + sext_i32_i32(group_sizze_33812)) - np.int32(1)),
                               sext_i32_i32(group_sizze_33812))
    if ((1 * (np.long(num_groups_33813) * np.long(group_sizze_33812))) != 0):
      self.scan_stage3_33809_var.set_args(np.int32(sizze_31492),
                                          np.int32(sizze_31493),
                                          np.int32(num_groups_32484), mem_33573,
                                          mem_33577)
      cl.enqueue_nd_range_kernel(self.queue, self.scan_stage3_33809_var,
                                 ((np.long(num_groups_33813) * np.long(group_sizze_33812)),),
                                 (np.long(group_sizze_33812),))
      if synchronous:
        self.queue.finish()
    mem_33569 = None
    mem_33588 = opencl_alloc(self, bytes_33566, "mem_33588")
    group_sizze_33819 = self.sizes["main.group_size_33819"]
    num_groups_33820 = squot32((((sizze_31493 * sizze_31492) + sext_i32_i32(group_sizze_33819)) - np.int32(1)),
                               sext_i32_i32(group_sizze_33819))
    if ((1 * (np.long(num_groups_33820) * np.long(group_sizze_33819))) != 0):
      self.replicate_33816_var.set_args(np.int32(sizze_31492),
                                        np.int32(sizze_31493), mem_33588)
      cl.enqueue_nd_range_kernel(self.queue, self.replicate_33816_var,
                                 ((np.long(num_groups_33820) * np.long(group_sizze_33819)),),
                                 (np.long(group_sizze_33819),))
      if synchronous:
        self.queue.finish()
    mem_33592 = opencl_alloc(self, bytes_33566, "mem_33592")
    group_sizze_33824 = self.sizes["main.group_size_33824"]
    num_groups_33825 = squot32((((sizze_31493 * sizze_31492) + sext_i32_i32(group_sizze_33824)) - np.int32(1)),
                               sext_i32_i32(group_sizze_33824))
    if ((1 * (np.long(num_groups_33825) * np.long(group_sizze_33824))) != 0):
      self.replicate_33821_var.set_args(np.int32(sizze_31492),
                                        np.int32(sizze_31493), mem_33592)
      cl.enqueue_nd_range_kernel(self.queue, self.replicate_33821_var,
                                 ((np.long(num_groups_33825) * np.long(group_sizze_33824)),),
                                 (np.long(group_sizze_33824),))
      if synchronous:
        self.queue.finish()
    group_sizze_32419 = self.sizes["main.group_size_32418"]
    y_32420 = (group_sizze_32419 - np.int32(1))
    x_32421 = (y_32420 + convop_x_33567)
    num_groups_32422 = squot32(x_32421, group_sizze_32419)
    num_threads_32423 = (group_sizze_32419 * num_groups_32422)
    if ((1 * (np.long(num_groups_32422) * np.long(group_sizze_32419))) != 0):
      self.map_32424_var.set_args(np.int32(sizze_31492), np.int32(sizze_31493),
                                  np.int32(i_31764), mem_33573, mem_33577,
                                  mem_33580, mem_33584, mem_33588, mem_33592)
      cl.enqueue_nd_range_kernel(self.queue, self.map_32424_var,
                                 ((np.long(num_groups_32422) * np.long(group_sizze_32419)),),
                                 (np.long(group_sizze_32419),))
      if synchronous:
        self.queue.finish()
    mem_33577 = None
    mem_33580 = None
    mem_33584 = None
    group_sizze_32513 = self.sizes["main.group_size_32512"]
    y_32514 = (group_sizze_32513 - np.int32(1))
    x_32515 = (sizze_31493 + y_32514)
    num_groups_32516 = squot32(x_32515, group_sizze_32513)
    num_threads_32517 = (group_sizze_32513 * num_groups_32516)
    mem_33596 = opencl_alloc(self, bytes_33566, "mem_33596")
    self.futhark__map_transpose_f32(mem_33596, np.int32(0), mem_33588,
                                    np.int32(0), np.int32(1), sizze_31492,
                                    sizze_31493, (sizze_31493 * sizze_31492),
                                    (sizze_31493 * sizze_31492))
    bytes_33597 = (np.int64(4) * bytes_33526)
    mem_33599 = opencl_alloc(self, bytes_33597, "mem_33599")
    mem_33602 = opencl_alloc(self, bytes_33597, "mem_33602")
    if ((1 * (np.long(num_groups_32516) * np.long(group_sizze_32513))) != 0):
      self.map_32518_var.set_args(np.int32(sizze_31493), np.int32(n_31497),
                                  np.int32(res_31514), mem_33517, mem_33596,
                                  mem_33599, mem_33602)
      cl.enqueue_nd_range_kernel(self.queue, self.map_32518_var,
                                 ((np.long(num_groups_32516) * np.long(group_sizze_32513)),),
                                 (np.long(group_sizze_32513),))
      if synchronous:
        self.queue.finish()
    mem_33517 = None
    mem_33596 = None
    group_sizze_32573 = self.sizes["main.group_size_32572"]
    y_32574 = (group_sizze_32573 - np.int32(1))
    x_32575 = (sizze_31493 + y_32574)
    num_groups_32576 = squot32(x_32575, group_sizze_32573)
    num_threads_32577 = (group_sizze_32573 * num_groups_32576)
    mem_33605 = opencl_alloc(self, bytes_33597, "mem_33605")
    mem_33608 = opencl_alloc(self, bytes_33597, "mem_33608")
    if ((1 * (np.long(num_groups_32576) * np.long(group_sizze_32573))) != 0):
      self.map_32578_var.set_args(np.int32(sizze_31492), np.int32(sizze_31493),
                                  np.float32(hfrac_31499), mem_33588, mem_33599,
                                  mem_33605, mem_33608)
      cl.enqueue_nd_range_kernel(self.queue, self.map_32578_var,
                                 ((np.long(num_groups_32576) * np.long(group_sizze_32573)),),
                                 (np.long(group_sizze_32573),))
      if synchronous:
        self.queue.finish()
    x_31878 = (sizze_31492 - n_31497)
    range_end_31879 = (x_31878 - np.int32(1))
    bounds_invalid_upwards_31880 = slt32(range_end_31879, np.int32(0))
    distance_31881 = (np.int32(1) + range_end_31879)
    if bounds_invalid_upwards_31880:
      num_elems_31882 = np.int32(0)
    else:
      num_elems_31882 = distance_31881
    x_31884 = (np.int32(1) + n_31497)
    x_31885 = sle32(np.int32(0), i_31764)
    index_certs_31888 = True
    assert x_31885, ("Error at bfastdistrib.fut:108:1-244:20 -> bfastdistrib.fut:192:15-196:33 -> bfastdistrib.fut:194:63-81: %s%d%s%d%s" % ("Index [",
                                                                                                                                             i_31764,
                                                                                                                                             "] out of bounds for array of shape [",
                                                                                                                                             sizze_31492,
                                                                                                                                             "]."))
    read_res_33891 = np.empty(1, dtype=ct.c_int32)
    cl.enqueue_copy(self.queue, read_res_33891, mappingindices_mem_33490,
                    device_offset=np.long((i_31764 * np.int32(4))),
                    is_blocking=True)
    arg_31889 = read_res_33891[0]
    res_31890 = sitofp_i32_f32(arg_31889)
    group_sizze_32607 = self.sizes["main.group_size_32606"]
    y_32608 = (group_sizze_32607 - np.int32(1))
    x_32609 = (num_elems_31882 + y_32608)
    num_groups_32610 = squot32(x_32609, group_sizze_32607)
    num_threads_32611 = (group_sizze_32607 * num_groups_32610)
    binop_x_33610 = sext_i32_i64(num_elems_31882)
    bytes_33609 = (np.int64(4) * binop_x_33610)
    mem_33611 = opencl_alloc(self, bytes_33609, "mem_33611")
    if ((1 * (np.long(num_groups_32610) * np.long(group_sizze_32607))) != 0):
      self.map_32612_var.set_args(np.float32(lam_31500),
                                  np.int32(num_elems_31882), np.int32(x_31884),
                                  np.float32(res_31890),
                                  mappingindices_mem_33490, mem_33611)
      cl.enqueue_nd_range_kernel(self.queue, self.map_32612_var,
                                 ((np.long(num_groups_32610) * np.long(group_sizze_32607)),),
                                 (np.long(group_sizze_32607),))
      if synchronous:
        self.queue.finish()
    group_sizze_32777 = self.sizes["main.group_size_32776"]
    y_32778 = (group_sizze_32777 - np.int32(1))
    x_32779 = (sizze_31493 + y_32778)
    num_groups_32780 = squot32(x_32779, group_sizze_32777)
    num_threads_32781 = (group_sizze_32777 * num_groups_32780)
    mem_33614 = opencl_alloc(self, bytes_33597, "mem_33614")
    mem_33617 = opencl_alloc(self, bytes_33597, "mem_33617")
    if ((1 * (np.long(num_groups_32780) * np.long(group_sizze_32777))) != 0):
      self.map_32782_var.set_args(np.int32(sizze_31492), np.int32(sizze_31493),
                                  np.int32(i_31764), mem_33573, mem_33599,
                                  mem_33602, mem_33614, mem_33617)
      cl.enqueue_nd_range_kernel(self.queue, self.map_32782_var,
                                 ((np.long(num_groups_32780) * np.long(group_sizze_32777)),),
                                 (np.long(group_sizze_32777),))
      if synchronous:
        self.queue.finish()
    mem_33573 = None
    mem_33602 = None
    total_num_elements_32740 = (sizze_31493 * num_elems_31882)
    total_num_elements_32742 = sext_i32_i64(total_num_elements_32740)
    group_sizze_32744 = self.sizes["main.group_size_32743"]
    max_num_groups_32746 = self.sizes["main.max_num_groups_32745"]
    group_sizze_32747 = sext_i32_i64(group_sizze_32744)
    max_num_groups_32748 = sext_i32_i64(max_num_groups_32746)
    y_32749 = (group_sizze_32747 - np.int64(1))
    x_32750 = (total_num_elements_32742 + y_32749)
    w_div_group_sizze_32751 = squot64(x_32750, group_sizze_32747)
    num_groups_maybe_zzero_32752 = smin64(max_num_groups_32748,
                                          w_div_group_sizze_32751)
    num_groups_32753 = smax64(np.int64(1), num_groups_maybe_zzero_32752)
    num_threads_32754 = (group_sizze_32747 * num_groups_32753)
    num_groups_32755 = sext_i64_i32(num_groups_32753)
    num_threads_32756 = sext_i64_i32(num_threads_32754)
    bytes_33618 = (np.int64(4) * total_num_elements_32742)
    mem_33621 = opencl_alloc(self, bytes_33618, "mem_33621")
    if ((1 * (np.long(num_groups_32755) * np.long(group_sizze_32744))) != 0):
      self.scan_stage1_32761_var.set_args(np.int32(sizze_31492),
                                          np.int32(sizze_31493),
                                          np.int32(num_elems_31882),
                                          np.int32(num_groups_32755), mem_33588,
                                          mem_33599, mem_33605, mem_33608,
                                          mem_33617, mem_33621)
      cl.enqueue_nd_range_kernel(self.queue, self.scan_stage1_32761_var,
                                 ((np.long(num_groups_32755) * np.long(group_sizze_32744)),),
                                 (np.long(group_sizze_32744),))
      if synchronous:
        self.queue.finish()
    if ((1 * (np.long(np.int32(1)) * np.long(num_groups_32755))) != 0):
      self.scan_stage2_33860_var.set_args(cl.LocalMemory(np.long((np.int32(4) * num_groups_32755))),
                                          np.int32(sizze_31493),
                                          np.int32(num_elems_31882),
                                          np.int32(num_groups_32755), mem_33621)
      cl.enqueue_nd_range_kernel(self.queue, self.scan_stage2_33860_var,
                                 ((np.long(np.int32(1)) * np.long(num_groups_32755)),),
                                 (np.long(num_groups_32755),))
      if synchronous:
        self.queue.finish()
    group_sizze_33876 = self.sizes["main.group_size_33876"]
    num_groups_33877 = squot32((((sizze_31493 * num_elems_31882) + sext_i32_i32(group_sizze_33876)) - np.int32(1)),
                               sext_i32_i32(group_sizze_33876))
    if ((1 * (np.long(num_groups_33877) * np.long(group_sizze_33876))) != 0):
      self.scan_stage3_33873_var.set_args(np.int32(sizze_31493),
                                          np.int32(num_elems_31882),
                                          np.int32(num_groups_32755), mem_33621)
      cl.enqueue_nd_range_kernel(self.queue, self.scan_stage3_33873_var,
                                 ((np.long(num_groups_33877) * np.long(group_sizze_33876)),),
                                 (np.long(group_sizze_33876),))
      if synchronous:
        self.queue.finish()
    mem_33588 = None
    mem_33605 = None
    mem_33608 = None
    group_sizze_32667 = self.sizes["main.group_size_32666"]
    y_32668 = (group_sizze_32667 - np.int32(1))
    x_32669 = (sizze_31493 + y_32668)
    num_groups_32670 = squot32(x_32669, group_sizze_32667)
    num_threads_32671 = (group_sizze_32667 * num_groups_32670)
    mem_33625 = opencl_alloc(self, bytes_33618, "mem_33625")
    self.futhark__map_transpose_f32(mem_33625, np.int32(0), mem_33621,
                                    np.int32(0), np.int32(1), num_elems_31882,
                                    sizze_31493,
                                    (sizze_31493 * num_elems_31882),
                                    (sizze_31493 * num_elems_31882))
    mem_33621 = None
    mem_33632 = opencl_alloc(self, bytes_33618, "mem_33632")
    mem_33634 = opencl_alloc(self, bytes_33526, "mem_33634")
    mem_33637 = opencl_alloc(self, bytes_33597, "mem_33637")
    mem_33640 = opencl_alloc(self, bytes_33597, "mem_33640")
    num_threads64_33667 = sext_i32_i64(num_threads_32671)
    total_sizze_33668 = (bytes_33609 * num_threads64_33667)
    mem_33628 = opencl_alloc(self, total_sizze_33668, "mem_33628")
    if ((1 * (np.long(num_groups_32670) * np.long(group_sizze_32667))) != 0):
      self.map_32672_var.set_args(np.int32(sizze_31492), np.int32(sizze_31493),
                                  np.int32(n_31497), np.int32(num_elems_31882),
                                  mem_33592, mem_33599, mem_33611, mem_33614,
                                  mem_33617, mem_33625, mem_33628, mem_33632,
                                  mem_33634, mem_33637, mem_33640)
      cl.enqueue_nd_range_kernel(self.queue, self.map_32672_var,
                                 ((np.long(num_groups_32670) * np.long(group_sizze_32667)),),
                                 (np.long(group_sizze_32667),))
      if synchronous:
        self.queue.finish()
    mem_33592 = None
    mem_33611 = None
    mem_33614 = None
    mem_33625 = None
    mem_33628 = None
    group_sizze_32632 = self.sizes["main.group_size_32631"]
    y_32633 = (group_sizze_32632 - np.int32(1))
    x_32634 = (sizze_31493 + y_32633)
    num_groups_32635 = squot32(x_32634, group_sizze_32632)
    num_threads_32636 = (group_sizze_32632 * num_groups_32635)
    mem_33644 = opencl_alloc(self, bytes_33618, "mem_33644")
    self.futhark__map_transpose_i32(mem_33644, np.int32(0), mem_33632,
                                    np.int32(0), np.int32(1), sizze_31493,
                                    num_elems_31882,
                                    (sizze_31493 * num_elems_31882),
                                    (sizze_31493 * num_elems_31882))
    mem_33632 = None
    mem_33647 = opencl_alloc(self, bytes_33597, "mem_33647")
    if ((1 * (np.long(num_groups_32635) * np.long(group_sizze_32632))) != 0):
      self.map_32637_var.set_args(np.int32(sizze_31493),
                                  np.int32(num_elems_31882), mem_33599,
                                  mem_33617, mem_33634, mem_33637, mem_33644,
                                  mem_33647)
      cl.enqueue_nd_range_kernel(self.queue, self.map_32637_var,
                                 ((np.long(num_groups_32635) * np.long(group_sizze_32632)),),
                                 (np.long(group_sizze_32632),))
      if synchronous:
        self.queue.finish()
    mem_33599 = None
    mem_33617 = None
    mem_33634 = None
    mem_33637 = None
    mem_33644 = None
    out_arrsizze_33680 = sizze_31493
    out_arrsizze_33682 = sizze_31493
    out_mem_33679 = mem_33647
    out_mem_33681 = mem_33640
    return (out_mem_33679, out_arrsizze_33680, out_mem_33681,
            out_arrsizze_33682)
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
  def futhark_remove_nans(self, images_mem_33490, sizze_31478, sizze_31479,
                          sizze_31480, nan_value_31481):
    nesting_sizze_32003 = (sizze_31479 * sizze_31480)
    nesting_sizze_32004 = (sizze_31478 * nesting_sizze_32003)
    group_sizze_32006 = self.sizes["remove_nans.group_size_32005"]
    y_32007 = (group_sizze_32006 - np.int32(1))
    x_32008 = (nesting_sizze_32004 + y_32007)
    num_groups_32009 = squot32(x_32008, group_sizze_32006)
    num_threads_32010 = (group_sizze_32006 * num_groups_32009)
    binop_x_33492 = (sizze_31478 * sizze_31479)
    convop_x_33493 = (sizze_31480 * binop_x_33492)
    binop_x_33494 = sext_i32_i64(convop_x_33493)
    bytes_33491 = (np.int64(4) * binop_x_33494)
    mem_33495 = opencl_alloc(self, bytes_33491, "mem_33495")
    if ((1 * (np.long(num_groups_32009) * np.long(group_sizze_32006))) != 0):
      self.map_32011_var.set_args(np.int32(sizze_31478), np.int32(sizze_31479),
                                  np.int32(sizze_31480),
                                  np.int16(nan_value_31481), images_mem_33490,
                                  mem_33495)
      cl.enqueue_nd_range_kernel(self.queue, self.map_32011_var,
                                 ((np.long(num_groups_32009) * np.long(group_sizze_32006)),),
                                 (np.long(group_sizze_32006),))
      if synchronous:
        self.queue.finish()
    out_arrsizze_33674 = sizze_31478
    out_arrsizze_33675 = sizze_31479
    out_arrsizze_33676 = sizze_31480
    out_mem_33673 = mem_33495
    return (out_mem_33673, out_arrsizze_33674, out_arrsizze_33675,
            out_arrsizze_33676)
  def futhark_reshapeTransp(self, images_mem_33490, sizze_31471, sizze_31472,
                            sizze_31473):
    flat_dim_31475 = (sizze_31472 * sizze_31473)
    convop_x_33492 = (sizze_31471 * flat_dim_31475)
    binop_x_33493 = sext_i32_i64(convop_x_33492)
    bytes_33491 = (np.int64(4) * binop_x_33493)
    mem_33494 = opencl_alloc(self, bytes_33491, "mem_33494")
    self.futhark__map_transpose_f32(mem_33494, np.int32(0), images_mem_33490,
                                    np.int32(0), np.int32(1), flat_dim_31475,
                                    sizze_31471, (flat_dim_31475 * sizze_31471),
                                    (flat_dim_31475 * sizze_31471))
    out_arrsizze_33671 = flat_dim_31475
    out_arrsizze_33672 = sizze_31471
    out_mem_33670 = mem_33494
    return (out_mem_33670, out_arrsizze_33671, out_arrsizze_33672)
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
  def main(self, trend_31495_ext, k_31496_ext, n_31497_ext, freq_31498_ext,
           hfrac_31499_ext, lam_31500_ext, mappingindices_mem_33490_ext,
           images_mem_33491_ext):
    try:
      trend_31495 = np.int32(ct.c_int32(trend_31495_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("i32",
                                                                                                                            type(trend_31495_ext),
                                                                                                                            trend_31495_ext))
    try:
      k_31496 = np.int32(ct.c_int32(k_31496_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("i32",
                                                                                                                            type(k_31496_ext),
                                                                                                                            k_31496_ext))
    try:
      n_31497 = np.int32(ct.c_int32(n_31497_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #2 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("i32",
                                                                                                                            type(n_31497_ext),
                                                                                                                            n_31497_ext))
    try:
      freq_31498 = np.float32(ct.c_float(freq_31498_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #3 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(freq_31498_ext),
                                                                                                                            freq_31498_ext))
    try:
      hfrac_31499 = np.float32(ct.c_float(hfrac_31499_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #4 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(hfrac_31499_ext),
                                                                                                                            hfrac_31499_ext))
    try:
      lam_31500 = np.float32(ct.c_float(lam_31500_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #5 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(lam_31500_ext),
                                                                                                                            lam_31500_ext))
    try:
      assert ((type(mappingindices_mem_33490_ext) in [np.ndarray,
                                                      cl.array.Array]) and (mappingindices_mem_33490_ext.dtype == np.int32)), "Parameter has unexpected type"
      sizze_31492 = np.int32(mappingindices_mem_33490_ext.shape[0])
      if (type(mappingindices_mem_33490_ext) == cl.array.Array):
        mappingindices_mem_33490 = mappingindices_mem_33490_ext.data
      else:
        mappingindices_mem_33490 = opencl_alloc(self,
                                                np.int64(mappingindices_mem_33490_ext.nbytes),
                                                "mappingindices_mem_33490")
        if (np.int64(mappingindices_mem_33490_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, mappingindices_mem_33490,
                          normaliseArray(mappingindices_mem_33490_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #6 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]i32",
                                                                                                                            type(mappingindices_mem_33490_ext),
                                                                                                                            mappingindices_mem_33490_ext))
    try:
      assert ((type(images_mem_33491_ext) in [np.ndarray,
                                              cl.array.Array]) and (images_mem_33491_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_31493 = np.int32(images_mem_33491_ext.shape[0])
      sizze_31494 = np.int32(images_mem_33491_ext.shape[1])
      if (type(images_mem_33491_ext) == cl.array.Array):
        images_mem_33491 = images_mem_33491_ext.data
      else:
        images_mem_33491 = opencl_alloc(self,
                                        np.int64(images_mem_33491_ext.nbytes),
                                        "images_mem_33491")
        if (np.int64(images_mem_33491_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, images_mem_33491,
                          normaliseArray(images_mem_33491_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #7 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f32",
                                                                                                                            type(images_mem_33491_ext),
                                                                                                                            images_mem_33491_ext))
    (out_mem_33679, out_arrsizze_33680, out_mem_33681,
     out_arrsizze_33682) = self.futhark_main(mappingindices_mem_33490,
                                             images_mem_33491, sizze_31492,
                                             sizze_31493, sizze_31494,
                                             trend_31495, k_31496, n_31497,
                                             freq_31498, hfrac_31499, lam_31500)
    return (cl.array.Array(self.queue, (out_arrsizze_33680,), ct.c_int32,
                           data=out_mem_33679), cl.array.Array(self.queue,
                                                               (out_arrsizze_33682,),
                                                               ct.c_float,
                                                               data=out_mem_33681))
  def remove_nans(self, nan_value_31481_ext, images_mem_33490_ext):
    try:
      nan_value_31481 = np.int16(ct.c_int16(nan_value_31481_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("i16",
                                                                                                                            type(nan_value_31481_ext),
                                                                                                                            nan_value_31481_ext))
    try:
      assert ((type(images_mem_33490_ext) in [np.ndarray,
                                              cl.array.Array]) and (images_mem_33490_ext.dtype == np.int16)), "Parameter has unexpected type"
      sizze_31478 = np.int32(images_mem_33490_ext.shape[0])
      sizze_31479 = np.int32(images_mem_33490_ext.shape[1])
      sizze_31480 = np.int32(images_mem_33490_ext.shape[2])
      if (type(images_mem_33490_ext) == cl.array.Array):
        images_mem_33490 = images_mem_33490_ext.data
      else:
        images_mem_33490 = opencl_alloc(self,
                                        np.int64(images_mem_33490_ext.nbytes),
                                        "images_mem_33490")
        if (np.int64(images_mem_33490_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, images_mem_33490,
                          normaliseArray(images_mem_33490_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][][]i16",
                                                                                                                            type(images_mem_33490_ext),
                                                                                                                            images_mem_33490_ext))
    (out_mem_33673, out_arrsizze_33674, out_arrsizze_33675,
     out_arrsizze_33676) = self.futhark_remove_nans(images_mem_33490,
                                                    sizze_31478, sizze_31479,
                                                    sizze_31480,
                                                    nan_value_31481)
    return cl.array.Array(self.queue, (out_arrsizze_33674, out_arrsizze_33675,
                                       out_arrsizze_33676), ct.c_float,
                          data=out_mem_33673)
  def reshapeTransp(self, images_mem_33490_ext):
    try:
      assert ((type(images_mem_33490_ext) in [np.ndarray,
                                              cl.array.Array]) and (images_mem_33490_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_31471 = np.int32(images_mem_33490_ext.shape[0])
      sizze_31472 = np.int32(images_mem_33490_ext.shape[1])
      sizze_31473 = np.int32(images_mem_33490_ext.shape[2])
      if (type(images_mem_33490_ext) == cl.array.Array):
        images_mem_33490 = images_mem_33490_ext.data
      else:
        images_mem_33490 = opencl_alloc(self,
                                        np.int64(images_mem_33490_ext.nbytes),
                                        "images_mem_33490")
        if (np.int64(images_mem_33490_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, images_mem_33490,
                          normaliseArray(images_mem_33490_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][][]f32",
                                                                                                                            type(images_mem_33490_ext),
                                                                                                                            images_mem_33490_ext))
    (out_mem_33670, out_arrsizze_33671,
     out_arrsizze_33672) = self.futhark_reshapeTransp(images_mem_33490,
                                                      sizze_31471, sizze_31472,
                                                      sizze_31473)
    return cl.array.Array(self.queue, (out_arrsizze_33671, out_arrsizze_33672),
                          ct.c_float, data=out_mem_33670)