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
__kernel void copy_34645(int32_t sizze_32280, int32_t res_32302,
                         int32_t j_32404, int32_t j_m_i_32405, __global
                         unsigned char *mem_34382, __global
                         unsigned char *mem_34407)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t copy_gtid_34645;
    int32_t copy_ltid_34646;
    int32_t copy_gid_34647;
    
    copy_gtid_34645 = get_global_id(0);
    copy_ltid_34646 = get_local_id(0);
    copy_gid_34647 = get_group_id(0);
    if (slt32(copy_gtid_34645, sizze_32280 * (res_32302 * j_m_i_32405))) {
        *(__global float *) &mem_34407[((copy_gtid_34645 -
                                         squot32(copy_gtid_34645, res_32302 *
                                                 j_m_i_32405) * (res_32302 *
                                                                 j_m_i_32405) -
                                         squot32(copy_gtid_34645 -
                                                 squot32(copy_gtid_34645,
                                                         res_32302 *
                                                         j_m_i_32405) *
                                                 (res_32302 * j_m_i_32405),
                                                 j_m_i_32405) * j_m_i_32405) *
                                        (res_32302 * sizze_32280) +
                                        squot32(copy_gtid_34645, res_32302 *
                                                j_m_i_32405) * res_32302 +
                                        squot32(copy_gtid_34645 -
                                                squot32(copy_gtid_34645,
                                                        res_32302 *
                                                        j_m_i_32405) *
                                                (res_32302 * j_m_i_32405),
                                                j_m_i_32405)) * 4] = *(__global
                                                                       float *) &mem_34382[(res_32302 +
                                                                                            (squot32(copy_gtid_34645,
                                                                                                     res_32302 *
                                                                                                     j_m_i_32405) *
                                                                                             (j_32404 *
                                                                                              res_32302) +
                                                                                             squot32(copy_gtid_34645 -
                                                                                                     squot32(copy_gtid_34645,
                                                                                                             res_32302 *
                                                                                                             j_m_i_32405) *
                                                                                                     (res_32302 *
                                                                                                      j_m_i_32405),
                                                                                                     j_m_i_32405) *
                                                                                             j_32404 +
                                                                                             (copy_gtid_34645 -
                                                                                              squot32(copy_gtid_34645,
                                                                                                      res_32302 *
                                                                                                      j_m_i_32405) *
                                                                                              (res_32302 *
                                                                                               j_m_i_32405) -
                                                                                              squot32(copy_gtid_34645 -
                                                                                                      squot32(copy_gtid_34645,
                                                                                                              res_32302 *
                                                                                                              j_m_i_32405) *
                                                                                                      (res_32302 *
                                                                                                       j_m_i_32405),
                                                                                                      j_m_i_32405) *
                                                                                              j_m_i_32405))) *
                                                                                           4];
    }
}
__kernel void copy_34721(int32_t sizze_32279, int32_t sizze_32280,
                         int32_t i_32550, __global unsigned char *mem_34431,
                         __global unsigned char *mem_34445)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t copy_gtid_34721;
    int32_t copy_ltid_34722;
    int32_t copy_gid_34723;
    
    copy_gtid_34721 = get_global_id(0);
    copy_ltid_34722 = get_local_id(0);
    copy_gid_34723 = get_group_id(0);
    if (slt32(copy_gtid_34721, sizze_32280)) {
        *(__global int32_t *) &mem_34445[copy_gtid_34721 * 4] = *(__global
                                                                  int32_t *) &mem_34431[(i_32550 +
                                                                                         copy_gtid_34721 *
                                                                                         sizze_32279) *
                                                                                        4];
    }
}
__kernel void map_32830(int32_t sizze_32265, int32_t sizze_32266,
                        int32_t sizze_32267, int16_t nan_value_32268, __global
                        unsigned char *images_mem_34348, __global
                        unsigned char *mem_34353)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_32830;
    int32_t local_tid_32831;
    int32_t group_sizze_34561;
    int32_t wave_sizze_34560;
    int32_t group_id_32832;
    
    global_tid_32830 = get_global_id(0);
    local_tid_32831 = get_local_id(0);
    group_sizze_34561 = get_local_size(0);
    wave_sizze_34560 = LOCKSTEP_WIDTH;
    group_id_32832 = get_group_id(0);
    
    int32_t gtid_32819;
    int32_t gtid_32820;
    int32_t gtid_32821;
    
    gtid_32819 = squot32(global_tid_32830, sizze_32266 * sizze_32267);
    gtid_32820 = squot32(global_tid_32830 - squot32(global_tid_32830,
                                                    sizze_32266 * sizze_32267) *
                         (sizze_32266 * sizze_32267), sizze_32267);
    gtid_32821 = global_tid_32830 - squot32(global_tid_32830, sizze_32266 *
                                            sizze_32267) * (sizze_32266 *
                                                            sizze_32267) -
        squot32(global_tid_32830 - squot32(global_tid_32830, sizze_32266 *
                                           sizze_32267) * (sizze_32266 *
                                                           sizze_32267),
                sizze_32267) * sizze_32267;
    
    int16_t x_32833;
    bool cond_32834;
    float res_32835;
    
    if ((slt32(gtid_32819, sizze_32265) && slt32(gtid_32820, sizze_32266)) &&
        slt32(gtid_32821, sizze_32267)) {
        x_32833 = *(__global int16_t *) &images_mem_34348[(gtid_32819 *
                                                           (sizze_32267 *
                                                            sizze_32266) +
                                                           gtid_32820 *
                                                           sizze_32267 +
                                                           gtid_32821) * 2];
        cond_32834 = x_32833 == nan_value_32268;
        if (cond_32834) {
            res_32835 = NAN;
        } else {
            float res_32836 = sitofp_i16_f32(x_32833);
            
            res_32835 = res_32836;
        }
    }
    if ((slt32(gtid_32819, sizze_32265) && slt32(gtid_32820, sizze_32266)) &&
        slt32(gtid_32821, sizze_32267)) {
        *(__global float *) &mem_34353[(gtid_32819 * (sizze_32267 *
                                                      sizze_32266) +
                                        gtid_32820 * sizze_32267 + gtid_32821) *
                                       4] = res_32835;
    }
}
__kernel void map_32850(int32_t sizze_32279, float freq_32285,
                        int32_t res_32302, __global
                        unsigned char *mappingindices_mem_34348, __global
                        unsigned char *mem_34353)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_32850;
    int32_t local_tid_32851;
    int32_t group_sizze_34589;
    int32_t wave_sizze_34588;
    int32_t group_id_32852;
    
    global_tid_32850 = get_global_id(0);
    local_tid_32851 = get_local_id(0);
    group_sizze_34589 = get_local_size(0);
    wave_sizze_34588 = LOCKSTEP_WIDTH;
    group_id_32852 = get_group_id(0);
    
    int32_t gtid_32841;
    int32_t gtid_32842;
    
    gtid_32841 = squot32(global_tid_32850, sizze_32279);
    gtid_32842 = global_tid_32850 - squot32(global_tid_32850, sizze_32279) *
        sizze_32279;
    
    bool index_primexp_33749;
    bool index_primexp_33748;
    int32_t cmpop_x_33746;
    bool index_primexp_33747;
    int32_t convop_x_33743;
    float binop_y_33744;
    float index_primexp_33745;
    int32_t x_32857;
    float res_32858;
    
    if (slt32(gtid_32841, res_32302) && slt32(gtid_32842, sizze_32279)) {
        index_primexp_33749 = gtid_32841 == 0;
        index_primexp_33748 = gtid_32841 == 1;
        cmpop_x_33746 = smod32(gtid_32841, 2);
        index_primexp_33747 = cmpop_x_33746 == 0;
        convop_x_33743 = sdiv32(gtid_32841, 2);
        binop_y_33744 = sitofp_i32_f32(convop_x_33743);
        index_primexp_33745 = 6.2831855F * binop_y_33744;
        x_32857 = *(__global int32_t *) &mappingindices_mem_34348[gtid_32842 *
                                                                  4];
        if (index_primexp_33749) {
            res_32858 = 1.0F;
        } else {
            float res_32859;
            
            if (index_primexp_33748) {
                float res_32860 = sitofp_i32_f32(x_32857);
                
                res_32859 = res_32860;
            } else {
                float res_32861;
                float x_32862;
                float res_32863;
                float res_32864;
                
                res_32861 = sitofp_i32_f32(x_32857);
                x_32862 = res_32861 * index_primexp_33745;
                res_32863 = x_32862 / freq_32285;
                if (index_primexp_33747) {
                    float res_32865;
                    
                    res_32865 = futrts_sin32(res_32863);
                    res_32864 = res_32865;
                } else {
                    float res_32866;
                    
                    res_32866 = futrts_cos32(res_32863);
                    res_32864 = res_32866;
                }
                res_32859 = res_32864;
            }
            res_32858 = res_32859;
        }
    }
    if (slt32(gtid_32841, res_32302) && slt32(gtid_32842, sizze_32279)) {
        *(__global float *) &mem_34353[(gtid_32841 * sizze_32279 + gtid_32842) *
                                       4] = res_32858;
    }
}
__kernel void map_32897(int32_t sizze_32279, float freq_32285,
                        int32_t res_32302, __global
                        unsigned char *mappingindices_mem_34348, __global
                        unsigned char *mem_34357)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_32897;
    int32_t local_tid_32898;
    int32_t group_sizze_34591;
    int32_t wave_sizze_34590;
    int32_t group_id_32899;
    
    global_tid_32897 = get_global_id(0);
    local_tid_32898 = get_local_id(0);
    group_sizze_34591 = get_local_size(0);
    wave_sizze_34590 = LOCKSTEP_WIDTH;
    group_id_32899 = get_group_id(0);
    
    int32_t gtid_32888;
    int32_t gtid_32889;
    
    gtid_32888 = squot32(global_tid_32897, sizze_32279);
    gtid_32889 = global_tid_32897 - squot32(global_tid_32897, sizze_32279) *
        sizze_32279;
    
    bool index_primexp_33757;
    int32_t binop_x_33754;
    int32_t cmpop_x_33755;
    bool index_primexp_33756;
    int32_t convop_x_33751;
    float binop_y_33752;
    float index_primexp_33753;
    int32_t x_32903;
    float res_32904;
    
    if (slt32(gtid_32888, res_32302) && slt32(gtid_32889, sizze_32279)) {
        index_primexp_33757 = gtid_32888 == 0;
        binop_x_33754 = 1 + gtid_32888;
        cmpop_x_33755 = smod32(binop_x_33754, 2);
        index_primexp_33756 = cmpop_x_33755 == 0;
        convop_x_33751 = sdiv32(binop_x_33754, 2);
        binop_y_33752 = sitofp_i32_f32(convop_x_33751);
        index_primexp_33753 = 6.2831855F * binop_y_33752;
        x_32903 = *(__global int32_t *) &mappingindices_mem_34348[gtid_32889 *
                                                                  4];
        if (index_primexp_33757) {
            res_32904 = 1.0F;
        } else {
            float res_32905;
            float x_32906;
            float res_32907;
            float res_32908;
            
            res_32905 = sitofp_i32_f32(x_32903);
            x_32906 = res_32905 * index_primexp_33753;
            res_32907 = x_32906 / freq_32285;
            if (index_primexp_33756) {
                float res_32909;
                
                res_32909 = futrts_sin32(res_32907);
                res_32908 = res_32909;
            } else {
                float res_32910;
                
                res_32910 = futrts_cos32(res_32907);
                res_32908 = res_32910;
            }
            res_32904 = res_32908;
        }
    }
    if (slt32(gtid_32888, res_32302) && slt32(gtid_32889, sizze_32279)) {
        *(__global float *) &mem_34357[(gtid_32888 * sizze_32279 + gtid_32889) *
                                       4] = res_32904;
    }
}
__kernel void map_32938(int32_t sizze_32279, int32_t res_32302, float res_32375,
                        __global unsigned char *mem_34362, __global
                        unsigned char *mem_34366)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_32938;
    int32_t local_tid_32939;
    int32_t group_sizze_34593;
    int32_t wave_sizze_34592;
    int32_t group_id_32940;
    
    global_tid_32938 = get_global_id(0);
    local_tid_32939 = get_local_id(0);
    group_sizze_34593 = get_local_size(0);
    wave_sizze_34592 = LOCKSTEP_WIDTH;
    group_id_32940 = get_group_id(0);
    
    int32_t gtid_32929;
    int32_t gtid_32930;
    
    gtid_32929 = squot32(global_tid_32938, res_32302);
    gtid_32930 = global_tid_32938 - squot32(global_tid_32938, res_32302) *
        res_32302;
    
    float x_32941;
    float res_32942;
    
    if (slt32(gtid_32929, sizze_32279) && slt32(gtid_32930, res_32302)) {
        x_32941 = *(__global float *) &mem_34362[(gtid_32929 * res_32302 +
                                                  gtid_32930) * 4];
        res_32942 = res_32375 + x_32941;
    }
    if (slt32(gtid_32929, sizze_32279) && slt32(gtid_32930, res_32302)) {
        *(__global float *) &mem_34366[(gtid_32929 * res_32302 + gtid_32930) *
                                       4] = res_32942;
    }
}
__kernel void map_32967(__local volatile int64_t *mem_34378_backing_aligned_0,
                        int32_t sizze_32280, int32_t n_32284, int32_t res_32302,
                        int32_t gidzz_range_33776, int32_t tile_sizze_x_33780,
                        int32_t tiled_group_sizze_33782, __global
                        unsigned char *mem_34362, __global
                        unsigned char *mem_34366, __global
                        unsigned char *mem_34371, __global
                        unsigned char *mem_34375)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict mem_34378_backing_0 =
                          mem_34378_backing_aligned_0;
    int32_t global_tid_32967;
    int32_t local_tid_32968;
    int32_t group_sizze_34595;
    int32_t wave_sizze_34594;
    int32_t group_id_32969;
    
    global_tid_32967 = get_global_id(0);
    local_tid_32968 = get_local_id(0);
    group_sizze_34595 = get_local_size(0);
    wave_sizze_34594 = LOCKSTEP_WIDTH;
    group_id_32969 = get_group_id(0);
    
    int32_t gtid_32956;
    int32_t gtid_32957;
    int32_t gtid_32958;
    int32_t ltid_33783;
    int32_t ltid_33784;
    int32_t ltid_33785;
    
    gtid_32956 = squot32(srem32(global_tid_32967, tile_sizze_x_33780 *
                                tile_sizze_x_33780), tile_sizze_x_33780 *
                         tile_sizze_x_33780) + squot32(squot32(global_tid_32967,
                                                               tile_sizze_x_33780 *
                                                               tile_sizze_x_33780),
                                                       squot32(res_32302 +
                                                               tile_sizze_x_33780 -
                                                               1,
                                                               tile_sizze_x_33780) *
                                                       squot32(res_32302 +
                                                               tile_sizze_x_33780 -
                                                               1,
                                                               tile_sizze_x_33780));
    gtid_32957 = squot32(srem32(global_tid_32967, tile_sizze_x_33780 *
                                tile_sizze_x_33780) -
                         squot32(srem32(global_tid_32967, tile_sizze_x_33780 *
                                        tile_sizze_x_33780),
                                 tile_sizze_x_33780 * tile_sizze_x_33780) *
                         (tile_sizze_x_33780 * tile_sizze_x_33780),
                         tile_sizze_x_33780) + squot32(squot32(global_tid_32967,
                                                               tile_sizze_x_33780 *
                                                               tile_sizze_x_33780) -
                                                       squot32(squot32(global_tid_32967,
                                                                       tile_sizze_x_33780 *
                                                                       tile_sizze_x_33780),
                                                               squot32(res_32302 +
                                                                       tile_sizze_x_33780 -
                                                                       1,
                                                                       tile_sizze_x_33780) *
                                                               squot32(res_32302 +
                                                                       tile_sizze_x_33780 -
                                                                       1,
                                                                       tile_sizze_x_33780)) *
                                                       (squot32(res_32302 +
                                                                tile_sizze_x_33780 -
                                                                1,
                                                                tile_sizze_x_33780) *
                                                        squot32(res_32302 +
                                                                tile_sizze_x_33780 -
                                                                1,
                                                                tile_sizze_x_33780)),
                                                       squot32(res_32302 +
                                                               tile_sizze_x_33780 -
                                                               1,
                                                               tile_sizze_x_33780)) *
        tile_sizze_x_33780;
    gtid_32958 = srem32(global_tid_32967, tile_sizze_x_33780 *
                        tile_sizze_x_33780) - squot32(srem32(global_tid_32967,
                                                             tile_sizze_x_33780 *
                                                             tile_sizze_x_33780),
                                                      tile_sizze_x_33780 *
                                                      tile_sizze_x_33780) *
        (tile_sizze_x_33780 * tile_sizze_x_33780) -
        squot32(srem32(global_tid_32967, tile_sizze_x_33780 *
                       tile_sizze_x_33780) - squot32(srem32(global_tid_32967,
                                                            tile_sizze_x_33780 *
                                                            tile_sizze_x_33780),
                                                     tile_sizze_x_33780 *
                                                     tile_sizze_x_33780) *
                (tile_sizze_x_33780 * tile_sizze_x_33780), tile_sizze_x_33780) *
        tile_sizze_x_33780 + (squot32(global_tid_32967, tile_sizze_x_33780 *
                                      tile_sizze_x_33780) -
                              squot32(squot32(global_tid_32967,
                                              tile_sizze_x_33780 *
                                              tile_sizze_x_33780),
                                      squot32(res_32302 + tile_sizze_x_33780 -
                                              1, tile_sizze_x_33780) *
                                      squot32(res_32302 + tile_sizze_x_33780 -
                                              1, tile_sizze_x_33780)) *
                              (squot32(res_32302 + tile_sizze_x_33780 - 1,
                                       tile_sizze_x_33780) * squot32(res_32302 +
                                                                     tile_sizze_x_33780 -
                                                                     1,
                                                                     tile_sizze_x_33780)) -
                              squot32(squot32(global_tid_32967,
                                              tile_sizze_x_33780 *
                                              tile_sizze_x_33780) -
                                      squot32(squot32(global_tid_32967,
                                                      tile_sizze_x_33780 *
                                                      tile_sizze_x_33780),
                                              squot32(res_32302 +
                                                      tile_sizze_x_33780 - 1,
                                                      tile_sizze_x_33780) *
                                              squot32(res_32302 +
                                                      tile_sizze_x_33780 - 1,
                                                      tile_sizze_x_33780)) *
                                      (squot32(res_32302 + tile_sizze_x_33780 -
                                               1, tile_sizze_x_33780) *
                                       squot32(res_32302 + tile_sizze_x_33780 -
                                               1, tile_sizze_x_33780)),
                                      squot32(res_32302 + tile_sizze_x_33780 -
                                              1, tile_sizze_x_33780)) *
                              squot32(res_32302 + tile_sizze_x_33780 - 1,
                                      tile_sizze_x_33780)) * tile_sizze_x_33780;
    ltid_33783 = squot32(srem32(global_tid_32967, tile_sizze_x_33780 *
                                tile_sizze_x_33780), tile_sizze_x_33780 *
                         tile_sizze_x_33780);
    ltid_33784 = squot32(srem32(global_tid_32967, tile_sizze_x_33780 *
                                tile_sizze_x_33780) -
                         squot32(srem32(global_tid_32967, tile_sizze_x_33780 *
                                        tile_sizze_x_33780),
                                 tile_sizze_x_33780 * tile_sizze_x_33780) *
                         (tile_sizze_x_33780 * tile_sizze_x_33780),
                         tile_sizze_x_33780);
    ltid_33785 = srem32(global_tid_32967, tile_sizze_x_33780 *
                        tile_sizze_x_33780) - squot32(srem32(global_tid_32967,
                                                             tile_sizze_x_33780 *
                                                             tile_sizze_x_33780),
                                                      tile_sizze_x_33780 *
                                                      tile_sizze_x_33780) *
        (tile_sizze_x_33780 * tile_sizze_x_33780) -
        squot32(srem32(global_tid_32967, tile_sizze_x_33780 *
                       tile_sizze_x_33780) - squot32(srem32(global_tid_32967,
                                                            tile_sizze_x_33780 *
                                                            tile_sizze_x_33780),
                                                     tile_sizze_x_33780 *
                                                     tile_sizze_x_33780) *
                (tile_sizze_x_33780 * tile_sizze_x_33780), tile_sizze_x_33780) *
        tile_sizze_x_33780;
    
    int32_t mm_33773;
    int32_t m_33803;
    bool is_active_34310;
    bool is_active_34311;
    bool active_34313;
    
    if ((slt32(gtid_32956, gidzz_range_33776) && slt32(gtid_32957,
                                                       res_32302)) &&
        slt32(gtid_32958, res_32302)) {
        mm_33773 = 30 * gtid_32956;
        m_33803 = local_tid_32968 + mm_33773;
        is_active_34310 = slt32(local_tid_32968, 30);
        is_active_34311 = slt32(m_33803, sizze_32280);
        active_34313 = is_active_34310 && is_active_34311;
    }
    
    __local char *mem_34378;
    
    mem_34378 = (__local char *) mem_34378_backing_0;
    
    float res_34139;
    float res_34140;
    float res_34141;
    float res_34142;
    float res_34143;
    float res_34144;
    float res_34145;
    float res_34146;
    float res_34147;
    float res_34148;
    float res_34149;
    float res_34150;
    float res_34151;
    float res_34152;
    float res_34153;
    float res_34154;
    float res_34155;
    float res_34156;
    float res_34157;
    float res_34158;
    float res_34159;
    float res_34160;
    float res_34161;
    float res_34162;
    float res_34163;
    float res_34164;
    float res_34165;
    float res_34166;
    float res_34167;
    float res_34168;
    int32_t m_34174;
    int32_t m_34177;
    int32_t m_34180;
    int32_t m_34183;
    int32_t m_34186;
    int32_t m_34189;
    int32_t m_34192;
    int32_t m_34195;
    int32_t m_34198;
    int32_t m_34201;
    int32_t m_34204;
    int32_t m_34207;
    int32_t m_34210;
    int32_t m_34213;
    int32_t m_34216;
    int32_t m_34219;
    int32_t m_34222;
    int32_t m_34225;
    int32_t m_34228;
    int32_t m_34231;
    int32_t m_34234;
    int32_t m_34237;
    int32_t m_34240;
    int32_t m_34243;
    int32_t m_34246;
    int32_t m_34249;
    int32_t m_34252;
    int32_t m_34255;
    int32_t m_34258;
    
    if ((slt32(gtid_32956, gidzz_range_33776) && slt32(gtid_32957,
                                                       res_32302)) &&
        slt32(gtid_32958, res_32302)) {
        float acc_clone_33809;
        float acc_clone_33820;
        float acc_clone_33831;
        float acc_clone_33842;
        float acc_clone_33853;
        float acc_clone_33864;
        float acc_clone_33875;
        float acc_clone_33886;
        float acc_clone_33897;
        float acc_clone_33908;
        float acc_clone_33919;
        float acc_clone_33930;
        float acc_clone_33941;
        float acc_clone_33952;
        float acc_clone_33963;
        float acc_clone_33974;
        float acc_clone_33985;
        float acc_clone_33996;
        float acc_clone_34007;
        float acc_clone_34018;
        float acc_clone_34029;
        float acc_clone_34040;
        float acc_clone_34051;
        float acc_clone_34062;
        float acc_clone_34073;
        float acc_clone_34084;
        float acc_clone_34095;
        float acc_clone_34106;
        float acc_clone_34117;
        float acc_clone_34128;
        
        acc_clone_33809 = 0.0F;
        acc_clone_33820 = 0.0F;
        acc_clone_33831 = 0.0F;
        acc_clone_33842 = 0.0F;
        acc_clone_33853 = 0.0F;
        acc_clone_33864 = 0.0F;
        acc_clone_33875 = 0.0F;
        acc_clone_33886 = 0.0F;
        acc_clone_33897 = 0.0F;
        acc_clone_33908 = 0.0F;
        acc_clone_33919 = 0.0F;
        acc_clone_33930 = 0.0F;
        acc_clone_33941 = 0.0F;
        acc_clone_33952 = 0.0F;
        acc_clone_33963 = 0.0F;
        acc_clone_33974 = 0.0F;
        acc_clone_33985 = 0.0F;
        acc_clone_33996 = 0.0F;
        acc_clone_34007 = 0.0F;
        acc_clone_34018 = 0.0F;
        acc_clone_34029 = 0.0F;
        acc_clone_34040 = 0.0F;
        acc_clone_34051 = 0.0F;
        acc_clone_34062 = 0.0F;
        acc_clone_34073 = 0.0F;
        acc_clone_34084 = 0.0F;
        acc_clone_34095 = 0.0F;
        acc_clone_34106 = 0.0F;
        acc_clone_34117 = 0.0F;
        acc_clone_34128 = 0.0F;
        for (int32_t loop_ind_34138 = 0; loop_ind_34138 < n_32284;
             loop_ind_34138++) {
            int32_t i_32982;
            
            i_32982 = loop_ind_34138;
            barrier(CLK_LOCAL_MEM_FENCE);
            
            float x_32988;
            float x_32989;
            float x_32987;
            
            x_32988 = *(__global float *) &mem_34362[(i_32982 * res_32302 +
                                                      gtid_32957) * 4];
            x_32989 = *(__global float *) &mem_34366[(i_32982 * res_32302 +
                                                      gtid_32958) * 4];
            if (active_34313) {
                float x_34314 = *(__global float *) &mem_34375[(i_32982 *
                                                                sizze_32280 +
                                                                m_33803) * 4];
                
                x_32987 = x_34314;
            } else {
                x_32987 = 0.0F;
            }
            for (int32_t comb_iter_34626 = 0; comb_iter_34626 < 1;
                 comb_iter_34626++) {
                int32_t cid_33807;
                int32_t flat_comb_id_34627 = comb_iter_34626 *
                        tiled_group_sizze_33782 + local_tid_32968;
                
                cid_33807 = flat_comb_id_34627;
                if (slt32(cid_33807, tiled_group_sizze_33782) &&
                    (slt32(local_tid_32968, 30) && slt32(m_33803,
                                                         sizze_32280))) {
                    *(__local float *) &mem_34378[cid_33807 * 4] = x_32987;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            float x_33814;
            bool res_33815;
            float res_33816;
            float res_33818;
            float x_33825;
            bool res_33826;
            float res_33827;
            float res_33829;
            float x_33836;
            bool res_33837;
            float res_33838;
            float res_33840;
            float x_33847;
            bool res_33848;
            float res_33849;
            float res_33851;
            float x_33858;
            bool res_33859;
            float res_33860;
            float res_33862;
            float x_33869;
            bool res_33870;
            float res_33871;
            float res_33873;
            float x_33880;
            bool res_33881;
            float res_33882;
            float res_33884;
            float x_33891;
            bool res_33892;
            float res_33893;
            float res_33895;
            float x_33902;
            bool res_33903;
            float res_33904;
            float res_33906;
            float x_33913;
            bool res_33914;
            float res_33915;
            float res_33917;
            float x_33924;
            bool res_33925;
            float res_33926;
            float res_33928;
            float x_33935;
            bool res_33936;
            float res_33937;
            float res_33939;
            float x_33946;
            bool res_33947;
            float res_33948;
            float res_33950;
            float x_33957;
            bool res_33958;
            float res_33959;
            float res_33961;
            float x_33968;
            bool res_33969;
            float res_33970;
            float res_33972;
            float x_33979;
            bool res_33980;
            float res_33981;
            float res_33983;
            float x_33990;
            bool res_33991;
            float res_33992;
            float res_33994;
            float x_34001;
            bool res_34002;
            float res_34003;
            float res_34005;
            float x_34012;
            bool res_34013;
            float res_34014;
            float res_34016;
            float x_34023;
            bool res_34024;
            float res_34025;
            float res_34027;
            float x_34034;
            bool res_34035;
            float res_34036;
            float res_34038;
            float x_34045;
            bool res_34046;
            float res_34047;
            float res_34049;
            float x_34056;
            bool res_34057;
            float res_34058;
            float res_34060;
            float x_34067;
            bool res_34068;
            float res_34069;
            float res_34071;
            float x_34078;
            bool res_34079;
            float res_34080;
            float res_34082;
            float x_34089;
            bool res_34090;
            float res_34091;
            float res_34093;
            float x_34100;
            bool res_34101;
            float res_34102;
            float res_34104;
            float x_34111;
            bool res_34112;
            float res_34113;
            float res_34115;
            float x_34122;
            bool res_34123;
            float res_34124;
            float res_34126;
            float x_34133;
            bool res_34134;
            float res_34135;
            float res_34137;
            
            x_33814 = *(__local float *) &mem_34378[0];
            res_33815 = futrts_isnan32(x_33814);
            if (res_33815) {
                res_33816 = 0.0F;
            } else {
                float res_33817 = x_32988 * x_32989;
                
                res_33816 = res_33817;
            }
            res_33818 = acc_clone_33809 + res_33816;
            x_33825 = *(__local float *) &mem_34378[4];
            res_33826 = futrts_isnan32(x_33825);
            if (res_33826) {
                res_33827 = 0.0F;
            } else {
                float res_33828 = x_32988 * x_32989;
                
                res_33827 = res_33828;
            }
            res_33829 = acc_clone_33820 + res_33827;
            x_33836 = *(__local float *) &mem_34378[8];
            res_33837 = futrts_isnan32(x_33836);
            if (res_33837) {
                res_33838 = 0.0F;
            } else {
                float res_33839 = x_32988 * x_32989;
                
                res_33838 = res_33839;
            }
            res_33840 = acc_clone_33831 + res_33838;
            x_33847 = *(__local float *) &mem_34378[12];
            res_33848 = futrts_isnan32(x_33847);
            if (res_33848) {
                res_33849 = 0.0F;
            } else {
                float res_33850 = x_32988 * x_32989;
                
                res_33849 = res_33850;
            }
            res_33851 = acc_clone_33842 + res_33849;
            x_33858 = *(__local float *) &mem_34378[16];
            res_33859 = futrts_isnan32(x_33858);
            if (res_33859) {
                res_33860 = 0.0F;
            } else {
                float res_33861 = x_32988 * x_32989;
                
                res_33860 = res_33861;
            }
            res_33862 = acc_clone_33853 + res_33860;
            x_33869 = *(__local float *) &mem_34378[20];
            res_33870 = futrts_isnan32(x_33869);
            if (res_33870) {
                res_33871 = 0.0F;
            } else {
                float res_33872 = x_32988 * x_32989;
                
                res_33871 = res_33872;
            }
            res_33873 = acc_clone_33864 + res_33871;
            x_33880 = *(__local float *) &mem_34378[24];
            res_33881 = futrts_isnan32(x_33880);
            if (res_33881) {
                res_33882 = 0.0F;
            } else {
                float res_33883 = x_32988 * x_32989;
                
                res_33882 = res_33883;
            }
            res_33884 = acc_clone_33875 + res_33882;
            x_33891 = *(__local float *) &mem_34378[28];
            res_33892 = futrts_isnan32(x_33891);
            if (res_33892) {
                res_33893 = 0.0F;
            } else {
                float res_33894 = x_32988 * x_32989;
                
                res_33893 = res_33894;
            }
            res_33895 = acc_clone_33886 + res_33893;
            x_33902 = *(__local float *) &mem_34378[32];
            res_33903 = futrts_isnan32(x_33902);
            if (res_33903) {
                res_33904 = 0.0F;
            } else {
                float res_33905 = x_32988 * x_32989;
                
                res_33904 = res_33905;
            }
            res_33906 = acc_clone_33897 + res_33904;
            x_33913 = *(__local float *) &mem_34378[36];
            res_33914 = futrts_isnan32(x_33913);
            if (res_33914) {
                res_33915 = 0.0F;
            } else {
                float res_33916 = x_32988 * x_32989;
                
                res_33915 = res_33916;
            }
            res_33917 = acc_clone_33908 + res_33915;
            x_33924 = *(__local float *) &mem_34378[40];
            res_33925 = futrts_isnan32(x_33924);
            if (res_33925) {
                res_33926 = 0.0F;
            } else {
                float res_33927 = x_32988 * x_32989;
                
                res_33926 = res_33927;
            }
            res_33928 = acc_clone_33919 + res_33926;
            x_33935 = *(__local float *) &mem_34378[44];
            res_33936 = futrts_isnan32(x_33935);
            if (res_33936) {
                res_33937 = 0.0F;
            } else {
                float res_33938 = x_32988 * x_32989;
                
                res_33937 = res_33938;
            }
            res_33939 = acc_clone_33930 + res_33937;
            x_33946 = *(__local float *) &mem_34378[48];
            res_33947 = futrts_isnan32(x_33946);
            if (res_33947) {
                res_33948 = 0.0F;
            } else {
                float res_33949 = x_32988 * x_32989;
                
                res_33948 = res_33949;
            }
            res_33950 = acc_clone_33941 + res_33948;
            x_33957 = *(__local float *) &mem_34378[52];
            res_33958 = futrts_isnan32(x_33957);
            if (res_33958) {
                res_33959 = 0.0F;
            } else {
                float res_33960 = x_32988 * x_32989;
                
                res_33959 = res_33960;
            }
            res_33961 = acc_clone_33952 + res_33959;
            x_33968 = *(__local float *) &mem_34378[56];
            res_33969 = futrts_isnan32(x_33968);
            if (res_33969) {
                res_33970 = 0.0F;
            } else {
                float res_33971 = x_32988 * x_32989;
                
                res_33970 = res_33971;
            }
            res_33972 = acc_clone_33963 + res_33970;
            x_33979 = *(__local float *) &mem_34378[60];
            res_33980 = futrts_isnan32(x_33979);
            if (res_33980) {
                res_33981 = 0.0F;
            } else {
                float res_33982 = x_32988 * x_32989;
                
                res_33981 = res_33982;
            }
            res_33983 = acc_clone_33974 + res_33981;
            x_33990 = *(__local float *) &mem_34378[64];
            res_33991 = futrts_isnan32(x_33990);
            if (res_33991) {
                res_33992 = 0.0F;
            } else {
                float res_33993 = x_32988 * x_32989;
                
                res_33992 = res_33993;
            }
            res_33994 = acc_clone_33985 + res_33992;
            x_34001 = *(__local float *) &mem_34378[68];
            res_34002 = futrts_isnan32(x_34001);
            if (res_34002) {
                res_34003 = 0.0F;
            } else {
                float res_34004 = x_32988 * x_32989;
                
                res_34003 = res_34004;
            }
            res_34005 = acc_clone_33996 + res_34003;
            x_34012 = *(__local float *) &mem_34378[72];
            res_34013 = futrts_isnan32(x_34012);
            if (res_34013) {
                res_34014 = 0.0F;
            } else {
                float res_34015 = x_32988 * x_32989;
                
                res_34014 = res_34015;
            }
            res_34016 = acc_clone_34007 + res_34014;
            x_34023 = *(__local float *) &mem_34378[76];
            res_34024 = futrts_isnan32(x_34023);
            if (res_34024) {
                res_34025 = 0.0F;
            } else {
                float res_34026 = x_32988 * x_32989;
                
                res_34025 = res_34026;
            }
            res_34027 = acc_clone_34018 + res_34025;
            x_34034 = *(__local float *) &mem_34378[80];
            res_34035 = futrts_isnan32(x_34034);
            if (res_34035) {
                res_34036 = 0.0F;
            } else {
                float res_34037 = x_32988 * x_32989;
                
                res_34036 = res_34037;
            }
            res_34038 = acc_clone_34029 + res_34036;
            x_34045 = *(__local float *) &mem_34378[84];
            res_34046 = futrts_isnan32(x_34045);
            if (res_34046) {
                res_34047 = 0.0F;
            } else {
                float res_34048 = x_32988 * x_32989;
                
                res_34047 = res_34048;
            }
            res_34049 = acc_clone_34040 + res_34047;
            x_34056 = *(__local float *) &mem_34378[88];
            res_34057 = futrts_isnan32(x_34056);
            if (res_34057) {
                res_34058 = 0.0F;
            } else {
                float res_34059 = x_32988 * x_32989;
                
                res_34058 = res_34059;
            }
            res_34060 = acc_clone_34051 + res_34058;
            x_34067 = *(__local float *) &mem_34378[92];
            res_34068 = futrts_isnan32(x_34067);
            if (res_34068) {
                res_34069 = 0.0F;
            } else {
                float res_34070 = x_32988 * x_32989;
                
                res_34069 = res_34070;
            }
            res_34071 = acc_clone_34062 + res_34069;
            x_34078 = *(__local float *) &mem_34378[96];
            res_34079 = futrts_isnan32(x_34078);
            if (res_34079) {
                res_34080 = 0.0F;
            } else {
                float res_34081 = x_32988 * x_32989;
                
                res_34080 = res_34081;
            }
            res_34082 = acc_clone_34073 + res_34080;
            x_34089 = *(__local float *) &mem_34378[100];
            res_34090 = futrts_isnan32(x_34089);
            if (res_34090) {
                res_34091 = 0.0F;
            } else {
                float res_34092 = x_32988 * x_32989;
                
                res_34091 = res_34092;
            }
            res_34093 = acc_clone_34084 + res_34091;
            x_34100 = *(__local float *) &mem_34378[104];
            res_34101 = futrts_isnan32(x_34100);
            if (res_34101) {
                res_34102 = 0.0F;
            } else {
                float res_34103 = x_32988 * x_32989;
                
                res_34102 = res_34103;
            }
            res_34104 = acc_clone_34095 + res_34102;
            x_34111 = *(__local float *) &mem_34378[108];
            res_34112 = futrts_isnan32(x_34111);
            if (res_34112) {
                res_34113 = 0.0F;
            } else {
                float res_34114 = x_32988 * x_32989;
                
                res_34113 = res_34114;
            }
            res_34115 = acc_clone_34106 + res_34113;
            x_34122 = *(__local float *) &mem_34378[112];
            res_34123 = futrts_isnan32(x_34122);
            if (res_34123) {
                res_34124 = 0.0F;
            } else {
                float res_34125 = x_32988 * x_32989;
                
                res_34124 = res_34125;
            }
            res_34126 = acc_clone_34117 + res_34124;
            x_34133 = *(__local float *) &mem_34378[116];
            res_34134 = futrts_isnan32(x_34133);
            if (res_34134) {
                res_34135 = 0.0F;
            } else {
                float res_34136 = x_32988 * x_32989;
                
                res_34135 = res_34136;
            }
            res_34137 = acc_clone_34128 + res_34135;
            
            float acc_clone_tmp_34596 = res_33818;
            float acc_clone_tmp_34597 = res_33829;
            float acc_clone_tmp_34598 = res_33840;
            float acc_clone_tmp_34599 = res_33851;
            float acc_clone_tmp_34600 = res_33862;
            float acc_clone_tmp_34601 = res_33873;
            float acc_clone_tmp_34602 = res_33884;
            float acc_clone_tmp_34603 = res_33895;
            float acc_clone_tmp_34604 = res_33906;
            float acc_clone_tmp_34605 = res_33917;
            float acc_clone_tmp_34606 = res_33928;
            float acc_clone_tmp_34607 = res_33939;
            float acc_clone_tmp_34608 = res_33950;
            float acc_clone_tmp_34609 = res_33961;
            float acc_clone_tmp_34610 = res_33972;
            float acc_clone_tmp_34611 = res_33983;
            float acc_clone_tmp_34612 = res_33994;
            float acc_clone_tmp_34613 = res_34005;
            float acc_clone_tmp_34614 = res_34016;
            float acc_clone_tmp_34615 = res_34027;
            float acc_clone_tmp_34616 = res_34038;
            float acc_clone_tmp_34617 = res_34049;
            float acc_clone_tmp_34618 = res_34060;
            float acc_clone_tmp_34619 = res_34071;
            float acc_clone_tmp_34620 = res_34082;
            float acc_clone_tmp_34621 = res_34093;
            float acc_clone_tmp_34622 = res_34104;
            float acc_clone_tmp_34623 = res_34115;
            float acc_clone_tmp_34624 = res_34126;
            float acc_clone_tmp_34625;
            
            acc_clone_tmp_34625 = res_34137;
            acc_clone_33809 = acc_clone_tmp_34596;
            acc_clone_33820 = acc_clone_tmp_34597;
            acc_clone_33831 = acc_clone_tmp_34598;
            acc_clone_33842 = acc_clone_tmp_34599;
            acc_clone_33853 = acc_clone_tmp_34600;
            acc_clone_33864 = acc_clone_tmp_34601;
            acc_clone_33875 = acc_clone_tmp_34602;
            acc_clone_33886 = acc_clone_tmp_34603;
            acc_clone_33897 = acc_clone_tmp_34604;
            acc_clone_33908 = acc_clone_tmp_34605;
            acc_clone_33919 = acc_clone_tmp_34606;
            acc_clone_33930 = acc_clone_tmp_34607;
            acc_clone_33941 = acc_clone_tmp_34608;
            acc_clone_33952 = acc_clone_tmp_34609;
            acc_clone_33963 = acc_clone_tmp_34610;
            acc_clone_33974 = acc_clone_tmp_34611;
            acc_clone_33985 = acc_clone_tmp_34612;
            acc_clone_33996 = acc_clone_tmp_34613;
            acc_clone_34007 = acc_clone_tmp_34614;
            acc_clone_34018 = acc_clone_tmp_34615;
            acc_clone_34029 = acc_clone_tmp_34616;
            acc_clone_34040 = acc_clone_tmp_34617;
            acc_clone_34051 = acc_clone_tmp_34618;
            acc_clone_34062 = acc_clone_tmp_34619;
            acc_clone_34073 = acc_clone_tmp_34620;
            acc_clone_34084 = acc_clone_tmp_34621;
            acc_clone_34095 = acc_clone_tmp_34622;
            acc_clone_34106 = acc_clone_tmp_34623;
            acc_clone_34117 = acc_clone_tmp_34624;
            acc_clone_34128 = acc_clone_tmp_34625;
        }
        res_34139 = acc_clone_33809;
        res_34140 = acc_clone_33820;
        res_34141 = acc_clone_33831;
        res_34142 = acc_clone_33842;
        res_34143 = acc_clone_33853;
        res_34144 = acc_clone_33864;
        res_34145 = acc_clone_33875;
        res_34146 = acc_clone_33886;
        res_34147 = acc_clone_33897;
        res_34148 = acc_clone_33908;
        res_34149 = acc_clone_33919;
        res_34150 = acc_clone_33930;
        res_34151 = acc_clone_33941;
        res_34152 = acc_clone_33952;
        res_34153 = acc_clone_33963;
        res_34154 = acc_clone_33974;
        res_34155 = acc_clone_33985;
        res_34156 = acc_clone_33996;
        res_34157 = acc_clone_34007;
        res_34158 = acc_clone_34018;
        res_34159 = acc_clone_34029;
        res_34160 = acc_clone_34040;
        res_34161 = acc_clone_34051;
        res_34162 = acc_clone_34062;
        res_34163 = acc_clone_34073;
        res_34164 = acc_clone_34084;
        res_34165 = acc_clone_34095;
        res_34166 = acc_clone_34106;
        res_34167 = acc_clone_34117;
        res_34168 = acc_clone_34128;
        m_34174 = 1 + mm_33773;
        m_34177 = 2 + mm_33773;
        m_34180 = 3 + mm_33773;
        m_34183 = 4 + mm_33773;
        m_34186 = 5 + mm_33773;
        m_34189 = 6 + mm_33773;
        m_34192 = 7 + mm_33773;
        m_34195 = 8 + mm_33773;
        m_34198 = 9 + mm_33773;
        m_34201 = 10 + mm_33773;
        m_34204 = 11 + mm_33773;
        m_34207 = 12 + mm_33773;
        m_34210 = 13 + mm_33773;
        m_34213 = 14 + mm_33773;
        m_34216 = 15 + mm_33773;
        m_34219 = 16 + mm_33773;
        m_34222 = 17 + mm_33773;
        m_34225 = 18 + mm_33773;
        m_34228 = 19 + mm_33773;
        m_34231 = 20 + mm_33773;
        m_34234 = 21 + mm_33773;
        m_34237 = 22 + mm_33773;
        m_34240 = 23 + mm_33773;
        m_34243 = 24 + mm_33773;
        m_34246 = 25 + mm_33773;
        m_34249 = 26 + mm_33773;
        m_34252 = 27 + mm_33773;
        m_34255 = 28 + mm_33773;
        m_34258 = 29 + mm_33773;
    }
    if (((((slt32(gtid_32956, gidzz_range_33776) && slt32(gtid_32957,
                                                          res_32302)) &&
           slt32(gtid_32958, res_32302)) && (sle32(0, mm_33773) &&
                                             slt32(mm_33773, sizze_32280))) &&
         (sle32(0, gtid_32957) && slt32(gtid_32957, res_32302))) && (sle32(0,
                                                                           gtid_32958) &&
                                                                     slt32(gtid_32958,
                                                                           res_32302))) {
        *(__global float *) &mem_34371[(mm_33773 * (res_32302 * res_32302) +
                                        gtid_32957 * res_32302 + gtid_32958) *
                                       4] = res_34139;
    }
    if (((((slt32(gtid_32956, gidzz_range_33776) && slt32(gtid_32957,
                                                          res_32302)) &&
           slt32(gtid_32958, res_32302)) && (sle32(0, m_34174) && slt32(m_34174,
                                                                        sizze_32280))) &&
         (sle32(0, gtid_32957) && slt32(gtid_32957, res_32302))) && (sle32(0,
                                                                           gtid_32958) &&
                                                                     slt32(gtid_32958,
                                                                           res_32302))) {
        *(__global float *) &mem_34371[(m_34174 * (res_32302 * res_32302) +
                                        gtid_32957 * res_32302 + gtid_32958) *
                                       4] = res_34140;
    }
    if (((((slt32(gtid_32956, gidzz_range_33776) && slt32(gtid_32957,
                                                          res_32302)) &&
           slt32(gtid_32958, res_32302)) && (sle32(0, m_34177) && slt32(m_34177,
                                                                        sizze_32280))) &&
         (sle32(0, gtid_32957) && slt32(gtid_32957, res_32302))) && (sle32(0,
                                                                           gtid_32958) &&
                                                                     slt32(gtid_32958,
                                                                           res_32302))) {
        *(__global float *) &mem_34371[(m_34177 * (res_32302 * res_32302) +
                                        gtid_32957 * res_32302 + gtid_32958) *
                                       4] = res_34141;
    }
    if (((((slt32(gtid_32956, gidzz_range_33776) && slt32(gtid_32957,
                                                          res_32302)) &&
           slt32(gtid_32958, res_32302)) && (sle32(0, m_34180) && slt32(m_34180,
                                                                        sizze_32280))) &&
         (sle32(0, gtid_32957) && slt32(gtid_32957, res_32302))) && (sle32(0,
                                                                           gtid_32958) &&
                                                                     slt32(gtid_32958,
                                                                           res_32302))) {
        *(__global float *) &mem_34371[(m_34180 * (res_32302 * res_32302) +
                                        gtid_32957 * res_32302 + gtid_32958) *
                                       4] = res_34142;
    }
    if (((((slt32(gtid_32956, gidzz_range_33776) && slt32(gtid_32957,
                                                          res_32302)) &&
           slt32(gtid_32958, res_32302)) && (sle32(0, m_34183) && slt32(m_34183,
                                                                        sizze_32280))) &&
         (sle32(0, gtid_32957) && slt32(gtid_32957, res_32302))) && (sle32(0,
                                                                           gtid_32958) &&
                                                                     slt32(gtid_32958,
                                                                           res_32302))) {
        *(__global float *) &mem_34371[(m_34183 * (res_32302 * res_32302) +
                                        gtid_32957 * res_32302 + gtid_32958) *
                                       4] = res_34143;
    }
    if (((((slt32(gtid_32956, gidzz_range_33776) && slt32(gtid_32957,
                                                          res_32302)) &&
           slt32(gtid_32958, res_32302)) && (sle32(0, m_34186) && slt32(m_34186,
                                                                        sizze_32280))) &&
         (sle32(0, gtid_32957) && slt32(gtid_32957, res_32302))) && (sle32(0,
                                                                           gtid_32958) &&
                                                                     slt32(gtid_32958,
                                                                           res_32302))) {
        *(__global float *) &mem_34371[(m_34186 * (res_32302 * res_32302) +
                                        gtid_32957 * res_32302 + gtid_32958) *
                                       4] = res_34144;
    }
    if (((((slt32(gtid_32956, gidzz_range_33776) && slt32(gtid_32957,
                                                          res_32302)) &&
           slt32(gtid_32958, res_32302)) && (sle32(0, m_34189) && slt32(m_34189,
                                                                        sizze_32280))) &&
         (sle32(0, gtid_32957) && slt32(gtid_32957, res_32302))) && (sle32(0,
                                                                           gtid_32958) &&
                                                                     slt32(gtid_32958,
                                                                           res_32302))) {
        *(__global float *) &mem_34371[(m_34189 * (res_32302 * res_32302) +
                                        gtid_32957 * res_32302 + gtid_32958) *
                                       4] = res_34145;
    }
    if (((((slt32(gtid_32956, gidzz_range_33776) && slt32(gtid_32957,
                                                          res_32302)) &&
           slt32(gtid_32958, res_32302)) && (sle32(0, m_34192) && slt32(m_34192,
                                                                        sizze_32280))) &&
         (sle32(0, gtid_32957) && slt32(gtid_32957, res_32302))) && (sle32(0,
                                                                           gtid_32958) &&
                                                                     slt32(gtid_32958,
                                                                           res_32302))) {
        *(__global float *) &mem_34371[(m_34192 * (res_32302 * res_32302) +
                                        gtid_32957 * res_32302 + gtid_32958) *
                                       4] = res_34146;
    }
    if (((((slt32(gtid_32956, gidzz_range_33776) && slt32(gtid_32957,
                                                          res_32302)) &&
           slt32(gtid_32958, res_32302)) && (sle32(0, m_34195) && slt32(m_34195,
                                                                        sizze_32280))) &&
         (sle32(0, gtid_32957) && slt32(gtid_32957, res_32302))) && (sle32(0,
                                                                           gtid_32958) &&
                                                                     slt32(gtid_32958,
                                                                           res_32302))) {
        *(__global float *) &mem_34371[(m_34195 * (res_32302 * res_32302) +
                                        gtid_32957 * res_32302 + gtid_32958) *
                                       4] = res_34147;
    }
    if (((((slt32(gtid_32956, gidzz_range_33776) && slt32(gtid_32957,
                                                          res_32302)) &&
           slt32(gtid_32958, res_32302)) && (sle32(0, m_34198) && slt32(m_34198,
                                                                        sizze_32280))) &&
         (sle32(0, gtid_32957) && slt32(gtid_32957, res_32302))) && (sle32(0,
                                                                           gtid_32958) &&
                                                                     slt32(gtid_32958,
                                                                           res_32302))) {
        *(__global float *) &mem_34371[(m_34198 * (res_32302 * res_32302) +
                                        gtid_32957 * res_32302 + gtid_32958) *
                                       4] = res_34148;
    }
    if (((((slt32(gtid_32956, gidzz_range_33776) && slt32(gtid_32957,
                                                          res_32302)) &&
           slt32(gtid_32958, res_32302)) && (sle32(0, m_34201) && slt32(m_34201,
                                                                        sizze_32280))) &&
         (sle32(0, gtid_32957) && slt32(gtid_32957, res_32302))) && (sle32(0,
                                                                           gtid_32958) &&
                                                                     slt32(gtid_32958,
                                                                           res_32302))) {
        *(__global float *) &mem_34371[(m_34201 * (res_32302 * res_32302) +
                                        gtid_32957 * res_32302 + gtid_32958) *
                                       4] = res_34149;
    }
    if (((((slt32(gtid_32956, gidzz_range_33776) && slt32(gtid_32957,
                                                          res_32302)) &&
           slt32(gtid_32958, res_32302)) && (sle32(0, m_34204) && slt32(m_34204,
                                                                        sizze_32280))) &&
         (sle32(0, gtid_32957) && slt32(gtid_32957, res_32302))) && (sle32(0,
                                                                           gtid_32958) &&
                                                                     slt32(gtid_32958,
                                                                           res_32302))) {
        *(__global float *) &mem_34371[(m_34204 * (res_32302 * res_32302) +
                                        gtid_32957 * res_32302 + gtid_32958) *
                                       4] = res_34150;
    }
    if (((((slt32(gtid_32956, gidzz_range_33776) && slt32(gtid_32957,
                                                          res_32302)) &&
           slt32(gtid_32958, res_32302)) && (sle32(0, m_34207) && slt32(m_34207,
                                                                        sizze_32280))) &&
         (sle32(0, gtid_32957) && slt32(gtid_32957, res_32302))) && (sle32(0,
                                                                           gtid_32958) &&
                                                                     slt32(gtid_32958,
                                                                           res_32302))) {
        *(__global float *) &mem_34371[(m_34207 * (res_32302 * res_32302) +
                                        gtid_32957 * res_32302 + gtid_32958) *
                                       4] = res_34151;
    }
    if (((((slt32(gtid_32956, gidzz_range_33776) && slt32(gtid_32957,
                                                          res_32302)) &&
           slt32(gtid_32958, res_32302)) && (sle32(0, m_34210) && slt32(m_34210,
                                                                        sizze_32280))) &&
         (sle32(0, gtid_32957) && slt32(gtid_32957, res_32302))) && (sle32(0,
                                                                           gtid_32958) &&
                                                                     slt32(gtid_32958,
                                                                           res_32302))) {
        *(__global float *) &mem_34371[(m_34210 * (res_32302 * res_32302) +
                                        gtid_32957 * res_32302 + gtid_32958) *
                                       4] = res_34152;
    }
    if (((((slt32(gtid_32956, gidzz_range_33776) && slt32(gtid_32957,
                                                          res_32302)) &&
           slt32(gtid_32958, res_32302)) && (sle32(0, m_34213) && slt32(m_34213,
                                                                        sizze_32280))) &&
         (sle32(0, gtid_32957) && slt32(gtid_32957, res_32302))) && (sle32(0,
                                                                           gtid_32958) &&
                                                                     slt32(gtid_32958,
                                                                           res_32302))) {
        *(__global float *) &mem_34371[(m_34213 * (res_32302 * res_32302) +
                                        gtid_32957 * res_32302 + gtid_32958) *
                                       4] = res_34153;
    }
    if (((((slt32(gtid_32956, gidzz_range_33776) && slt32(gtid_32957,
                                                          res_32302)) &&
           slt32(gtid_32958, res_32302)) && (sle32(0, m_34216) && slt32(m_34216,
                                                                        sizze_32280))) &&
         (sle32(0, gtid_32957) && slt32(gtid_32957, res_32302))) && (sle32(0,
                                                                           gtid_32958) &&
                                                                     slt32(gtid_32958,
                                                                           res_32302))) {
        *(__global float *) &mem_34371[(m_34216 * (res_32302 * res_32302) +
                                        gtid_32957 * res_32302 + gtid_32958) *
                                       4] = res_34154;
    }
    if (((((slt32(gtid_32956, gidzz_range_33776) && slt32(gtid_32957,
                                                          res_32302)) &&
           slt32(gtid_32958, res_32302)) && (sle32(0, m_34219) && slt32(m_34219,
                                                                        sizze_32280))) &&
         (sle32(0, gtid_32957) && slt32(gtid_32957, res_32302))) && (sle32(0,
                                                                           gtid_32958) &&
                                                                     slt32(gtid_32958,
                                                                           res_32302))) {
        *(__global float *) &mem_34371[(m_34219 * (res_32302 * res_32302) +
                                        gtid_32957 * res_32302 + gtid_32958) *
                                       4] = res_34155;
    }
    if (((((slt32(gtid_32956, gidzz_range_33776) && slt32(gtid_32957,
                                                          res_32302)) &&
           slt32(gtid_32958, res_32302)) && (sle32(0, m_34222) && slt32(m_34222,
                                                                        sizze_32280))) &&
         (sle32(0, gtid_32957) && slt32(gtid_32957, res_32302))) && (sle32(0,
                                                                           gtid_32958) &&
                                                                     slt32(gtid_32958,
                                                                           res_32302))) {
        *(__global float *) &mem_34371[(m_34222 * (res_32302 * res_32302) +
                                        gtid_32957 * res_32302 + gtid_32958) *
                                       4] = res_34156;
    }
    if (((((slt32(gtid_32956, gidzz_range_33776) && slt32(gtid_32957,
                                                          res_32302)) &&
           slt32(gtid_32958, res_32302)) && (sle32(0, m_34225) && slt32(m_34225,
                                                                        sizze_32280))) &&
         (sle32(0, gtid_32957) && slt32(gtid_32957, res_32302))) && (sle32(0,
                                                                           gtid_32958) &&
                                                                     slt32(gtid_32958,
                                                                           res_32302))) {
        *(__global float *) &mem_34371[(m_34225 * (res_32302 * res_32302) +
                                        gtid_32957 * res_32302 + gtid_32958) *
                                       4] = res_34157;
    }
    if (((((slt32(gtid_32956, gidzz_range_33776) && slt32(gtid_32957,
                                                          res_32302)) &&
           slt32(gtid_32958, res_32302)) && (sle32(0, m_34228) && slt32(m_34228,
                                                                        sizze_32280))) &&
         (sle32(0, gtid_32957) && slt32(gtid_32957, res_32302))) && (sle32(0,
                                                                           gtid_32958) &&
                                                                     slt32(gtid_32958,
                                                                           res_32302))) {
        *(__global float *) &mem_34371[(m_34228 * (res_32302 * res_32302) +
                                        gtid_32957 * res_32302 + gtid_32958) *
                                       4] = res_34158;
    }
    if (((((slt32(gtid_32956, gidzz_range_33776) && slt32(gtid_32957,
                                                          res_32302)) &&
           slt32(gtid_32958, res_32302)) && (sle32(0, m_34231) && slt32(m_34231,
                                                                        sizze_32280))) &&
         (sle32(0, gtid_32957) && slt32(gtid_32957, res_32302))) && (sle32(0,
                                                                           gtid_32958) &&
                                                                     slt32(gtid_32958,
                                                                           res_32302))) {
        *(__global float *) &mem_34371[(m_34231 * (res_32302 * res_32302) +
                                        gtid_32957 * res_32302 + gtid_32958) *
                                       4] = res_34159;
    }
    if (((((slt32(gtid_32956, gidzz_range_33776) && slt32(gtid_32957,
                                                          res_32302)) &&
           slt32(gtid_32958, res_32302)) && (sle32(0, m_34234) && slt32(m_34234,
                                                                        sizze_32280))) &&
         (sle32(0, gtid_32957) && slt32(gtid_32957, res_32302))) && (sle32(0,
                                                                           gtid_32958) &&
                                                                     slt32(gtid_32958,
                                                                           res_32302))) {
        *(__global float *) &mem_34371[(m_34234 * (res_32302 * res_32302) +
                                        gtid_32957 * res_32302 + gtid_32958) *
                                       4] = res_34160;
    }
    if (((((slt32(gtid_32956, gidzz_range_33776) && slt32(gtid_32957,
                                                          res_32302)) &&
           slt32(gtid_32958, res_32302)) && (sle32(0, m_34237) && slt32(m_34237,
                                                                        sizze_32280))) &&
         (sle32(0, gtid_32957) && slt32(gtid_32957, res_32302))) && (sle32(0,
                                                                           gtid_32958) &&
                                                                     slt32(gtid_32958,
                                                                           res_32302))) {
        *(__global float *) &mem_34371[(m_34237 * (res_32302 * res_32302) +
                                        gtid_32957 * res_32302 + gtid_32958) *
                                       4] = res_34161;
    }
    if (((((slt32(gtid_32956, gidzz_range_33776) && slt32(gtid_32957,
                                                          res_32302)) &&
           slt32(gtid_32958, res_32302)) && (sle32(0, m_34240) && slt32(m_34240,
                                                                        sizze_32280))) &&
         (sle32(0, gtid_32957) && slt32(gtid_32957, res_32302))) && (sle32(0,
                                                                           gtid_32958) &&
                                                                     slt32(gtid_32958,
                                                                           res_32302))) {
        *(__global float *) &mem_34371[(m_34240 * (res_32302 * res_32302) +
                                        gtid_32957 * res_32302 + gtid_32958) *
                                       4] = res_34162;
    }
    if (((((slt32(gtid_32956, gidzz_range_33776) && slt32(gtid_32957,
                                                          res_32302)) &&
           slt32(gtid_32958, res_32302)) && (sle32(0, m_34243) && slt32(m_34243,
                                                                        sizze_32280))) &&
         (sle32(0, gtid_32957) && slt32(gtid_32957, res_32302))) && (sle32(0,
                                                                           gtid_32958) &&
                                                                     slt32(gtid_32958,
                                                                           res_32302))) {
        *(__global float *) &mem_34371[(m_34243 * (res_32302 * res_32302) +
                                        gtid_32957 * res_32302 + gtid_32958) *
                                       4] = res_34163;
    }
    if (((((slt32(gtid_32956, gidzz_range_33776) && slt32(gtid_32957,
                                                          res_32302)) &&
           slt32(gtid_32958, res_32302)) && (sle32(0, m_34246) && slt32(m_34246,
                                                                        sizze_32280))) &&
         (sle32(0, gtid_32957) && slt32(gtid_32957, res_32302))) && (sle32(0,
                                                                           gtid_32958) &&
                                                                     slt32(gtid_32958,
                                                                           res_32302))) {
        *(__global float *) &mem_34371[(m_34246 * (res_32302 * res_32302) +
                                        gtid_32957 * res_32302 + gtid_32958) *
                                       4] = res_34164;
    }
    if (((((slt32(gtid_32956, gidzz_range_33776) && slt32(gtid_32957,
                                                          res_32302)) &&
           slt32(gtid_32958, res_32302)) && (sle32(0, m_34249) && slt32(m_34249,
                                                                        sizze_32280))) &&
         (sle32(0, gtid_32957) && slt32(gtid_32957, res_32302))) && (sle32(0,
                                                                           gtid_32958) &&
                                                                     slt32(gtid_32958,
                                                                           res_32302))) {
        *(__global float *) &mem_34371[(m_34249 * (res_32302 * res_32302) +
                                        gtid_32957 * res_32302 + gtid_32958) *
                                       4] = res_34165;
    }
    if (((((slt32(gtid_32956, gidzz_range_33776) && slt32(gtid_32957,
                                                          res_32302)) &&
           slt32(gtid_32958, res_32302)) && (sle32(0, m_34252) && slt32(m_34252,
                                                                        sizze_32280))) &&
         (sle32(0, gtid_32957) && slt32(gtid_32957, res_32302))) && (sle32(0,
                                                                           gtid_32958) &&
                                                                     slt32(gtid_32958,
                                                                           res_32302))) {
        *(__global float *) &mem_34371[(m_34252 * (res_32302 * res_32302) +
                                        gtid_32957 * res_32302 + gtid_32958) *
                                       4] = res_34166;
    }
    if (((((slt32(gtid_32956, gidzz_range_33776) && slt32(gtid_32957,
                                                          res_32302)) &&
           slt32(gtid_32958, res_32302)) && (sle32(0, m_34255) && slt32(m_34255,
                                                                        sizze_32280))) &&
         (sle32(0, gtid_32957) && slt32(gtid_32957, res_32302))) && (sle32(0,
                                                                           gtid_32958) &&
                                                                     slt32(gtid_32958,
                                                                           res_32302))) {
        *(__global float *) &mem_34371[(m_34255 * (res_32302 * res_32302) +
                                        gtid_32957 * res_32302 + gtid_32958) *
                                       4] = res_34167;
    }
    if (((((slt32(gtid_32956, gidzz_range_33776) && slt32(gtid_32957,
                                                          res_32302)) &&
           slt32(gtid_32958, res_32302)) && (sle32(0, m_34258) && slt32(m_34258,
                                                                        sizze_32280))) &&
         (sle32(0, gtid_32957) && slt32(gtid_32957, res_32302))) && (sle32(0,
                                                                           gtid_32958) &&
                                                                     slt32(gtid_32958,
                                                                           res_32302))) {
        *(__global float *) &mem_34371[(m_34258 * (res_32302 * res_32302) +
                                        gtid_32957 * res_32302 + gtid_32958) *
                                       4] = res_34168;
    }
}
__kernel void map_33022(int32_t sizze_32280, int32_t arg_32408,
                        int32_t arg_32422, __global unsigned char *mem_34382,
                        __global unsigned char *mem_34389)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_33022;
    int32_t local_tid_33023;
    int32_t group_sizze_34636;
    int32_t wave_sizze_34635;
    int32_t group_id_33024;
    
    global_tid_33022 = get_global_id(0);
    local_tid_33023 = get_local_id(0);
    group_sizze_34636 = get_local_size(0);
    wave_sizze_34635 = LOCKSTEP_WIDTH;
    group_id_33024 = get_group_id(0);
    
    int32_t gtid_33013;
    int32_t gtid_33014;
    
    gtid_33013 = squot32(global_tid_33022, arg_32422);
    gtid_33014 = global_tid_33022 - squot32(global_tid_33022, arg_32422) *
        arg_32422;
    
    float write_value_32501;
    
    if (slt32(gtid_33013, sizze_32280) && slt32(gtid_33014, arg_32422)) {
        write_value_32501 = *(__global float *) &mem_34389[(gtid_33013 *
                                                            arg_32422 +
                                                            gtid_33014) * 4];
    }
    if (((slt32(gtid_33013, sizze_32280) && slt32(gtid_33014, arg_32422)) &&
         (sle32(0, gtid_33013) && slt32(gtid_33013, sizze_32280))) && (sle32(0,
                                                                             gtid_33014) &&
                                                                       slt32(gtid_33014,
                                                                             arg_32408))) {
        *(__global float *) &mem_34382[(gtid_33013 * arg_32408 + gtid_33014) *
                                       4] = write_value_32501;
    }
}
__kernel void map_33036(int32_t sizze_32280, int32_t arg_32408,
                        int32_t res_32421, int32_t arg_32422, int32_t m_32438,
                        int32_t i_32472, __global unsigned char *mem_34382,
                        __global unsigned char *mem_34385, __global
                        unsigned char *mem_34389)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_33036;
    int32_t local_tid_33037;
    int32_t group_sizze_34634;
    int32_t wave_sizze_34633;
    int32_t group_id_33038;
    
    global_tid_33036 = get_global_id(0);
    local_tid_33037 = get_local_id(0);
    group_sizze_34634 = get_local_size(0);
    wave_sizze_34633 = LOCKSTEP_WIDTH;
    group_id_33038 = get_group_id(0);
    
    int32_t gtid_33027;
    int32_t gtid_33028;
    
    gtid_33027 = squot32(global_tid_33036, arg_32422);
    gtid_33028 = global_tid_33036 - squot32(global_tid_33036, arg_32422) *
        arg_32422;
    
    float res_33040;
    bool cond_33041;
    int32_t res_33043;
    int32_t res_33044;
    float res_33045;
    
    if (slt32(gtid_33027, sizze_32280) && slt32(gtid_33028, arg_32422)) {
        res_33040 = *(__global float *) &mem_34382[(gtid_33027 * arg_32408 +
                                                    i_32472) * 4];
        cond_33041 = *(__global bool *) &mem_34385[gtid_33027];
        res_33043 = sdiv32(gtid_33028, res_32421);
        res_33044 = smod32(gtid_33028, res_32421);
        if (cond_33041) {
            int32_t x_33046;
            int32_t i_33047;
            float res_33048;
            
            x_33046 = res_32421 * res_33043;
            i_33047 = res_33044 + x_33046;
            res_33048 = *(__global float *) &mem_34382[(gtid_33027 * arg_32408 +
                                                        i_33047) * 4];
            res_33045 = res_33048;
        } else {
            float x_33049;
            float res_33050;
            bool cond_33051;
            float res_33052;
            
            x_33049 = *(__global float *) &mem_34382[(gtid_33027 * arg_32408 +
                                                      res_33044) * 4];
            res_33050 = x_33049 / res_33040;
            cond_33051 = slt32(res_33043, m_32438);
            if (cond_33051) {
                int32_t x_33053;
                int32_t x_33054;
                int32_t i_33055;
                float x_33056;
                int32_t i_33057;
                float x_33058;
                float y_33059;
                float res_33060;
                
                x_33053 = 1 + res_33043;
                x_33054 = res_32421 * x_33053;
                i_33055 = res_33044 + x_33054;
                x_33056 = *(__global float *) &mem_34382[(gtid_33027 *
                                                          arg_32408 + i_33055) *
                                                         4];
                i_33057 = i_32472 + x_33054;
                x_33058 = *(__global float *) &mem_34382[(gtid_33027 *
                                                          arg_32408 + i_33057) *
                                                         4];
                y_33059 = res_33050 * x_33058;
                res_33060 = x_33056 - y_33059;
                res_33052 = res_33060;
            } else {
                res_33052 = res_33050;
            }
            res_33045 = res_33052;
        }
    }
    if (slt32(gtid_33027, sizze_32280) && slt32(gtid_33028, arg_32422)) {
        *(__global float *) &mem_34389[(gtid_33027 * arg_32422 + gtid_33028) *
                                       4] = res_33045;
    }
}
__kernel void map_33068(int32_t sizze_32280, int32_t arg_32408, int32_t i_32472,
                        __global unsigned char *mem_34382, __global
                        unsigned char *mem_34385)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_33068;
    int32_t local_tid_33069;
    int32_t group_sizze_34632;
    int32_t wave_sizze_34631;
    int32_t group_id_33070;
    
    global_tid_33068 = get_global_id(0);
    local_tid_33069 = get_local_id(0);
    group_sizze_34632 = get_local_size(0);
    wave_sizze_34631 = LOCKSTEP_WIDTH;
    group_id_33070 = get_group_id(0);
    
    int32_t gtid_33061;
    
    gtid_33061 = global_tid_33068;
    
    float res_33072;
    bool cond_33073;
    
    if (slt32(gtid_33061, sizze_32280)) {
        res_33072 = *(__global float *) &mem_34382[(gtid_33061 * arg_32408 +
                                                    i_32472) * 4];
        cond_33073 = res_33072 == 0.0F;
    }
    if (slt32(gtid_33061, sizze_32280)) {
        *(__global bool *) &mem_34385[gtid_33061] = cond_33073;
    }
}
__kernel void map_33083(int32_t sizze_32280, int32_t res_32302, int32_t j_32404,
                        int32_t arg_32408, __global unsigned char *mem_34371,
                        __global unsigned char *mem_34382)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_33083;
    int32_t local_tid_33084;
    int32_t group_sizze_34629;
    int32_t wave_sizze_34628;
    int32_t group_id_33085;
    
    global_tid_33083 = get_global_id(0);
    local_tid_33084 = get_local_id(0);
    group_sizze_34629 = get_local_size(0);
    wave_sizze_34628 = LOCKSTEP_WIDTH;
    group_id_33085 = get_group_id(0);
    
    int32_t gtid_33074;
    int32_t gtid_33075;
    
    gtid_33074 = squot32(global_tid_33083, arg_32408);
    gtid_33075 = global_tid_33083 - squot32(global_tid_33083, arg_32408) *
        arg_32408;
    
    int32_t res_33088;
    int32_t res_33089;
    bool cond_33090;
    float res_33091;
    
    if (slt32(gtid_33074, sizze_32280) && slt32(gtid_33075, arg_32408)) {
        res_33088 = sdiv32(gtid_33075, j_32404);
        res_33089 = smod32(gtid_33075, j_32404);
        cond_33090 = slt32(res_33089, res_32302);
        if (cond_33090) {
            float res_33092 = *(__global float *) &mem_34371[(gtid_33074 *
                                                              (res_32302 *
                                                               res_32302) +
                                                              res_33088 *
                                                              res_32302 +
                                                              res_33089) * 4];
            
            res_33091 = res_33092;
        } else {
            int32_t y_33093;
            bool cond_33094;
            float res_33095;
            
            y_33093 = res_32302 + res_33088;
            cond_33094 = res_33089 == y_33093;
            if (cond_33094) {
                res_33095 = 1.0F;
            } else {
                res_33095 = 0.0F;
            }
            res_33091 = res_33095;
        }
    }
    if (slt32(gtid_33074, sizze_32280) && slt32(gtid_33075, arg_32408)) {
        *(__global float *) &mem_34382[(gtid_33074 * arg_32408 + gtid_33075) *
                                       4] = res_33091;
    }
}
__kernel void map_33116(int32_t sizze_32280, int32_t sizze_32281,
                        int32_t n_32284, int32_t res_32302, __global
                        unsigned char *images_mem_34349, __global
                        unsigned char *mem_34362, __global
                        unsigned char *mem_34402)
{
    const int32_t tile_sizze_34261 = mainzitile_sizze_34260;
    const int32_t tiled_group_sizze_34262 = mainzitile_sizze_34260 *
                  mainzitile_sizze_34260;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    ALIGNED_LOCAL_MEMORY(mem_34394_backing_0, 4 *
                         sext_i32_i64(mainzitile_sizze_34260 *
                         mainzitile_sizze_34260));
    ALIGNED_LOCAL_MEMORY(mem_34398_backing_1, 4 *
                         sext_i32_i64(mainzitile_sizze_34260 *
                         mainzitile_sizze_34260));
    
    int32_t global_tid_33116;
    int32_t local_tid_33117;
    int32_t group_sizze_34638;
    int32_t wave_sizze_34637;
    int32_t group_id_33118;
    
    global_tid_33116 = get_global_id(0);
    local_tid_33117 = get_local_id(0);
    group_sizze_34638 = get_local_size(0);
    wave_sizze_34637 = LOCKSTEP_WIDTH;
    group_id_33118 = get_group_id(0);
    
    int32_t gtid_33107;
    int32_t gtid_33108;
    int32_t ltid_34263;
    int32_t ltid_34264;
    
    gtid_33107 = squot32(srem32(global_tid_33116, tile_sizze_34261 *
                                tile_sizze_34261), tile_sizze_34261) +
        squot32(squot32(global_tid_33116, tile_sizze_34261 * tile_sizze_34261),
                squot32(res_32302 + tile_sizze_34261 - 1, tile_sizze_34261)) *
        tile_sizze_34261;
    gtid_33108 = srem32(global_tid_33116, tile_sizze_34261 * tile_sizze_34261) -
        squot32(srem32(global_tid_33116, tile_sizze_34261 * tile_sizze_34261),
                tile_sizze_34261) * tile_sizze_34261 +
        (squot32(global_tid_33116, tile_sizze_34261 * tile_sizze_34261) -
         squot32(squot32(global_tid_33116, tile_sizze_34261 * tile_sizze_34261),
                 squot32(res_32302 + tile_sizze_34261 - 1, tile_sizze_34261)) *
         squot32(res_32302 + tile_sizze_34261 - 1, tile_sizze_34261)) *
        tile_sizze_34261;
    ltid_34263 = squot32(srem32(global_tid_33116, tile_sizze_34261 *
                                tile_sizze_34261), tile_sizze_34261);
    ltid_34264 = srem32(global_tid_33116, tile_sizze_34261 * tile_sizze_34261) -
        squot32(srem32(global_tid_33116, tile_sizze_34261 * tile_sizze_34261),
                tile_sizze_34261) * tile_sizze_34261;
    if (slt32(gtid_33107, sizze_32280) && slt32(gtid_33108, res_32302)) { }
    
    __local char *mem_34394;
    __local char *mem_34398;
    float res_33121;
    
    mem_34394 = (__local char *) mem_34394_backing_0;
    mem_34398 = (__local char *) mem_34398_backing_1;
    
    float x_33124 = 0.0F;
    int32_t chunk_sizze_33122;
    int32_t chunk_offset_33123 = 0;
    
    while (slt32(chunk_offset_33123, n_32284)) {
        if (slt32(n_32284 - chunk_offset_33123, tile_sizze_34261)) {
            chunk_sizze_33122 = n_32284 - chunk_offset_33123;
        } else {
            chunk_sizze_33122 = tile_sizze_34261;
        }
        for (int32_t comb_iter_34639 = 0; comb_iter_34639 <
             squot32(tile_sizze_34261 * tile_sizze_34261 +
                     tiled_group_sizze_34262 - 1, tiled_group_sizze_34262);
             comb_iter_34639++) {
            int32_t cid_34276;
            int32_t cid_34277;
            int32_t flat_comb_id_34640 = comb_iter_34639 *
                    tiled_group_sizze_34262 + local_tid_33117;
            
            cid_34276 = squot32(flat_comb_id_34640, tile_sizze_34261);
            cid_34277 = flat_comb_id_34640 - squot32(flat_comb_id_34640,
                                                     tile_sizze_34261) *
                tile_sizze_34261;
            if ((slt32(cid_34276, chunk_sizze_33122) && slt32(cid_34277,
                                                              tile_sizze_34261)) &&
                slt32(gtid_33108, res_32302)) {
                float x_chunk_outer_elem_34275 = *(__global
                                                   float *) &mem_34362[(res_32302 *
                                                                        0 +
                                                                        gtid_33108 +
                                                                        res_32302 *
                                                                        chunk_offset_33123 +
                                                                        ltid_34263 *
                                                                        res_32302) *
                                                                       4];
                
                *(__local float *) &mem_34394[(cid_34276 * tile_sizze_34261 +
                                               cid_34277) * 4] =
                    x_chunk_outer_elem_34275;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (slt32(gtid_33107, sizze_32280) && slt32(gtid_33108, res_32302)) { }
        for (int32_t comb_iter_34641 = 0; comb_iter_34641 <
             squot32(tile_sizze_34261 * tile_sizze_34261 +
                     tiled_group_sizze_34262 - 1, tiled_group_sizze_34262);
             comb_iter_34641++) {
            int32_t cid_34281;
            int32_t cid_34282;
            int32_t flat_comb_id_34642 = comb_iter_34641 *
                    tiled_group_sizze_34262 + local_tid_33117;
            
            cid_34281 = squot32(flat_comb_id_34642, tile_sizze_34261);
            cid_34282 = flat_comb_id_34642 - squot32(flat_comb_id_34642,
                                                     tile_sizze_34261) *
                tile_sizze_34261;
            if ((slt32(cid_34281, tile_sizze_34261) && slt32(cid_34282,
                                                             chunk_sizze_33122)) &&
                slt32(gtid_33107, sizze_32280)) {
                float x_chunk_outer_elem_34280 = *(__global
                                                   float *) &images_mem_34349[(gtid_33107 *
                                                                               sizze_32281 +
                                                                               chunk_offset_33123 +
                                                                               ltid_34264) *
                                                                              4];
                
                *(__local float *) &mem_34398[(cid_34281 * tile_sizze_34261 +
                                               cid_34282) * 4] =
                    x_chunk_outer_elem_34280;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (slt32(gtid_33107, sizze_32280) && slt32(gtid_33108, res_32302)) { }
        
        float res_33127;
        float sync_34284;
        float acc_33130 = x_33124;
        int32_t groupstream_mapaccum_dummy_chunk_sizze_33128;
        
        groupstream_mapaccum_dummy_chunk_sizze_33128 = 1;
        if (slt32(gtid_33107, sizze_32280) && slt32(gtid_33108, res_32302)) {
            if (chunk_sizze_33122 == tile_sizze_34261) {
                for (int32_t i_33129 = 0; i_33129 < tile_sizze_34261;
                     i_33129++) {
                    float x_33133;
                    float x_33134;
                    bool res_33136;
                    float res_33137;
                    float res_33140;
                    
                    x_33133 = *(__local float *) &mem_34394[(tile_sizze_34261 *
                                                             0 + ltid_34264 +
                                                             tile_sizze_34261 *
                                                             i_33129 + 0 *
                                                             tile_sizze_34261) *
                                                            4];
                    x_33134 = *(__local float *) &mem_34398[(ltid_34263 *
                                                             tile_sizze_34261 +
                                                             i_33129) * 4];
                    res_33136 = futrts_isnan32(x_33134);
                    if (res_33136) {
                        res_33137 = 0.0F;
                    } else {
                        float res_33138 = x_33133 * x_33134;
                        
                        res_33137 = res_33138;
                    }
                    res_33140 = acc_33130 + res_33137;
                    
                    float acc_tmp_34643 = res_33140;
                    
                    acc_33130 = acc_tmp_34643;
                }
            } else {
                for (int32_t i_33129 = 0; i_33129 < chunk_sizze_33122;
                     i_33129++) {
                    float x_33133;
                    float x_33134;
                    bool res_33136;
                    float res_33137;
                    float res_33140;
                    
                    x_33133 = *(__local float *) &mem_34394[(tile_sizze_34261 *
                                                             0 + ltid_34264 +
                                                             tile_sizze_34261 *
                                                             i_33129 + 0 *
                                                             tile_sizze_34261) *
                                                            4];
                    x_33134 = *(__local float *) &mem_34398[(ltid_34263 *
                                                             tile_sizze_34261 +
                                                             i_33129) * 4];
                    res_33136 = futrts_isnan32(x_33134);
                    if (res_33136) {
                        res_33137 = 0.0F;
                    } else {
                        float res_33138 = x_33133 * x_33134;
                        
                        res_33137 = res_33138;
                    }
                    res_33140 = acc_33130 + res_33137;
                    
                    float acc_tmp_34644 = res_33140;
                    
                    acc_33130 = acc_tmp_34644;
                }
            }
        }
        res_33127 = acc_33130;
        sync_34284 = res_33127;
        barrier(CLK_LOCAL_MEM_FENCE);
        x_33124 = sync_34284;
        chunk_offset_33123 += tile_sizze_34261;
    }
    res_33121 = x_33124;
    if (slt32(gtid_33107, sizze_32280) && slt32(gtid_33108, res_32302)) {
        *(__global float *) &mem_34402[(gtid_33107 * res_32302 + gtid_33108) *
                                       4] = res_33121;
    }
}
__kernel void map_33162(int32_t sizze_32280, int32_t res_32302,
                        int32_t j_m_i_32405, __global unsigned char *mem_34402,
                        __global unsigned char *mem_34407, __global
                        unsigned char *mem_34411)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_33162;
    int32_t local_tid_33163;
    int32_t group_sizze_34651;
    int32_t wave_sizze_34650;
    int32_t group_id_33164;
    
    global_tid_33162 = get_global_id(0);
    local_tid_33163 = get_local_id(0);
    group_sizze_34651 = get_local_size(0);
    wave_sizze_34650 = LOCKSTEP_WIDTH;
    group_id_33164 = get_group_id(0);
    
    int32_t gtid_33153;
    int32_t gtid_33154;
    
    gtid_33153 = squot32(global_tid_33162, res_32302);
    gtid_33154 = global_tid_33162 - squot32(global_tid_33162, res_32302) *
        res_32302;
    
    int32_t binop_x_34332;
    float res_33167;
    
    if (slt32(gtid_33153, sizze_32280) && slt32(gtid_33154, res_32302)) {
        binop_x_34332 = j_m_i_32405 * gtid_33153;
        
        float x_33170 = 0.0F;
        
        for (int32_t chunk_offset_33169 = 0; chunk_offset_33169 < j_m_i_32405;
             chunk_offset_33169++) {
            int32_t binop_x_34333;
            int32_t new_index_34334;
            int32_t binop_y_34340;
            int32_t new_index_34341;
            float x_33179;
            float x_33180;
            float res_33182;
            float res_33184;
            
            binop_x_34333 = chunk_offset_33169 + binop_x_34332;
            new_index_34334 = squot32(binop_x_34333, res_32302);
            binop_y_34340 = res_32302 * new_index_34334;
            new_index_34341 = binop_x_34333 - binop_y_34340;
            x_33179 = *(__global float *) &mem_34402[(new_index_34334 *
                                                      res_32302 +
                                                      new_index_34341) * 4];
            x_33180 = *(__global float *) &mem_34407[(chunk_offset_33169 *
                                                      (res_32302 *
                                                       sizze_32280) +
                                                      gtid_33153 * res_32302 +
                                                      gtid_33154) * 4];
            res_33182 = x_33179 * x_33180;
            res_33184 = x_33170 + res_33182;
            
            float x_tmp_34652 = res_33184;
            
            x_33170 = x_tmp_34652;
        }
        res_33167 = x_33170;
    }
    if (slt32(gtid_33153, sizze_32280) && slt32(gtid_33154, res_32302)) {
        *(__global float *) &mem_34411[(gtid_33153 * res_32302 + gtid_33154) *
                                       4] = res_33167;
    }
}
__kernel void map_33205(int32_t sizze_32279, int32_t sizze_32280,
                        int32_t res_32302, __global unsigned char *mem_34411,
                        __global unsigned char *mem_34415, __global
                        unsigned char *mem_34427)
{
    const int32_t tile_sizze_34286 = mainzitile_sizze_34285;
    const int32_t tiled_group_sizze_34287 = mainzitile_sizze_34285 *
                  mainzitile_sizze_34285;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    ALIGNED_LOCAL_MEMORY(mem_34419_backing_0, 4 *
                         sext_i32_i64(mainzitile_sizze_34285 *
                         mainzitile_sizze_34285));
    ALIGNED_LOCAL_MEMORY(mem_34423_backing_1, 4 *
                         sext_i32_i64(mainzitile_sizze_34285 *
                         mainzitile_sizze_34285));
    
    int32_t global_tid_33205;
    int32_t local_tid_33206;
    int32_t group_sizze_34654;
    int32_t wave_sizze_34653;
    int32_t group_id_33207;
    
    global_tid_33205 = get_global_id(0);
    local_tid_33206 = get_local_id(0);
    group_sizze_34654 = get_local_size(0);
    wave_sizze_34653 = LOCKSTEP_WIDTH;
    group_id_33207 = get_group_id(0);
    
    int32_t gtid_33196;
    int32_t gtid_33197;
    int32_t ltid_34288;
    int32_t ltid_34289;
    
    gtid_33196 = squot32(srem32(global_tid_33205, tile_sizze_34286 *
                                tile_sizze_34286), tile_sizze_34286) +
        squot32(squot32(global_tid_33205, tile_sizze_34286 * tile_sizze_34286),
                squot32(sizze_32279 + tile_sizze_34286 - 1, tile_sizze_34286)) *
        tile_sizze_34286;
    gtid_33197 = srem32(global_tid_33205, tile_sizze_34286 * tile_sizze_34286) -
        squot32(srem32(global_tid_33205, tile_sizze_34286 * tile_sizze_34286),
                tile_sizze_34286) * tile_sizze_34286 +
        (squot32(global_tid_33205, tile_sizze_34286 * tile_sizze_34286) -
         squot32(squot32(global_tid_33205, tile_sizze_34286 * tile_sizze_34286),
                 squot32(sizze_32279 + tile_sizze_34286 - 1,
                         tile_sizze_34286)) * squot32(sizze_32279 +
                                                      tile_sizze_34286 - 1,
                                                      tile_sizze_34286)) *
        tile_sizze_34286;
    ltid_34288 = squot32(srem32(global_tid_33205, tile_sizze_34286 *
                                tile_sizze_34286), tile_sizze_34286);
    ltid_34289 = srem32(global_tid_33205, tile_sizze_34286 * tile_sizze_34286) -
        squot32(srem32(global_tid_33205, tile_sizze_34286 * tile_sizze_34286),
                tile_sizze_34286) * tile_sizze_34286;
    if (slt32(gtid_33196, sizze_32280) && slt32(gtid_33197, sizze_32279)) { }
    
    __local char *mem_34419;
    __local char *mem_34423;
    float res_33210;
    
    mem_34419 = (__local char *) mem_34419_backing_0;
    mem_34423 = (__local char *) mem_34423_backing_1;
    
    float x_33213 = 0.0F;
    int32_t chunk_sizze_33211;
    int32_t chunk_offset_33212 = 0;
    
    while (slt32(chunk_offset_33212, res_32302)) {
        if (slt32(res_32302 - chunk_offset_33212, tile_sizze_34286)) {
            chunk_sizze_33211 = res_32302 - chunk_offset_33212;
        } else {
            chunk_sizze_33211 = tile_sizze_34286;
        }
        for (int32_t comb_iter_34655 = 0; comb_iter_34655 <
             squot32(tile_sizze_34286 * tile_sizze_34286 +
                     tiled_group_sizze_34287 - 1, tiled_group_sizze_34287);
             comb_iter_34655++) {
            int32_t cid_34301;
            int32_t cid_34302;
            int32_t flat_comb_id_34656 = comb_iter_34655 *
                    tiled_group_sizze_34287 + local_tid_33206;
            
            cid_34301 = squot32(flat_comb_id_34656, tile_sizze_34286);
            cid_34302 = flat_comb_id_34656 - squot32(flat_comb_id_34656,
                                                     tile_sizze_34286) *
                tile_sizze_34286;
            if ((slt32(cid_34301, tile_sizze_34286) && slt32(cid_34302,
                                                             chunk_sizze_33211)) &&
                slt32(gtid_33196, sizze_32280)) {
                float x_chunk_outer_elem_34300 = *(__global
                                                   float *) &mem_34411[(gtid_33196 *
                                                                        res_32302 +
                                                                        chunk_offset_33212 +
                                                                        ltid_34289) *
                                                                       4];
                
                *(__local float *) &mem_34419[(cid_34301 * tile_sizze_34286 +
                                               cid_34302) * 4] =
                    x_chunk_outer_elem_34300;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (slt32(gtid_33196, sizze_32280) && slt32(gtid_33197,
                                                    sizze_32279)) { }
        for (int32_t comb_iter_34657 = 0; comb_iter_34657 <
             squot32(tile_sizze_34286 * tile_sizze_34286 +
                     tiled_group_sizze_34287 - 1, tiled_group_sizze_34287);
             comb_iter_34657++) {
            int32_t cid_34306;
            int32_t cid_34307;
            int32_t flat_comb_id_34658 = comb_iter_34657 *
                    tiled_group_sizze_34287 + local_tid_33206;
            
            cid_34306 = squot32(flat_comb_id_34658, tile_sizze_34286);
            cid_34307 = flat_comb_id_34658 - squot32(flat_comb_id_34658,
                                                     tile_sizze_34286) *
                tile_sizze_34286;
            if ((slt32(cid_34306, chunk_sizze_33211) && slt32(cid_34307,
                                                              tile_sizze_34286)) &&
                slt32(gtid_33197, sizze_32279)) {
                float x_chunk_outer_elem_34305 = *(__global
                                                   float *) &mem_34415[(gtid_33197 +
                                                                        sizze_32279 *
                                                                        chunk_offset_33212 +
                                                                        ltid_34288 *
                                                                        sizze_32279) *
                                                                       4];
                
                *(__local float *) &mem_34423[(cid_34306 * tile_sizze_34286 +
                                               cid_34307) * 4] =
                    x_chunk_outer_elem_34305;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (slt32(gtid_33196, sizze_32280) && slt32(gtid_33197,
                                                    sizze_32279)) { }
        
        float res_33216;
        float sync_34309;
        float acc_33219 = x_33213;
        int32_t groupstream_mapaccum_dummy_chunk_sizze_33217;
        
        groupstream_mapaccum_dummy_chunk_sizze_33217 = 1;
        if (slt32(gtid_33196, sizze_32280) && slt32(gtid_33197, sizze_32279)) {
            if (chunk_sizze_33211 == tile_sizze_34286) {
                for (int32_t i_33218 = 0; i_33218 < tile_sizze_34286;
                     i_33218++) {
                    float x_33222;
                    float x_33223;
                    float res_33225;
                    float res_33227;
                    
                    x_33222 = *(__local float *) &mem_34419[(ltid_34288 *
                                                             tile_sizze_34286 +
                                                             i_33218) * 4];
                    x_33223 = *(__local float *) &mem_34423[(tile_sizze_34286 *
                                                             0 + ltid_34289 +
                                                             tile_sizze_34286 *
                                                             i_33218 + 0 *
                                                             tile_sizze_34286) *
                                                            4];
                    res_33225 = x_33222 * x_33223;
                    res_33227 = acc_33219 + res_33225;
                    
                    float acc_tmp_34659 = res_33227;
                    
                    acc_33219 = acc_tmp_34659;
                }
            } else {
                for (int32_t i_33218 = 0; i_33218 < chunk_sizze_33211;
                     i_33218++) {
                    float x_33222;
                    float x_33223;
                    float res_33225;
                    float res_33227;
                    
                    x_33222 = *(__local float *) &mem_34419[(ltid_34288 *
                                                             tile_sizze_34286 +
                                                             i_33218) * 4];
                    x_33223 = *(__local float *) &mem_34423[(tile_sizze_34286 *
                                                             0 + ltid_34289 +
                                                             tile_sizze_34286 *
                                                             i_33218 + 0 *
                                                             tile_sizze_34286) *
                                                            4];
                    res_33225 = x_33222 * x_33223;
                    res_33227 = acc_33219 + res_33225;
                    
                    float acc_tmp_34660 = res_33227;
                    
                    acc_33219 = acc_tmp_34660;
                }
            }
        }
        res_33216 = acc_33219;
        sync_34309 = res_33216;
        barrier(CLK_LOCAL_MEM_FENCE);
        x_33213 = sync_34309;
        chunk_offset_33212 += tile_sizze_34286;
    }
    res_33210 = x_33213;
    if (slt32(gtid_33196, sizze_32280) && slt32(gtid_33197, sizze_32279)) {
        *(__global float *) &mem_34427[(gtid_33196 * sizze_32279 + gtid_33197) *
                                       4] = res_33210;
    }
}
__kernel void map_33243(int32_t sizze_32279, int32_t sizze_32280,
                        int32_t i_32550, __global unsigned char *mem_34431,
                        __global unsigned char *mem_34435, __global
                        unsigned char *mem_34438, __global
                        unsigned char *mem_34442, __global
                        unsigned char *mem_34449, __global
                        unsigned char *mem_34453)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_33243;
    int32_t local_tid_33244;
    int32_t group_sizze_34737;
    int32_t wave_sizze_34736;
    int32_t group_id_33245;
    
    global_tid_33243 = get_global_id(0);
    local_tid_33244 = get_local_id(0);
    group_sizze_34737 = get_local_size(0);
    wave_sizze_34736 = LOCKSTEP_WIDTH;
    group_id_33245 = get_group_id(0);
    
    int32_t gtid_33234;
    int32_t gtid_33235;
    
    gtid_33234 = squot32(global_tid_33243, sizze_32279);
    gtid_33235 = global_tid_33243 - squot32(global_tid_33243, sizze_32279) *
        sizze_32279;
    
    int32_t res_32591;
    int32_t x_32596;
    bool x_32597;
    int32_t x_32598;
    float write_value_32599;
    int32_t res_32601;
    int32_t res_32602;
    
    if (slt32(gtid_33234, sizze_32280) && slt32(gtid_33235, sizze_32279)) {
        res_32591 = *(__global int32_t *) &mem_34431[(gtid_33234 * sizze_32279 +
                                                      i_32550) * 4];
        x_32596 = *(__global int32_t *) &mem_34435[(gtid_33234 * sizze_32279 +
                                                    gtid_33235) * 4];
        x_32597 = *(__global bool *) &mem_34438[gtid_33234 * sizze_32279 +
                                                gtid_33235];
        x_32598 = *(__global int32_t *) &mem_34431[(gtid_33234 * sizze_32279 +
                                                    gtid_33235) * 4];
        write_value_32599 = *(__global float *) &mem_34442[(gtid_33234 *
                                                            sizze_32279 +
                                                            gtid_33235) * 4];
        res_32601 = res_32591 + x_32596;
        if (x_32597) {
            int32_t res_32603 = x_32598 - 1;
            
            res_32602 = res_32603;
        } else {
            int32_t res_32604 = res_32601 - 1;
            
            res_32602 = res_32604;
        }
    }
    if (((slt32(gtid_33234, sizze_32280) && slt32(gtid_33235, sizze_32279)) &&
         (sle32(0, gtid_33234) && slt32(gtid_33234, sizze_32280))) && (sle32(0,
                                                                             res_32602) &&
                                                                       slt32(res_32602,
                                                                             sizze_32279))) {
        *(__global float *) &mem_34449[(gtid_33234 * sizze_32279 + res_32602) *
                                       4] = write_value_32599;
    }
    if (((slt32(gtid_33234, sizze_32280) && slt32(gtid_33235, sizze_32279)) &&
         (sle32(0, gtid_33234) && slt32(gtid_33234, sizze_32280))) && (sle32(0,
                                                                             res_32602) &&
                                                                       slt32(res_32602,
                                                                             sizze_32279))) {
        *(__global int32_t *) &mem_34453[(gtid_33234 * sizze_32279 +
                                          res_32602) * 4] = gtid_33235;
    }
}
__kernel void map_33337(int32_t sizze_32280, int32_t n_32284, int32_t res_32300,
                        __global unsigned char *mem_34375, __global
                        unsigned char *mem_34457, __global
                        unsigned char *mem_34460, __global
                        unsigned char *mem_34463)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_33337;
    int32_t local_tid_33338;
    int32_t group_sizze_34739;
    int32_t wave_sizze_34738;
    int32_t group_id_33339;
    
    global_tid_33337 = get_global_id(0);
    local_tid_33338 = get_local_id(0);
    group_sizze_34739 = get_local_size(0);
    wave_sizze_34738 = LOCKSTEP_WIDTH;
    group_id_33339 = get_group_id(0);
    
    int32_t gtid_33330;
    
    gtid_33330 = global_tid_33337;
    
    int32_t res_33342;
    float res_33359;
    int32_t arg_33377;
    float res_33378;
    float arg_33379;
    float res_33380;
    
    if (slt32(gtid_33330, sizze_32280)) {
        int32_t x_33345 = 0;
        
        for (int32_t chunk_offset_33344 = 0; chunk_offset_33344 < n_32284;
             chunk_offset_33344++) {
            float x_33352;
            bool res_33354;
            bool cond_33355;
            int32_t res_33356;
            int32_t res_33358;
            
            x_33352 = *(__global float *) &mem_34375[(chunk_offset_33344 *
                                                      sizze_32280 +
                                                      gtid_33330) * 4];
            res_33354 = futrts_isnan32(x_33352);
            cond_33355 = !res_33354;
            if (cond_33355) {
                res_33356 = 1;
            } else {
                res_33356 = 0;
            }
            res_33358 = x_33345 + res_33356;
            
            int32_t x_tmp_34740 = res_33358;
            
            x_33345 = x_tmp_34740;
        }
        res_33342 = x_33345;
        
        float x_33362 = 0.0F;
        
        for (int32_t chunk_offset_33361 = 0; chunk_offset_33361 < n_32284;
             chunk_offset_33361++) {
            bool cond_33371;
            float res_33372;
            float res_33374;
            float res_33376;
            
            cond_33371 = slt32(chunk_offset_33361, res_33342);
            if (cond_33371) {
                float res_33373 = *(__global
                                    float *) &mem_34457[(chunk_offset_33361 *
                                                         sizze_32280 +
                                                         gtid_33330) * 4];
                
                res_33372 = res_33373;
            } else {
                res_33372 = 0.0F;
            }
            res_33374 = res_33372 * res_33372;
            res_33376 = x_33362 + res_33374;
            
            float x_tmp_34741 = res_33376;
            
            x_33362 = x_tmp_34741;
        }
        res_33359 = x_33362;
        arg_33377 = res_33342 - res_32300;
        res_33378 = sitofp_i32_f32(arg_33377);
        arg_33379 = res_33359 / res_33378;
        res_33380 = futrts_sqrt32(arg_33379);
    }
    if (slt32(gtid_33330, sizze_32280)) {
        *(__global int32_t *) &mem_34460[gtid_33330 * 4] = res_33342;
    }
    if (slt32(gtid_33330, sizze_32280)) {
        *(__global float *) &mem_34463[gtid_33330 * 4] = res_33380;
    }
}
__kernel void map_33397(int32_t sizze_32279, int32_t sizze_32280,
                        float hfrac_32286, __global unsigned char *mem_34449,
                        __global unsigned char *mem_34460, __global
                        unsigned char *mem_34466, __global
                        unsigned char *mem_34469)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_33397;
    int32_t local_tid_33398;
    int32_t group_sizze_34743;
    int32_t wave_sizze_34742;
    int32_t group_id_33399;
    
    global_tid_33397 = get_global_id(0);
    local_tid_33398 = get_local_id(0);
    group_sizze_34743 = get_local_size(0);
    wave_sizze_34742 = LOCKSTEP_WIDTH;
    group_id_33399 = get_group_id(0);
    
    int32_t gtid_33390;
    
    gtid_33390 = global_tid_33397;
    
    int32_t x_33400;
    float res_33402;
    float arg_33403;
    int32_t res_33404;
    float res_33406;
    
    if (slt32(gtid_33390, sizze_32280)) {
        x_33400 = *(__global int32_t *) &mem_34460[gtid_33390 * 4];
        res_33402 = sitofp_i32_f32(x_33400);
        arg_33403 = hfrac_32286 * res_33402;
        res_33404 = fptosi_f32_i32(arg_33403);
        
        float x_33409 = 0.0F;
        
        for (int32_t chunk_offset_33408 = 0; chunk_offset_33408 < res_33404;
             chunk_offset_33408++) {
            int32_t x_33418;
            int32_t x_33419;
            int32_t i_33420;
            float res_33421;
            float res_33423;
            
            x_33418 = x_33400 + chunk_offset_33408;
            x_33419 = x_33418 - res_33404;
            i_33420 = 1 + x_33419;
            res_33421 = *(__global float *) &mem_34449[(gtid_33390 *
                                                        sizze_32279 + i_33420) *
                                                       4];
            res_33423 = x_33409 + res_33421;
            
            float x_tmp_34744 = res_33423;
            
            x_33409 = x_tmp_34744;
        }
        res_33406 = x_33409;
    }
    if (slt32(gtid_33390, sizze_32280)) {
        *(__global float *) &mem_34466[gtid_33390 * 4] = res_33406;
    }
    if (slt32(gtid_33390, sizze_32280)) {
        *(__global int32_t *) &mem_34469[gtid_33390 * 4] = res_33404;
    }
}
__kernel void map_33431(float lam_32287, int32_t num_elems_32669,
                        int32_t x_32671, float res_32677, __global
                        unsigned char *mappingindices_mem_34348, __global
                        unsigned char *mem_34472)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_33431;
    int32_t local_tid_33432;
    int32_t group_sizze_34746;
    int32_t wave_sizze_34745;
    int32_t group_id_33433;
    
    global_tid_33431 = get_global_id(0);
    local_tid_33432 = get_local_id(0);
    group_sizze_34746 = get_local_size(0);
    wave_sizze_34745 = LOCKSTEP_WIDTH;
    group_id_33433 = get_group_id(0);
    
    int32_t gtid_33424;
    
    gtid_33424 = global_tid_33431;
    
    int32_t res_33435;
    int32_t i_33436;
    int32_t res_33437;
    float res_33438;
    float arg_33439;
    bool cond_33440;
    float res_33441;
    float res_33443;
    float res_33444;
    
    if (slt32(gtid_33424, num_elems_32669)) {
        res_33435 = x_32671 + gtid_33424;
        i_33436 = res_33435 - 1;
        res_33437 = *(__global int32_t *) &mappingindices_mem_34348[i_33436 *
                                                                    4];
        res_33438 = sitofp_i32_f32(res_33437);
        arg_33439 = res_33438 / res_32677;
        cond_33440 = 2.7182817F < arg_33439;
        if (cond_33440) {
            float res_33442;
            
            res_33442 = futrts_log32(arg_33439);
            res_33441 = res_33442;
        } else {
            res_33441 = 1.0F;
        }
        res_33443 = futrts_sqrt32(res_33441);
        res_33444 = lam_32287 * res_33443;
    }
    if (slt32(gtid_33424, num_elems_32669)) {
        *(__global float *) &mem_34472[gtid_33424 * 4] = res_33444;
    }
}
__kernel void map_33456(int32_t sizze_32280, int32_t x_32665,
                        int32_t num_elems_32669, __global
                        unsigned char *mem_34512, __global
                        unsigned char *mem_34519, __global
                        unsigned char *mem_34523)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_33456;
    int32_t local_tid_33457;
    int32_t group_sizze_34807;
    int32_t wave_sizze_34806;
    int32_t group_id_33458;
    
    global_tid_33456 = get_global_id(0);
    local_tid_33457 = get_local_id(0);
    group_sizze_34807 = get_local_size(0);
    wave_sizze_34806 = LOCKSTEP_WIDTH;
    group_id_33458 = get_group_id(0);
    
    int32_t gtid_33447;
    int32_t gtid_33448;
    
    gtid_33447 = squot32(global_tid_33456, num_elems_32669);
    gtid_33448 = global_tid_33456 - squot32(global_tid_33456, num_elems_32669) *
        num_elems_32669;
    
    int32_t write_index_32817;
    float write_value_32818;
    
    if (slt32(gtid_33447, sizze_32280) && slt32(gtid_33448, num_elems_32669)) {
        write_index_32817 = *(__global int32_t *) &mem_34512[(gtid_33447 *
                                                              num_elems_32669 +
                                                              gtid_33448) * 4];
        write_value_32818 = *(__global float *) &mem_34523[(gtid_33447 *
                                                            num_elems_32669 +
                                                            gtid_33448) * 4];
    }
    if (((slt32(gtid_33447, sizze_32280) && slt32(gtid_33448,
                                                  num_elems_32669)) && (sle32(0,
                                                                              gtid_33447) &&
                                                                        slt32(gtid_33447,
                                                                              sizze_32280))) &&
        (sle32(0, write_index_32817) && slt32(write_index_32817, x_32665))) {
        *(__global float *) &mem_34519[(gtid_33447 * x_32665 +
                                        write_index_32817) * 4] =
            write_value_32818;
    }
}
__kernel void map_33483(int32_t sizze_32280, int32_t num_elems_32669, __global
                        unsigned char *mem_34460, __global
                        unsigned char *mem_34478, __global
                        unsigned char *mem_34502, __global
                        unsigned char *mem_34505, __global
                        unsigned char *mem_34512, __global
                        unsigned char *mem_34515)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_33483;
    int32_t local_tid_33484;
    int32_t group_sizze_34800;
    int32_t wave_sizze_34799;
    int32_t group_id_33485;
    
    global_tid_33483 = get_global_id(0);
    local_tid_33484 = get_local_id(0);
    group_sizze_34800 = get_local_size(0);
    wave_sizze_34799 = LOCKSTEP_WIDTH;
    group_id_33485 = get_group_id(0);
    
    int32_t gtid_33476;
    
    gtid_33476 = global_tid_33483;
    
    int32_t x_33486;
    int32_t y_33487;
    bool res_33488;
    int32_t res_33489;
    int32_t res_33491;
    bool cond_33493;
    bool res_33494;
    bool x_33495;
    bool y_33496;
    bool cond_33497;
    int32_t res_33498;
    
    if (slt32(gtid_33476, sizze_32280)) {
        x_33486 = *(__global int32_t *) &mem_34460[gtid_33476 * 4];
        y_33487 = *(__global int32_t *) &mem_34478[gtid_33476 * 4];
        res_33488 = *(__global bool *) &mem_34502[gtid_33476];
        res_33489 = *(__global int32_t *) &mem_34505[gtid_33476 * 4];
        if (res_33488) {
            int32_t res_33492 = *(__global int32_t *) &mem_34512[(gtid_33476 *
                                                                  num_elems_32669 +
                                                                  res_33489) *
                                                                 4];
            
            res_33491 = res_33492;
        } else {
            res_33491 = -1;
        }
        cond_33493 = sle32(x_33486, 5);
        res_33494 = sle32(y_33487, 5);
        x_33495 = !cond_33493;
        y_33496 = res_33494 && x_33495;
        cond_33497 = cond_33493 || y_33496;
        if (cond_33497) {
            res_33498 = -2;
        } else {
            res_33498 = res_33491;
        }
    }
    if (slt32(gtid_33476, sizze_32280)) {
        *(__global int32_t *) &mem_34515[gtid_33476 * 4] = res_33498;
    }
}
__kernel void map_33522(int32_t sizze_32279, int32_t sizze_32280,
                        int32_t n_32284, int32_t num_elems_32669, __global
                        unsigned char *mem_34453, __global
                        unsigned char *mem_34460, __global
                        unsigned char *mem_34472, __global
                        unsigned char *mem_34475, __global
                        unsigned char *mem_34478, __global
                        unsigned char *mem_34486, __global
                        unsigned char *mem_34489, __global
                        unsigned char *mem_34492, __global
                        unsigned char *mem_34496, __global
                        unsigned char *mem_34500, __global
                        unsigned char *mem_34502, __global
                        unsigned char *mem_34505, __global
                        unsigned char *mem_34508)
{
    const int32_t group_sizze_33517 = mainzigroup_sizze_33516;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_33522;
    int32_t local_tid_33523;
    int32_t group_sizze_34791;
    int32_t wave_sizze_34790;
    int32_t group_id_33524;
    
    global_tid_33522 = get_global_id(0);
    local_tid_33523 = get_local_id(0);
    group_sizze_34791 = get_local_size(0);
    wave_sizze_34790 = LOCKSTEP_WIDTH;
    group_id_33524 = get_group_id(0);
    
    int32_t gtid_33515;
    
    gtid_33515 = global_tid_33522;
    
    int32_t x_33525;
    int32_t y_33527;
    float y_33528;
    bool acc0_33539;
    int32_t acc0_33540;
    float acc0_33541;
    int32_t res_33587;
    
    if (slt32(gtid_33515, sizze_32280)) {
        x_33525 = *(__global int32_t *) &mem_34460[gtid_33515 * 4];
        y_33527 = *(__global int32_t *) &mem_34478[gtid_33515 * 4];
        y_33528 = *(__global float *) &mem_34475[gtid_33515 * 4];
        
        bool redout_33544;
        int32_t redout_33545;
        float redout_33546;
        
        redout_33544 = 0;
        redout_33545 = -1;
        redout_33546 = 0.0F;
        for (int32_t i_33549 = 0; i_33549 < num_elems_32669; i_33549++) {
            float x_33550;
            float x_33552;
            float res_33554;
            bool cond_33555;
            int32_t res_33556;
            bool res_33560;
            bool res_33561;
            bool x_33562;
            float res_33563;
            bool res_33564;
            bool x_33565;
            float res_33566;
            bool res_33573;
            int32_t res_33574;
            float res_33579;
            
            x_33550 = *(__global float *) &mem_34486[(i_33549 * sizze_32280 +
                                                      gtid_33515) * 4];
            x_33552 = *(__global float *) &mem_34472[i_33549 * 4];
            res_33554 = x_33550 / y_33528;
            cond_33555 = slt32(i_33549, y_33527);
            if (cond_33555) {
                int32_t i_33557;
                int32_t x_33558;
                int32_t res_33559;
                
                i_33557 = x_33525 + i_33549;
                x_33558 = *(__global int32_t *) &mem_34453[(gtid_33515 *
                                                            sizze_32279 +
                                                            i_33557) * 4];
                res_33559 = x_33558 - n_32284;
                res_33556 = res_33559;
            } else {
                res_33556 = -1;
            }
            res_33560 = futrts_isnan32(res_33554);
            res_33561 = !res_33560;
            x_33562 = cond_33555 && res_33561;
            res_33563 = (float) fabs(res_33554);
            res_33564 = x_33552 < res_33563;
            x_33565 = x_33562 && res_33564;
            if (cond_33555) {
                res_33566 = res_33554;
            } else {
                res_33566 = 0.0F;
            }
            if (redout_33544) {
                res_33573 = redout_33544;
                res_33574 = redout_33545;
            } else {
                bool x_33575;
                bool y_33576;
                bool res_33577;
                int32_t res_33578;
                
                x_33575 = !x_33565;
                y_33576 = redout_33544 && x_33575;
                res_33577 = x_33565 || y_33576;
                if (x_33565) {
                    res_33578 = i_33549;
                } else {
                    res_33578 = redout_33545;
                }
                res_33573 = res_33577;
                res_33574 = res_33578;
            }
            res_33579 = redout_33546 + res_33566;
            *(__global int32_t *) &mem_34489[(group_id_33524 *
                                              (group_sizze_33517 *
                                               num_elems_32669) +
                                              local_tid_33523 + i_33549 *
                                              group_sizze_33517) * 4] =
                res_33556;
            *(__global float *) &mem_34492[(group_id_33524 *
                                            (group_sizze_33517 *
                                             num_elems_32669) +
                                            local_tid_33523 + i_33549 *
                                            group_sizze_33517) * 4] = res_33554;
            
            bool redout_tmp_34792 = res_33573;
            int32_t redout_tmp_34793 = res_33574;
            float redout_tmp_34794;
            
            redout_tmp_34794 = res_33579;
            redout_33544 = redout_tmp_34792;
            redout_33545 = redout_tmp_34793;
            redout_33546 = redout_tmp_34794;
        }
        acc0_33539 = redout_33544;
        acc0_33540 = redout_33545;
        acc0_33541 = redout_33546;
        if (acc0_33539) {
            res_33587 = acc0_33540;
        } else {
            res_33587 = -1;
        }
    }
    if (slt32(gtid_33515, sizze_32280)) {
        for (int32_t i_34797 = 0; i_34797 < num_elems_32669; i_34797++) {
            *(__global int32_t *) &mem_34496[(gtid_33515 + i_34797 *
                                              sizze_32280) * 4] = *(__global
                                                                    int32_t *) &mem_34489[(group_id_33524 *
                                                                                           (group_sizze_33517 *
                                                                                            num_elems_32669) +
                                                                                           local_tid_33523 +
                                                                                           i_34797 *
                                                                                           group_sizze_33517) *
                                                                                          4];
        }
    }
    if (slt32(gtid_33515, sizze_32280)) {
        for (int32_t i_34798 = 0; i_34798 < num_elems_32669; i_34798++) {
            *(__global float *) &mem_34500[(gtid_33515 + i_34798 *
                                            sizze_32280) * 4] = *(__global
                                                                  float *) &mem_34492[(group_id_33524 *
                                                                                       (group_sizze_33517 *
                                                                                        num_elems_32669) +
                                                                                       local_tid_33523 +
                                                                                       i_34798 *
                                                                                       group_sizze_33517) *
                                                                                      4];
        }
    }
    if (slt32(gtid_33515, sizze_32280)) {
        *(__global bool *) &mem_34502[gtid_33515] = acc0_33539;
    }
    if (slt32(gtid_33515, sizze_32280)) {
        *(__global int32_t *) &mem_34505[gtid_33515 * 4] = res_33587;
    }
    if (slt32(gtid_33515, sizze_32280)) {
        *(__global float *) &mem_34508[gtid_33515 * 4] = acc0_33541;
    }
}
__kernel void map_33636(int32_t sizze_32280, __global unsigned char *mem_34445,
                        __global unsigned char *mem_34460, __global
                        unsigned char *mem_34463, __global
                        unsigned char *mem_34475, __global
                        unsigned char *mem_34478)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_33636;
    int32_t local_tid_33637;
    int32_t group_sizze_34748;
    int32_t wave_sizze_34747;
    int32_t group_id_33638;
    
    global_tid_33636 = get_global_id(0);
    local_tid_33637 = get_local_id(0);
    group_sizze_34748 = get_local_size(0);
    wave_sizze_34747 = LOCKSTEP_WIDTH;
    group_id_33638 = get_group_id(0);
    
    int32_t gtid_33629;
    
    gtid_33629 = global_tid_33636;
    
    int32_t x_33639;
    float x_33640;
    int32_t copy_p_33641;
    int32_t y_33642;
    float res_33643;
    float res_33644;
    float y_33645;
    
    if (slt32(gtid_33629, sizze_32280)) {
        x_33639 = *(__global int32_t *) &mem_34460[gtid_33629 * 4];
        x_33640 = *(__global float *) &mem_34463[gtid_33629 * 4];
        copy_p_33641 = *(__global int32_t *) &mem_34445[gtid_33629 * 4];
        y_33642 = copy_p_33641 - x_33639;
        res_33643 = sitofp_i32_f32(x_33639);
        res_33644 = futrts_sqrt32(res_33643);
        y_33645 = x_33640 * res_33644;
    }
    if (slt32(gtid_33629, sizze_32280)) {
        *(__global float *) &mem_34475[gtid_33629 * 4] = y_33645;
    }
    if (slt32(gtid_33629, sizze_32280)) {
        *(__global int32_t *) &mem_34478[gtid_33629 * 4] = y_33642;
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
__kernel void replicate_34726(int32_t sizze_32279, int32_t sizze_32280, __global
                              unsigned char *mem_34449)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t replicate_gtid_34726;
    int32_t replicate_ltid_34727;
    int32_t replicate_gid_34728;
    
    replicate_gtid_34726 = get_global_id(0);
    replicate_ltid_34727 = get_local_id(0);
    replicate_gid_34728 = get_group_id(0);
    if (slt32(replicate_gtid_34726, sizze_32280 * sizze_32279)) {
        *(__global float *) &mem_34449[(squot32(replicate_gtid_34726,
                                                sizze_32279) * sizze_32279 +
                                        (replicate_gtid_34726 -
                                         squot32(replicate_gtid_34726,
                                                 sizze_32279) * sizze_32279)) *
                                       4] = 0.0F;
    }
}
__kernel void replicate_34731(int32_t sizze_32279, int32_t sizze_32280, __global
                              unsigned char *mem_34453)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t replicate_gtid_34731;
    int32_t replicate_ltid_34732;
    int32_t replicate_gid_34733;
    
    replicate_gtid_34731 = get_global_id(0);
    replicate_ltid_34732 = get_local_id(0);
    replicate_gid_34733 = get_group_id(0);
    if (slt32(replicate_gtid_34731, sizze_32280 * sizze_32279)) {
        *(__global int32_t *) &mem_34453[(squot32(replicate_gtid_34731,
                                                  sizze_32279) * sizze_32279 +
                                          (replicate_gtid_34731 -
                                           squot32(replicate_gtid_34731,
                                                   sizze_32279) *
                                           sizze_32279)) * 4] = 0;
    }
}
__kernel void replicate_34801(int32_t sizze_32280, int32_t x_32665, __global
                              unsigned char *mem_34519)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t replicate_gtid_34801;
    int32_t replicate_ltid_34802;
    int32_t replicate_gid_34803;
    
    replicate_gtid_34801 = get_global_id(0);
    replicate_ltid_34802 = get_local_id(0);
    replicate_gid_34803 = get_group_id(0);
    if (slt32(replicate_gtid_34801, sizze_32280 * x_32665)) {
        *(__global float *) &mem_34519[(squot32(replicate_gtid_34801, x_32665) *
                                        x_32665 + (replicate_gtid_34801 -
                                                   squot32(replicate_gtid_34801,
                                                           x_32665) *
                                                   x_32665)) * 4] = NAN;
    }
}
__kernel void scan_stage1_33309(int32_t sizze_32279, int32_t sizze_32280,
                                int32_t sizze_32281, int32_t num_groups_33303,
                                __global unsigned char *images_mem_34349,
                                __global unsigned char *mem_34427, __global
                                unsigned char *mem_34431, __global
                                unsigned char *mem_34435, __global
                                unsigned char *mem_34438, __global
                                unsigned char *mem_34442)
{
    const int32_t group_sizze_33292 = mainzigroup_sizze_33291;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    ALIGNED_LOCAL_MEMORY(scan_arr_mem_34669_backing_0, 4 *
                         mainzigroup_sizze_33291);
    ALIGNED_LOCAL_MEMORY(scan_arr_mem_34671_backing_1, 4 *
                         mainzigroup_sizze_33291);
    
    int32_t global_tid_33309;
    int32_t local_tid_33310;
    int32_t group_sizze_34662;
    int32_t wave_sizze_34661;
    int32_t group_id_33311;
    
    global_tid_33309 = get_global_id(0);
    local_tid_33310 = get_local_id(0);
    group_sizze_34662 = get_local_size(0);
    wave_sizze_34661 = LOCKSTEP_WIDTH;
    group_id_33311 = get_group_id(0);
    
    int32_t gtid_33286;
    int32_t gtid_33308;
    __local char *scan_arr_mem_34669;
    
    scan_arr_mem_34669 = (__local char *) scan_arr_mem_34669_backing_0;
    
    __local char *scan_arr_mem_34671;
    
    scan_arr_mem_34671 = (__local char *) scan_arr_mem_34671_backing_1;
    
    int32_t x_32571;
    int32_t x_32572;
    int32_t x_32573;
    int32_t x_32574;
    
    x_32571 = 0;
    x_32572 = 0;
    for (int32_t j_34673 = 0; j_34673 < squot32(sizze_32280 * sizze_32279 +
                                                group_sizze_33292 *
                                                num_groups_33303 - 1,
                                                group_sizze_33292 *
                                                num_groups_33303); j_34673++) {
        int32_t chunk_offset_34674 = group_sizze_33292 * j_34673 +
                group_id_33311 * (group_sizze_33292 * squot32(sizze_32280 *
                                                              sizze_32279 +
                                                              group_sizze_33292 *
                                                              num_groups_33303 -
                                                              1,
                                                              group_sizze_33292 *
                                                              num_groups_33303));
        int32_t flat_idx_34675 = chunk_offset_34674 + local_tid_33310;
        
        gtid_33286 = squot32(flat_idx_34675, sizze_32279);
        gtid_33308 = flat_idx_34675 - squot32(flat_idx_34675, sizze_32279) *
            sizze_32279;
        // threads in bounds read input; others get neutral element
        {
            if (slt32(gtid_33286, sizze_32280) && slt32(gtid_33308,
                                                        sizze_32279)) {
                float x_32577;
                float x_32578;
                bool res_32579;
                bool cond_32580;
                float res_32581;
                bool res_32583;
                bool res_32584;
                int32_t res_32585;
                int32_t res_32586;
                
                x_32577 = *(__global float *) &images_mem_34349[(gtid_33286 *
                                                                 sizze_32281 +
                                                                 gtid_33308) *
                                                                4];
                x_32578 = *(__global float *) &mem_34427[(gtid_33286 *
                                                          sizze_32279 +
                                                          gtid_33308) * 4];
                res_32579 = futrts_isnan32(x_32577);
                cond_32580 = !res_32579;
                if (cond_32580) {
                    float res_32582 = x_32577 - x_32578;
                    
                    res_32581 = res_32582;
                } else {
                    res_32581 = NAN;
                }
                res_32583 = futrts_isnan32(res_32581);
                res_32584 = !res_32583;
                if (res_32584) {
                    res_32585 = 1;
                } else {
                    res_32585 = 0;
                }
                if (res_32584) {
                    res_32586 = 0;
                } else {
                    res_32586 = 1;
                }
                // write to-scan values to parameters
                {
                    x_32573 = res_32585;
                    x_32574 = res_32586;
                }
                // write mapped values results to global memory
                {
                    *(__global bool *) &mem_34438[gtid_33286 * sizze_32279 +
                                                  gtid_33308] = res_32584;
                    *(__global float *) &mem_34442[(gtid_33286 * sizze_32279 +
                                                    gtid_33308) * 4] =
                        res_32581;
                }
            } else {
                x_32573 = 0;
                x_32574 = 0;
            }
        }
        // combine with carry and write to local memory
        {
            int32_t res_32575;
            int32_t res_32576;
            
            res_32575 = x_32571 + x_32573;
            res_32576 = x_32572 + x_32574;
            *(__local int32_t *) &scan_arr_mem_34669[local_tid_33310 * 4] =
                res_32575;
            *(__local int32_t *) &scan_arr_mem_34671[local_tid_33310 * 4] =
                res_32576;
        }
        
        int32_t x_34663;
        int32_t x_34664;
        int32_t x_34665;
        int32_t x_34666;
        int32_t x_34676;
        int32_t x_34677;
        int32_t x_34678;
        int32_t x_34679;
        int32_t skip_threads_34682;
        
        if (slt32(local_tid_33310, group_sizze_33292)) {
            x_34665 = *(volatile __local
                        int32_t *) &scan_arr_mem_34669[local_tid_33310 *
                                                       sizeof(int32_t)];
            x_34666 = *(volatile __local
                        int32_t *) &scan_arr_mem_34671[local_tid_33310 *
                                                       sizeof(int32_t)];
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_34682 = 1;
            while (slt32(skip_threads_34682, 32)) {
                if (sle32(skip_threads_34682, local_tid_33310 -
                          squot32(local_tid_33310, 32) * 32) &&
                    slt32(local_tid_33310, group_sizze_33292)) {
                    // read operands
                    {
                        x_34663 = *(volatile __local
                                    int32_t *) &scan_arr_mem_34669[(local_tid_33310 -
                                                                    skip_threads_34682) *
                                                                   sizeof(int32_t)];
                        x_34664 = *(volatile __local
                                    int32_t *) &scan_arr_mem_34671[(local_tid_33310 -
                                                                    skip_threads_34682) *
                                                                   sizeof(int32_t)];
                    }
                    // perform operation
                    {
                        if (!slt32(srem32(local_tid_33310 + chunk_offset_34674,
                                          sizze_32279), local_tid_33310 +
                                   chunk_offset_34674 - (local_tid_33310 -
                                                         skip_threads_34682 +
                                                         chunk_offset_34674))) {
                            int32_t res_34667;
                            int32_t res_34668;
                            
                            res_34667 = x_34663 + x_34665;
                            res_34668 = x_34664 + x_34666;
                            x_34665 = res_34667;
                            x_34666 = res_34668;
                        }
                    }
                }
                if (sle32(wave_sizze_34661, skip_threads_34682)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_34682, local_tid_33310 -
                          squot32(local_tid_33310, 32) * 32) &&
                    slt32(local_tid_33310, group_sizze_33292)) {
                    // write result
                    {
                        *(volatile __local
                          int32_t *) &scan_arr_mem_34669[local_tid_33310 *
                                                         sizeof(int32_t)] =
                            x_34665;
                        *(volatile __local
                          int32_t *) &scan_arr_mem_34671[local_tid_33310 *
                                                         sizeof(int32_t)] =
                            x_34666;
                    }
                }
                if (sle32(wave_sizze_34661, skip_threads_34682)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_34682 *= 2;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // last thread of block 'i' writes its result to offset 'i'
        {
            if ((local_tid_33310 - squot32(local_tid_33310, 32) * 32) == 31 &&
                slt32(local_tid_33310, group_sizze_33292)) {
                *(volatile __local
                  int32_t *) &scan_arr_mem_34669[squot32(local_tid_33310, 32) *
                                                 sizeof(int32_t)] = x_34665;
                *(volatile __local
                  int32_t *) &scan_arr_mem_34671[squot32(local_tid_33310, 32) *
                                                 sizeof(int32_t)] = x_34666;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'
        {
            int32_t skip_threads_34683;
            
            if (squot32(local_tid_33310, 32) == 0 && slt32(local_tid_33310,
                                                           group_sizze_33292)) {
                x_34678 = *(volatile __local
                            int32_t *) &scan_arr_mem_34669[local_tid_33310 *
                                                           sizeof(int32_t)];
                x_34679 = *(volatile __local
                            int32_t *) &scan_arr_mem_34671[local_tid_33310 *
                                                           sizeof(int32_t)];
            }
            // in-block scan (hopefully no barriers needed)
            {
                skip_threads_34683 = 1;
                while (slt32(skip_threads_34683, 32)) {
                    if (sle32(skip_threads_34683, local_tid_33310 -
                              squot32(local_tid_33310, 32) * 32) &&
                        (squot32(local_tid_33310, 32) == 0 &&
                         slt32(local_tid_33310, group_sizze_33292))) {
                        // read operands
                        {
                            x_34676 = *(volatile __local
                                        int32_t *) &scan_arr_mem_34669[(local_tid_33310 -
                                                                        skip_threads_34683) *
                                                                       sizeof(int32_t)];
                            x_34677 = *(volatile __local
                                        int32_t *) &scan_arr_mem_34671[(local_tid_33310 -
                                                                        skip_threads_34683) *
                                                                       sizeof(int32_t)];
                        }
                        // perform operation
                        {
                            if (!slt32(srem32(local_tid_33310 * 32 + 32 - 1 +
                                              chunk_offset_34674, sizze_32279),
                                       local_tid_33310 * 32 + 32 - 1 +
                                       chunk_offset_34674 - ((local_tid_33310 -
                                                              skip_threads_34683) *
                                                             32 + 32 - 1 +
                                                             chunk_offset_34674))) {
                                int32_t res_34680;
                                int32_t res_34681;
                                
                                res_34680 = x_34676 + x_34678;
                                res_34681 = x_34677 + x_34679;
                                x_34678 = res_34680;
                                x_34679 = res_34681;
                            }
                        }
                    }
                    if (sle32(wave_sizze_34661, skip_threads_34683)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    if (sle32(skip_threads_34683, local_tid_33310 -
                              squot32(local_tid_33310, 32) * 32) &&
                        (squot32(local_tid_33310, 32) == 0 &&
                         slt32(local_tid_33310, group_sizze_33292))) {
                        // write result
                        {
                            *(volatile __local
                              int32_t *) &scan_arr_mem_34669[local_tid_33310 *
                                                             sizeof(int32_t)] =
                                x_34678;
                            *(volatile __local
                              int32_t *) &scan_arr_mem_34671[local_tid_33310 *
                                                             sizeof(int32_t)] =
                                x_34679;
                        }
                    }
                    if (sle32(wave_sizze_34661, skip_threads_34683)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    skip_threads_34683 *= 2;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // carry-in for every block except the first
        {
            if (!(squot32(local_tid_33310, 32) == 0 || !slt32(local_tid_33310,
                                                              group_sizze_33292))) {
                // read operands
                {
                    x_34663 = *(volatile __local
                                int32_t *) &scan_arr_mem_34669[(squot32(local_tid_33310,
                                                                        32) -
                                                                1) *
                                                               sizeof(int32_t)];
                    x_34664 = *(volatile __local
                                int32_t *) &scan_arr_mem_34671[(squot32(local_tid_33310,
                                                                        32) -
                                                                1) *
                                                               sizeof(int32_t)];
                }
                // perform operation
                {
                    if (!slt32(srem32(local_tid_33310 + chunk_offset_34674,
                                      sizze_32279), local_tid_33310 +
                               chunk_offset_34674 - (squot32(local_tid_33310,
                                                             32) * 32 - 1 +
                                                     chunk_offset_34674))) {
                        int32_t res_34667;
                        int32_t res_34668;
                        
                        res_34667 = x_34663 + x_34665;
                        res_34668 = x_34664 + x_34666;
                        x_34665 = res_34667;
                        x_34666 = res_34668;
                    }
                }
                // write final result
                {
                    *(volatile __local
                      int32_t *) &scan_arr_mem_34669[local_tid_33310 *
                                                     sizeof(int32_t)] = x_34665;
                    *(volatile __local
                      int32_t *) &scan_arr_mem_34671[local_tid_33310 *
                                                     sizeof(int32_t)] = x_34666;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // restore correct values for first block
        {
            if (squot32(local_tid_33310, 32) == 0) {
                *(volatile __local
                  int32_t *) &scan_arr_mem_34669[local_tid_33310 *
                                                 sizeof(int32_t)] = x_34665;
                *(volatile __local
                  int32_t *) &scan_arr_mem_34671[local_tid_33310 *
                                                 sizeof(int32_t)] = x_34666;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // threads in bounds write partial scan result
        {
            if (slt32(gtid_33286, sizze_32280) && slt32(gtid_33308,
                                                        sizze_32279)) {
                *(__global int32_t *) &mem_34431[(gtid_33286 * sizze_32279 +
                                                  gtid_33308) * 4] = *(__local
                                                                       int32_t *) &scan_arr_mem_34669[local_tid_33310 *
                                                                                                      4];
                *(__global int32_t *) &mem_34435[(gtid_33286 * sizze_32279 +
                                                  gtid_33308) * 4] = *(__local
                                                                       int32_t *) &scan_arr_mem_34671[local_tid_33310 *
                                                                                                      4];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // first thread reads last element as carry-in for next iteration
        {
            if (local_tid_33310 == 0) {
                if (slt32(srem32(chunk_offset_34674 + group_sizze_33292,
                                 sizze_32279), chunk_offset_34674 +
                          group_sizze_33292 - (chunk_offset_34674 +
                                               group_sizze_33292 - 1))) {
                    x_32571 = 0;
                    x_32572 = 0;
                } else {
                    x_32571 = *(__local
                                int32_t *) &scan_arr_mem_34669[(group_sizze_33292 -
                                                                1) * 4];
                    x_32572 = *(__local
                                int32_t *) &scan_arr_mem_34671[(group_sizze_33292 -
                                                                1) * 4];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
__kernel void scan_stage1_33615(int32_t sizze_32279, int32_t sizze_32280,
                                int32_t num_elems_32669,
                                int32_t num_groups_33609, __global
                                unsigned char *mem_34449, __global
                                unsigned char *mem_34460, __global
                                unsigned char *mem_34466, __global
                                unsigned char *mem_34469, __global
                                unsigned char *mem_34478, __global
                                unsigned char *mem_34482)
{
    const int32_t group_sizze_33598 = mainzigroup_sizze_33597;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    ALIGNED_LOCAL_MEMORY(scan_arr_mem_34754_backing_0, 4 *
                         mainzigroup_sizze_33597);
    
    int32_t global_tid_33615;
    int32_t local_tid_33616;
    int32_t group_sizze_34750;
    int32_t wave_sizze_34749;
    int32_t group_id_33617;
    
    global_tid_33615 = get_global_id(0);
    local_tid_33616 = get_local_id(0);
    group_sizze_34750 = get_local_size(0);
    wave_sizze_34749 = LOCKSTEP_WIDTH;
    group_id_33617 = get_group_id(0);
    
    int32_t gtid_33593;
    int32_t gtid_33614;
    __local char *scan_arr_mem_34754;
    
    scan_arr_mem_34754 = (__local char *) scan_arr_mem_34754_backing_0;
    
    float x_32743;
    float x_32744;
    
    x_32743 = 0.0F;
    for (int32_t j_34756 = 0; j_34756 < squot32(sizze_32280 * num_elems_32669 +
                                                group_sizze_33598 *
                                                num_groups_33609 - 1,
                                                group_sizze_33598 *
                                                num_groups_33609); j_34756++) {
        int32_t chunk_offset_34757 = group_sizze_33598 * j_34756 +
                group_id_33617 * (group_sizze_33598 * squot32(sizze_32280 *
                                                              num_elems_32669 +
                                                              group_sizze_33598 *
                                                              num_groups_33609 -
                                                              1,
                                                              group_sizze_33598 *
                                                              num_groups_33609));
        int32_t flat_idx_34758 = chunk_offset_34757 + local_tid_33616;
        
        gtid_33593 = squot32(flat_idx_34758, num_elems_32669);
        gtid_33614 = flat_idx_34758 - squot32(flat_idx_34758, num_elems_32669) *
            num_elems_32669;
        // threads in bounds read input; others get neutral element
        {
            if (slt32(gtid_33593, sizze_32280) && slt32(gtid_33614,
                                                        num_elems_32669)) {
                int32_t x_32718;
                int32_t x_32720;
                float x_32721;
                int32_t y_32725;
                bool cond_32747;
                float res_32748;
                
                x_32718 = *(__global int32_t *) &mem_34460[gtid_33593 * 4];
                x_32720 = *(__global int32_t *) &mem_34469[gtid_33593 * 4];
                x_32721 = *(__global float *) &mem_34466[gtid_33593 * 4];
                y_32725 = *(__global int32_t *) &mem_34478[gtid_33593 * 4];
                cond_32747 = sle32(y_32725, gtid_33614);
                if (cond_32747) {
                    res_32748 = 0.0F;
                } else {
                    bool cond_32749;
                    float res_32750;
                    
                    cond_32749 = gtid_33614 == 0;
                    if (cond_32749) {
                        res_32750 = x_32721;
                    } else {
                        int32_t x_32751;
                        int32_t i_32752;
                        float negate_arg_32753;
                        float x_32754;
                        int32_t i_32755;
                        float y_32756;
                        float res_32757;
                        
                        x_32751 = x_32718 - x_32720;
                        i_32752 = x_32751 + gtid_33614;
                        negate_arg_32753 = *(__global
                                             float *) &mem_34449[(gtid_33593 *
                                                                  sizze_32279 +
                                                                  i_32752) * 4];
                        x_32754 = 0.0F - negate_arg_32753;
                        i_32755 = x_32718 + gtid_33614;
                        y_32756 = *(__global float *) &mem_34449[(gtid_33593 *
                                                                  sizze_32279 +
                                                                  i_32755) * 4];
                        res_32757 = x_32754 + y_32756;
                        res_32750 = res_32757;
                    }
                    res_32748 = res_32750;
                }
                // write to-scan values to parameters
                {
                    x_32744 = res_32748;
                }
                // write mapped values results to global memory
                { }
            } else {
                x_32744 = 0.0F;
            }
        }
        // combine with carry and write to local memory
        {
            float res_32745 = x_32743 + x_32744;
            
            *(__local float *) &scan_arr_mem_34754[local_tid_33616 * 4] =
                res_32745;
        }
        
        float x_34751;
        float x_34752;
        float x_34759;
        float x_34760;
        int32_t skip_threads_34762;
        
        if (slt32(local_tid_33616, group_sizze_33598)) {
            x_34752 = *(volatile __local
                        float *) &scan_arr_mem_34754[local_tid_33616 *
                                                     sizeof(float)];
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_34762 = 1;
            while (slt32(skip_threads_34762, 32)) {
                if (sle32(skip_threads_34762, local_tid_33616 -
                          squot32(local_tid_33616, 32) * 32) &&
                    slt32(local_tid_33616, group_sizze_33598)) {
                    // read operands
                    {
                        x_34751 = *(volatile __local
                                    float *) &scan_arr_mem_34754[(local_tid_33616 -
                                                                  skip_threads_34762) *
                                                                 sizeof(float)];
                    }
                    // perform operation
                    {
                        if (!slt32(srem32(local_tid_33616 + chunk_offset_34757,
                                          num_elems_32669), local_tid_33616 +
                                   chunk_offset_34757 - (local_tid_33616 -
                                                         skip_threads_34762 +
                                                         chunk_offset_34757))) {
                            float res_34753 = x_34751 + x_34752;
                            
                            x_34752 = res_34753;
                        }
                    }
                }
                if (sle32(wave_sizze_34749, skip_threads_34762)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_34762, local_tid_33616 -
                          squot32(local_tid_33616, 32) * 32) &&
                    slt32(local_tid_33616, group_sizze_33598)) {
                    // write result
                    {
                        *(volatile __local
                          float *) &scan_arr_mem_34754[local_tid_33616 *
                                                       sizeof(float)] = x_34752;
                    }
                }
                if (sle32(wave_sizze_34749, skip_threads_34762)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_34762 *= 2;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // last thread of block 'i' writes its result to offset 'i'
        {
            if ((local_tid_33616 - squot32(local_tid_33616, 32) * 32) == 31 &&
                slt32(local_tid_33616, group_sizze_33598)) {
                *(volatile __local
                  float *) &scan_arr_mem_34754[squot32(local_tid_33616, 32) *
                                               sizeof(float)] = x_34752;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'
        {
            int32_t skip_threads_34763;
            
            if (squot32(local_tid_33616, 32) == 0 && slt32(local_tid_33616,
                                                           group_sizze_33598)) {
                x_34760 = *(volatile __local
                            float *) &scan_arr_mem_34754[local_tid_33616 *
                                                         sizeof(float)];
            }
            // in-block scan (hopefully no barriers needed)
            {
                skip_threads_34763 = 1;
                while (slt32(skip_threads_34763, 32)) {
                    if (sle32(skip_threads_34763, local_tid_33616 -
                              squot32(local_tid_33616, 32) * 32) &&
                        (squot32(local_tid_33616, 32) == 0 &&
                         slt32(local_tid_33616, group_sizze_33598))) {
                        // read operands
                        {
                            x_34759 = *(volatile __local
                                        float *) &scan_arr_mem_34754[(local_tid_33616 -
                                                                      skip_threads_34763) *
                                                                     sizeof(float)];
                        }
                        // perform operation
                        {
                            if (!slt32(srem32(local_tid_33616 * 32 + 32 - 1 +
                                              chunk_offset_34757,
                                              num_elems_32669),
                                       local_tid_33616 * 32 + 32 - 1 +
                                       chunk_offset_34757 - ((local_tid_33616 -
                                                              skip_threads_34763) *
                                                             32 + 32 - 1 +
                                                             chunk_offset_34757))) {
                                float res_34761 = x_34759 + x_34760;
                                
                                x_34760 = res_34761;
                            }
                        }
                    }
                    if (sle32(wave_sizze_34749, skip_threads_34763)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    if (sle32(skip_threads_34763, local_tid_33616 -
                              squot32(local_tid_33616, 32) * 32) &&
                        (squot32(local_tid_33616, 32) == 0 &&
                         slt32(local_tid_33616, group_sizze_33598))) {
                        // write result
                        {
                            *(volatile __local
                              float *) &scan_arr_mem_34754[local_tid_33616 *
                                                           sizeof(float)] =
                                x_34760;
                        }
                    }
                    if (sle32(wave_sizze_34749, skip_threads_34763)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    skip_threads_34763 *= 2;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // carry-in for every block except the first
        {
            if (!(squot32(local_tid_33616, 32) == 0 || !slt32(local_tid_33616,
                                                              group_sizze_33598))) {
                // read operands
                {
                    x_34751 = *(volatile __local
                                float *) &scan_arr_mem_34754[(squot32(local_tid_33616,
                                                                      32) - 1) *
                                                             sizeof(float)];
                }
                // perform operation
                {
                    if (!slt32(srem32(local_tid_33616 + chunk_offset_34757,
                                      num_elems_32669), local_tid_33616 +
                               chunk_offset_34757 - (squot32(local_tid_33616,
                                                             32) * 32 - 1 +
                                                     chunk_offset_34757))) {
                        float res_34753 = x_34751 + x_34752;
                        
                        x_34752 = res_34753;
                    }
                }
                // write final result
                {
                    *(volatile __local
                      float *) &scan_arr_mem_34754[local_tid_33616 *
                                                   sizeof(float)] = x_34752;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // restore correct values for first block
        {
            if (squot32(local_tid_33616, 32) == 0) {
                *(volatile __local
                  float *) &scan_arr_mem_34754[local_tid_33616 *
                                               sizeof(float)] = x_34752;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // threads in bounds write partial scan result
        {
            if (slt32(gtid_33593, sizze_32280) && slt32(gtid_33614,
                                                        num_elems_32669)) {
                *(__global float *) &mem_34482[(gtid_33593 * num_elems_32669 +
                                                gtid_33614) * 4] = *(__local
                                                                     float *) &scan_arr_mem_34754[local_tid_33616 *
                                                                                                  4];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // first thread reads last element as carry-in for next iteration
        {
            if (local_tid_33616 == 0) {
                if (slt32(srem32(chunk_offset_34757 + group_sizze_33598,
                                 num_elems_32669), chunk_offset_34757 +
                          group_sizze_33598 - (chunk_offset_34757 +
                                               group_sizze_33598 - 1))) {
                    x_32743 = 0.0F;
                } else {
                    x_32743 = *(__local
                                float *) &scan_arr_mem_34754[(group_sizze_33598 -
                                                              1) * 4];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
__kernel void scan_stage2_34696(__local volatile
                                int64_t *scan_arr_mem_34701_backing_aligned_0,
                                __local volatile
                                int64_t *scan_arr_mem_34703_backing_aligned_1,
                                int32_t sizze_32279, int32_t sizze_32280,
                                int32_t num_groups_33303, __global
                                unsigned char *mem_34431, __global
                                unsigned char *mem_34435)
{
    const int32_t group_sizze_33292 = mainzigroup_sizze_33291;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_34701_backing_0 =
                          scan_arr_mem_34701_backing_aligned_0;
    __local volatile char *restrict scan_arr_mem_34703_backing_1 =
                          scan_arr_mem_34703_backing_aligned_1;
    int32_t global_tid_34696;
    int32_t local_tid_34697;
    int32_t group_sizze_34700;
    int32_t wave_sizze_34699;
    int32_t group_id_34698;
    
    global_tid_34696 = get_global_id(0);
    local_tid_34697 = get_local_id(0);
    group_sizze_34700 = get_local_size(0);
    wave_sizze_34699 = LOCKSTEP_WIDTH;
    group_id_34698 = get_group_id(0);
    
    __local char *scan_arr_mem_34701;
    
    scan_arr_mem_34701 = (__local char *) scan_arr_mem_34701_backing_0;
    
    __local char *scan_arr_mem_34703;
    
    scan_arr_mem_34703 = (__local char *) scan_arr_mem_34703_backing_1;
    
    int32_t flat_idx_34705 = (local_tid_34697 + 1) * (group_sizze_33292 *
                                                      squot32(sizze_32280 *
                                                              sizze_32279 +
                                                              group_sizze_33292 *
                                                              num_groups_33303 -
                                                              1,
                                                              group_sizze_33292 *
                                                              num_groups_33303)) -
            1;
    int32_t gtid_33286 = squot32(flat_idx_34705, sizze_32279);
    int32_t gtid_33308;
    
    gtid_33308 = flat_idx_34705 - squot32(flat_idx_34705, sizze_32279) *
        sizze_32279;
    // threads in bound read carries; others get neutral element
    {
        if (slt32(gtid_33286, sizze_32280) && slt32(gtid_33308, sizze_32279)) {
            *(__local int32_t *) &scan_arr_mem_34701[local_tid_34697 * 4] =
                *(__global int32_t *) &mem_34431[(gtid_33286 * sizze_32279 +
                                                  gtid_33308) * 4];
            *(__local int32_t *) &scan_arr_mem_34703[local_tid_34697 * 4] =
                *(__global int32_t *) &mem_34435[(gtid_33286 * sizze_32279 +
                                                  gtid_33308) * 4];
        } else {
            *(__local int32_t *) &scan_arr_mem_34701[local_tid_34697 * 4] = 0;
            *(__local int32_t *) &scan_arr_mem_34703[local_tid_34697 * 4] = 0;
        }
    }
    
    int32_t x_34684;
    int32_t x_34685;
    int32_t x_34686;
    int32_t x_34687;
    int32_t x_34706;
    int32_t x_34707;
    int32_t x_34708;
    int32_t x_34709;
    int32_t skip_threads_34712;
    
    if (slt32(local_tid_34697, num_groups_33303)) {
        x_34686 = *(volatile __local
                    int32_t *) &scan_arr_mem_34701[local_tid_34697 *
                                                   sizeof(int32_t)];
        x_34687 = *(volatile __local
                    int32_t *) &scan_arr_mem_34703[local_tid_34697 *
                                                   sizeof(int32_t)];
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_34712 = 1;
        while (slt32(skip_threads_34712, 32)) {
            if (sle32(skip_threads_34712, local_tid_34697 -
                      squot32(local_tid_34697, 32) * 32) &&
                slt32(local_tid_34697, num_groups_33303)) {
                // read operands
                {
                    x_34684 = *(volatile __local
                                int32_t *) &scan_arr_mem_34701[(local_tid_34697 -
                                                                skip_threads_34712) *
                                                               sizeof(int32_t)];
                    x_34685 = *(volatile __local
                                int32_t *) &scan_arr_mem_34703[(local_tid_34697 -
                                                                skip_threads_34712) *
                                                               sizeof(int32_t)];
                }
                // perform operation
                {
                    if (!slt32(srem32((local_tid_34697 + 1) *
                                      (group_sizze_33292 * squot32(sizze_32280 *
                                                                   sizze_32279 +
                                                                   group_sizze_33292 *
                                                                   num_groups_33303 -
                                                                   1,
                                                                   group_sizze_33292 *
                                                                   num_groups_33303)) -
                                      1, sizze_32279), (local_tid_34697 + 1) *
                               (group_sizze_33292 * squot32(sizze_32280 *
                                                            sizze_32279 +
                                                            group_sizze_33292 *
                                                            num_groups_33303 -
                                                            1,
                                                            group_sizze_33292 *
                                                            num_groups_33303)) -
                               1 - ((local_tid_34697 - skip_threads_34712 + 1) *
                                    (group_sizze_33292 * squot32(sizze_32280 *
                                                                 sizze_32279 +
                                                                 group_sizze_33292 *
                                                                 num_groups_33303 -
                                                                 1,
                                                                 group_sizze_33292 *
                                                                 num_groups_33303)) -
                                    1))) {
                        int32_t res_34688;
                        int32_t res_34689;
                        
                        res_34688 = x_34684 + x_34686;
                        res_34689 = x_34685 + x_34687;
                        x_34686 = res_34688;
                        x_34687 = res_34689;
                    }
                }
            }
            if (sle32(wave_sizze_34699, skip_threads_34712)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_34712, local_tid_34697 -
                      squot32(local_tid_34697, 32) * 32) &&
                slt32(local_tid_34697, num_groups_33303)) {
                // write result
                {
                    *(volatile __local
                      int32_t *) &scan_arr_mem_34701[local_tid_34697 *
                                                     sizeof(int32_t)] = x_34686;
                    *(volatile __local
                      int32_t *) &scan_arr_mem_34703[local_tid_34697 *
                                                     sizeof(int32_t)] = x_34687;
                }
            }
            if (sle32(wave_sizze_34699, skip_threads_34712)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_34712 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_34697 - squot32(local_tid_34697, 32) * 32) == 31 &&
            slt32(local_tid_34697, num_groups_33303)) {
            *(volatile __local
              int32_t *) &scan_arr_mem_34701[squot32(local_tid_34697, 32) *
                                             sizeof(int32_t)] = x_34686;
            *(volatile __local
              int32_t *) &scan_arr_mem_34703[squot32(local_tid_34697, 32) *
                                             sizeof(int32_t)] = x_34687;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'
    {
        int32_t skip_threads_34713;
        
        if (squot32(local_tid_34697, 32) == 0 && slt32(local_tid_34697,
                                                       num_groups_33303)) {
            x_34708 = *(volatile __local
                        int32_t *) &scan_arr_mem_34701[local_tid_34697 *
                                                       sizeof(int32_t)];
            x_34709 = *(volatile __local
                        int32_t *) &scan_arr_mem_34703[local_tid_34697 *
                                                       sizeof(int32_t)];
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_34713 = 1;
            while (slt32(skip_threads_34713, 32)) {
                if (sle32(skip_threads_34713, local_tid_34697 -
                          squot32(local_tid_34697, 32) * 32) &&
                    (squot32(local_tid_34697, 32) == 0 && slt32(local_tid_34697,
                                                                num_groups_33303))) {
                    // read operands
                    {
                        x_34706 = *(volatile __local
                                    int32_t *) &scan_arr_mem_34701[(local_tid_34697 -
                                                                    skip_threads_34713) *
                                                                   sizeof(int32_t)];
                        x_34707 = *(volatile __local
                                    int32_t *) &scan_arr_mem_34703[(local_tid_34697 -
                                                                    skip_threads_34713) *
                                                                   sizeof(int32_t)];
                    }
                    // perform operation
                    {
                        if (!slt32(srem32((local_tid_34697 * 32 + 32 - 1 + 1) *
                                          (group_sizze_33292 *
                                           squot32(sizze_32280 * sizze_32279 +
                                                   group_sizze_33292 *
                                                   num_groups_33303 - 1,
                                                   group_sizze_33292 *
                                                   num_groups_33303)) - 1,
                                          sizze_32279), (local_tid_34697 * 32 +
                                                         32 - 1 + 1) *
                                   (group_sizze_33292 * squot32(sizze_32280 *
                                                                sizze_32279 +
                                                                group_sizze_33292 *
                                                                num_groups_33303 -
                                                                1,
                                                                group_sizze_33292 *
                                                                num_groups_33303)) -
                                   1 - (((local_tid_34697 -
                                          skip_threads_34713) * 32 + 32 - 1 +
                                         1) * (group_sizze_33292 *
                                               squot32(sizze_32280 *
                                                       sizze_32279 +
                                                       group_sizze_33292 *
                                                       num_groups_33303 - 1,
                                                       group_sizze_33292 *
                                                       num_groups_33303)) -
                                        1))) {
                            int32_t res_34710;
                            int32_t res_34711;
                            
                            res_34710 = x_34706 + x_34708;
                            res_34711 = x_34707 + x_34709;
                            x_34708 = res_34710;
                            x_34709 = res_34711;
                        }
                    }
                }
                if (sle32(wave_sizze_34699, skip_threads_34713)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_34713, local_tid_34697 -
                          squot32(local_tid_34697, 32) * 32) &&
                    (squot32(local_tid_34697, 32) == 0 && slt32(local_tid_34697,
                                                                num_groups_33303))) {
                    // write result
                    {
                        *(volatile __local
                          int32_t *) &scan_arr_mem_34701[local_tid_34697 *
                                                         sizeof(int32_t)] =
                            x_34708;
                        *(volatile __local
                          int32_t *) &scan_arr_mem_34703[local_tid_34697 *
                                                         sizeof(int32_t)] =
                            x_34709;
                    }
                }
                if (sle32(wave_sizze_34699, skip_threads_34713)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_34713 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_34697, 32) == 0 || !slt32(local_tid_34697,
                                                          num_groups_33303))) {
            // read operands
            {
                x_34684 = *(volatile __local
                            int32_t *) &scan_arr_mem_34701[(squot32(local_tid_34697,
                                                                    32) - 1) *
                                                           sizeof(int32_t)];
                x_34685 = *(volatile __local
                            int32_t *) &scan_arr_mem_34703[(squot32(local_tid_34697,
                                                                    32) - 1) *
                                                           sizeof(int32_t)];
            }
            // perform operation
            {
                if (!slt32(srem32((local_tid_34697 + 1) * (group_sizze_33292 *
                                                           squot32(sizze_32280 *
                                                                   sizze_32279 +
                                                                   group_sizze_33292 *
                                                                   num_groups_33303 -
                                                                   1,
                                                                   group_sizze_33292 *
                                                                   num_groups_33303)) -
                                  1, sizze_32279), (local_tid_34697 + 1) *
                           (group_sizze_33292 * squot32(sizze_32280 *
                                                        sizze_32279 +
                                                        group_sizze_33292 *
                                                        num_groups_33303 - 1,
                                                        group_sizze_33292 *
                                                        num_groups_33303)) - 1 -
                           ((squot32(local_tid_34697, 32) * 32 - 1 + 1) *
                            (group_sizze_33292 * squot32(sizze_32280 *
                                                         sizze_32279 +
                                                         group_sizze_33292 *
                                                         num_groups_33303 - 1,
                                                         group_sizze_33292 *
                                                         num_groups_33303)) -
                            1))) {
                    int32_t res_34688;
                    int32_t res_34689;
                    
                    res_34688 = x_34684 + x_34686;
                    res_34689 = x_34685 + x_34687;
                    x_34686 = res_34688;
                    x_34687 = res_34689;
                }
            }
            // write final result
            {
                *(volatile __local
                  int32_t *) &scan_arr_mem_34701[local_tid_34697 *
                                                 sizeof(int32_t)] = x_34686;
                *(volatile __local
                  int32_t *) &scan_arr_mem_34703[local_tid_34697 *
                                                 sizeof(int32_t)] = x_34687;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_34697, 32) == 0) {
            *(volatile __local int32_t *) &scan_arr_mem_34701[local_tid_34697 *
                                                              sizeof(int32_t)] =
                x_34686;
            *(volatile __local int32_t *) &scan_arr_mem_34703[local_tid_34697 *
                                                              sizeof(int32_t)] =
                x_34687;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // threads in bounds write scanned carries
    {
        if (slt32(gtid_33286, sizze_32280) && slt32(gtid_33308, sizze_32279)) {
            *(__global int32_t *) &mem_34431[(gtid_33286 * sizze_32279 +
                                              gtid_33308) * 4] = *(__local
                                                                   int32_t *) &scan_arr_mem_34701[local_tid_34697 *
                                                                                                  4];
            *(__global int32_t *) &mem_34435[(gtid_33286 * sizze_32279 +
                                              gtid_33308) * 4] = *(__local
                                                                   int32_t *) &scan_arr_mem_34703[local_tid_34697 *
                                                                                                  4];
        }
    }
}
__kernel void scan_stage2_34770(__local volatile
                                int64_t *scan_arr_mem_34775_backing_aligned_0,
                                int32_t sizze_32280, int32_t num_elems_32669,
                                int32_t num_groups_33609, __global
                                unsigned char *mem_34482)
{
    const int32_t group_sizze_33598 = mainzigroup_sizze_33597;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_34775_backing_0 =
                          scan_arr_mem_34775_backing_aligned_0;
    int32_t global_tid_34770;
    int32_t local_tid_34771;
    int32_t group_sizze_34774;
    int32_t wave_sizze_34773;
    int32_t group_id_34772;
    
    global_tid_34770 = get_global_id(0);
    local_tid_34771 = get_local_id(0);
    group_sizze_34774 = get_local_size(0);
    wave_sizze_34773 = LOCKSTEP_WIDTH;
    group_id_34772 = get_group_id(0);
    
    __local char *scan_arr_mem_34775;
    
    scan_arr_mem_34775 = (__local char *) scan_arr_mem_34775_backing_0;
    
    int32_t flat_idx_34777 = (local_tid_34771 + 1) * (group_sizze_33598 *
                                                      squot32(sizze_32280 *
                                                              num_elems_32669 +
                                                              group_sizze_33598 *
                                                              num_groups_33609 -
                                                              1,
                                                              group_sizze_33598 *
                                                              num_groups_33609)) -
            1;
    int32_t gtid_33593 = squot32(flat_idx_34777, num_elems_32669);
    int32_t gtid_33614;
    
    gtid_33614 = flat_idx_34777 - squot32(flat_idx_34777, num_elems_32669) *
        num_elems_32669;
    // threads in bound read carries; others get neutral element
    {
        if (slt32(gtid_33593, sizze_32280) && slt32(gtid_33614,
                                                    num_elems_32669)) {
            *(__local float *) &scan_arr_mem_34775[local_tid_34771 * 4] =
                *(__global float *) &mem_34482[(gtid_33593 * num_elems_32669 +
                                                gtid_33614) * 4];
        } else {
            *(__local float *) &scan_arr_mem_34775[local_tid_34771 * 4] = 0.0F;
        }
    }
    
    float x_34764;
    float x_34765;
    float x_34778;
    float x_34779;
    int32_t skip_threads_34781;
    
    if (slt32(local_tid_34771, num_groups_33609)) {
        x_34765 = *(volatile __local
                    float *) &scan_arr_mem_34775[local_tid_34771 *
                                                 sizeof(float)];
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_34781 = 1;
        while (slt32(skip_threads_34781, 32)) {
            if (sle32(skip_threads_34781, local_tid_34771 -
                      squot32(local_tid_34771, 32) * 32) &&
                slt32(local_tid_34771, num_groups_33609)) {
                // read operands
                {
                    x_34764 = *(volatile __local
                                float *) &scan_arr_mem_34775[(local_tid_34771 -
                                                              skip_threads_34781) *
                                                             sizeof(float)];
                }
                // perform operation
                {
                    if (!slt32(srem32((local_tid_34771 + 1) *
                                      (group_sizze_33598 * squot32(sizze_32280 *
                                                                   num_elems_32669 +
                                                                   group_sizze_33598 *
                                                                   num_groups_33609 -
                                                                   1,
                                                                   group_sizze_33598 *
                                                                   num_groups_33609)) -
                                      1, num_elems_32669), (local_tid_34771 +
                                                            1) *
                               (group_sizze_33598 * squot32(sizze_32280 *
                                                            num_elems_32669 +
                                                            group_sizze_33598 *
                                                            num_groups_33609 -
                                                            1,
                                                            group_sizze_33598 *
                                                            num_groups_33609)) -
                               1 - ((local_tid_34771 - skip_threads_34781 + 1) *
                                    (group_sizze_33598 * squot32(sizze_32280 *
                                                                 num_elems_32669 +
                                                                 group_sizze_33598 *
                                                                 num_groups_33609 -
                                                                 1,
                                                                 group_sizze_33598 *
                                                                 num_groups_33609)) -
                                    1))) {
                        float res_34766 = x_34764 + x_34765;
                        
                        x_34765 = res_34766;
                    }
                }
            }
            if (sle32(wave_sizze_34773, skip_threads_34781)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_34781, local_tid_34771 -
                      squot32(local_tid_34771, 32) * 32) &&
                slt32(local_tid_34771, num_groups_33609)) {
                // write result
                {
                    *(volatile __local
                      float *) &scan_arr_mem_34775[local_tid_34771 *
                                                   sizeof(float)] = x_34765;
                }
            }
            if (sle32(wave_sizze_34773, skip_threads_34781)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_34781 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_34771 - squot32(local_tid_34771, 32) * 32) == 31 &&
            slt32(local_tid_34771, num_groups_33609)) {
            *(volatile __local
              float *) &scan_arr_mem_34775[squot32(local_tid_34771, 32) *
                                           sizeof(float)] = x_34765;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'
    {
        int32_t skip_threads_34782;
        
        if (squot32(local_tid_34771, 32) == 0 && slt32(local_tid_34771,
                                                       num_groups_33609)) {
            x_34779 = *(volatile __local
                        float *) &scan_arr_mem_34775[local_tid_34771 *
                                                     sizeof(float)];
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_34782 = 1;
            while (slt32(skip_threads_34782, 32)) {
                if (sle32(skip_threads_34782, local_tid_34771 -
                          squot32(local_tid_34771, 32) * 32) &&
                    (squot32(local_tid_34771, 32) == 0 && slt32(local_tid_34771,
                                                                num_groups_33609))) {
                    // read operands
                    {
                        x_34778 = *(volatile __local
                                    float *) &scan_arr_mem_34775[(local_tid_34771 -
                                                                  skip_threads_34782) *
                                                                 sizeof(float)];
                    }
                    // perform operation
                    {
                        if (!slt32(srem32((local_tid_34771 * 32 + 32 - 1 + 1) *
                                          (group_sizze_33598 *
                                           squot32(sizze_32280 *
                                                   num_elems_32669 +
                                                   group_sizze_33598 *
                                                   num_groups_33609 - 1,
                                                   group_sizze_33598 *
                                                   num_groups_33609)) - 1,
                                          num_elems_32669), (local_tid_34771 *
                                                             32 + 32 - 1 + 1) *
                                   (group_sizze_33598 * squot32(sizze_32280 *
                                                                num_elems_32669 +
                                                                group_sizze_33598 *
                                                                num_groups_33609 -
                                                                1,
                                                                group_sizze_33598 *
                                                                num_groups_33609)) -
                                   1 - (((local_tid_34771 -
                                          skip_threads_34782) * 32 + 32 - 1 +
                                         1) * (group_sizze_33598 *
                                               squot32(sizze_32280 *
                                                       num_elems_32669 +
                                                       group_sizze_33598 *
                                                       num_groups_33609 - 1,
                                                       group_sizze_33598 *
                                                       num_groups_33609)) -
                                        1))) {
                            float res_34780 = x_34778 + x_34779;
                            
                            x_34779 = res_34780;
                        }
                    }
                }
                if (sle32(wave_sizze_34773, skip_threads_34782)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_34782, local_tid_34771 -
                          squot32(local_tid_34771, 32) * 32) &&
                    (squot32(local_tid_34771, 32) == 0 && slt32(local_tid_34771,
                                                                num_groups_33609))) {
                    // write result
                    {
                        *(volatile __local
                          float *) &scan_arr_mem_34775[local_tid_34771 *
                                                       sizeof(float)] = x_34779;
                    }
                }
                if (sle32(wave_sizze_34773, skip_threads_34782)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_34782 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_34771, 32) == 0 || !slt32(local_tid_34771,
                                                          num_groups_33609))) {
            // read operands
            {
                x_34764 = *(volatile __local
                            float *) &scan_arr_mem_34775[(squot32(local_tid_34771,
                                                                  32) - 1) *
                                                         sizeof(float)];
            }
            // perform operation
            {
                if (!slt32(srem32((local_tid_34771 + 1) * (group_sizze_33598 *
                                                           squot32(sizze_32280 *
                                                                   num_elems_32669 +
                                                                   group_sizze_33598 *
                                                                   num_groups_33609 -
                                                                   1,
                                                                   group_sizze_33598 *
                                                                   num_groups_33609)) -
                                  1, num_elems_32669), (local_tid_34771 + 1) *
                           (group_sizze_33598 * squot32(sizze_32280 *
                                                        num_elems_32669 +
                                                        group_sizze_33598 *
                                                        num_groups_33609 - 1,
                                                        group_sizze_33598 *
                                                        num_groups_33609)) - 1 -
                           ((squot32(local_tid_34771, 32) * 32 - 1 + 1) *
                            (group_sizze_33598 * squot32(sizze_32280 *
                                                         num_elems_32669 +
                                                         group_sizze_33598 *
                                                         num_groups_33609 - 1,
                                                         group_sizze_33598 *
                                                         num_groups_33609)) -
                            1))) {
                    float res_34766 = x_34764 + x_34765;
                    
                    x_34765 = res_34766;
                }
            }
            // write final result
            {
                *(volatile __local
                  float *) &scan_arr_mem_34775[local_tid_34771 *
                                               sizeof(float)] = x_34765;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_34771, 32) == 0) {
            *(volatile __local float *) &scan_arr_mem_34775[local_tid_34771 *
                                                            sizeof(float)] =
                x_34765;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // threads in bounds write scanned carries
    {
        if (slt32(gtid_33593, sizze_32280) && slt32(gtid_33614,
                                                    num_elems_32669)) {
            *(__global float *) &mem_34482[(gtid_33593 * num_elems_32669 +
                                            gtid_33614) * 4] = *(__local
                                                                 float *) &scan_arr_mem_34775[local_tid_34771 *
                                                                                              4];
        }
    }
}
__kernel void scan_stage3_34714(int32_t sizze_32279, int32_t sizze_32280,
                                int32_t num_groups_33303, __global
                                unsigned char *mem_34431, __global
                                unsigned char *mem_34435)
{
    const int32_t group_sizze_33292 = mainzigroup_sizze_33291;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t scan_gtid_34714;
    int32_t scan_ltid_34715;
    int32_t scan_gid_34716;
    
    scan_gtid_34714 = get_global_id(0);
    scan_ltid_34715 = get_local_id(0);
    scan_gid_34716 = get_group_id(0);
    
    int32_t gtid_33286 = squot32(scan_gtid_34714, sizze_32279);
    int32_t gtid_33308;
    
    gtid_33308 = scan_gtid_34714 - squot32(scan_gtid_34714, sizze_32279) *
        sizze_32279;
    
    int32_t orig_group_34719 = squot32(scan_gtid_34714, group_sizze_33292 *
                                       squot32(sizze_32280 * sizze_32279 +
                                               group_sizze_33292 *
                                               num_groups_33303 - 1,
                                               group_sizze_33292 *
                                               num_groups_33303));
    int32_t carry_in_flat_idx_34720 = orig_group_34719 * (group_sizze_33292 *
                                                          squot32(sizze_32280 *
                                                                  sizze_32279 +
                                                                  group_sizze_33292 *
                                                                  num_groups_33303 -
                                                                  1,
                                                                  group_sizze_33292 *
                                                                  num_groups_33303)) -
            1;
    
    if (slt32(scan_gtid_34714, sizze_32280 * sizze_32279)) {
        if (!(orig_group_34719 == 0 || (scan_gtid_34714 == (orig_group_34719 +
                                                            1) *
                                        (group_sizze_33292 *
                                         squot32(sizze_32280 * sizze_32279 +
                                                 group_sizze_33292 *
                                                 num_groups_33303 - 1,
                                                 group_sizze_33292 *
                                                 num_groups_33303)) - 1 ||
                                        slt32(srem32(scan_gtid_34714,
                                                     sizze_32279),
                                              scan_gtid_34714 -
                                              carry_in_flat_idx_34720)))) {
            int32_t x_34690;
            int32_t x_34691;
            int32_t x_34692;
            int32_t x_34693;
            
            x_34690 = *(__global
                        int32_t *) &mem_34431[(squot32(carry_in_flat_idx_34720,
                                                       sizze_32279) *
                                               sizze_32279 +
                                               (carry_in_flat_idx_34720 -
                                                squot32(carry_in_flat_idx_34720,
                                                        sizze_32279) *
                                                sizze_32279)) * 4];
            x_34691 = *(__global
                        int32_t *) &mem_34435[(squot32(carry_in_flat_idx_34720,
                                                       sizze_32279) *
                                               sizze_32279 +
                                               (carry_in_flat_idx_34720 -
                                                squot32(carry_in_flat_idx_34720,
                                                        sizze_32279) *
                                                sizze_32279)) * 4];
            x_34692 = *(__global int32_t *) &mem_34431[(gtid_33286 *
                                                        sizze_32279 +
                                                        gtid_33308) * 4];
            x_34693 = *(__global int32_t *) &mem_34435[(gtid_33286 *
                                                        sizze_32279 +
                                                        gtid_33308) * 4];
            
            int32_t res_34694;
            int32_t res_34695;
            
            if (slt32(scan_gtid_34714, sizze_32280 * sizze_32279)) {
                res_34694 = x_34690 + x_34692;
                res_34695 = x_34691 + x_34693;
            }
            x_34690 = res_34694;
            x_34691 = res_34695;
            *(__global int32_t *) &mem_34431[(gtid_33286 * sizze_32279 +
                                              gtid_33308) * 4] = x_34690;
            *(__global int32_t *) &mem_34435[(gtid_33286 * sizze_32279 +
                                              gtid_33308) * 4] = x_34691;
        }
    }
}
__kernel void scan_stage3_34783(int32_t sizze_32280, int32_t num_elems_32669,
                                int32_t num_groups_33609, __global
                                unsigned char *mem_34482)
{
    const int32_t group_sizze_33598 = mainzigroup_sizze_33597;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t scan_gtid_34783;
    int32_t scan_ltid_34784;
    int32_t scan_gid_34785;
    
    scan_gtid_34783 = get_global_id(0);
    scan_ltid_34784 = get_local_id(0);
    scan_gid_34785 = get_group_id(0);
    
    int32_t gtid_33593 = squot32(scan_gtid_34783, num_elems_32669);
    int32_t gtid_33614;
    
    gtid_33614 = scan_gtid_34783 - squot32(scan_gtid_34783, num_elems_32669) *
        num_elems_32669;
    
    int32_t orig_group_34788 = squot32(scan_gtid_34783, group_sizze_33598 *
                                       squot32(sizze_32280 * num_elems_32669 +
                                               group_sizze_33598 *
                                               num_groups_33609 - 1,
                                               group_sizze_33598 *
                                               num_groups_33609));
    int32_t carry_in_flat_idx_34789 = orig_group_34788 * (group_sizze_33598 *
                                                          squot32(sizze_32280 *
                                                                  num_elems_32669 +
                                                                  group_sizze_33598 *
                                                                  num_groups_33609 -
                                                                  1,
                                                                  group_sizze_33598 *
                                                                  num_groups_33609)) -
            1;
    
    if (slt32(scan_gtid_34783, sizze_32280 * num_elems_32669)) {
        if (!(orig_group_34788 == 0 || (scan_gtid_34783 == (orig_group_34788 +
                                                            1) *
                                        (group_sizze_33598 *
                                         squot32(sizze_32280 * num_elems_32669 +
                                                 group_sizze_33598 *
                                                 num_groups_33609 - 1,
                                                 group_sizze_33598 *
                                                 num_groups_33609)) - 1 ||
                                        slt32(srem32(scan_gtid_34783,
                                                     num_elems_32669),
                                              scan_gtid_34783 -
                                              carry_in_flat_idx_34789)))) {
            float x_34767;
            float x_34768;
            
            x_34767 = *(__global
                        float *) &mem_34482[(squot32(carry_in_flat_idx_34789,
                                                     num_elems_32669) *
                                             num_elems_32669 +
                                             (carry_in_flat_idx_34789 -
                                              squot32(carry_in_flat_idx_34789,
                                                      num_elems_32669) *
                                              num_elems_32669)) * 4];
            x_34768 = *(__global float *) &mem_34482[(gtid_33593 *
                                                      num_elems_32669 +
                                                      gtid_33614) * 4];
            
            float res_34769;
            
            if (slt32(scan_gtid_34783, sizze_32280 * num_elems_32669)) {
                res_34769 = x_34767 + x_34768;
            }
            x_34767 = res_34769;
            *(__global float *) &mem_34482[(gtid_33593 * num_elems_32669 +
                                            gtid_33614) * 4] = x_34767;
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
class bfastdistribdetailed:
  entry_points = {"main": (["i32", "i32", "i32", "f32", "f32", "f32", "[]i32",
                            "[][]f32"], ["[]f32", "[]i32", "[]i32", "[]f32",
                                         "[][]f32", "[][]f32", "[]f32", "[]i32",
                                         "[]f32", "[][]f32", "[][]f32"]),
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
                                       all_sizes={"main.group_size_32844": {"class": "group_size", "value": None},
                                        "main.group_size_32891": {"class": "group_size", "value": None},
                                        "main.group_size_32932": {"class": "group_size", "value": None},
                                        "main.group_size_33016": {"class": "group_size", "value": None},
                                        "main.group_size_33030": {"class": "group_size", "value": None},
                                        "main.group_size_33062": {"class": "group_size", "value": None},
                                        "main.group_size_33077": {"class": "group_size", "value": None},
                                        "main.group_size_33156": {"class": "group_size", "value": None},
                                        "main.group_size_33237": {"class": "group_size", "value": None},
                                        "main.group_size_33291": {"class": "group_size", "value": None},
                                        "main.group_size_33331": {"class": "group_size", "value": None},
                                        "main.group_size_33391": {"class": "group_size", "value": None},
                                        "main.group_size_33425": {"class": "group_size", "value": None},
                                        "main.group_size_33450": {"class": "group_size", "value": None},
                                        "main.group_size_33477": {"class": "group_size", "value": None},
                                        "main.group_size_33516": {"class": "group_size", "value": None},
                                        "main.group_size_33597": {"class": "group_size", "value": None},
                                        "main.group_size_33630": {"class": "group_size", "value": None},
                                        "main.group_size_34648": {"class": "group_size", "value": None},
                                        "main.group_size_34717": {"class": "group_size", "value": None},
                                        "main.group_size_34724": {"class": "group_size", "value": None},
                                        "main.group_size_34729": {"class": "group_size", "value": None},
                                        "main.group_size_34734": {"class": "group_size", "value": None},
                                        "main.group_size_34786": {"class": "group_size", "value": None},
                                        "main.group_size_34804": {"class": "group_size", "value": None},
                                        "main.max_num_groups_33293": {"class": "num_groups", "value": None},
                                        "main.max_num_groups_33599": {"class": "num_groups", "value": None},
                                        "main.tile_size_33778": {"class": "tile_size", "value": None},
                                        "main.tile_size_34260": {"class": "tile_size", "value": None},
                                        "main.tile_size_34285": {"class": "tile_size", "value": None},
                                        "remove_nans.group_size_32824": {"class": "group_size", "value": None}})
    self.copy_34645_var = program.copy_34645
    self.copy_34721_var = program.copy_34721
    self.map_32830_var = program.map_32830
    self.map_32850_var = program.map_32850
    self.map_32897_var = program.map_32897
    self.map_32938_var = program.map_32938
    self.map_32967_var = program.map_32967
    self.map_33022_var = program.map_33022
    self.map_33036_var = program.map_33036
    self.map_33068_var = program.map_33068
    self.map_33083_var = program.map_33083
    self.map_33116_var = program.map_33116
    self.map_33162_var = program.map_33162
    self.map_33205_var = program.map_33205
    self.map_33243_var = program.map_33243
    self.map_33337_var = program.map_33337
    self.map_33397_var = program.map_33397
    self.map_33431_var = program.map_33431
    self.map_33456_var = program.map_33456
    self.map_33483_var = program.map_33483
    self.map_33522_var = program.map_33522
    self.map_33636_var = program.map_33636
    self.map_transpose_f32_var = program.map_transpose_f32
    self.map_transpose_f32_low_height_var = program.map_transpose_f32_low_height
    self.map_transpose_f32_low_width_var = program.map_transpose_f32_low_width
    self.map_transpose_f32_small_var = program.map_transpose_f32_small
    self.map_transpose_i32_var = program.map_transpose_i32
    self.map_transpose_i32_low_height_var = program.map_transpose_i32_low_height
    self.map_transpose_i32_low_width_var = program.map_transpose_i32_low_width
    self.map_transpose_i32_small_var = program.map_transpose_i32_small
    self.replicate_34726_var = program.replicate_34726
    self.replicate_34731_var = program.replicate_34731
    self.replicate_34801_var = program.replicate_34801
    self.scan_stage1_33309_var = program.scan_stage1_33309
    self.scan_stage1_33615_var = program.scan_stage1_33615
    self.scan_stage2_34696_var = program.scan_stage2_34696
    self.scan_stage2_34770_var = program.scan_stage2_34770
    self.scan_stage3_34714_var = program.scan_stage3_34714
    self.scan_stage3_34783_var = program.scan_stage3_34783
  def futhark_main(self, mappingindices_mem_34348, images_mem_34349,
                   sizze_32279, sizze_32280, sizze_32281, trend_32282, k_32283,
                   n_32284, freq_32285, hfrac_32286, lam_32287):
    dim_zzero_32290 = (np.int32(0) == sizze_32280)
    dim_zzero_32291 = (np.int32(0) == sizze_32281)
    old_empty_32292 = (dim_zzero_32290 or dim_zzero_32291)
    dim_zzero_32293 = (np.int32(0) == sizze_32279)
    new_empty_32294 = (dim_zzero_32290 or dim_zzero_32293)
    both_empty_32295 = (old_empty_32292 and new_empty_32294)
    dim_match_32296 = (sizze_32279 == sizze_32281)
    empty_or_match_32297 = (both_empty_32295 or dim_match_32296)
    empty_or_match_cert_32298 = True
    assert empty_or_match_32297, ("Error at bfastdistribdetailed.fut:108:1-243:86: %s" % ("function arguments of wrong shape",))
    x_32299 = (np.int32(2) * k_32283)
    res_32300 = (np.int32(2) + x_32299)
    cond_32301 = slt32(np.int32(0), trend_32282)
    if cond_32301:
      res_32302 = res_32300
    else:
      res_32303 = (res_32300 - np.int32(1))
      res_32302 = res_32303
    bounds_invalid_upwards_32304 = slt32(res_32302, np.int32(0))
    convop_x_34351 = (sizze_32279 * res_32302)
    binop_x_34352 = sext_i32_i64(convop_x_34351)
    bytes_34350 = (np.int64(4) * binop_x_34352)
    if cond_32301:
      eq_x_zz_32306 = (np.int32(0) == res_32302)
      not_p_32307 = not(bounds_invalid_upwards_32304)
      p_and_eq_x_y_32308 = (eq_x_zz_32306 and not_p_32307)
      dim_zzero_32309 = (bounds_invalid_upwards_32304 or p_and_eq_x_y_32308)
      both_empty_32310 = (eq_x_zz_32306 and dim_zzero_32309)
      empty_or_match_32314 = (not_p_32307 or both_empty_32310)
      empty_or_match_cert_32315 = True
      assert empty_or_match_32314, ("Error at bfastdistribdetailed.fut:108:1-243:86 -> bfastdistribdetailed.fut:119:16-55 -> bfastdistribdetailed.fut:40:10-18 -> /futlib/array.fut:61:1-62:12: %s%s%s%d%s%s" % ("Function return value does not match shape of type ",
                                                                                                                                                                                                                 "*",
                                                                                                                                                                                                                 "[",
                                                                                                                                                                                                                 res_32302,
                                                                                                                                                                                                                 "]",
                                                                                                                                                                                                                 "intrinsics.i32"))
      group_sizze_32845 = self.sizes["main.group_size_32844"]
      y_32846 = (group_sizze_32845 - np.int32(1))
      x_32847 = (y_32846 + convop_x_34351)
      num_groups_32848 = squot32(x_32847, group_sizze_32845)
      num_threads_32849 = (group_sizze_32845 * num_groups_32848)
      mem_34353 = opencl_alloc(self, bytes_34350, "mem_34353")
      if ((1 * (np.long(num_groups_32848) * np.long(group_sizze_32845))) != 0):
        self.map_32850_var.set_args(np.int32(sizze_32279),
                                    np.float32(freq_32285), np.int32(res_32302),
                                    mappingindices_mem_34348, mem_34353)
        cl.enqueue_nd_range_kernel(self.queue, self.map_32850_var,
                                   ((np.long(num_groups_32848) * np.long(group_sizze_32845)),),
                                   (np.long(group_sizze_32845),))
        if synchronous:
          self.queue.finish()
      arg_mem_34358 = mem_34353
    else:
      eq_x_zz_32337 = (np.int32(0) == res_32302)
      not_p_32338 = not(bounds_invalid_upwards_32304)
      p_and_eq_x_y_32339 = (eq_x_zz_32337 and not_p_32338)
      dim_zzero_32340 = (bounds_invalid_upwards_32304 or p_and_eq_x_y_32339)
      both_empty_32341 = (eq_x_zz_32337 and dim_zzero_32340)
      empty_or_match_32345 = (not_p_32338 or both_empty_32341)
      empty_or_match_cert_32346 = True
      assert empty_or_match_32345, ("Error at bfastdistribdetailed.fut:108:1-243:86 -> bfastdistribdetailed.fut:120:16-55 -> bfastdistribdetailed.fut:52:10-20 -> /futlib/array.fut:61:1-62:12: %s%s%s%d%s%s" % ("Function return value does not match shape of type ",
                                                                                                                                                                                                                 "*",
                                                                                                                                                                                                                 "[",
                                                                                                                                                                                                                 res_32302,
                                                                                                                                                                                                                 "]",
                                                                                                                                                                                                                 "intrinsics.i32"))
      group_sizze_32892 = self.sizes["main.group_size_32891"]
      y_32893 = (group_sizze_32892 - np.int32(1))
      x_32894 = (y_32893 + convop_x_34351)
      num_groups_32895 = squot32(x_32894, group_sizze_32892)
      num_threads_32896 = (group_sizze_32892 * num_groups_32895)
      mem_34357 = opencl_alloc(self, bytes_34350, "mem_34357")
      if ((1 * (np.long(num_groups_32895) * np.long(group_sizze_32892))) != 0):
        self.map_32897_var.set_args(np.int32(sizze_32279),
                                    np.float32(freq_32285), np.int32(res_32302),
                                    mappingindices_mem_34348, mem_34357)
        cl.enqueue_nd_range_kernel(self.queue, self.map_32897_var,
                                   ((np.long(num_groups_32895) * np.long(group_sizze_32892)),),
                                   (np.long(group_sizze_32892),))
        if synchronous:
          self.queue.finish()
      arg_mem_34358 = mem_34357
    x_32367 = (sizze_32279 * sizze_32279)
    y_32368 = (np.int32(2) * sizze_32279)
    x_32369 = (x_32367 + y_32368)
    x_32370 = (np.int32(1) + x_32369)
    y_32371 = (np.int32(1) + sizze_32279)
    x_32372 = sdiv32(x_32370, y_32371)
    x_32373 = (x_32372 - sizze_32279)
    arg_32374 = (x_32373 - np.int32(1))
    res_32375 = sitofp_i32_f32(arg_32374)
    group_sizze_32933 = self.sizes["main.group_size_32932"]
    y_32934 = (group_sizze_32933 - np.int32(1))
    x_32935 = (y_32934 + convop_x_34351)
    num_groups_32936 = squot32(x_32935, group_sizze_32933)
    num_threads_32937 = (group_sizze_32933 * num_groups_32936)
    mem_34362 = opencl_alloc(self, bytes_34350, "mem_34362")
    self.futhark__map_transpose_f32(mem_34362, np.int32(0), arg_mem_34358,
                                    np.int32(0), np.int32(1), sizze_32279,
                                    res_32302, (res_32302 * sizze_32279),
                                    (res_32302 * sizze_32279))
    arg_mem_34358 = None
    mem_34366 = opencl_alloc(self, bytes_34350, "mem_34366")
    if ((1 * (np.long(num_groups_32936) * np.long(group_sizze_32933))) != 0):
      self.map_32938_var.set_args(np.int32(sizze_32279), np.int32(res_32302),
                                  np.float32(res_32375), mem_34362, mem_34366)
      cl.enqueue_nd_range_kernel(self.queue, self.map_32938_var,
                                 ((np.long(num_groups_32936) * np.long(group_sizze_32933)),),
                                 (np.long(group_sizze_32933),))
      if synchronous:
        self.queue.finish()
    tmp_33777 = (np.int32(29) + sizze_32280)
    gidzz_range_33776 = squot32(tmp_33777, np.int32(30))
    tile_sizze_33779 = self.sizes["main.tile_size_33778"]
    tile_sizze_x_33780 = smin32(res_32302, tile_sizze_33779)
    tiled_group_sizze_33782 = (tile_sizze_x_33780 * tile_sizze_x_33780)
    y_33789 = (tile_sizze_x_33780 - np.int32(1))
    x_33790 = (res_32302 + y_33789)
    groups_in_dim_33791 = squot32(x_33790, tile_sizze_x_33780)
    y_33796 = (groups_in_dim_33791 * groups_in_dim_33791)
    num_groups_33797 = (gidzz_range_33776 * y_33796)
    num_threads_33798 = (tiled_group_sizze_33782 * num_groups_33797)
    binop_x_34368 = (sizze_32280 * res_32302)
    convop_x_34369 = (res_32302 * binop_x_34368)
    binop_x_34370 = sext_i32_i64(convop_x_34369)
    bytes_34367 = (np.int64(4) * binop_x_34370)
    mem_34371 = opencl_alloc(self, bytes_34367, "mem_34371")
    convop_x_34373 = (sizze_32280 * sizze_32281)
    binop_x_34374 = sext_i32_i64(convop_x_34373)
    bytes_34372 = (np.int64(4) * binop_x_34374)
    mem_34375 = opencl_alloc(self, bytes_34372, "mem_34375")
    self.futhark__map_transpose_f32(mem_34375, np.int32(0), images_mem_34349,
                                    np.int32(0), np.int32(1), sizze_32281,
                                    sizze_32280, (sizze_32280 * sizze_32281),
                                    (sizze_32280 * sizze_32281))
    binop_x_34377 = sext_i32_i64(tiled_group_sizze_33782)
    bytes_34376 = (np.int64(4) * binop_x_34377)
    if ((1 * (np.long(num_groups_33797) * np.long(tiled_group_sizze_33782))) != 0):
      self.map_32967_var.set_args(cl.LocalMemory(np.long(bytes_34376)),
                                  np.int32(sizze_32280), np.int32(n_32284),
                                  np.int32(res_32302),
                                  np.int32(gidzz_range_33776),
                                  np.int32(tile_sizze_x_33780),
                                  np.int32(tiled_group_sizze_33782), mem_34362,
                                  mem_34366, mem_34371, mem_34375)
      cl.enqueue_nd_range_kernel(self.queue, self.map_32967_var,
                                 ((np.long(num_groups_33797) * np.long(tiled_group_sizze_33782)),),
                                 (np.long(tiled_group_sizze_33782),))
      if synchronous:
        self.queue.finish()
    j_32404 = (np.int32(2) * res_32302)
    j_m_i_32405 = (j_32404 - res_32302)
    arg_32408 = (res_32302 * j_32404)
    res_32421 = sdiv32(arg_32408, res_32302)
    arg_32422 = (res_32302 * res_32421)
    m_32438 = (res_32302 - np.int32(1))
    nesting_sizze_33076 = (sizze_32280 * arg_32408)
    group_sizze_33078 = self.sizes["main.group_size_33077"]
    y_33079 = (group_sizze_33078 - np.int32(1))
    x_33080 = (nesting_sizze_33076 + y_33079)
    num_groups_33081 = squot32(x_33080, group_sizze_33078)
    num_threads_33082 = (group_sizze_33078 * num_groups_33081)
    binop_x_34381 = sext_i32_i64(nesting_sizze_33076)
    bytes_34379 = (np.int64(4) * binop_x_34381)
    mem_34382 = opencl_alloc(self, bytes_34379, "mem_34382")
    if ((1 * (np.long(num_groups_33081) * np.long(group_sizze_33078))) != 0):
      self.map_33083_var.set_args(np.int32(sizze_32280), np.int32(res_32302),
                                  np.int32(j_32404), np.int32(arg_32408),
                                  mem_34371, mem_34382)
      cl.enqueue_nd_range_kernel(self.queue, self.map_33083_var,
                                 ((np.long(num_groups_33081) * np.long(group_sizze_33078)),),
                                 (np.long(group_sizze_33078),))
      if synchronous:
        self.queue.finish()
    mem_34371 = None
    loop_nonempty_33680 = slt32(np.int32(0), res_32302)
    group_sizze_33063 = self.sizes["main.group_size_33062"]
    y_33064 = (group_sizze_33063 - np.int32(1))
    x_33065 = (sizze_32280 + y_33064)
    if loop_nonempty_33680:
      x_33681 = squot32(x_33065, group_sizze_33063)
      num_groups_33066 = x_33681
    else:
      num_groups_33066 = np.int32(0)
    num_threads_33067 = (group_sizze_33063 * num_groups_33066)
    nesting_sizze_33029 = (sizze_32280 * arg_32422)
    group_sizze_33031 = self.sizes["main.group_size_33030"]
    y_33032 = (group_sizze_33031 - np.int32(1))
    x_33033 = (nesting_sizze_33029 + y_33032)
    if loop_nonempty_33680:
      x_33683 = squot32(x_33033, group_sizze_33031)
      num_groups_33034 = x_33683
    else:
      num_groups_33034 = np.int32(0)
    num_threads_33035 = (group_sizze_33031 * num_groups_33034)
    group_sizze_33017 = self.sizes["main.group_size_33016"]
    y_33018 = (group_sizze_33017 - np.int32(1))
    x_33019 = (y_33018 + nesting_sizze_33029)
    if loop_nonempty_33680:
      x_33685 = squot32(x_33019, group_sizze_33017)
      num_groups_33020 = x_33685
    else:
      num_groups_33020 = np.int32(0)
    num_threads_33021 = (group_sizze_33017 * num_groups_33020)
    bytes_34384 = sext_i32_i64(sizze_32280)
    mem_34385 = opencl_alloc(self, bytes_34384, "mem_34385")
    binop_x_34388 = sext_i32_i64(nesting_sizze_33029)
    bytes_34386 = (np.int64(4) * binop_x_34388)
    mem_34389 = opencl_alloc(self, bytes_34386, "mem_34389")
    i_32472 = np.int32(0)
    one_34809 = np.int32(1)
    for counter_34808 in range(res_32302):
      if ((1 * (np.long(num_groups_33066) * np.long(group_sizze_33063))) != 0):
        self.map_33068_var.set_args(np.int32(sizze_32280), np.int32(arg_32408),
                                    np.int32(i_32472), mem_34382, mem_34385)
        cl.enqueue_nd_range_kernel(self.queue, self.map_33068_var,
                                   ((np.long(num_groups_33066) * np.long(group_sizze_33063)),),
                                   (np.long(group_sizze_33063),))
        if synchronous:
          self.queue.finish()
      if ((1 * (np.long(num_groups_33034) * np.long(group_sizze_33031))) != 0):
        self.map_33036_var.set_args(np.int32(sizze_32280), np.int32(arg_32408),
                                    np.int32(res_32421), np.int32(arg_32422),
                                    np.int32(m_32438), np.int32(i_32472),
                                    mem_34382, mem_34385, mem_34389)
        cl.enqueue_nd_range_kernel(self.queue, self.map_33036_var,
                                   ((np.long(num_groups_33034) * np.long(group_sizze_33031)),),
                                   (np.long(group_sizze_33031),))
        if synchronous:
          self.queue.finish()
      if ((1 * (np.long(num_groups_33020) * np.long(group_sizze_33017))) != 0):
        self.map_33022_var.set_args(np.int32(sizze_32280), np.int32(arg_32408),
                                    np.int32(arg_32422), mem_34382, mem_34389)
        cl.enqueue_nd_range_kernel(self.queue, self.map_33022_var,
                                   ((np.long(num_groups_33020) * np.long(group_sizze_33017)),),
                                   (np.long(group_sizze_33017),))
        if synchronous:
          self.queue.finish()
      i_32472 += one_34809
    mem_34385 = None
    mem_34389 = None
    tile_sizze_34261 = self.sizes["main.tile_size_34260"]
    tiled_group_sizze_34262 = (tile_sizze_34261 * tile_sizze_34261)
    y_34265 = (tile_sizze_34261 - np.int32(1))
    x_34266 = (sizze_32280 + y_34265)
    groups_in_dim_34267 = squot32(x_34266, tile_sizze_34261)
    x_34269 = (res_32302 + y_34265)
    groups_in_dim_34270 = squot32(x_34269, tile_sizze_34261)
    num_groups_34272 = (groups_in_dim_34267 * groups_in_dim_34270)
    num_threads_34273 = (tiled_group_sizze_34262 * num_groups_34272)
    binop_x_34401 = sext_i32_i64(binop_x_34368)
    bytes_34399 = (np.int64(4) * binop_x_34401)
    mem_34402 = opencl_alloc(self, bytes_34399, "mem_34402")
    binop_x_34393 = sext_i32_i64(tiled_group_sizze_34262)
    bytes_34391 = (np.int64(4) * binop_x_34393)
    if ((1 * (np.long(num_groups_34272) * np.long(tiled_group_sizze_34262))) != 0):
      self.map_33116_var.set_args(np.int32(sizze_32280), np.int32(sizze_32281),
                                  np.int32(n_32284), np.int32(res_32302),
                                  images_mem_34349, mem_34362, mem_34402)
      cl.enqueue_nd_range_kernel(self.queue, self.map_33116_var,
                                 ((np.long(num_groups_34272) * np.long(tiled_group_sizze_34262)),),
                                 (np.long(tiled_group_sizze_34262),))
      if synchronous:
        self.queue.finish()
    mem_34362 = None
    group_sizze_33157 = self.sizes["main.group_size_33156"]
    y_33158 = (group_sizze_33157 - np.int32(1))
    x_33159 = (y_33158 + binop_x_34368)
    num_groups_33160 = squot32(x_33159, group_sizze_33157)
    num_threads_33161 = (group_sizze_33157 * num_groups_33160)
    binop_x_34404 = (sizze_32280 * j_m_i_32405)
    convop_x_34405 = (res_32302 * binop_x_34404)
    binop_x_34406 = sext_i32_i64(convop_x_34405)
    bytes_34403 = (np.int64(4) * binop_x_34406)
    mem_34407 = opencl_alloc(self, bytes_34403, "mem_34407")
    group_sizze_34648 = self.sizes["main.group_size_34648"]
    num_groups_34649 = squot32((((sizze_32280 * (res_32302 * j_m_i_32405)) + sext_i32_i32(group_sizze_34648)) - np.int32(1)),
                               sext_i32_i32(group_sizze_34648))
    if ((1 * (np.long(num_groups_34649) * np.long(group_sizze_34648))) != 0):
      self.copy_34645_var.set_args(np.int32(sizze_32280), np.int32(res_32302),
                                   np.int32(j_32404), np.int32(j_m_i_32405),
                                   mem_34382, mem_34407)
      cl.enqueue_nd_range_kernel(self.queue, self.copy_34645_var,
                                 ((np.long(num_groups_34649) * np.long(group_sizze_34648)),),
                                 (np.long(group_sizze_34648),))
      if synchronous:
        self.queue.finish()
    mem_34382 = None
    mem_34411 = opencl_alloc(self, bytes_34399, "mem_34411")
    if ((1 * (np.long(num_groups_33160) * np.long(group_sizze_33157))) != 0):
      self.map_33162_var.set_args(np.int32(sizze_32280), np.int32(res_32302),
                                  np.int32(j_m_i_32405), mem_34402, mem_34407,
                                  mem_34411)
      cl.enqueue_nd_range_kernel(self.queue, self.map_33162_var,
                                 ((np.long(num_groups_33160) * np.long(group_sizze_33157)),),
                                 (np.long(group_sizze_33157),))
      if synchronous:
        self.queue.finish()
    mem_34402 = None
    mem_34407 = None
    mem_34415 = opencl_alloc(self, bytes_34350, "mem_34415")
    self.futhark__map_transpose_f32(mem_34415, np.int32(0), mem_34366,
                                    np.int32(0), np.int32(1), res_32302,
                                    sizze_32279, (sizze_32279 * res_32302),
                                    (sizze_32279 * res_32302))
    mem_34366 = None
    tile_sizze_34286 = self.sizes["main.tile_size_34285"]
    tiled_group_sizze_34287 = (tile_sizze_34286 * tile_sizze_34286)
    y_34290 = (tile_sizze_34286 - np.int32(1))
    x_34291 = (sizze_32280 + y_34290)
    groups_in_dim_34292 = squot32(x_34291, tile_sizze_34286)
    x_34294 = (sizze_32279 + y_34290)
    groups_in_dim_34295 = squot32(x_34294, tile_sizze_34286)
    num_groups_34297 = (groups_in_dim_34292 * groups_in_dim_34295)
    num_threads_34298 = (tiled_group_sizze_34287 * num_groups_34297)
    convop_x_34425 = (sizze_32279 * sizze_32280)
    binop_x_34426 = sext_i32_i64(convop_x_34425)
    bytes_34424 = (np.int64(4) * binop_x_34426)
    mem_34427 = opencl_alloc(self, bytes_34424, "mem_34427")
    binop_x_34418 = sext_i32_i64(tiled_group_sizze_34287)
    bytes_34416 = (np.int64(4) * binop_x_34418)
    if ((1 * (np.long(num_groups_34297) * np.long(tiled_group_sizze_34287))) != 0):
      self.map_33205_var.set_args(np.int32(sizze_32279), np.int32(sizze_32280),
                                  np.int32(res_32302), mem_34411, mem_34415,
                                  mem_34427)
      cl.enqueue_nd_range_kernel(self.queue, self.map_33205_var,
                                 ((np.long(num_groups_34297) * np.long(tiled_group_sizze_34287)),),
                                 (np.long(tiled_group_sizze_34287),))
      if synchronous:
        self.queue.finish()
    mem_34411 = None
    mem_34415 = None
    i_32550 = (sizze_32279 - np.int32(1))
    group_sizze_33292 = self.sizes["main.group_size_33291"]
    max_num_groups_33294 = self.sizes["main.max_num_groups_33293"]
    group_sizze_33295 = sext_i32_i64(group_sizze_33292)
    max_num_groups_33296 = sext_i32_i64(max_num_groups_33294)
    y_33297 = (group_sizze_33295 - np.int64(1))
    x_33298 = (y_33297 + binop_x_34426)
    w_div_group_sizze_33299 = squot64(x_33298, group_sizze_33295)
    num_groups_maybe_zzero_33300 = smin64(max_num_groups_33296,
                                          w_div_group_sizze_33299)
    num_groups_33301 = smax64(np.int64(1), num_groups_maybe_zzero_33300)
    num_threads_33302 = (group_sizze_33295 * num_groups_33301)
    num_groups_33303 = sext_i64_i32(num_groups_33301)
    num_threads_33304 = sext_i64_i32(num_threads_33302)
    mem_34431 = opencl_alloc(self, bytes_34424, "mem_34431")
    mem_34435 = opencl_alloc(self, bytes_34424, "mem_34435")
    mem_34438 = opencl_alloc(self, binop_x_34426, "mem_34438")
    mem_34442 = opencl_alloc(self, bytes_34424, "mem_34442")
    if ((1 * (np.long(num_groups_33303) * np.long(group_sizze_33292))) != 0):
      self.scan_stage1_33309_var.set_args(np.int32(sizze_32279),
                                          np.int32(sizze_32280),
                                          np.int32(sizze_32281),
                                          np.int32(num_groups_33303),
                                          images_mem_34349, mem_34427,
                                          mem_34431, mem_34435, mem_34438,
                                          mem_34442)
      cl.enqueue_nd_range_kernel(self.queue, self.scan_stage1_33309_var,
                                 ((np.long(num_groups_33303) * np.long(group_sizze_33292)),),
                                 (np.long(group_sizze_33292),))
      if synchronous:
        self.queue.finish()
    if ((1 * (np.long(np.int32(1)) * np.long(num_groups_33303))) != 0):
      self.scan_stage2_34696_var.set_args(cl.LocalMemory(np.long((np.int32(4) * num_groups_33303))),
                                          cl.LocalMemory(np.long((np.int32(4) * num_groups_33303))),
                                          np.int32(sizze_32279),
                                          np.int32(sizze_32280),
                                          np.int32(num_groups_33303), mem_34431,
                                          mem_34435)
      cl.enqueue_nd_range_kernel(self.queue, self.scan_stage2_34696_var,
                                 ((np.long(np.int32(1)) * np.long(num_groups_33303)),),
                                 (np.long(num_groups_33303),))
      if synchronous:
        self.queue.finish()
    group_sizze_34717 = self.sizes["main.group_size_34717"]
    num_groups_34718 = squot32((((sizze_32280 * sizze_32279) + sext_i32_i32(group_sizze_34717)) - np.int32(1)),
                               sext_i32_i32(group_sizze_34717))
    if ((1 * (np.long(num_groups_34718) * np.long(group_sizze_34717))) != 0):
      self.scan_stage3_34714_var.set_args(np.int32(sizze_32279),
                                          np.int32(sizze_32280),
                                          np.int32(num_groups_33303), mem_34431,
                                          mem_34435)
      cl.enqueue_nd_range_kernel(self.queue, self.scan_stage3_34714_var,
                                 ((np.long(num_groups_34718) * np.long(group_sizze_34717)),),
                                 (np.long(group_sizze_34717),))
      if synchronous:
        self.queue.finish()
    bytes_34443 = (np.int64(4) * bytes_34384)
    mem_34445 = opencl_alloc(self, bytes_34443, "mem_34445")
    group_sizze_34724 = self.sizes["main.group_size_34724"]
    num_groups_34725 = squot32(((sizze_32280 + sext_i32_i32(group_sizze_34724)) - np.int32(1)),
                               sext_i32_i32(group_sizze_34724))
    if ((1 * (np.long(num_groups_34725) * np.long(group_sizze_34724))) != 0):
      self.copy_34721_var.set_args(np.int32(sizze_32279), np.int32(sizze_32280),
                                   np.int32(i_32550), mem_34431, mem_34445)
      cl.enqueue_nd_range_kernel(self.queue, self.copy_34721_var,
                                 ((np.long(num_groups_34725) * np.long(group_sizze_34724)),),
                                 (np.long(group_sizze_34724),))
      if synchronous:
        self.queue.finish()
    mem_34449 = opencl_alloc(self, bytes_34424, "mem_34449")
    group_sizze_34729 = self.sizes["main.group_size_34729"]
    num_groups_34730 = squot32((((sizze_32280 * sizze_32279) + sext_i32_i32(group_sizze_34729)) - np.int32(1)),
                               sext_i32_i32(group_sizze_34729))
    if ((1 * (np.long(num_groups_34730) * np.long(group_sizze_34729))) != 0):
      self.replicate_34726_var.set_args(np.int32(sizze_32279),
                                        np.int32(sizze_32280), mem_34449)
      cl.enqueue_nd_range_kernel(self.queue, self.replicate_34726_var,
                                 ((np.long(num_groups_34730) * np.long(group_sizze_34729)),),
                                 (np.long(group_sizze_34729),))
      if synchronous:
        self.queue.finish()
    mem_34453 = opencl_alloc(self, bytes_34424, "mem_34453")
    group_sizze_34734 = self.sizes["main.group_size_34734"]
    num_groups_34735 = squot32((((sizze_32280 * sizze_32279) + sext_i32_i32(group_sizze_34734)) - np.int32(1)),
                               sext_i32_i32(group_sizze_34734))
    if ((1 * (np.long(num_groups_34735) * np.long(group_sizze_34734))) != 0):
      self.replicate_34731_var.set_args(np.int32(sizze_32279),
                                        np.int32(sizze_32280), mem_34453)
      cl.enqueue_nd_range_kernel(self.queue, self.replicate_34731_var,
                                 ((np.long(num_groups_34735) * np.long(group_sizze_34734)),),
                                 (np.long(group_sizze_34734),))
      if synchronous:
        self.queue.finish()
    group_sizze_33238 = self.sizes["main.group_size_33237"]
    y_33239 = (group_sizze_33238 - np.int32(1))
    x_33240 = (y_33239 + convop_x_34425)
    num_groups_33241 = squot32(x_33240, group_sizze_33238)
    num_threads_33242 = (group_sizze_33238 * num_groups_33241)
    if ((1 * (np.long(num_groups_33241) * np.long(group_sizze_33238))) != 0):
      self.map_33243_var.set_args(np.int32(sizze_32279), np.int32(sizze_32280),
                                  np.int32(i_32550), mem_34431, mem_34435,
                                  mem_34438, mem_34442, mem_34449, mem_34453)
      cl.enqueue_nd_range_kernel(self.queue, self.map_33243_var,
                                 ((np.long(num_groups_33241) * np.long(group_sizze_33238)),),
                                 (np.long(group_sizze_33238),))
      if synchronous:
        self.queue.finish()
    mem_34431 = None
    mem_34435 = None
    mem_34438 = None
    mem_34442 = None
    group_sizze_33332 = self.sizes["main.group_size_33331"]
    y_33333 = (group_sizze_33332 - np.int32(1))
    x_33334 = (sizze_32280 + y_33333)
    num_groups_33335 = squot32(x_33334, group_sizze_33332)
    num_threads_33336 = (group_sizze_33332 * num_groups_33335)
    mem_34457 = opencl_alloc(self, bytes_34424, "mem_34457")
    self.futhark__map_transpose_f32(mem_34457, np.int32(0), mem_34449,
                                    np.int32(0), np.int32(1), sizze_32279,
                                    sizze_32280, (sizze_32280 * sizze_32279),
                                    (sizze_32280 * sizze_32279))
    mem_34460 = opencl_alloc(self, bytes_34443, "mem_34460")
    mem_34463 = opencl_alloc(self, bytes_34443, "mem_34463")
    if ((1 * (np.long(num_groups_33335) * np.long(group_sizze_33332))) != 0):
      self.map_33337_var.set_args(np.int32(sizze_32280), np.int32(n_32284),
                                  np.int32(res_32300), mem_34375, mem_34457,
                                  mem_34460, mem_34463)
      cl.enqueue_nd_range_kernel(self.queue, self.map_33337_var,
                                 ((np.long(num_groups_33335) * np.long(group_sizze_33332)),),
                                 (np.long(group_sizze_33332),))
      if synchronous:
        self.queue.finish()
    mem_34375 = None
    mem_34457 = None
    group_sizze_33392 = self.sizes["main.group_size_33391"]
    y_33393 = (group_sizze_33392 - np.int32(1))
    x_33394 = (sizze_32280 + y_33393)
    num_groups_33395 = squot32(x_33394, group_sizze_33392)
    num_threads_33396 = (group_sizze_33392 * num_groups_33395)
    mem_34466 = opencl_alloc(self, bytes_34443, "mem_34466")
    mem_34469 = opencl_alloc(self, bytes_34443, "mem_34469")
    if ((1 * (np.long(num_groups_33395) * np.long(group_sizze_33392))) != 0):
      self.map_33397_var.set_args(np.int32(sizze_32279), np.int32(sizze_32280),
                                  np.float32(hfrac_32286), mem_34449, mem_34460,
                                  mem_34466, mem_34469)
      cl.enqueue_nd_range_kernel(self.queue, self.map_33397_var,
                                 ((np.long(num_groups_33395) * np.long(group_sizze_33392)),),
                                 (np.long(group_sizze_33392),))
      if synchronous:
        self.queue.finish()
    x_32665 = (sizze_32279 - n_32284)
    range_end_32666 = (x_32665 - np.int32(1))
    bounds_invalid_upwards_32667 = slt32(range_end_32666, np.int32(0))
    distance_32668 = (np.int32(1) + range_end_32666)
    if bounds_invalid_upwards_32667:
      num_elems_32669 = np.int32(0)
    else:
      num_elems_32669 = distance_32668
    x_32671 = (np.int32(1) + n_32284)
    x_32672 = sle32(np.int32(0), i_32550)
    index_certs_32675 = True
    assert x_32672, ("Error at bfastdistribdetailed.fut:108:1-243:86 -> bfastdistribdetailed.fut:192:15-196:33 -> bfastdistribdetailed.fut:194:63-81: %s%d%s%d%s" % ("Index [",
                                                                                                                                                                     i_32550,
                                                                                                                                                                     "] out of bounds for array of shape [",
                                                                                                                                                                     sizze_32279,
                                                                                                                                                                     "]."))
    read_res_34810 = np.empty(1, dtype=ct.c_int32)
    cl.enqueue_copy(self.queue, read_res_34810, mappingindices_mem_34348,
                    device_offset=np.long((i_32550 * np.int32(4))),
                    is_blocking=True)
    arg_32676 = read_res_34810[0]
    res_32677 = sitofp_i32_f32(arg_32676)
    group_sizze_33426 = self.sizes["main.group_size_33425"]
    y_33427 = (group_sizze_33426 - np.int32(1))
    x_33428 = (num_elems_32669 + y_33427)
    num_groups_33429 = squot32(x_33428, group_sizze_33426)
    num_threads_33430 = (group_sizze_33426 * num_groups_33429)
    binop_x_34471 = sext_i32_i64(num_elems_32669)
    bytes_34470 = (np.int64(4) * binop_x_34471)
    mem_34472 = opencl_alloc(self, bytes_34470, "mem_34472")
    if ((1 * (np.long(num_groups_33429) * np.long(group_sizze_33426))) != 0):
      self.map_33431_var.set_args(np.float32(lam_32287),
                                  np.int32(num_elems_32669), np.int32(x_32671),
                                  np.float32(res_32677),
                                  mappingindices_mem_34348, mem_34472)
      cl.enqueue_nd_range_kernel(self.queue, self.map_33431_var,
                                 ((np.long(num_groups_33429) * np.long(group_sizze_33426)),),
                                 (np.long(group_sizze_33426),))
      if synchronous:
        self.queue.finish()
    eq_x_zz_32691 = (np.int32(0) == distance_32668)
    not_p_32692 = not(bounds_invalid_upwards_32667)
    p_and_eq_x_y_32693 = (eq_x_zz_32691 and not_p_32692)
    dim_zzero_32694 = (bounds_invalid_upwards_32667 or p_and_eq_x_y_32693)
    both_empty_32695 = (dim_zzero_32694 and dim_zzero_32694)
    eq_x_y_32696 = (distance_32668 == np.int32(0))
    p_and_eq_x_y_32697 = (bounds_invalid_upwards_32667 and eq_x_y_32696)
    eq_x_zz_32698 = (not_p_32692 or p_and_eq_x_y_32697)
    p_and_eq_x_y_32699 = (bounds_invalid_upwards_32667 and dim_zzero_32694)
    p_and_eq_x_y_32700 = (not_p_32692 and eq_x_zz_32698)
    dim_match_32701 = (p_and_eq_x_y_32699 or p_and_eq_x_y_32700)
    empty_or_match_32702 = (both_empty_32695 or dim_match_32701)
    empty_or_match_cert_32703 = True
    assert empty_or_match_32702, ("Error at bfastdistribdetailed.fut:108:1-243:86 -> bfastdistribdetailed.fut:207:38-242:9 -> /futlib/functional.fut:7:42-44 -> bfastdistribdetailed.fut:220:21-223:36: %s" % ("function arguments of wrong shape",))
    group_sizze_33631 = self.sizes["main.group_size_33630"]
    y_33632 = (group_sizze_33631 - np.int32(1))
    x_33633 = (sizze_32280 + y_33632)
    num_groups_33634 = squot32(x_33633, group_sizze_33631)
    num_threads_33635 = (group_sizze_33631 * num_groups_33634)
    mem_34475 = opencl_alloc(self, bytes_34443, "mem_34475")
    mem_34478 = opencl_alloc(self, bytes_34443, "mem_34478")
    if ((1 * (np.long(num_groups_33634) * np.long(group_sizze_33631))) != 0):
      self.map_33636_var.set_args(np.int32(sizze_32280), mem_34445, mem_34460,
                                  mem_34463, mem_34475, mem_34478)
      cl.enqueue_nd_range_kernel(self.queue, self.map_33636_var,
                                 ((np.long(num_groups_33634) * np.long(group_sizze_33631)),),
                                 (np.long(group_sizze_33631),))
      if synchronous:
        self.queue.finish()
    total_num_elements_33594 = (sizze_32280 * num_elems_32669)
    total_num_elements_33596 = sext_i32_i64(total_num_elements_33594)
    group_sizze_33598 = self.sizes["main.group_size_33597"]
    max_num_groups_33600 = self.sizes["main.max_num_groups_33599"]
    group_sizze_33601 = sext_i32_i64(group_sizze_33598)
    max_num_groups_33602 = sext_i32_i64(max_num_groups_33600)
    y_33603 = (group_sizze_33601 - np.int64(1))
    x_33604 = (total_num_elements_33596 + y_33603)
    w_div_group_sizze_33605 = squot64(x_33604, group_sizze_33601)
    num_groups_maybe_zzero_33606 = smin64(max_num_groups_33602,
                                          w_div_group_sizze_33605)
    num_groups_33607 = smax64(np.int64(1), num_groups_maybe_zzero_33606)
    num_threads_33608 = (group_sizze_33601 * num_groups_33607)
    num_groups_33609 = sext_i64_i32(num_groups_33607)
    num_threads_33610 = sext_i64_i32(num_threads_33608)
    bytes_34479 = (np.int64(4) * total_num_elements_33596)
    mem_34482 = opencl_alloc(self, bytes_34479, "mem_34482")
    if ((1 * (np.long(num_groups_33609) * np.long(group_sizze_33598))) != 0):
      self.scan_stage1_33615_var.set_args(np.int32(sizze_32279),
                                          np.int32(sizze_32280),
                                          np.int32(num_elems_32669),
                                          np.int32(num_groups_33609), mem_34449,
                                          mem_34460, mem_34466, mem_34469,
                                          mem_34478, mem_34482)
      cl.enqueue_nd_range_kernel(self.queue, self.scan_stage1_33615_var,
                                 ((np.long(num_groups_33609) * np.long(group_sizze_33598)),),
                                 (np.long(group_sizze_33598),))
      if synchronous:
        self.queue.finish()
    if ((1 * (np.long(np.int32(1)) * np.long(num_groups_33609))) != 0):
      self.scan_stage2_34770_var.set_args(cl.LocalMemory(np.long((np.int32(4) * num_groups_33609))),
                                          np.int32(sizze_32280),
                                          np.int32(num_elems_32669),
                                          np.int32(num_groups_33609), mem_34482)
      cl.enqueue_nd_range_kernel(self.queue, self.scan_stage2_34770_var,
                                 ((np.long(np.int32(1)) * np.long(num_groups_33609)),),
                                 (np.long(num_groups_33609),))
      if synchronous:
        self.queue.finish()
    group_sizze_34786 = self.sizes["main.group_size_34786"]
    num_groups_34787 = squot32((((sizze_32280 * num_elems_32669) + sext_i32_i32(group_sizze_34786)) - np.int32(1)),
                               sext_i32_i32(group_sizze_34786))
    if ((1 * (np.long(num_groups_34787) * np.long(group_sizze_34786))) != 0):
      self.scan_stage3_34783_var.set_args(np.int32(sizze_32280),
                                          np.int32(num_elems_32669),
                                          np.int32(num_groups_33609), mem_34482)
      cl.enqueue_nd_range_kernel(self.queue, self.scan_stage3_34783_var,
                                 ((np.long(num_groups_34787) * np.long(group_sizze_34786)),),
                                 (np.long(group_sizze_34786),))
      if synchronous:
        self.queue.finish()
    mem_34469 = None
    group_sizze_33517 = self.sizes["main.group_size_33516"]
    y_33518 = (group_sizze_33517 - np.int32(1))
    x_33519 = (sizze_32280 + y_33518)
    num_groups_33520 = squot32(x_33519, group_sizze_33517)
    num_threads_33521 = (group_sizze_33517 * num_groups_33520)
    mem_34486 = opencl_alloc(self, bytes_34479, "mem_34486")
    self.futhark__map_transpose_f32(mem_34486, np.int32(0), mem_34482,
                                    np.int32(0), np.int32(1), num_elems_32669,
                                    sizze_32280,
                                    (sizze_32280 * num_elems_32669),
                                    (sizze_32280 * num_elems_32669))
    mem_34482 = None
    mem_34496 = opencl_alloc(self, bytes_34479, "mem_34496")
    mem_34500 = opencl_alloc(self, bytes_34479, "mem_34500")
    mem_34502 = opencl_alloc(self, bytes_34384, "mem_34502")
    mem_34505 = opencl_alloc(self, bytes_34443, "mem_34505")
    mem_34508 = opencl_alloc(self, bytes_34443, "mem_34508")
    num_threads64_34548 = sext_i32_i64(num_threads_33521)
    total_sizze_34549 = (bytes_34470 * num_threads64_34548)
    mem_34489 = opencl_alloc(self, total_sizze_34549, "mem_34489")
    total_sizze_34550 = (bytes_34470 * num_threads64_34548)
    mem_34492 = opencl_alloc(self, total_sizze_34550, "mem_34492")
    if ((1 * (np.long(num_groups_33520) * np.long(group_sizze_33517))) != 0):
      self.map_33522_var.set_args(np.int32(sizze_32279), np.int32(sizze_32280),
                                  np.int32(n_32284), np.int32(num_elems_32669),
                                  mem_34453, mem_34460, mem_34472, mem_34475,
                                  mem_34478, mem_34486, mem_34489, mem_34492,
                                  mem_34496, mem_34500, mem_34502, mem_34505,
                                  mem_34508)
      cl.enqueue_nd_range_kernel(self.queue, self.map_33522_var,
                                 ((np.long(num_groups_33520) * np.long(group_sizze_33517)),),
                                 (np.long(group_sizze_33517),))
      if synchronous:
        self.queue.finish()
    mem_34453 = None
    mem_34475 = None
    mem_34486 = None
    mem_34489 = None
    mem_34492 = None
    group_sizze_33478 = self.sizes["main.group_size_33477"]
    y_33479 = (group_sizze_33478 - np.int32(1))
    x_33480 = (sizze_32280 + y_33479)
    num_groups_33481 = squot32(x_33480, group_sizze_33478)
    num_threads_33482 = (group_sizze_33478 * num_groups_33481)
    mem_34512 = opencl_alloc(self, bytes_34479, "mem_34512")
    self.futhark__map_transpose_i32(mem_34512, np.int32(0), mem_34496,
                                    np.int32(0), np.int32(1), sizze_32280,
                                    num_elems_32669,
                                    (sizze_32280 * num_elems_32669),
                                    (sizze_32280 * num_elems_32669))
    mem_34496 = None
    mem_34515 = opencl_alloc(self, bytes_34443, "mem_34515")
    if ((1 * (np.long(num_groups_33481) * np.long(group_sizze_33478))) != 0):
      self.map_33483_var.set_args(np.int32(sizze_32280),
                                  np.int32(num_elems_32669), mem_34460,
                                  mem_34478, mem_34502, mem_34505, mem_34512,
                                  mem_34515)
      cl.enqueue_nd_range_kernel(self.queue, self.map_33483_var,
                                 ((np.long(num_groups_33481) * np.long(group_sizze_33478)),),
                                 (np.long(group_sizze_33478),))
      if synchronous:
        self.queue.finish()
    mem_34478 = None
    mem_34502 = None
    mem_34505 = None
    convop_x_34517 = (sizze_32280 * x_32665)
    binop_x_34518 = sext_i32_i64(convop_x_34517)
    bytes_34516 = (np.int64(4) * binop_x_34518)
    mem_34519 = opencl_alloc(self, bytes_34516, "mem_34519")
    group_sizze_34804 = self.sizes["main.group_size_34804"]
    num_groups_34805 = squot32((((sizze_32280 * x_32665) + sext_i32_i32(group_sizze_34804)) - np.int32(1)),
                               sext_i32_i32(group_sizze_34804))
    if ((1 * (np.long(num_groups_34805) * np.long(group_sizze_34804))) != 0):
      self.replicate_34801_var.set_args(np.int32(sizze_32280),
                                        np.int32(x_32665), mem_34519)
      cl.enqueue_nd_range_kernel(self.queue, self.replicate_34801_var,
                                 ((np.long(num_groups_34805) * np.long(group_sizze_34804)),),
                                 (np.long(group_sizze_34804),))
      if synchronous:
        self.queue.finish()
    group_sizze_33451 = self.sizes["main.group_size_33450"]
    y_33452 = (group_sizze_33451 - np.int32(1))
    x_33453 = (y_33452 + total_num_elements_33594)
    num_groups_33454 = squot32(x_33453, group_sizze_33451)
    num_threads_33455 = (group_sizze_33451 * num_groups_33454)
    mem_34523 = opencl_alloc(self, bytes_34479, "mem_34523")
    self.futhark__map_transpose_f32(mem_34523, np.int32(0), mem_34500,
                                    np.int32(0), np.int32(1), sizze_32280,
                                    num_elems_32669,
                                    (sizze_32280 * num_elems_32669),
                                    (sizze_32280 * num_elems_32669))
    if ((1 * (np.long(num_groups_33454) * np.long(group_sizze_33451))) != 0):
      self.map_33456_var.set_args(np.int32(sizze_32280), np.int32(x_32665),
                                  np.int32(num_elems_32669), mem_34512,
                                  mem_34519, mem_34523)
      cl.enqueue_nd_range_kernel(self.queue, self.map_33456_var,
                                 ((np.long(num_groups_33454) * np.long(group_sizze_33451)),),
                                 (np.long(group_sizze_33451),))
      if synchronous:
        self.queue.finish()
    mem_34512 = None
    mem_34523 = None
    mem_34527 = opencl_alloc(self, bytes_34479, "mem_34527")
    self.futhark__map_transpose_f32(mem_34527, np.int32(0), mem_34500,
                                    np.int32(0), np.int32(1), sizze_32280,
                                    num_elems_32669,
                                    (sizze_32280 * num_elems_32669),
                                    (sizze_32280 * num_elems_32669))
    mem_34500 = None
    out_arrsizze_34563 = sizze_32280
    out_arrsizze_34565 = sizze_32280
    out_arrsizze_34567 = sizze_32280
    out_arrsizze_34569 = sizze_32280
    out_arrsizze_34571 = sizze_32280
    out_arrsizze_34572 = x_32665
    out_arrsizze_34574 = sizze_32280
    out_arrsizze_34575 = num_elems_32669
    out_arrsizze_34577 = num_elems_32669
    out_arrsizze_34579 = sizze_32280
    out_arrsizze_34581 = sizze_32280
    out_arrsizze_34583 = sizze_32280
    out_arrsizze_34584 = sizze_32279
    out_arrsizze_34586 = sizze_32280
    out_arrsizze_34587 = sizze_32279
    out_mem_34562 = mem_34466
    out_mem_34564 = mem_34445
    out_mem_34566 = mem_34460
    out_mem_34568 = mem_34463
    out_mem_34570 = mem_34519
    out_mem_34573 = mem_34527
    out_mem_34576 = mem_34472
    out_mem_34578 = mem_34515
    out_mem_34580 = mem_34508
    out_mem_34582 = mem_34449
    out_mem_34585 = mem_34427
    return (out_mem_34562, out_arrsizze_34563, out_mem_34564,
            out_arrsizze_34565, out_mem_34566, out_arrsizze_34567,
            out_mem_34568, out_arrsizze_34569, out_mem_34570,
            out_arrsizze_34571, out_arrsizze_34572, out_mem_34573,
            out_arrsizze_34574, out_arrsizze_34575, out_mem_34576,
            out_arrsizze_34577, out_mem_34578, out_arrsizze_34579,
            out_mem_34580, out_arrsizze_34581, out_mem_34582,
            out_arrsizze_34583, out_arrsizze_34584, out_mem_34585,
            out_arrsizze_34586, out_arrsizze_34587)
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
  def futhark_remove_nans(self, images_mem_34348, sizze_32265, sizze_32266,
                          sizze_32267, nan_value_32268):
    nesting_sizze_32822 = (sizze_32266 * sizze_32267)
    nesting_sizze_32823 = (sizze_32265 * nesting_sizze_32822)
    group_sizze_32825 = self.sizes["remove_nans.group_size_32824"]
    y_32826 = (group_sizze_32825 - np.int32(1))
    x_32827 = (nesting_sizze_32823 + y_32826)
    num_groups_32828 = squot32(x_32827, group_sizze_32825)
    num_threads_32829 = (group_sizze_32825 * num_groups_32828)
    binop_x_34350 = (sizze_32265 * sizze_32266)
    convop_x_34351 = (sizze_32267 * binop_x_34350)
    binop_x_34352 = sext_i32_i64(convop_x_34351)
    bytes_34349 = (np.int64(4) * binop_x_34352)
    mem_34353 = opencl_alloc(self, bytes_34349, "mem_34353")
    if ((1 * (np.long(num_groups_32828) * np.long(group_sizze_32825))) != 0):
      self.map_32830_var.set_args(np.int32(sizze_32265), np.int32(sizze_32266),
                                  np.int32(sizze_32267),
                                  np.int16(nan_value_32268), images_mem_34348,
                                  mem_34353)
      cl.enqueue_nd_range_kernel(self.queue, self.map_32830_var,
                                 ((np.long(num_groups_32828) * np.long(group_sizze_32825)),),
                                 (np.long(group_sizze_32825),))
      if synchronous:
        self.queue.finish()
    out_arrsizze_34557 = sizze_32265
    out_arrsizze_34558 = sizze_32266
    out_arrsizze_34559 = sizze_32267
    out_mem_34556 = mem_34353
    return (out_mem_34556, out_arrsizze_34557, out_arrsizze_34558,
            out_arrsizze_34559)
  def futhark_reshapeTransp(self, images_mem_34348, sizze_32258, sizze_32259,
                            sizze_32260):
    flat_dim_32262 = (sizze_32259 * sizze_32260)
    convop_x_34350 = (sizze_32258 * flat_dim_32262)
    binop_x_34351 = sext_i32_i64(convop_x_34350)
    bytes_34349 = (np.int64(4) * binop_x_34351)
    mem_34352 = opencl_alloc(self, bytes_34349, "mem_34352")
    self.futhark__map_transpose_f32(mem_34352, np.int32(0), images_mem_34348,
                                    np.int32(0), np.int32(1), flat_dim_32262,
                                    sizze_32258, (flat_dim_32262 * sizze_32258),
                                    (flat_dim_32262 * sizze_32258))
    out_arrsizze_34554 = flat_dim_32262
    out_arrsizze_34555 = sizze_32258
    out_mem_34553 = mem_34352
    return (out_mem_34553, out_arrsizze_34554, out_arrsizze_34555)
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
  def main(self, trend_32282_ext, k_32283_ext, n_32284_ext, freq_32285_ext,
           hfrac_32286_ext, lam_32287_ext, mappingindices_mem_34348_ext,
           images_mem_34349_ext):
    try:
      trend_32282 = np.int32(ct.c_int32(trend_32282_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("i32",
                                                                                                                            type(trend_32282_ext),
                                                                                                                            trend_32282_ext))
    try:
      k_32283 = np.int32(ct.c_int32(k_32283_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("i32",
                                                                                                                            type(k_32283_ext),
                                                                                                                            k_32283_ext))
    try:
      n_32284 = np.int32(ct.c_int32(n_32284_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #2 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("i32",
                                                                                                                            type(n_32284_ext),
                                                                                                                            n_32284_ext))
    try:
      freq_32285 = np.float32(ct.c_float(freq_32285_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #3 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(freq_32285_ext),
                                                                                                                            freq_32285_ext))
    try:
      hfrac_32286 = np.float32(ct.c_float(hfrac_32286_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #4 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(hfrac_32286_ext),
                                                                                                                            hfrac_32286_ext))
    try:
      lam_32287 = np.float32(ct.c_float(lam_32287_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #5 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(lam_32287_ext),
                                                                                                                            lam_32287_ext))
    try:
      assert ((type(mappingindices_mem_34348_ext) in [np.ndarray,
                                                      cl.array.Array]) and (mappingindices_mem_34348_ext.dtype == np.int32)), "Parameter has unexpected type"
      sizze_32279 = np.int32(mappingindices_mem_34348_ext.shape[0])
      if (type(mappingindices_mem_34348_ext) == cl.array.Array):
        mappingindices_mem_34348 = mappingindices_mem_34348_ext.data
      else:
        mappingindices_mem_34348 = opencl_alloc(self,
                                                np.int64(mappingindices_mem_34348_ext.nbytes),
                                                "mappingindices_mem_34348")
        if (np.int64(mappingindices_mem_34348_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, mappingindices_mem_34348,
                          normaliseArray(mappingindices_mem_34348_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #6 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]i32",
                                                                                                                            type(mappingindices_mem_34348_ext),
                                                                                                                            mappingindices_mem_34348_ext))
    try:
      assert ((type(images_mem_34349_ext) in [np.ndarray,
                                              cl.array.Array]) and (images_mem_34349_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_32280 = np.int32(images_mem_34349_ext.shape[0])
      sizze_32281 = np.int32(images_mem_34349_ext.shape[1])
      if (type(images_mem_34349_ext) == cl.array.Array):
        images_mem_34349 = images_mem_34349_ext.data
      else:
        images_mem_34349 = opencl_alloc(self,
                                        np.int64(images_mem_34349_ext.nbytes),
                                        "images_mem_34349")
        if (np.int64(images_mem_34349_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, images_mem_34349,
                          normaliseArray(images_mem_34349_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #7 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f32",
                                                                                                                            type(images_mem_34349_ext),
                                                                                                                            images_mem_34349_ext))
    (out_mem_34562, out_arrsizze_34563, out_mem_34564, out_arrsizze_34565,
     out_mem_34566, out_arrsizze_34567, out_mem_34568, out_arrsizze_34569,
     out_mem_34570, out_arrsizze_34571, out_arrsizze_34572, out_mem_34573,
     out_arrsizze_34574, out_arrsizze_34575, out_mem_34576, out_arrsizze_34577,
     out_mem_34578, out_arrsizze_34579, out_mem_34580, out_arrsizze_34581,
     out_mem_34582, out_arrsizze_34583, out_arrsizze_34584, out_mem_34585,
     out_arrsizze_34586,
     out_arrsizze_34587) = self.futhark_main(mappingindices_mem_34348,
                                             images_mem_34349, sizze_32279,
                                             sizze_32280, sizze_32281,
                                             trend_32282, k_32283, n_32284,
                                             freq_32285, hfrac_32286, lam_32287)
    return (cl.array.Array(self.queue, (out_arrsizze_34563,), ct.c_float,
                           data=out_mem_34562), cl.array.Array(self.queue,
                                                               (out_arrsizze_34565,),
                                                               ct.c_int32,
                                                               data=out_mem_34564),
            cl.array.Array(self.queue, (out_arrsizze_34567,), ct.c_int32,
                           data=out_mem_34566), cl.array.Array(self.queue,
                                                               (out_arrsizze_34569,),
                                                               ct.c_float,
                                                               data=out_mem_34568),
            cl.array.Array(self.queue, (out_arrsizze_34571, out_arrsizze_34572),
                           ct.c_float, data=out_mem_34570),
            cl.array.Array(self.queue, (out_arrsizze_34574, out_arrsizze_34575),
                           ct.c_float, data=out_mem_34573),
            cl.array.Array(self.queue, (out_arrsizze_34577,), ct.c_float,
                           data=out_mem_34576), cl.array.Array(self.queue,
                                                               (out_arrsizze_34579,),
                                                               ct.c_int32,
                                                               data=out_mem_34578),
            cl.array.Array(self.queue, (out_arrsizze_34581,), ct.c_float,
                           data=out_mem_34580), cl.array.Array(self.queue,
                                                               (out_arrsizze_34583,
                                                                out_arrsizze_34584),
                                                               ct.c_float,
                                                               data=out_mem_34582),
            cl.array.Array(self.queue, (out_arrsizze_34586, out_arrsizze_34587),
                           ct.c_float, data=out_mem_34585))
  def remove_nans(self, nan_value_32268_ext, images_mem_34348_ext):
    try:
      nan_value_32268 = np.int16(ct.c_int16(nan_value_32268_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("i16",
                                                                                                                            type(nan_value_32268_ext),
                                                                                                                            nan_value_32268_ext))
    try:
      assert ((type(images_mem_34348_ext) in [np.ndarray,
                                              cl.array.Array]) and (images_mem_34348_ext.dtype == np.int16)), "Parameter has unexpected type"
      sizze_32265 = np.int32(images_mem_34348_ext.shape[0])
      sizze_32266 = np.int32(images_mem_34348_ext.shape[1])
      sizze_32267 = np.int32(images_mem_34348_ext.shape[2])
      if (type(images_mem_34348_ext) == cl.array.Array):
        images_mem_34348 = images_mem_34348_ext.data
      else:
        images_mem_34348 = opencl_alloc(self,
                                        np.int64(images_mem_34348_ext.nbytes),
                                        "images_mem_34348")
        if (np.int64(images_mem_34348_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, images_mem_34348,
                          normaliseArray(images_mem_34348_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][][]i16",
                                                                                                                            type(images_mem_34348_ext),
                                                                                                                            images_mem_34348_ext))
    (out_mem_34556, out_arrsizze_34557, out_arrsizze_34558,
     out_arrsizze_34559) = self.futhark_remove_nans(images_mem_34348,
                                                    sizze_32265, sizze_32266,
                                                    sizze_32267,
                                                    nan_value_32268)
    return cl.array.Array(self.queue, (out_arrsizze_34557, out_arrsizze_34558,
                                       out_arrsizze_34559), ct.c_float,
                          data=out_mem_34556)
  def reshapeTransp(self, images_mem_34348_ext):
    try:
      assert ((type(images_mem_34348_ext) in [np.ndarray,
                                              cl.array.Array]) and (images_mem_34348_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_32258 = np.int32(images_mem_34348_ext.shape[0])
      sizze_32259 = np.int32(images_mem_34348_ext.shape[1])
      sizze_32260 = np.int32(images_mem_34348_ext.shape[2])
      if (type(images_mem_34348_ext) == cl.array.Array):
        images_mem_34348 = images_mem_34348_ext.data
      else:
        images_mem_34348 = opencl_alloc(self,
                                        np.int64(images_mem_34348_ext.nbytes),
                                        "images_mem_34348")
        if (np.int64(images_mem_34348_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, images_mem_34348,
                          normaliseArray(images_mem_34348_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][][]f32",
                                                                                                                            type(images_mem_34348_ext),
                                                                                                                            images_mem_34348_ext))
    (out_mem_34553, out_arrsizze_34554,
     out_arrsizze_34555) = self.futhark_reshapeTransp(images_mem_34348,
                                                      sizze_32258, sizze_32259,
                                                      sizze_32260)
    return cl.array.Array(self.queue, (out_arrsizze_34554, out_arrsizze_34555),
                          ct.c_float, data=out_mem_34553)