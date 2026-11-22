import cutlass.cute as cute
from collections.abc import Callable

from typing import Optional
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op
from cutlass import Float32, Int16

def warp_reduction_partial(value: cute.typing.Numeric, op: Callable, width: int) -> cute.typing.Numeric:
    assert 1 <= width <= 32, f"width must be in [1,32], got {width}"
    assert (width & (width - 1)) == 0, f"width must be a power of 2, got {width}"
    result = value
    for i in range(width.bit_length() - 1):
        result = op(result, cute.arch.shuffle_sync_bfly(result,1 << i))
    return result

@dsl_user_op
def cvt_bf16_f32_alt(
    src_bf16: ir.Value,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> Float32:
    v = src_bf16 if isinstance(src_bf16, ir.Value) else src_bf16.ir_value(loc=loc, ip=ip)
    v_i16 = llvm.bitcast(Int16.mlir_type, v, loc=loc, ip=ip)
    instruction = "mov.b32 $0, {0, $1};"# "cvt.f32.bf16 $0, $1;"
    result = llvm.inline_asm(
        Float32.mlir_type, [v_i16],
        instruction, "=f,h",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    return Float32(result)