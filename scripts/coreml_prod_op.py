"""Register aten::prod for coremltools 7.2 (missing from its torch frontend).

MatAnyone's `aggregate()` uses torch.prod(1-prob, dim, keepdim); coremltools 7.2 has no
`prod` op. Map it to MIL `reduce_prod`. Import this module before ct.convert().
"""
from coremltools.converters.mil.frontend.torch.torch_op_registry import register_torch_op
from coremltools.converters.mil.frontend.torch.ops import _get_inputs
from coremltools.converters.mil import Builder as mb


@register_torch_op(torch_alias=["prod"], override=True)
def prod(context, node):
    inputs = _get_inputs(context, node)
    x = inputs[0]
    if len(inputs) >= 2 and inputs[1] is not None and inputs[1].val is not None:
        axis = int(inputs[1].val)
        keepdim = bool(inputs[2].val) if len(inputs) >= 3 and inputs[2] is not None else False
        res = mb.reduce_prod(x=x, axes=[axis], keep_dims=keepdim, name=node.name)
    else:
        res = mb.reduce_prod(x=x, name=node.name)
    context.add(res)
