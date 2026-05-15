import math
import re
from dataclasses import dataclass
from typing import Union


IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def require_ident(name: object, context: str) -> str:
    if not isinstance(name, str) or not IDENT_RE.match(name):
        raise ValueError(f"{context}: '{name}' is not a valid Verilog identifier")
    return name


def parse_int(value: object, context: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{context}: boolean is not an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value, 0)
    raise ValueError(f"{context}: expected integer")


def ceil_log2_capacity(count: int) -> int:
    if count <= 1:
        return 1
    return math.ceil(math.log2(count))


def sv_const(*parts: str) -> str:
    return "_".join(parts).upper()


def sv_comment_list(names: list[str]) -> str:
    return "{" + ", ".join(names) + "}"


def width_range(width: int) -> str:
    return "" if width == 1 else f"[{width - 1}:0] "


@dataclass(frozen=True)
class Field:
    name: str
    width: int
    signed: bool = False
    default: Union[int, str] = 0


@dataclass(frozen=True)
class FieldPlacement:
    field: Field
    msb: int
    lsb: int


@dataclass(frozen=True)
class Command:
    module: str
    name: str
    command_id: int
    args: list[Field]
    arg_layout: list[FieldPlacement]
    command_id_msb: int
    command_id_lsb: int
    arg_width: int
    using: list[str]
    deps: list[str]


@dataclass(frozen=True)
class RegGroup:
    module: str
    name: str
    addr: int
    fields: list[Field]
    field_layout: list[FieldPlacement]
    addr_msb: int
    addr_lsb: int
    packed_width: int


@dataclass(frozen=True)
class Module:
    name: str
    command_id_width: int
    commands: list[Command]
    regs: list[RegGroup]


@dataclass(frozen=True)
class SystemOp:
    name: str
    opcode: int
    deps: list[str]


@dataclass(frozen=True)
class Model:
    instruction_width: int
    opcode_width: int
    reg_addr_width: int
    payload_width: int
    system_ops: list[SystemOp]
    pingpongs: list[str]
    modules: list[Module]
    command_opcodes: dict[str, int]
    regs_opcode: int | None

    @property
    def opcode_msb(self) -> int:
        return self.instruction_width - 1

    @property
    def opcode_lsb(self) -> int:
        return self.payload_width

    @property
    def payload_msb(self) -> int:
        return self.payload_width - 1


def field_width_sum(fields: list[Field]) -> int:
    return sum(field.width for field in fields)


def field_layout(fields: list[Field], total_width: int, prefix_width: int = 0) -> list[FieldPlacement]:
    pos = total_width - prefix_width
    out: list[FieldPlacement] = []
    for field in fields:
        pos -= field.width
        if pos < 0:
            raise ValueError("internal error: field layout overflow")
        out.append(FieldPlacement(field, pos + field.width - 1, pos))
    return out


def packed_width(fields: list[Field]) -> int:
    width = field_width_sum(fields)
    return max(width, 1)


def packed_default(fields: list[Field]) -> str:
    if not fields:
        return "'0"
    parts = []
    for field in fields:
        value = field.default
        if isinstance(value, str):
            parts.append(value)
            continue
        if field.signed:
            parts.append(f"{field.width}'sd{value}")
        else:
            parts.append(f"{field.width}'d{value}")
    return "{" + ", ".join(parts) + "}"
