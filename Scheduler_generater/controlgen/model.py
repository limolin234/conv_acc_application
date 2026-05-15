from .common import (
    Command,
    Field,
    Model,
    Module,
    RegGroup,
    SystemOp,
    ceil_log2_capacity,
    field_layout,
    field_width_sum,
    packed_width,
    parse_int,
)
from .validate import validate_spec


def get_key(obj: dict, *names: str, default=None):
    for name in names:
        if name in obj:
            return obj[name]
    return default


def parse_default(value, context: str):
    if isinstance(value, str):
        try:
            return parse_int(value, context)
        except ValueError:
            return value
    return parse_int(value, context)


def parse_field(item: dict) -> Field:
    name = get_key(item, "Name", "name")
    return Field(
        name=name,
        width=parse_int(get_key(item, "Width", "width"), f"{name}.Width"),
        signed=bool(get_key(item, "Signed", "signed", default=False)),
        default=parse_default(get_key(item, "Default", "default", default=0), f"{name}.Default"),
    )


def parse_model(spec: dict) -> Model:
    validate_spec(spec)

    iw = parse_int(spec["InstructionWidth"], "InstructionWidth")
    opw = parse_int(spec["OpcodeWidth"], "OpcodeWidth")
    raw = parse_int(spec["RegAddrWidth"], "RegAddrWidth")
    pw = parse_int(spec["PayloadWidth"], "PayloadWidth")
    pingpongs = list(spec.get("Pingpongs", {}).keys())

    modules = []
    next_reg_addr = 0
    for module_name, module_body in spec["Modules"].items():
        command_obj = module_body.get("Command", {})
        command_names = list(command_obj.keys())
        cmd_width = parse_int(module_body["CommandOpcodeWidth"], f"{module_name}.CommandOpcodeWidth") if "CommandOpcodeWidth" in module_body else ceil_log2_capacity(len(command_names))
        commands = []
        for command_id, (command_name, command_body) in enumerate(command_obj.items()):
            args = [parse_field(item) for item in command_body.get("Args", [])]
            commands.append(Command(
                module=module_name,
                name=command_name,
                command_id=command_id,
                args=args,
                arg_layout=field_layout(args, pw, cmd_width),
                command_id_msb=pw - 1,
                command_id_lsb=pw - cmd_width,
                arg_width=field_width_sum(args),
                using=list(command_body.get("Using", [])),
                deps=list(command_body.get("Deps", [])),
            ))
        regs = []
        for reg_name, fields_raw in module_body.get("Regs", {}).items():
            fields = [parse_field(item) for item in fields_raw]
            regs.append(RegGroup(
                module=module_name,
                name=reg_name,
                addr=next_reg_addr,
                fields=fields,
                field_layout=field_layout(fields, pw, raw),
                addr_msb=pw - 1,
                addr_lsb=pw - raw,
                packed_width=packed_width(fields),
            ))
            next_reg_addr += 1
        modules.append(Module(module_name, cmd_width, commands, regs))

    opcode = 0
    system_ops = []
    for name, body in spec.get("System", {}).items():
        body = body or {}
        system_ops.append(SystemOp(name=name, opcode=opcode, deps=list(body.get("Deps", []))))
        opcode += 1

    regs_opcode = opcode if any(module.regs for module in modules) else None
    if regs_opcode is not None:
        opcode += 1

    command_opcodes = {}
    for module in modules:
        if module.commands:
            command_opcodes[module.name] = opcode
            opcode += 1

    return Model(
        instruction_width=iw,
        opcode_width=opw,
        reg_addr_width=raw,
        payload_width=pw,
        system_ops=system_ops,
        pingpongs=pingpongs,
        modules=modules,
        command_opcodes=command_opcodes,
        regs_opcode=regs_opcode,
    )
