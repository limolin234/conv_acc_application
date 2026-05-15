import re

from .common import ceil_log2_capacity, field_width_sum, parse_int, require_ident


VERILOG_DEFAULT_RE = re.compile(
    r"^[+-]?([0-9]+)?'s?[bBoOdDhH][0-9a-fA-F_xXzZ?]+$|^[+-]?[0-9]+$"
)


def get_key(obj: dict, *names: str, default=None):
    for name in names:
        if name in obj:
            return obj[name]
    return default


class SpecValidator:
    def __init__(self, spec: dict):
        self.spec = spec
        self.errors: list[str] = []

    def error(self, text: str) -> None:
        self.errors.append(text)

    def require_int(self, key: str) -> int | None:
        if key not in self.spec:
            self.error(f"missing required key: {key}")
            return None
        try:
            value = parse_int(self.spec[key], key)
        except ValueError as exc:
            self.error(str(exc))
            return None
        if value <= 0:
            self.error(f"{key}: must be positive")
            return None
        return value

    def validate(self) -> None:
        if not isinstance(self.spec, dict):
            raise ValueError("top-level spec must be an object")

        iw = self.require_int("InstructionWidth")
        opw = self.require_int("OpcodeWidth")
        raw = self.require_int("RegAddrWidth")
        pw = self.require_int("PayloadWidth")
        if iw is not None and iw != 128:
            self.error(f"InstructionWidth: only 128-bit instructions are supported now, got {iw}")
        if None not in (iw, opw, raw, pw) and iw != opw + pw:
            self.error(f"InstructionWidth must equal OpcodeWidth + PayloadWidth ({iw} != {opw} + {pw})")

        pingpongs_obj = self.spec.get("Pingpongs", {})
        if not isinstance(pingpongs_obj, dict):
            self.error("Pingpongs: must be an object")
            pingpongs_obj = {}
        pingpongs = set()
        for name in pingpongs_obj:
            try:
                require_ident(name, "Pingpongs")
            except ValueError as exc:
                self.error(str(exc))
            if name in pingpongs:
                self.error(f"Pingpongs: duplicate name {name}")
            pingpongs.add(name)

        modules_obj = self.spec.get("Modules")
        if not isinstance(modules_obj, dict) or not modules_obj:
            self.error("Modules: must be a non-empty object")
            modules_obj = {}
        module_names = set(modules_obj.keys())
        seen_modules = set()
        total_regs = 0
        for module_name, module_body in modules_obj.items():
            self.validate_module(module_name, module_body, pingpongs, module_names, seen_modules, raw, pw)
            seen_modules.add(module_name)
            if isinstance(module_body, dict) and isinstance(module_body.get("Regs", {}), dict):
                total_regs += len(module_body.get("Regs", {}))
        if raw is not None and total_regs > (1 << raw):
            self.error(f"Modules: {total_regs} total register groups do not fit RegAddrWidth={raw}")

        system_obj = self.spec.get("System", {})
        if not isinstance(system_obj, dict):
            self.error("System: must be an object")
            system_obj = {}
        for name, body in system_obj.items():
            try:
                require_ident(name, "System")
            except ValueError as exc:
                self.error(str(exc))
            if body is None:
                body = {}
            if not isinstance(body, dict):
                self.error(f"System.{name}: must be an object")
                continue
            self.validate_deps(body.get("Deps", []), module_names, pingpongs, f"System.{name}.Deps")

        if opw is not None:
            system_count = len(system_obj)
            command_count = sum(1 for body in modules_obj.values() if isinstance(body, dict) and body.get("Command"))
            opcodes_needed = system_count + (1 if self.any_regs(modules_obj) else 0) + command_count
            if opcodes_needed > (1 << opw):
                self.error(f"OpcodeWidth={opw} cannot encode {opcodes_needed} generated opcodes")

        if self.errors:
            raise ValueError("\n".join(self.errors))

    @staticmethod
    def any_regs(modules_obj: dict) -> bool:
        for body in modules_obj.values():
            if isinstance(body, dict) and body.get("Regs"):
                return True
        return False

    def validate_module(self, module_name, module_body, pingpongs, module_names, seen_modules, raw, pw) -> None:
        try:
            require_ident(module_name, "Modules")
        except ValueError as exc:
            self.error(str(exc))
        if module_name in seen_modules:
            self.error(f"Modules: duplicate module name {module_name}")
        if not isinstance(module_body, dict):
            self.error(f"Modules.{module_name}: must be an object")
            return

        commands_obj = module_body.get("Command", {})
        if not isinstance(commands_obj, dict):
            self.error(f"Modules.{module_name}.Command: must be an object")
            commands_obj = {}
        command_names = list(commands_obj.keys())
        for command_name in command_names:
            try:
                require_ident(command_name, f"Modules.{module_name}.Command")
            except ValueError as exc:
                self.error(str(exc))

        try:
            cmd_width = parse_int(module_body["CommandOpcodeWidth"], f"Modules.{module_name}.CommandOpcodeWidth") if "CommandOpcodeWidth" in module_body else ceil_log2_capacity(len(command_names))
            if cmd_width <= 0:
                self.error(f"Modules.{module_name}.CommandOpcodeWidth: must be positive")
        except ValueError as exc:
            self.error(str(exc))
            cmd_width = 1
        if len(command_names) > (1 << cmd_width):
            self.error(f"Modules.{module_name}: {len(command_names)} commands do not fit CommandOpcodeWidth={cmd_width}")

        for command_name, command_body in commands_obj.items():
            context = f"Modules.{module_name}.Command.{command_name}"
            if not isinstance(command_body, dict):
                self.error(f"{context}: must be an object")
                continue
            args = self.validate_fields(command_body.get("Args", []), f"{context}.Args")
            if pw is not None and cmd_width + field_width_sum(args) > pw:
                self.error(f"{context}: command id width {cmd_width} + args width {field_width_sum(args)} exceeds PayloadWidth={pw}")
            self.validate_pingpong_list(command_body.get("Using", []), pingpongs, f"{context}.Using")
            self.validate_deps(command_body.get("Deps", []), module_names, pingpongs, f"{context}.Deps")

        regs_obj = module_body.get("Regs", {})
        if not isinstance(regs_obj, dict):
            self.error(f"Modules.{module_name}.Regs: must be an object")
            regs_obj = {}
        for reg_name, fields_raw in regs_obj.items():
            context = f"Modules.{module_name}.Regs.{reg_name}"
            try:
                require_ident(reg_name, f"Modules.{module_name}.Regs")
            except ValueError as exc:
                self.error(str(exc))
            fields = self.validate_fields(fields_raw, context)
            if raw is not None and pw is not None and raw + field_width_sum(fields) > pw:
                self.error(f"{context}: RegAddrWidth {raw} + fields width {field_width_sum(fields)} exceeds PayloadWidth={pw}")

    def validate_fields(self, items, context: str):
        if not isinstance(items, list):
            self.error(f"{context}: must be a list")
            return []
        seen = set()
        out = []
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                self.error(f"{context}[{idx}]: field must be an object")
                continue
            name = get_key(item, "Name", "name")
            try:
                require_ident(name, f"{context}[{idx}].Name")
            except ValueError as exc:
                self.error(str(exc))
                continue
            if name in seen:
                self.error(f"{context}: duplicate field {name}")
            seen.add(name)
            try:
                width = parse_int(get_key(item, "Width", "width"), f"{context}.{name}.Width")
                if width <= 0:
                    self.error(f"{context}.{name}.Width: must be positive")
            except ValueError as exc:
                self.error(str(exc))
                width = 1
            signed = bool(get_key(item, "Signed", "signed", default=False))
            default = get_key(item, "Default", "default", default=0)
            if isinstance(default, str):
                try:
                    parsed_default = parse_int(default, f"{context}.{name}.Default")
                    lo = -(1 << (width - 1)) if signed else 0
                    hi = (1 << (width - 1)) - 1 if signed else (1 << width) - 1
                    if parsed_default < lo or parsed_default > hi:
                        self.error(f"{context}.{name}.Default={parsed_default} does not fit Width={width}")
                except ValueError:
                    if not default or not VERILOG_DEFAULT_RE.match(default):
                        self.error(
                            f"{context}.{name}.Default: string default must be a Verilog literal, got {default!r}"
                        )
            else:
                try:
                    default = parse_int(default, f"{context}.{name}.Default")
                    lo = -(1 << (width - 1)) if signed else 0
                    hi = (1 << (width - 1)) - 1 if signed else (1 << width) - 1
                    if default < lo or default > hi:
                        self.error(f"{context}.{name}.Default={default} does not fit Width={width}")
                except ValueError as exc:
                    self.error(str(exc))
            out.append(type("CheckedField", (), {"name": name, "width": width})())
        return out

    def validate_pingpong_list(self, items, pingpongs, context: str) -> None:
        if not isinstance(items, list):
            self.error(f"{context}: must be a list")
            return
        for name in items:
            if name not in pingpongs:
                self.error(f"{context}: unknown pingpong/resource {name}")

    def validate_deps(self, items, module_names, pingpongs, context: str) -> None:
        if not isinstance(items, list):
            self.error(f"{context}: must be a list")
            return
        valid = set(module_names) | set(pingpongs)
        for name in items:
            if name not in valid:
                self.error(f"{context}: unknown dependency {name}")


def validate_spec(spec: dict) -> None:
    SpecValidator(spec).validate()
