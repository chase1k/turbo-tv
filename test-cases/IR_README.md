# Extracting TurboFan IR from JavaScript Files

## Usage

### Python Script 

The `extract_ir.py` script is a merged Python implementation that combines IR extraction and format conversion:

```bash
python3 extract_ir.py <js_file> [function_name] [output_dir] [--no-convert]
```

Examples:
```bash
# Extract IR for function 'a' in current directory
python3 extract_ir.py func_call.js a .

# Extract IR for all functions (use '*' as function name)
python3 extract_ir.py func_call.js '*' .

# Skip format conversion (keep d8 format)
python3 extract_ir.py func_call.js a . --no-convert
```

### Output

- `src.ir` - IR before optimization (from TypedLowering phase)
- `tgt.ir` - IR after optimization (from SimplifiedLowering or Untyper phase)

## Flags used 

- `--trace-turbo` - Enable TurboFan IR tracing
- `--trace-turbo-graph` - Output graph format
- `--trace-turbo-filter="<function_name>"` - Filter to specific function
- `--allow-natives-syntax` - Allow V8 intrinsics like `%OptimizeFunctionOnNextCall`

## Format Conversion

The script automatically converts d8's IR format to turbo-tv's expected format rather than using the ./exp script in the docker container.

### d8 format
```
#33:Return(#32:NumberConstant, #58:NumberConstant, #26:JSStoreGlobal, #26:JSStoreGlobal)
```

### turbo-tv format
```
#33:Return(#32:NumberConstant, #58:NumberConstant)(#26:JSStoreGlobal)(#26:JSStoreGlobal)
```

(value edges)(effect edges)(control edges)

## Opcode limitations not recognized by turbo-tv. I.e.: `JSStoreGlobal`

Common output is

```sh
Result: Not Implemented 1
Opcodes: [JSStoreGlobal]
```

## Json output

```sh
d8 --trace-turbo --trace-turbo-path=./output_dir --trace-turbo-filter="a" --allow-natives-syntax func_call.js
```

