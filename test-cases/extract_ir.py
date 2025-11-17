#!/usr/bin/env python3
"""
Script to extract TurboFan IR before and after optimization and convert to turbo-tv format.

Usage: ./extract_ir.py <js_file> [function_name] [output_dir] [--no-convert]
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

# Opcodes that typically have no inputs (constants, Start)
NO_INPUT_OPCODES = {
    'Start', 'NumberConstant', 'Int32Constant', 'Int64Constant',
    'Float32Constant', 'Float64Constant', 'HeapConstant', 'ExternalConstant'
}

# Opcodes that only have value inputs (pure operations)
PURE_OPCODES = {
    'NumberModulus', 'Int32Mod',
    'NumberAdd', 'NumberSubtract', 'NumberMultiply', 'NumberDivide', 'NumberModulus',
    'Int32Add', 'Int32Subtract', 'Int32Multiply', 'Int32Divide', 'Int32Mod',
    'ChangeInt32ToTagged', 'ChangeInt31ToTaggedSigned', 'ChangeTaggedToInt32',
    'ChangeTaggedSignedToInt32', 'ChangeTaggedToFloat64', 'ChangeFloat64ToTagged',
    'StateValues', 'TypedStateValues'
}

# Speculative operations have value, effect, and control edges
SPECULATIVE_OPCODES = {
    'SpeculativeNumberAdd', 'SpeculativeNumberSubtract',
    'SpeculativeNumberMultiply', 'SpeculativeNumberDivide', 'SpeculativeNumberModulus',
    'SpeculativeNumberBitwiseAnd', 'SpeculativeNumberBitwiseOr', 'SpeculativeNumberBitwiseXor',
    'SpeculativeNumberShiftLeft', 'SpeculativeNumberShiftRight', 'SpeculativeNumberShiftRightLogical',
    'SpeculativeSafeIntegerAdd', 'SpeculativeNumberLessThan', 'SpeculativeNumberLessThanOrEqual',
    'SpeculativeNumberEqual', 'SpeculativeNumberGreaterThan', 'SpeculativeNumberGreaterThanOrEqual'
}

# Opcodes that have value inputs only (FrameState)
FRAMESTATE_OPCODES = {'FrameState'}

# Opcodes that have effect and control (no value inputs)
EFFECT_CONTROL_OPCODES = {'JSStackCheck', 'Checkpoint'}

# Opcodes that only have control input
CONTROL_ONLY_OPCODES = {'End'}

# Opcodes where last input is typically control
CONTROL_LAST_OPCODES = {
    'Return', 'JSStoreGlobal', 'JSLoadGlobal', 'StoreField', 'LoadField',
    'StoreElement', 'LoadElement', 'CheckedInt32Add', 'CheckedInt32Subtract',
    'CheckedTaggedSignedToInt32', 'SpeculativeSmallIntegerAdd'
}


def parse_ir_line(line):
    """Parse an IR line into components."""
    # Remove type annotation if present
    line = re.sub(r'\s+\[Type:.*?\]$', '', line)
    
    # Match: #ID:Opcode[...](inputs)
    match = re.match(r'^#(\d+):([A-Za-z0-9_]+)(\[.*?\])?\((.*?)\)(.*)$', line)
    if not match:
        return None
    
    node_id = match.group(1)
    opcode = match.group(2)
    opcode_suffix = match.group(3) or ''
    inputs_str = match.group(4)
    rest = match.group(5)
    
    # Parse inputs
    inputs = []
    if inputs_str.strip():
        # Split by comma, but be careful with nested structures
        current = ''
        depth = 0
        for char in inputs_str:
            if char == '(' or char == '[' or char == '<':
                depth += 1
                current += char
            elif char == ')' or char == ']' or char == '>':
                depth -= 1
                current += char
            elif char == ',' and depth == 0:
                if current.strip():
                    inputs.append(current.strip())
                current = ''
            else:
                current += char
        if current.strip():
            inputs.append(current.strip())
    
    return {
        'node_id': node_id,
        'opcode': opcode,
        'opcode_suffix': opcode_suffix,
        'inputs': inputs,
        'rest': rest
    }


def classify_edges(opcode, inputs):
    """Classify edges into value, effect, and control based on opcode semantics."""
    if not inputs:
        return [], [], []
    
    # Constants and Start have no inputs
    if opcode in NO_INPUT_OPCODES:
        return [], [], []
    
    # Pure operations: all inputs are value
    if opcode in PURE_OPCODES:
        return inputs, [], []
    
    # Speculative operations: value inputs, then effect, then control
    if opcode in SPECULATIVE_OPCODES:
        if len(inputs) == 1:
            return [inputs[0]], [], []
        elif len(inputs) == 2:
            return [inputs[0]], [inputs[1]], [inputs[1]]
        elif len(inputs) >= 3:
            # Heuristic: last input is control, second-to-last might be effect
            # For speculative ops, typically: value inputs, effect, control
            last = inputs[-1]
            is_control = any(ctrl in last for ctrl in ['Start', 'JSStackCheck', 'Checkpoint', 'Merge', 'End'])
            if is_control and len(inputs) >= 3:
                # Check if second-to-last is effect
                second_last = inputs[-2]
                is_effect = any(eff in second_last for eff in ['Checkpoint', 'JSStackCheck', 'FrameState'])
                if is_effect:
                    return inputs[:-2], [inputs[-2]], [inputs[-1]]
                else:
                    # All but last are value, last is control
                    return inputs[:-1], [inputs[-1]], [inputs[-1]]
            else:
                # Default: first N-2 are value, second-to-last is effect, last is control
                return inputs[:-2], [inputs[-2]], [inputs[-1]]
        else:
            return inputs, [], []
    
    # FrameState: all inputs are value
    if opcode in FRAMESTATE_OPCODES:
        return inputs, [], []
    
    # Effect and control only (no value)
    if opcode in EFFECT_CONTROL_OPCODES:
        if opcode == 'JSStackCheck':
            # JSStackCheck: find Start or JSStackCheck nodes for effect/control
            # Usually the last input(s) are the control/effect
            control_nodes = [inp for inp in inputs if any(ctrl in inp for ctrl in ['Start', 'JSStackCheck'])]
            if control_nodes:
                ctrl = control_nodes[-1]
                return [], [ctrl], [ctrl]
            elif inputs:
                # Default: last input is both effect and control
                return [], [inputs[-1]], [inputs[-1]]
            else:
                return [], [], []
        elif opcode == 'Checkpoint':
            # Checkpoint: first input is effect (FrameState), last is control
            if len(inputs) >= 2:
                return [], [inputs[0]], [inputs[-1]]
            elif len(inputs) == 1:
                return [], [inputs[0]], [inputs[0]]
            else:
                return [], [], []
        else:
            if len(inputs) == 1:
                return [], [inputs[0]], [inputs[0]]
            elif len(inputs) == 2:
                return [], [inputs[0]], [inputs[1]]
            elif len(inputs) >= 3:
                # First input is effect, last is control
                return [], [inputs[0]], [inputs[-1]]
            else:
                return [], [], []
    
    # Control only
    if opcode in CONTROL_ONLY_OPCODES:
        if inputs:
            return [], [], [inputs[0]]
        return [], [], []
    
    # Parameters: control from Start goes in value position
    if opcode == 'Parameter':
        if inputs:
            return [inputs[0]], [], []
        return [], [], []
    
    # Return: first N-2 are value, second-to-last is effect, last is control
    if opcode == 'Return':
        if len(inputs) == 1:
            return [inputs[0]], [], []
        elif len(inputs) == 2:
            return [inputs[0]], [inputs[1]], [inputs[1]]
        elif len(inputs) >= 3:
            # Last is control, second-to-last is effect, rest are value
            return inputs[:-2], [inputs[-2]], [inputs[-1]]
        else:
            return inputs, [], []
    
    # Operations with side effects: heuristic based on input types
    if opcode in CONTROL_LAST_OPCODES:
        if len(inputs) == 1:
            return [inputs[0]], [], []
        elif len(inputs) == 2:
            # Check if second is a control node
            if any(ctrl in inputs[1] for ctrl in ['Start', 'JSStackCheck', 'Checkpoint', 'Merge', 'End']):
                return [inputs[0]], [inputs[0]], [inputs[1]]
            else:
                return inputs, [], []
        elif len(inputs) >= 3:
            # Heuristic: if last input contains control keywords, it's control
            # Second-to-last might be effect, rest are value
            last = inputs[-1]
            is_control = any(ctrl in last for ctrl in ['Start', 'JSStackCheck', 'Checkpoint', 'Merge', 'End', 'Return'])
            
            if is_control:
                if len(inputs) >= 4:
                    # Check if second-to-last is effect (often Checkpoint or similar)
                    second_last = inputs[-2]
                    is_effect = any(eff in second_last for eff in ['Checkpoint', 'JSStackCheck', 'FrameState'])
                    if is_effect:
                        return inputs[:-2], [inputs[-2]], [inputs[-1]]
                    else:
                        # All but last are value, last is control
                        return inputs[:-1], [inputs[-1]], [inputs[-1]]
                else:
                    # 3 inputs: value, effect, control
                    return [inputs[0]], [inputs[1]], [inputs[2]]
            else:
                # No clear control, treat all as value
                return inputs, [], []
        else:
            return inputs, [], []
    
    # Default heuristic: try to identify control edges
    # Look for common control node patterns in inputs
    control_nodes = []
    effect_nodes = []
    value_nodes = []
    
    for inp in inputs:
        # Check if it's a control node
        if any(ctrl in inp for ctrl in ['Start', 'End', 'Return', 'Merge', 'Branch', 'IfTrue', 'IfFalse']):
            control_nodes.append(inp)
        # Check if it's an effect node
        elif any(eff in inp for eff in ['JSStackCheck', 'Checkpoint', 'FrameState']):
            effect_nodes.append(inp)
        else:
            value_nodes.append(inp)
    
    # If we found control nodes, use them
    if control_nodes:
        # Last control node is the control edge
        # Others might be effect
        if len(control_nodes) == 1 and len(inputs) > 1:
            # One control, rest are value/effect
            return value_nodes + effect_nodes, effect_nodes if effect_nodes else [control_nodes[0]], [control_nodes[-1]]
        else:
            return value_nodes, effect_nodes, [control_nodes[-1]]
    
    # If we found effect nodes but no control
    if effect_nodes:
        return value_nodes, effect_nodes, []
    
    # Default: all are value
    return inputs, [], []


def map_opcode_in_reference(ref):
    """No-op: return reference as-is (opcode mapping removed to allow turbo-tv to error on unsupported opcodes)."""
    return ref


def format_edges(value_edges, effect_edges, control_edges):
    """Format edges into three parentheses sets."""
    def format_edge_list(edges):
        if not edges:
            return ''
        return ', '.join(edges)
    
    value_str = format_edge_list(value_edges)
    effect_str = format_edge_list(effect_edges)
    control_str = format_edge_list(control_edges)
    
    return f"({value_str})({effect_str})({control_str})"


def convert_line(line):
    """Convert a single IR line from d8 format to turbo-tv format."""
    line = line.rstrip()
    
    # Skip empty lines and non-IR lines
    if not line or not line.startswith('#'):
        return line
    
    parsed = parse_ir_line(line)
    if not parsed:
        return line
    
    # Use opcode as-is (no mapping - let turbo-tv error on unsupported opcodes)
    opcode = parsed['opcode']
    
    # Classify edges
    value_edges, effect_edges, control_edges = classify_edges(opcode, parsed['inputs'])
    
    # Format the line
    opcode_part = f"#{parsed['node_id']}:{opcode}{parsed['opcode_suffix']}"
    edges_part = format_edges(value_edges, effect_edges, control_edges)
    
    result = f"{opcode_part}{edges_part}{parsed['rest']}"
    return result


def convert_ir_file(input_path, output_path):
    """Convert an IR file from d8 format to turbo-tv format."""
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    with open(output_path, 'w') as f:
        for line in lines:
            converted = convert_line(line)
            f.write(converted + '\n')


def extract_ir_from_trace(trace_content, phase_pattern, exclude_pattern):
    """Extract IR lines from trace content for a specific phase.
    
    This exactly mimics the awk range pattern behavior:
    /Graph after V8.TFTypedLowering/,/Graph after V8.TF[^T]/ {
        if (/Graph after V8.TF/ && !/TypedLowering/) exit
        if (/^#/) print
    }
    
    The awk range pattern /pattern1/,/pattern2/ means:
    - Start when pattern1 matches
    - Continue until pattern2 matches (inclusive of the line that matches pattern2)
    - But the inner check exits early if we see a different phase
    """
    lines = trace_content.split('\n')
    in_phase = False
    ir_lines = []
    
    # Extract the character we need to check (e.g., 'T' from "TypedLowering")
    # The pattern /Graph after V8.TF[^T]/ means "TF" followed by NOT 'T'
    exclude_char = exclude_pattern[0] if exclude_pattern else None
    
    # Compile the end pattern: Graph after V8.TF followed by NOT exclude_char
    if exclude_char:
        end_pattern = re.compile(r'Graph after V8\.TF[^{}]'.format(re.escape(exclude_char)))
    else:
        end_pattern = None
    
    for line in lines:
        # Check if we're entering the target phase (matches pattern1)
        if phase_pattern in line:
            in_phase = True
        
        # Check if we're leaving the phase (matches pattern2 or the inner check)
        if in_phase:
            # Inner check: if (/Graph after V8.TF/ && !/TypedLowering/) exit
            if 'Graph after V8.TF' in line:
                if exclude_pattern and exclude_pattern not in line:
                    # Early exit - we've moved to a different phase
                    break
                # Also check the [^X] pattern
                if end_pattern and end_pattern.search(line):
                    # Matched the end pattern - stop extracting
                    break
        
        # Collect IR lines (lines starting with #)
        # This matches: if (/^#/) print
        if in_phase and line.startswith('#'):
            ir_lines.append(line)
    
    return '\n'.join(ir_lines)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Extract TurboFan IR before and after optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s test.js
  %(prog)s test.js myFunction
  %(prog)s test.js myFunction output/
  %(prog)s test.js myFunction output/ --no-convert
        """
    )
    
    parser.add_argument('js_file', help='JavaScript file to process')
    parser.add_argument('function_name', nargs='?', default='*',
                       help='Function name to filter (default: *)')
    parser.add_argument('output_dir', nargs='?', default='.',
                       help='Output directory (default: current directory)')
    parser.add_argument('--no-convert', action='store_true',
                       help='Skip IR format conversion')
    
    args = parser.parse_args()
    
    js_file = args.js_file
    func_name = args.function_name
    output_dir = args.output_dir
    no_convert = args.no_convert
    
    # Validate inputs
    if not os.path.isfile(js_file):
        print(f"Error: JavaScript file not found: {js_file}", file=sys.stderr)
        sys.exit(1)
    
    # Default d8 path
    d8_path = os.path.expanduser("~/v8/v8/out/x64.debug/d8")
    
    if not os.path.isfile(d8_path):
        print(f"Error: d8 not found at {d8_path}", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Run d8 with trace-turbo and capture output
    print(f"Running d8 with trace-turbo for function: {func_name}")
    try:
        result = subprocess.run(
            [d8_path, '--trace-turbo', '--trace-turbo-graph',
             f'--trace-turbo-filter={func_name}', '--allow-natives-syntax', js_file],
            capture_output=True,
            text=True,
            check=False
        )
        # d8 outputs trace to stderr, but old script captures both with 2>&1
        # Combine both stdout and stderr to match old behavior
        trace_content = result.stdout + result.stderr
    except Exception as e:
        print(f"Error running d8: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Extract IR from TypedLowering phase (before optimization) -> src.ir.raw
    src_ir_raw = os.path.join(output_dir, 'src.ir.raw')
    src_ir_content = extract_ir_from_trace(trace_content, 'Graph after V8.TFTypedLowering', 'TypedLowering')
    with open(src_ir_raw, 'w') as f:
        f.write(src_ir_content)
    
    # Extract IR from SimplifiedLowering phase (after optimization) -> tgt.ir.raw
    tgt_ir_raw = os.path.join(output_dir, 'tgt.ir.raw')
    tgt_ir_content = extract_ir_from_trace(trace_content, 'Graph after V8.TFSimplifiedLowering', 'SimplifiedLowering')
    
    # If SimplifiedLowering doesn't exist, try Untyper phase
    if not tgt_ir_content.strip():
        tgt_ir_content = extract_ir_from_trace(trace_content, 'Graph after V8.TFUntyper', 'Untyper')
    
    with open(tgt_ir_raw, 'w') as f:
        f.write(tgt_ir_content)
    
    # Convert IR format if requested
    src_ir = os.path.join(output_dir, 'src.ir')
    tgt_ir = os.path.join(output_dir, 'tgt.ir')
    
    if not no_convert:
        print("Converting IR format to turbo-tv format...")
        convert_ir_file(src_ir_raw, src_ir)
        convert_ir_file(tgt_ir_raw, tgt_ir)
        # Remove raw files
        os.remove(src_ir_raw)
        os.remove(tgt_ir_raw)
    else:
        # Just rename raw files
        os.rename(src_ir_raw, src_ir)
        os.rename(tgt_ir_raw, tgt_ir)
    
    # Print results
    print("\nExtracted IR files:")
    print(f"  src.ir (before optimization): {src_ir}")
    print(f"  tgt.ir (after optimization): {tgt_ir}")
    if not no_convert:
        print("\nConverted")


if __name__ == '__main__':
    main()

