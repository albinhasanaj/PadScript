import re
import tkinter as tk
from tkinter import filedialog
import sys
from datetime import datetime

# DEBUG = True
DEBUG = False

# -----------------------------
# Token Definitions
# -----------------------------
TOKEN_SPECIFICATIONS = [
    ('WORD_WITH_TYPE', r'\w+(?:<[^<>]+>)+'),  # e.g., sum<WHOLE_NUMBER>
    ('DOT', r'\.'),                            # .
    ('PAD', r'\bpad\b'),                       # pad
    ('LOOP_COOL', r'\bLOOP_COOL\b'),           # LOOP_COOL
    ('GRAB', r'\bgrab\b'),                     # grab
    ('FROM', r'\bfrom\b'),                     # from
    ('ADOPT', r'\bADOPT\b'),                   # ADOPT
    ('MAKE_BIG_BLOCK', r'\bMAKE_BIG_BLOCK\b'), # MAKE_BIG_BLOCK
    ('MAKE_BIGGER_BLOCK', r'\bMAKE_BIGGER_BLOCK\b'), # MAKE_BIGGER_BLOCK
    ('RUN_THIS_BITCH', r'\brun_this_bitch\b'), # run_this_bitch
    ('ATTACK_THIS_BITCH', r'\battack_this_bitch\b'), # attack_this_bitch
    ('HAHA_U_CAUGHT', r'\bhaha_u_caught\b'),   # haha_u_caught
    ('TAKE_THIS', r'\bTAKE_THIS\b'),           # TAKE_THIS
    ('CONFUSING_WORD', r'\bconfusing_word\b'), # confusing_word
    ('UH', r'\buh\b'),                         # uh
    ('HUH', r'\bhuh\b'),                       # huh
    ('DO', r'\bdo\b'),                         # do
    ('DO_NOT_WORK', r'\bdo_not_work\b'),       # do_not_work
    ('DO_NOT_WORK_NOT_WORK', r'\bdo_not_work_not_work\b'), # do_not_work_not_work
    ('YES', r'\bYES\b'),                       # YES
    ('NO', r'\bNO\b'),                         # NO
    ('WHOLE_NUMBER', r'<WHOLE_NUMBER>'),
    ('WORDS', r'<WORDS>'), #string
    ('NOT_NICE_NUMBER', r'<NOT_NICE_NUMBER>'),
    ('MORE_WORDS', r'<MORE_WORDS>'),
    ('WORD_TYPE', r'<WORD>'),
    ('CHAR_TYPE', r'<CHAR>'),
    ('MORE_WHOLE_NUMBER', r'<MORE_WHOLE_NUMBER>'),
    ('YES_OR_NO', r'<YES_OR_NO>'),
    ('EQUALS', r'=='),
    ('ASSIGN', r'='),
    ('SEMICOLON', r';'),
    ('COLON', r':'),
    ('COMMA', r','),
    ('LBRACKET', r'\['),
    ('RBRACKET', r'\]'),
    ('LPAREN', r'\('),
    ('RPAREN', r'\)'),
    ('LBRACE', r'\{'),
    ('RBRACE', r'\}'),
    ('STRING', r'\".*?\"'),
    ('CHAR', r"'.*?'"),
    ('NUMBER', r'\d+'),
    ('WORD', r'[a-zA-Z_][a-zA-Z0-9_]*'),
    ('OPERATOR', r'==|!=|<=|>=|<|>'),
    ('ARITHMETIC', r'\+|\-|\*|/'),
    ('SKIP', r'[ \t\n]+'),
    ('UNKNOWN', r'.'),  # Any other character
]

# Compile regex patterns
TOKEN_REGEX = '|'.join('(?P<%s>%s)' % pair for pair in TOKEN_SPECIFICATIONS)
TOKEN_RE = re.compile(TOKEN_REGEX)

# -----------------------------
# Tokenizer
# -----------------------------
class Token:
    def __init__(self, type_, value, line, column):
        self.type = type_
        self.value = value
        self.line = line
        self.column = column
    
    def __repr__(self):
        return f'Token({self.type}, {self.value}, {self.line}, {self.column})'

def tokenize(code):
    tokens = []
    line_num = 1
    line_start = 0
    for mo in TOKEN_RE.finditer(code):
        kind = mo.lastgroup
        value = mo.group()
        column = mo.start() - line_start
        if kind == 'SKIP':
            if '\n' in value:
                line_num += value.count('\n')
        elif kind == 'UNKNOWN':
            raise SyntaxError(f'Unknown token {value!r} at line {line_num} column {column}')
        else:
            tokens.append(Token(kind, value, line_num, column)) 
        if '\n' in value:
            line_num += value.count('\n') 
            line_start = mo.end()
    tokens.append(Token('EOF', 'EOF', line_num, column)) 
    if DEBUG:
        print(f"Tokenizer: Completed tokenization with {len(tokens)} tokens.")
    return tokens

# -----------------------------
# Interpreter
# -----------------------------
class InterpreterError(Exception):
    pass

class ReturnException(Exception):
    def __init__(self, value):
        self.value = value

class ClassInstance:
    def __init__(self, class_def, args, interpreter):
        self.class_def = class_def
        self.attributes = {}
        self.interpreter = interpreter  # Reference to the interpreter for executing constructor
        
        # Initialize attributes based on constructor parameters
        for param, arg in zip(class_def['params'], args):
            self.attributes[param['name']] = arg
            if DEBUG:
                print(f"ClassInstance: Set attribute '{param['name']}' to '{arg}'")
        
        # Push a new scope for constructor execution
        interpreter.push_scope()
        if DEBUG:
            print(f"ClassInstance: New scope pushed for constructor of '{self.class_def['name']}'")
        
        # Set constructor parameters as variables in the new scope
        for param, arg in zip(class_def['params'], args):
            interpreter.set_variable(param['name'], arg)
            if DEBUG:
                print(f"ClassInstance: Parameter '{param['name']}' set to '{arg}' in constructor scope")
        
        # Execute the constructor body
        if 'body' in class_def:
            if DEBUG:
                print(f"ClassInstance: Executing constructor for class '{self.class_def['name']}'")
            try:
                interpreter.execute_tokens(class_def['body'])
            except ReturnException:
                pass  # Constructors typically do not return values
        else:
            if DEBUG:
                print(f"ClassInstance: No constructor body found for class '{self.class_def['name']}'")
        
        # Pop the constructor scope after execution
        interpreter.pop_scope()
        if DEBUG:
            print(f"ClassInstance: Scope popped after constructor execution for '{self.class_def['name']}'")
    
    def get_attribute(self, attr_name):
        if attr_name in self.attributes:
            return self.attributes[attr_name]
        else:
            raise InterpreterError(f"Attribute '{attr_name}' not found in class '{self.class_def['name']}'.")

    def __repr__(self):
        return f"<Instance of {self.class_def['name']} with attributes {self.attributes}>"

class Interpreter:
    def __init__(self):
        self.scope_stack = [{}]
        self.functions = {}
        self.classes = {}
        self.current_class = None
        self.imported_modules = {}
        self.block_executed = False  # Initialize the flag
        if DEBUG:
            print("Interpreter initialized.")
    
    def handle_do(self, tokens, i):
        """
        Handle 'do' statement (equivalent to 'if').
        Syntax:
            do(condition) {
                // block
            }
        """
        if DEBUG:
            print(f"handle_do: Handling 'do' statement starting at index {i}")

        # Ensure the next token is LPAREN
        if tokens[i + 1].type != 'LPAREN':
            raise InterpreterError(f"Expected '(' after 'do' at line {tokens[i].line}")

        # Parse the condition inside parentheses
        j = i + 2
        condition_tokens = []
        while j < len(tokens) and tokens[j].type != 'RPAREN':
            condition_tokens.append(tokens[j])
            j += 1

        if j >= len(tokens):
            raise InterpreterError(f"Expected ')' to close condition for 'do' at line {tokens[j - 1].line}")

        # Evaluate the condition
        condition_result = self.evaluate_condition(condition_tokens)
        if DEBUG:
            print(f"handle_do: Condition evaluated to {condition_result}")

        # Expect LBRACE
        if tokens[j + 1].type != 'LBRACE':
            raise InterpreterError(f"Expected '{{' to start 'do' block at line {tokens[j + 1].line}")

        # Extract the block tokens
        block_tokens = []
        brace_count = 1
        k = j + 2
        while k < len(tokens) and brace_count > 0:
            if tokens[k].type == 'LBRACE':
                brace_count += 1
            elif tokens[k].type == 'RBRACE':
                brace_count -= 1
            if brace_count > 0:
                block_tokens.append(tokens[k])
            k += 1

        if brace_count != 0:
            raise InterpreterError(f"Unmatched '{{' in 'do' block starting at line {tokens[i].line}")

        # Execute the block if condition is true
        if condition_result:
            if DEBUG:
                print(f"handle_do: Executing 'do' block at index {i}")
            self.push_scope()
            self.execute_tokens(block_tokens)
            self.pop_scope()
            self.block_executed = True  # Set the flag
        else:
            if DEBUG:
                print(f"handle_do: Condition false; 'do' block not executed")

        # Return the new index
        return k

    def handle_do_not_work(self, tokens, i):
        """
        Handle 'do_not_work' statement (equivalent to 'else if').
        Syntax:
            do_not_work(condition) {
                // block
            }
        """
        if DEBUG:
            print(f"handle_do_not_work: Handling 'do_not_work' statement starting at index {i}")

        # Ensure the next token is LPAREN
        if tokens[i + 1].type != 'LPAREN':
            raise InterpreterError(f"Expected '(' after 'do_not_work' at line {tokens[i].line}")

        # Parse the condition inside parentheses
        j = i + 2
        condition_tokens = []
        while j < len(tokens) and tokens[j].type != 'RPAREN':
            condition_tokens.append(tokens[j])
            j += 1

        if j >= len(tokens):
            raise InterpreterError(f"Expected ')' to close condition for 'do_not_work' at line {tokens[j - 1].line}")

        # Evaluate the condition only if no previous block has been executed
        if not self.block_executed:
            condition_result = self.evaluate_condition(condition_tokens)
            if DEBUG:
                print(f"handle_do_not_work: Condition evaluated to {condition_result}")

            # Expect LBRACE
            if tokens[j + 1].type != 'LBRACE':
                raise InterpreterError(f"Expected '{{' to start 'do_not_work' block at line {tokens[j + 1].line}")

            # Extract the block tokens
            block_tokens = []
            brace_count = 1
            k = j + 2
            while k < len(tokens) and brace_count > 0:
                if tokens[k].type == 'LBRACE':
                    brace_count += 1
                elif tokens[k].type == 'RBRACE':
                    brace_count -= 1
                if brace_count > 0:
                    block_tokens.append(tokens[k])
                k += 1

            if brace_count != 0:
                raise InterpreterError(f"Unmatched '{{' in 'do_not_work' block starting at line {tokens[i].line}")

            # Execute the block if condition is true
            if condition_result:
                if DEBUG:
                    print(f"handle_do_not_work: Executing 'do_not_work' block at index {i}")
                self.push_scope()
                self.execute_tokens(block_tokens)
                self.pop_scope()
                self.block_executed = True  # Set the flag
                return k, True
            else:
                if DEBUG:
                    print(f"handle_do_not_work: Condition false; 'do_not_work' block not executed")
                return k, False
        else:
            if DEBUG:
                print(f"handle_do_not_work: Previous block already executed; skipping 'do_not_work'")
            # Skip the 'do_not_work' block
            if tokens[j + 1].type != 'LBRACE':
                raise InterpreterError(f"Expected '{{' to start 'do_not_work' block at line {tokens[j + 1].line}")
            brace_count = 1
            k = j + 2
            while k < len(tokens) and brace_count > 0:
                if tokens[k].type == 'LBRACE':
                    brace_count += 1
                elif tokens[k].type == 'RBRACE':
                    brace_count -= 1
                k += 1
            return k, False

    def handle_do_not_work_not_work(self, tokens, i):
        """
        Handle 'do_not_work_not_work' statement (equivalent to 'else').
        Syntax:
            do_not_work_not_work {
                // block
            }
        """
        if DEBUG:
            print(f"handle_do_not_work_not_work: Handling 'do_not_work_not_work' statement starting at index {i}")

        # Execute only if no previous block has been executed
        if not self.block_executed:
            # Expect LBRACE
            if tokens[i + 1].type != 'LBRACE':
                raise InterpreterError(f"Expected '{{' to start 'do_not_work_not_work' block at line {tokens[i + 1].line}")

            # Extract the block tokens
            block_tokens = []
            brace_count = 1
            j = i + 2
            while j < len(tokens) and brace_count > 0:
                if tokens[j].type == 'LBRACE':
                    brace_count += 1
                elif tokens[j].type == 'RBRACE':
                    brace_count -= 1
                if brace_count > 0:
                    block_tokens.append(tokens[j])
                j += 1

            if brace_count != 0:
                raise InterpreterError(f"Unmatched '{{' in 'do_not_work_not_work' block starting at line {tokens[i].line}")

            # Execute the block
            if DEBUG:
                print(f"handle_do_not_work_not_work: Executing 'do_not_work_not_work' block at index {i}")
            self.push_scope()
            self.execute_tokens(block_tokens)
            self.pop_scope()

        else:
            if DEBUG:
                print(f"handle_do_not_work_not_work: Previous block already executed; skipping 'do_not_work_not_work'")
        # Reset the flag after the entire if-elif-else chain
        self.block_executed = False
        return j

    def handle_loop_cool(self, tokens, i):
        if DEBUG:
            print(f"handle_loop_cool: Handling 'LOOP_COOL' starting at index {i}")
        
        # Expect ':' after 'LOOP_COOL'
        if tokens[i + 1].type != 'COLON':
            raise InterpreterError(f"Expected ':' after 'LOOP_COOL' at line {tokens[i + 1].line}")
        
        # Expect 'GRAB' token
        if tokens[i + 2].type != 'GRAB':
            raise InterpreterError(f"Expected 'grab' after ':' in 'LOOP_COOL' at line {tokens[i + 2].line}")
        
        # Expect loop variable (WORD)
        if tokens[i + 3].type != 'WORD':
            raise InterpreterError(f"Expected loop variable name after 'grab' in 'LOOP_COOL' at line {tokens[i + 3].line}")
        
        loop_var = tokens[i + 3].value
        if DEBUG:
            print(f"handle_loop_cool: Loop variable '{loop_var}'")
        
        # Expect 'FROM' token
        if tokens[i + 4].type != 'FROM':
            raise InterpreterError(f"Expected 'from' after loop variable in 'LOOP_COOL' at line {tokens[i + 4].line}")
        
        # Expect list variable (WORD)
        if tokens[i + 5].type != 'WORD':
            raise InterpreterError(f"Expected list variable name after 'from' in 'LOOP_COOL' at line {tokens[i + 5].line}")
        
        list_var = tokens[i + 5].value
        if DEBUG:
            print(f"handle_loop_cool: Source list variable '{list_var}'")
        
        # Expect 'LBRACE' token to start the loop block
        if tokens[i + 6].type != 'LBRACE':
            raise InterpreterError(f"Expected '{{' to start 'LOOP_COOL' block at line {tokens[i + 6].line}")
        
        # Extract the loop block tokens
        loop_block = []
        brace_count = 1
        j = i + 7  # Start after '{'
        while j < len(tokens) and brace_count > 0:
            if tokens[j].type == 'LBRACE':
                brace_count += 1
            elif tokens[j].type == 'RBRACE':
                brace_count -= 1
            if brace_count > 0:
                loop_block.append(tokens[j])
            j += 1
        
        if brace_count != 0:
            raise InterpreterError(f"Unmatched '{{' in 'LOOP_COOL' block starting at line {tokens[i].line}")
        
        # Retrieve the source list
        source_list = self.get_variable(list_var)
        if not isinstance(source_list, list):
            raise InterpreterError(f"Variable '{list_var}' is not a list at line {tokens[i].line}")
        if DEBUG:
            print(f"handle_loop_cool: Retrieved list '{list_var}': {source_list}")
        
        # Iterate over the list and execute the loop block for each element
        for element in source_list:
            if DEBUG:
                print(f"handle_loop_cool: Iterating with '{loop_var}' = {element}")
            self.push_scope()
            self.set_variable(loop_var, element)
            if DEBUG:
                print(f"handle_loop_cool: Set loop variable '{loop_var}' to '{element}' in new scope")
            self.execute_tokens(loop_block)
            self.pop_scope()
            if DEBUG:
                print(f"handle_loop_cool: Popped scope after iterating '{loop_var}' = '{element}'")
        
        if DEBUG:
            print(f"handle_loop_cool: Completed 'LOOP_COOL' starting at index {i}")
        
        return j  # Return the index after the loop block

    def get_variable(self, name):
        # Handle 'this.attribute' keyword
        if self.current_class and '.' in name:
            parts = name.split('.', 1)
            if parts[0] == 'this':
                attr = parts[1]
                if DEBUG:
                    print(f"get_variable: Accessing attribute '{attr}' of current class '{self.current_class.class_def['name']}'")
                return self.current_class.get_attribute(attr)
        for scope in reversed(self.scope_stack):
            if name in scope:
                value = scope[name]
                if DEBUG:
                    print(f"get_variable: Found variable '{name}' with value '{value}'")
                return value
        raise InterpreterError(f"Undefined variable: {name}")

    def set_variable(self, name, value):
        # Handle 'this.attribute' keyword
        if self.current_class and '.' in name:
            parts = name.split('.', 1)
            if parts[0] == 'this':
                attr = parts[1]
                self.current_class.attributes[attr] = value
                if DEBUG:
                    print(f"set_variable: Set attribute '{attr}' of class '{self.current_class.class_def['name']}' to '{value}'")
                return
        # Set variable in the current scope
        self.scope_stack[-1][name] = value
        if DEBUG:
            print(f"set_variable: Variable '{name}' set in current scope to '{value}'")

    def push_scope(self):
        self.scope_stack.append({})
        if DEBUG:
            print("push_scope: New scope pushed.")

    def pop_scope(self):
        self.scope_stack.pop()
        if DEBUG:
            print("pop_scope: Scope popped.")

    def apply_operator(self, a, b, op):
        if DEBUG:
            print(f"apply_operator: Applying operator '{op}' on operands '{a}' and '{b}'")
        if op == '+':
            return a + b
        if op == '-':
            return a - b
        if op == '*':
            return a * b
        if op == '/':
            if b == 0:
                raise InterpreterError("Division by zero.")
            return a // b  # Integer division
        raise InterpreterError(f"Operator {op} not implemented.")

    def evaluate_condition(self, condition_tokens):
        """
        Evaluate a condition expression.
        Supports simple comparisons: ==, !=, <, >, <=, >=
        """
        if DEBUG:
            print(f"evaluate_condition: Evaluating condition tokens: {condition_tokens}")

        # Find the operator in the condition tokens
        operators = ['==', '!=', '<=', '>=', '<', '>']
        operator = None
        op_token = None
        for tok in condition_tokens:
            if tok.type == 'OPERATOR' and tok.value in operators:
                operator = tok.value
                op_token = tok
                break

        if not operator:
            raise InterpreterError("Invalid condition: No comparison operator found.")

        # Split the tokens into left and right expressions
        op_index = condition_tokens.index(op_token)
        left_tokens = condition_tokens[:op_index]
        right_tokens = condition_tokens[op_index + 1:]

        # Evaluate left and right expressions
        left_value = self.evaluate_arithmetic(left_tokens)
        right_value = self.evaluate_arithmetic(right_tokens)

        # Perform the comparison
        if operator == '==':
            result = left_value == right_value
        elif operator == '!=':
            result = left_value != right_value
        elif operator == '<':
            result = left_value < right_value
        elif operator == '>':
            result = left_value > right_value
        elif operator == '<=':
            result = left_value <= right_value
        elif operator == '>=':
            result = left_value >= right_value
        else:
            raise InterpreterError(f"Unsupported operator '{operator}' in condition.")

        if DEBUG:
            print(f"evaluate_condition: Condition evaluated to {result}")
        return result

    def evaluate_arithmetic(self, expr_tokens):
        if DEBUG:
            print("evaluate_arithmetic: Evaluating arithmetic expression.")
        # Shunting Yard algorithm for operator precedence
        def precedence(op):
            return {'+':1, '-':1, '*':2, '/':2}.get(op, 0)
        
        output = []
        operators = []
        
        for token in expr_tokens:
            if DEBUG:
                print(f"evaluate_arithmetic: Processing token {token}")
            if token.type == 'NUMBER':
                output.append(int(token.value))
                if DEBUG:
                    print(f"evaluate_arithmetic: Pushed number {int(token.value)} to output")
            elif token.type == 'WORD':
                var_value = self.get_variable(token.value)
                if isinstance(var_value, int):
                    output.append(var_value)
                    if DEBUG:
                        print(f"evaluate_arithmetic: Pushed variable '{token.value}' with value {var_value} to output")
                else:
                    raise InterpreterError(f"Variable '{token.value}' is not a number.")
            elif token.type == 'ARITHMETIC':
                if DEBUG:
                    print(f"evaluate_arithmetic: Encountered operator '{token.value}'")
                while (operators and precedence(operators[-1]) >= precedence(token.value)):
                    op = operators.pop()
                    if DEBUG:
                        print(f"evaluate_arithmetic: Popped operator '{op}' from operators stack")
                    if len(output) < 2:
                        raise InterpreterError("Invalid arithmetic expression.")
                    b = output.pop()
                    a = output.pop()
                    result = self.apply_operator(a, b, op)
                    output.append(result)
                    if DEBUG:
                        print(f"evaluate_arithmetic: Result of {a} {op} {b} is {result}")
                operators.append(token.value)
                if DEBUG:
                    print(f"evaluate_arithmetic: Pushed operator '{token.value}' to operators stack")
            else:
                raise InterpreterError(f"Invalid token in arithmetic expression: {token}")
        
        while operators:
            op = operators.pop()
            if DEBUG:
                print(f"evaluate_arithmetic: Popped operator '{op}' from operators stack for final evaluation")
            if len(output) < 2:
                raise InterpreterError("Invalid arithmetic expression.")
            b = output.pop()
            a = output.pop()
            result = self.apply_operator(a, b, op)
            output.append(result)
            if DEBUG:
                print(f"evaluate_arithmetic: Final result of {a} {op} {b} is {result}")
        
        if len(output) != 1:
            raise InterpreterError("Invalid arithmetic expression.")
        final_result = output[0]
        if DEBUG:
            print(f"evaluate_arithmetic: Final evaluated result is {final_result}")
        return final_result

    def handle_pad(self, tokens, i):
        if DEBUG:
            print(f"handle_pad: Handling 'pad' statement starting at index {i}")
        # pad("Hello World");
        if tokens[i + 1].type != 'LPAREN':
            raise InterpreterError(f"Expected '(' after 'pad' at line {tokens[i].line}")
        j = i + 2
        pad_content = []
        while j < len(tokens) and tokens[j].type != 'RPAREN':
            pad_content.append(tokens[j])
            j += 1
        if j >= len(tokens):
            raise InterpreterError("Expected ')' to close 'pad' statement.")

        # Evaluate pad_content
        output = ''
        for token in pad_content:
            if token.type == 'STRING':
                output += token.value.strip('"')
                if DEBUG:
                    print(f"handle_pad: Appended string '{token.value.strip('\"')}' to output")
            elif token.type == 'WORD':
                value = self.get_variable(token.value)
                output += str(value)
                if DEBUG:
                    print(f"handle_pad: Appended variable '{token.value}' with value '{value}' to output")
            elif token.type == 'COMMA':
                output += ' '  # Changed from ', ' to ' '
                if DEBUG:
                    print("handle_pad: Appended space to output")
            else:
                output += token.value
                if DEBUG:
                    print(f"handle_pad: Appended token '{token.value}' to output")
        if DEBUG:
            print(f"handle_pad: Final output to print: '{output}'")
        print(output)  # Actual pad output

        # Expect SEMICOLON
        if j + 1 >= len(tokens) or tokens[j + 1].type != 'SEMICOLON':
            raise InterpreterError(f"Expected ';' after 'pad' statement at line {tokens[j].line}")
        if DEBUG:
            print(f"handle_pad: 'pad' statement successfully handled up to index {j + 1}")
        return j + 2

    def handle_function_call(self, tokens, i):
        if DEBUG:
            print(f"handle_function_call: Handling function call starting at index {i}")
        func_name_token = tokens[i]
        func_name_match = re.match(r'(\w+)<([^<>]+)>', func_name_token.value)
        if not func_name_match:
            raise InterpreterError(f"Invalid function call syntax at line {func_name_token.line}")
        func_name = func_name_match.group(1)
        if DEBUG:
            print(f"handle_function_call: Function name extracted: '{func_name}'")
        
        # Check if it's a class constructor
        if func_name in self.classes:
            # Handle as class constructor
            class_def = self.classes[func_name]
            # Expect LPAREN
            if tokens[i + 1].type != 'LPAREN':
                raise InterpreterError(f"Expected '(' after constructor name at line {tokens[i + 1].line}")
            # Parse constructor arguments
            args = []
            j = i + 2
            while j < len(tokens) and tokens[j].type != 'RPAREN':
                if tokens[j].type == 'COMMA':
                    j += 1
                    continue
                arg_token = tokens[j]
                if arg_token.type == 'NUMBER':
                    args.append(int(arg_token.value))
                    if DEBUG:
                        print(f"handle_function_call: Parsed constructor argument {int(arg_token.value)}")
                elif arg_token.type == 'WORD':
                    arg_value = self.get_variable(arg_token.value)
                    args.append(arg_value)
                    if DEBUG:
                        print(f"handle_function_call: Parsed constructor argument '{arg_token.value}' with value '{arg_value}'")
                else:
                    raise InterpreterError(f"Invalid constructor argument type at line {arg_token.line}")
                j += 1
            if j >= len(tokens):
                raise InterpreterError(f"Expected ')' to close constructor arguments for class '{func_name}' at line {func_name_token.line}")
            # Create class instance
            class_def = self.classes[func_name]
            instance = ClassInstance(class_def, args, self)
            if DEBUG:
                print(f"handle_function_call: Created instance of class '{func_name}' with arguments {args}")
            # Expect RPAREN
            j += 1
            return instance, j
        elif func_name in self.functions:
            # Handle as regular function
            func_def = self.functions[func_name]
            # Expect LPAREN
            if tokens[i + 1].type != 'LPAREN':
                raise InterpreterError(f"Expected '(' after function name at line {tokens[i + 1].line}")
            
            # Parse arguments
            args = []
            j = i + 2
            while j < len(tokens) and tokens[j].type != 'RPAREN':
                if tokens[j].type == 'COMMA':
                    j += 1
                    continue
                arg_token = tokens[j]
                if arg_token.type == 'NUMBER':
                    args.append(int(arg_token.value))
                    if DEBUG:
                        print(f"handle_function_call: Parsed argument {int(arg_token.value)}")
                elif arg_token.type == 'WORD':
                    arg_value = self.get_variable(arg_token.value)
                    args.append(arg_value)
                    if DEBUG:
                        print(f"handle_function_call: Parsed argument '{arg_token.value}' with value '{arg_value}'")
                else:
                    raise InterpreterError(f"Invalid argument type at line {arg_token.line}")
                j += 1
            
            if j >= len(tokens):
                raise InterpreterError(f"Expected ')' to close function arguments at line {func_name_token.line}")
            
            # Check argument count
            if len(args) != len(func_def['params']):
                raise InterpreterError(f"Function '{func_name}' expects {len(func_def['params'])} arguments, got {len(args)}.")
            
            if DEBUG:
                print(f"handle_function_call: Function '{func_name}' called with arguments {args}")
            
            # Create a new scope
            self.push_scope()
            if DEBUG:
                print(f"handle_function_call: New scope created for function '{func_name}'")
            
            # Assign arguments to parameters
            for param, arg in zip(func_def['params'], args):
                self.set_variable(param['name'], arg)
                if DEBUG:
                    print(f"handle_function_call: Parameter '{param['name']}' set to '{arg}'")
            
            # Execute function body
            try:
                self.execute_tokens(func_def['body'])
            except ReturnException as ret:
                return_value = ret.value
                if DEBUG:
                    print(f"handle_function_call: Function '{func_name}' returned with value '{ret.value}'")
            else:
                return_value = None  # If no return statement is encountered
                if DEBUG:
                    print(f"handle_function_call: Function '{func_name}' executed without return statement")
            finally:
                self.pop_scope()
                if DEBUG:
                    print(f"handle_function_call: Scope popped after function '{func_name}' execution")
            
            # Return the current index 'j + 1' to let 'handle_variable_declaration' handle the semicolon
            return return_value, j + 1
        else:
            raise InterpreterError(f"Function or class '{func_name}' is not defined.")

    def handle_adopt(self, tokens, i):
        if DEBUG:
            print(f"handle_adopt: Handling 'ADOPT' statement starting at index {i}")
        # ADOPT expression;
        if tokens[i].type != 'ADOPT':
            raise InterpreterError(f"Expected 'ADOPT' at line {tokens[i].line}")
        j = i + 1
        expr_tokens = []
        while j < len(tokens) and tokens[j].type != 'SEMICOLON':
            expr_tokens.append(tokens[j])
            j += 1
        if j >= len(tokens):
            raise InterpreterError(f"Expected ';' after 'ADOPT' statement at line {tokens[j-1].line}")
        
        if DEBUG:
            print(f"handle_adopt: Collected expression tokens for 'ADOPT': {expr_tokens}")
        
        if not expr_tokens:
            raise InterpreterError(f"Expected return value after 'ADOPT' at line {tokens[i].line}")
        
        # Check if it's a function call
        if expr_tokens[0].type in ('WORD', 'WORD_WITH_TYPE') and len(expr_tokens) > 1 and expr_tokens[1].type == 'LPAREN':
            if DEBUG:
                print("handle_adopt: Detected function call within 'ADOPT'")
            return_value, _ = self.handle_function_call(expr_tokens, 0)
        else:
            # Treat as arithmetic expression
            if DEBUG:
                print("handle_adopt: Treating 'ADOPT' expression as arithmetic")
            return_value = self.evaluate_arithmetic(expr_tokens)
            if DEBUG:
                print(f"handle_adopt: 'ADOPT' expression evaluated to {return_value}")
        
        if DEBUG:
            print(f"handle_adopt: Raising ReturnException with value {return_value}")
        raise ReturnException(return_value)

    def handle_variable_declaration(self, tokens, i, is_constant):
        var_type_str = 'constant' if is_constant else 'variable'
        if DEBUG:
            print(f"handle_variable_declaration: Handling {var_type_str} declaration starting at index {i}")
        # uh sum<WHOLE_NUMBER> = add<WHOLE_NUMBER>(10, 20);
        var_decl_token = tokens[i]
        var_name_token = tokens[i + 1]
        
        # Check for class attribute
        if var_name_token.type == 'WORD_WITH_TYPE' and var_name_token.value.startswith('confusing_word<this>'):
            remaining = var_name_token.value.replace('confusing_word<this>', '')
            var_match = re.match(r'(\w+)<([^<>]+)>', remaining)
            if not var_match:
                raise InterpreterError(f"Invalid class attribute declaration at line {var_name_token.line}")
            var_name = var_match.group(1)
            var_type = var_match.group(2)
            is_class_attribute = True
            if DEBUG:
                print(f"handle_variable_declaration: Detected class attribute '{var_name}' of type '{var_type}'")
        else:
            var_match = re.match(r'(\w+)<([^<>]+)>', var_name_token.value)
            if not var_match:
                raise InterpreterError(f"Invalid variable declaration syntax at line {var_name_token.line}")
            var_name = var_match.group(1)
            var_type = var_match.group(2)
            is_class_attribute = False
            if DEBUG:
                print(f"handle_variable_declaration: Detected {var_type_str} '{var_name}' of type '{var_type}'")
        
        # Check if var_type is a predefined type or a class
        predefined_types = {
            'WHOLE_NUMBER',
            'NOT_NICE_NUMBER',
            'MORE_WORDS',
            'WORD',
            'CHAR',
            'MORE_WHOLE_NUMBER',
            'YES_OR_NO',
            'STR<WORD>',
            'STR<CHAR>',
            'WORD<WORD>',
            'WORD<CHAR>',
            'WORDS'  # Added 'WORDS' type
        }

        
        if var_type.upper() in predefined_types:
            var_type_upper = var_type.upper()
            if DEBUG:
                print(f"handle_variable_declaration: Identified predefined type '{var_type_upper}'")
        elif var_type in self.classes:
            var_type_upper = var_type  # Keep as class name
            if DEBUG:
                print(f"handle_variable_declaration: Identified class type '{var_type_upper}'")
        else:
            raise InterpreterError(f"Variable type '{var_type}' not implemented.")
        
        # Expect ASSIGN
        if tokens[i + 2].type != 'ASSIGN':
            raise InterpreterError(f"Expected '=' after variable declaration at line {tokens[i + 2].line}")
        
        # Parse the value based on type
        var_value = None
        j = i + 3
        
        if DEBUG:
            print(f"handle_variable_declaration: Parsing value for '{var_name}' based on type '{var_type}'")
        
        if var_type_upper in predefined_types:
            if var_type_upper == 'MORE_WHOLE_NUMBER':
                # Handle list initialization
                if j >= len(tokens):
                    raise InterpreterError(f"Unexpected end of tokens while parsing list for '{var_name}' at line {var_decl_token.line}")
                if tokens[j].type != 'LBRACKET':
                    raise InterpreterError(f"Expected '[' to start list at line {tokens[j].line}")
                if DEBUG:
                    print(f"handle_variable_declaration: Found '[' to start list for variable '{var_name}'")
                list_values = []
                j += 1
                while j < len(tokens) and tokens[j].type != 'RBRACKET':
                    token = tokens[j]
                    if token.type == 'NUMBER':
                        list_values.append(int(token.value))
                        if DEBUG:
                            print(f"handle_variable_declaration: Appended number {int(token.value)} to list")
                    elif token.type in ('STRING', 'CHAR'):
                        cleaned_value = token.value.strip('"').strip("'")
                        list_values.append(cleaned_value)
                        if DEBUG:
                            print(f'handle_variable_declaration: Appended string/char "{cleaned_value}" to list')
                    elif token.type == 'WORD':
                        var_val = self.get_variable(token.value)
                        list_values.append(var_val)
                        if DEBUG:
                            print(f"handle_variable_declaration: Appended variable '{token.value}' with value '{var_val}' to list")
                    elif token.type == 'COMMA':
                        if DEBUG:
                            print("handle_variable_declaration: Encountered comma in list, skipping")
                        pass  # Skip commas
                    else:
                        raise InterpreterError(f"Invalid token '{token.value}' in list at line {token.line}")
                    j += 1
                if j >= len(tokens):
                    raise InterpreterError(f"Expected ']' to close list for '{var_name}' at line {var_decl_token.line}")
                var_value = list_values
                if DEBUG:
                    print(f"handle_variable_declaration: Parsed list value for '{var_name}': {var_value}")
                j += 1  # Move past 'RBRACKET'
            elif var_type_upper == 'WHOLE_NUMBER':
                # Check for function call
                if j < len(tokens) -1 and tokens[j].type in ('WORD', 'WORD_WITH_TYPE') and tokens[j + 1].type == 'LPAREN':
                    if DEBUG:
                        print("handle_variable_declaration: Detected function call for 'WHOLE_NUMBER' type")
                    return_value, j = self.handle_function_call(tokens, j)
                    var_value = return_value
                    if DEBUG:
                        print(f"handle_variable_declaration: Function call returned value {var_value}")
                else:
                    # Arithmetic expression
                    expr_tokens = []
                    while j < len(tokens) and tokens[j].type != 'SEMICOLON':
                        expr_tokens.append(tokens[j])
                        j += 1
                    if not expr_tokens:
                        raise InterpreterError(f"Expected value after '=' at line {tokens[j-1].line}")
                    var_value = self.evaluate_arithmetic(expr_tokens)
                    if DEBUG:
                        print(f"handle_variable_declaration: Evaluated arithmetic expression to {var_value}")
            elif var_type_upper == 'YES_OR_NO':
                if tokens[j].type == 'YES':
                    var_value = True
                    if DEBUG:
                        print(f"handle_variable_declaration: Set 'YES_OR_NO' variable '{var_name}' to True")
                elif tokens[j].type == 'NO':
                    var_value = False
                    if DEBUG:
                        print(f"handle_variable_declaration: Set 'YES_OR_NO' variable '{var_name}' to False")
                else:
                    raise InterpreterError(f"Invalid value for YES_OR_NO at line {tokens[j].line}")
                j += 1
            elif var_type_upper in ('STR<WORD>', 'STR<CHAR>', 'WORD<WORD>', 'WORD<CHAR>', 'WORDS'):
                # Handle string types including 'WORDS'
                if tokens[j].type == 'STRING':
                    var_value = tokens[j].value.strip('"').strip("'")
                    if DEBUG:
                        print(f"handle_variable_declaration: Set '{var_type_upper}' variable '{var_name}' to '{var_value}'")
                    j += 1
                else:
                    raise InterpreterError(f"Invalid value for '{var_type_upper}' at line {tokens[j].line}")
            elif var_type_upper == 'CHAR':  # **Added handling for CHAR**
                if tokens[j].type in ('STRING', 'CHAR'):
                    var_value = tokens[j].value.strip('"').strip("'")
                    if len(var_value) != 1:
                        raise InterpreterError(f"Invalid CHAR value '{var_value}' at line {tokens[j].line}")
                    if DEBUG:
                        print(f"handle_variable_declaration: Set 'CHAR' variable '{var_name}' to '{var_value}'")
                    j += 1
                else:
                    raise InterpreterError(f"Invalid value for 'CHAR' at line {tokens[j].line}")
            elif var_type_upper == 'MORE_WORDS':
                # Handle list initialization for MORE_WORDS
                if j >= len(tokens):
                    raise InterpreterError(f"Unexpected end of tokens while parsing list for '{var_name}' at line {var_decl_token.line}")
                if tokens[j].type != 'LBRACKET':
                    raise InterpreterError(f"Expected '[' to start list at line {tokens[j].line}")
                if DEBUG:
                    print(f"handle_variable_declaration: Found '[' to start list for variable '{var_name}'")
                list_values = []
                j += 1
                while j < len(tokens) and tokens[j].type != 'RBRACKET':
                    token = tokens[j]
                    if token.type == 'STRING':
                        cleaned_value = token.value.strip('"').strip("'")
                        list_values.append(cleaned_value)
                        if DEBUG:
                            print(f"handle_variable_declaration: Appended string '{cleaned_value}' to list")
                    elif token.type == 'CHAR':
                        cleaned_value = token.value.strip('"').strip("'")
                        list_values.append(cleaned_value)
                        if DEBUG:
                            print(f"handle_variable_declaration: Appended char '{cleaned_value}' to list")
                    elif token.type == 'WORD':
                        var_val = self.get_variable(token.value)
                        list_values.append(var_val)
                        if DEBUG:
                            print(f"handle_variable_declaration: Appended variable '{token.value}' with value '{var_val}' to list")
                    elif token.type == 'COMMA':
                        if DEBUG:
                            print("handle_variable_declaration: Encountered comma in list, skipping")
                        pass  # Skip commas
                    else:
                        raise InterpreterError(f"Invalid token '{token.value}' in list at line {token.line}")
                    j += 1
                if j >= len(tokens):
                    raise InterpreterError(f"Expected ']' to close list for '{var_name}' at line {var_decl_token.line}")
                var_value = list_values
                if DEBUG:
                    print(f"handle_variable_declaration: Parsed list value for '{var_name}': {var_value}")
                j += 1  # Move past 'RBRACKET'
   
            
        elif var_type_upper in self.classes:
            # Handle class instantiation
            if tokens[j].type != 'WORD' or tokens[j].value != var_type_upper:
                raise InterpreterError(f"Expected constructor call '{var_type_upper}' for class '{var_type_upper}' at line {tokens[j].line}")
            # Expect LPAREN
            if tokens[j + 1].type != 'LPAREN':
                raise InterpreterError(f"Expected '(' after constructor name at line {tokens[j + 1].line}")
            # Parse constructor arguments
            args = []
            k = j + 2
            while k < len(tokens) and tokens[k].type != 'RPAREN':
                if tokens[k].type == 'COMMA':
                    k += 1
                    continue
                arg_token = tokens[k]
                if arg_token.type == 'NUMBER':
                    args.append(int(arg_token.value))
                    if DEBUG:
                        print(f"handle_variable_declaration: Parsed constructor argument {int(arg_token.value)}")
                elif arg_token.type == 'WORD':
                    arg_value = self.get_variable(arg_token.value)
                    args.append(arg_value)
                    if DEBUG:
                        print(f"handle_variable_declaration: Parsed constructor argument '{arg_token.value}' with value '{arg_value}'")
                else:
                    raise InterpreterError(f"Invalid constructor argument type at line {arg_token.line}")
                k += 1
            if k >= len(tokens):
                raise InterpreterError(f"Expected ')' to close constructor arguments for class '{var_type_upper}' at line {var_decl_token.line}")
            # Create class instance
            class_def = self.classes[var_type_upper]
            instance = ClassInstance(class_def, args, self)
            if DEBUG:
                print(f"handle_variable_declaration: Created instance of class '{var_type_upper}' with arguments {args}")
            var_value = instance
            j = k + 1  # Move past ')'
        
        else:
            raise InterpreterError(f"Variable type '{var_type}' not implemented.")
        
        # Assign the variable
        if is_class_attribute:
            if self.current_class:
                self.current_class.attributes[var_name] = var_value
                if DEBUG:
                    print(f"handle_variable_declaration: Assigned '{var_name}' to class '{self.current_class.class_def['name']}' with value '{var_value}'")
            else:
                raise InterpreterError(f"Class attribute '{var_name}' declared outside of class context.")
        
        else:
            self.set_variable(var_name, var_value)
            if DEBUG:
                print(f"handle_variable_declaration: Assigned variable '{var_name}' with value '{var_value}'")
        
        # Expect SEMICOLON
        if j >= len(tokens) or tokens[j].type != 'SEMICOLON':
            raise InterpreterError(f"Expected ';' after variable declaration at line {tokens[j].line}")
        
        if DEBUG:
            print(f"handle_variable_declaration: Variable declaration handled up to index {j}")
        return j + 1  # Move past 'SEMICOLON'

    def parse_function_definition(self, tokens, i):
        if DEBUG:
            print(f"parse_function_definition: Parsing function definition starting at index {i}")
        # MAKE_BIG_BLOCK functionName<RETURN_TYPE>(params) { body }
        if tokens[i].type != 'MAKE_BIG_BLOCK':
            raise InterpreterError(f"Expected 'MAKE_BIG_BLOCK' at line {tokens[i].line}")
        
        func_name_token = tokens[i + 1]
        func_match = re.match(r'(\w+)<([^<>]+)>', func_name_token.value)
        if not func_match:
            raise InterpreterError(f"Invalid function name syntax at line {func_name_token.line}")
        func_name = func_match.group(1)
        return_type = func_match.group(2)
        if DEBUG:
            print(f"parse_function_definition: Function name '{func_name}' with return type '{return_type}'")
        
        # Expect LPAREN
        if tokens[i + 2].type != 'LPAREN':
            raise InterpreterError(f"Expected '(' after function name at line {tokens[i + 2].line}")
        
        # Parse parameters
        params = []
        j = i + 3
        while j < len(tokens) and tokens[j].type != 'RPAREN':
            if tokens[j].type == 'COMMA':
                j += 1
                continue
            param_token = tokens[j]
            param_match = re.match(r'(\w+)<([^<>]+)>', param_token.value)
            if not param_match:
                raise InterpreterError(f"Invalid parameter syntax at line {param_token.line}")
            param_name = param_match.group(1)
            param_type = param_match.group(2)
            params.append({'name': param_name, 'type': param_type})
            if DEBUG:
                print(f"parse_function_definition: Parsed parameter '{param_name}' of type '{param_type}'")
            j += 1
        
        if j >= len(tokens):
            raise InterpreterError("Expected ')' to close function parameters.")
        
        # Expect LBRACE
        if tokens[j + 1].type != 'LBRACE':
            raise InterpreterError(f"Expected '{{' to start function body at line {tokens[j + 1].line}")
        
        # Extract function body
        body_tokens = []
        brace_count = 1
        k = j + 2
        while k < len(tokens) and brace_count > 0:
            if tokens[k].type == 'LBRACE':
                brace_count += 1
                if DEBUG:
                    print(f"parse_function_definition: Encountered '{{' at index {k}, brace_count={brace_count}")
            elif tokens[k].type == 'RBRACE':
                brace_count -= 1
                if DEBUG:
                    print(f"parse_function_definition: Encountered '}}' at index {k}, brace_count={brace_count}")
            if brace_count > 0:
                body_tokens.append(tokens[k])
                if DEBUG:
                    print(f"parse_function_definition: Added token {tokens[k]} to function body")
            k += 1
        
        # Register the function
        self.functions[func_name] = {
            'return_type': return_type,
            'params': params,
            'body': body_tokens
        }
        if DEBUG:
            print(f"parse_function_definition: Registered function '{func_name}' with parameters {params}")
        
        return k

    def parse_class_definition(self, tokens, i):
        if DEBUG:
            print(f"parse_class_definition: Parsing class definition starting at index {i}")
        # MAKE_BIGGER_BLOCK ClassName(params) { body }
        if tokens[i].type != 'MAKE_BIGGER_BLOCK':
            raise InterpreterError(f"Expected 'MAKE_BIGGER_BLOCK' at line {tokens[i].line}")
        
        class_name_token = tokens[i + 1]
        class_match = re.match(r'(\w+)', class_name_token.value)
        if not class_match:
            raise InterpreterError(f"Invalid class name syntax at line {class_name_token.line}")
        class_name = class_match.group(1)
        if DEBUG:
            print(f"parse_class_definition: Class name '{class_name}'")
        
        # Expect LPAREN
        if tokens[i + 2].type != 'LPAREN':
            raise InterpreterError(f"Expected '(' after class name at line {tokens[i + 2].line}")
        
        # Parse constructor parameters
        params = []
        j = i + 3
        while j < len(tokens) and tokens[j].type != 'RPAREN':
            if tokens[j].type == 'COMMA':
                j += 1
                continue
            param_token = tokens[j]
            param_match = re.match(r'(\w+)<([^<>]+)>', param_token.value)
            if not param_match:
                raise InterpreterError(f"Invalid parameter syntax at line {param_token.line}")
            param_name = param_match.group(1)
            param_type = param_match.group(2)
            params.append({'name': param_name, 'type': param_type})
            if DEBUG:
                print(f"parse_class_definition: Parsed constructor parameter '{param_name}' of type '{param_type}'")
            j += 1
        
        if j >= len(tokens):
            raise InterpreterError("Expected ')' to close class constructor parameters.")
        
        # Expect LBRACE
        if tokens[j + 1].type != 'LBRACE':
            raise InterpreterError(f"Expected '{{' to start class body at line {tokens[j + 1].line}")
        
        # Extract class body
        body_tokens = []
        brace_count = 1
        k = j + 2
        while k < len(tokens) and brace_count > 0:
            if tokens[k].type == 'LBRACE':
                brace_count += 1
                if DEBUG:
                    print(f"parse_class_definition: Encountered '{{' at index {k}, brace_count={brace_count}")
            elif tokens[k].type == 'RBRACE':
                brace_count -= 1
                if DEBUG:
                    print(f"parse_class_definition: Encountered '}}' at index {k}, brace_count={brace_count}")
            if brace_count > 0:
                body_tokens.append(tokens[k])
                if DEBUG:
                    print(f"parse_class_definition: Added token {tokens[k]} to class body")
            k += 1
        
        # Register the class
        self.classes[class_name] = {
            'name': class_name,
            'params': params,
            'body': body_tokens
        }
        if DEBUG:
            print(f"parse_class_definition: Registered class '{class_name}' with constructor parameters {params}")
        
        return k

    def parse_import_statement(self, tokens, i):
        if DEBUG:
            print(f"parse_import_statement: Parsing import statement starting at index {i}")
        # TAKE_THIS "module_name";
        if tokens[i].type != 'TAKE_THIS':
            raise InterpreterError(f"Expected 'TAKE_THIS' at line {tokens[i].line}")
        module_token = tokens[i + 1]
        if module_token.type not in ('STRING', 'WORD'):
            raise InterpreterError(f"Expected module name after 'TAKE_THIS' at line {module_token.line}")
        module_name = module_token.value.strip('"').strip("'")
        if DEBUG:
            print(f"parse_import_statement: Module to import '{module_name}'")
        
        # Expect SEMICOLON
        if tokens[i + 2].type != 'SEMICOLON':
            raise InterpreterError(f"Expected ';' after import statement at line {tokens[i + 2].line}")
        
        self.import_module(module_name)
        if DEBUG:
            print(f"parse_import_statement: Imported module '{module_name}' successfully")
        return i + 3

    def import_module(self, module_name):
        if DEBUG:
            print(f"import_module: Importing module '{module_name}'")
        if module_name in self.imported_modules:
            if DEBUG:
                print(f"import_module: Module '{module_name}' is already imported")
            return  # Already imported
        try:
            with open(f"{module_name}.ps", 'r') as file:
                code = file.read()
            tokens = tokenize(code)
            if DEBUG:
                print(f"import_module: Tokenized code of module '{module_name}'")
            self.execute_tokens(tokens)
            self.imported_modules[module_name] = True
            if DEBUG:
                print(f"import_module: Module '{module_name}' imported and executed successfully")
        except FileNotFoundError:
            raise InterpreterError(f"Module '{module_name}' not found.")

    def handle_pad_statement(self, tokens, i):
        if DEBUG:
            print(f"handle_pad_statement: Handling 'PAD' statement at index {i}")
        return self.handle_pad(tokens, i)

    def handle_function_call_statement(self, tokens, i):
        if DEBUG:
            print(f"handle_function_call_statement: Handling function call statement at index {i}")
        return self.handle_function_call(tokens, i)

    def handle_adopt_statement(self, tokens, i):
        if DEBUG:
            print(f"handle_adopt_statement: Handling 'ADOPT' statement at index {i}")
        return self.handle_adopt(tokens, i)

    def handle_variable_declaration_statement(self, tokens, i, is_constant):
        var_type_str = 'constant' if is_constant else 'variable'
        if DEBUG:
            print(f"handle_variable_declaration_statement: Handling {var_type_str} declaration at index {i}")
        return self.handle_variable_declaration(tokens, i, is_constant)

    def handle_function_definition(self, tokens, i):
        if DEBUG:
            print(f"handle_function_definition: Handling function definition at index {i}")
        return self.parse_function_definition(tokens, i)

    def handle_class_definition(self, tokens, i):
        if DEBUG:
            print(f"handle_class_definition: Handling class definition at index {i}")
        return self.parse_class_definition(tokens, i)

    def handle_import_statement(self, tokens, i):
        if DEBUG:
            print(f"handle_import_statement: Handling import statement at index {i}")
        return self.parse_import_statement(tokens, i)

    def execute_tokens(self, tokens):
        if DEBUG:
            print("execute_tokens: Starting execution of tokens.")
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if DEBUG:
                print(f"execute_tokens: Current token at index {i}: {token}")
            if token.type == 'EOF':
                if DEBUG:
                    print("execute_tokens: Reached EOF.")
                break  # Stop processing further tokens
            elif token.type in ('UH', 'HUH'):
                is_constant = (token.type == 'HUH')
                if DEBUG:
                    print(f"execute_tokens: Detected {'constant' if is_constant else 'variable'} declaration.")
                i = self.handle_variable_declaration_statement(tokens, i, is_constant)
            elif token.type == 'MAKE_BIG_BLOCK':
                if DEBUG:
                    print("execute_tokens: Detected function definition.")
                i = self.handle_function_definition(tokens, i)
            elif token.type == 'MAKE_BIGGER_BLOCK':
                if DEBUG:
                    print("execute_tokens: Detected class definition.")
                i = self.handle_class_definition(tokens, i)
            elif token.type == 'TAKE_THIS':
                if DEBUG:
                    print("execute_tokens: Detected import statement.")
                i = self.handle_import_statement(tokens, i)
            elif token.type == 'PAD':
                if DEBUG:
                    print("execute_tokens: Detected 'PAD' statement.")
                i = self.handle_pad_statement(tokens, i)
            elif token.type == 'ADOPT':
                if DEBUG:
                    print("execute_tokens: Detected 'ADOPT' statement.")
                i = self.handle_adopt_statement(tokens, i)
            elif token.type == 'DO':
                if DEBUG:
                    print("execute_tokens: Detected 'DO' statement.")
                i = self.handle_do(tokens, i)
            elif token.type == 'DO_NOT_WORK':
                if DEBUG:
                    print("execute_tokens: Detected 'do_not_work' statement.")
                i, executed = self.handle_do_not_work(tokens, i)
                # No need to do anything else; the flag is already set
            elif token.type == 'DO_NOT_WORK_NOT_WORK':
                if DEBUG:
                    print("execute_tokens: Detected 'do_not_work_not_work' statement.")
                if not self.block_executed:
                    i = self.handle_do_not_work_not_work(tokens, i)
                else:
                    # Skip the else block
                    # Find the closing brace to skip the block
                    j = i + 1
                    if tokens[j].type != 'LBRACE':
                        raise InterpreterError(f"Expected '{{' to start 'do_not_work_not_work' block at line {tokens[j].line}")
                    brace_count = 1
                    k = j + 1
                    while k < len(tokens) and brace_count > 0:
                        if tokens[k].type == 'LBRACE':
                            brace_count += 1
                        elif tokens[k].type == 'RBRACE':
                            brace_count -= 1
                        k += 1
                    i = k  # Move past the 'else' block
            
            elif token.type == 'LOOP_COOL':
                if DEBUG:
                    print("execute_tokens: Detected 'LOOP_COOL' statement.")
                i = self.handle_loop_cool(tokens, i)
            
            elif token.type == 'WORD':
                # Potential function call
                if i + 1 < len(tokens) and tokens[i + 1].type == 'LPAREN':
                    if DEBUG:
                        print(f"execute_tokens: Detected function call for word '{token.value}'")
                    return_value, new_i = self.handle_function_call_statement(tokens, i)
                    # Optionally handle return_value if needed
                    i = new_i
                else:
                    raise InterpreterError(f"Unexpected token '{token.value}' at line {token.line}")
            else:
                raise InterpreterError(f"Unknown or unsupported token: {token.type} at line {token.line}")
        if DEBUG:
            print("execute_tokens: Finished execution of tokens.")
        # Reset the flag after execution
        self.block_executed = False
        return

    def run(self, filename):
        if DEBUG:
            print(f"run: Running PadScript file '{filename}'")
        try:
            with open(filename, 'r') as file:
                code = file.read()
            if DEBUG:
                print(f"run: Read code from '{filename}'")
            tokens = tokenize(code)
            if DEBUG:
                print(f"run: Tokenized code from '{filename}'")
            self.execute_tokens(tokens)
            if DEBUG:
                print(f"run: Successfully executed '{filename}'")
        except InterpreterError as e:
            print(f"Error: {e}")  # Only essential error messages are printed
        except SyntaxError as se:
            print(f"Syntax Error: {se}")  # Only essential syntax errors are printed

# Instantiate the interpreter
interpreter = Interpreter()

# -----------------------------
# GUI for File Selection
# -----------------------------
def main():
    time = ""
    if DEBUG:
        print("main: Starting PadScript Interpreter GUI.")
    root = tk.Tk()
    root.withdraw()
    filename = filedialog.askopenfilename(
        title="Select PadScript File",
        filetypes=[("PadScript Files", "*.ps"), ("All Files", "*.*")]
    )
    if filename:
        if DEBUG:
            print(f"main: Selected file '{filename}'")
            
        time = datetime.now()
        interpreter.run(filename)
        time = datetime.now() - time
    else:
        if DEBUG:
            print("main: No file selected.")
            
    return time

# Entry point
if __name__ == "__main__":
    time = main()
    print(f"\nExecution time: {time.total_seconds()} seconds")
    input("Press any key to exit...")  # Keep the window open
