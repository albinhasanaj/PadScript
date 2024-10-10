import re
from constants import TOKEN_SPECIFICATIONS, DEBUG

# Recompile the regex
TOKEN_REGEX = '|'.join('(?P<%s>%s)' % pair for pair in TOKEN_SPECIFICATIONS)
TOKEN_RE = re.compile(TOKEN_REGEX)

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
        if kind == 'SKIP' or kind == 'COMMENT':
            if '\n' in value:
                line_num += value.count('\n')
        elif kind == 'UNKNOWN':
            raise SyntaxError(f'Unknown token {value!r} at line {line_num} column {column}')
        else:
            token = Token(kind, value, line_num, column)
            tokens.append(token)
            if DEBUG:
                print(f"Tokenized: {token} by pattern '{kind}'")  # Enhanced Debug Statement
        if '\n' in value:
            line_num += value.count('\n') 
            line_start = mo.end()

    tokens.append(Token('EOF', 'EOF', line_num, column)) 
    if DEBUG:
        print(f"Tokenizer: Completed tokenization with {len(tokens)} tokens.")
    return tokens
