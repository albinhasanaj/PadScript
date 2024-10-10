DEBUG = False

TOKEN_SPECIFICATIONS = [
    ('COMMENT', r'\#.*'),                     # Comments
    ('WORD_WITH_TYPE', r'\w+(?:<[^<>]+>)+'),  # e.g., sum<WHOLE_NUMBER>
    ('LOOP_COOL', r'\bLOOP_COOL\b'),          # LOOP_COOL
    ('GRAB', r'\bgrab\b'),                    # grab
    ('FROM', r'\bfrom\b'),                    # from
    ('ADOPT', r'\bADOPT\b'),                  # ADOPT
    ('MAKE_BIG_BLOCK', r'\bMAKE_BIG_BLOCK\b'),# MAKE_BIG_BLOCK
    ('MAKE_BIGGER_BLOCK', r'\bMAKE_BIGGER_BLOCK\b'), # MAKE_BIGGER_BLOCK
    ('TAKE_THIS', r'\bTAKE_THIS\b'),          # TAKE_THIS
    ('CONFUSING_WORD', r'\bconfusing_word\b'),# confusing_word
    ('UH', r'\buh\b'),                        # uh
    ('HUH', r'\bhuh\b'),                      # huh
    ('DO', r'\bdo\b'),                        # do
    ('DO_NOT_WORK', r'\bdo_not_work\b'),      # do_not_work
    ('DO_NOT_WORK_NOT_WORK', r'\bdo_not_work_not_work\b'),# do_not_work_not_work
    ('YES', r'\bYES\b'),                      # YES
    ('NO', r'\bNO\b'),                        # NO
    ('WHOLE_NUMBER', r'<WHOLE_NUMBER>'),
    ('WORDS', r'<WORDS>'),                    # string
    ('NOT_NICE_NUMBER', r'<NOT_NICE_NUMBER>'),
    ('MORE_WORDS', r'<MORE_WORDS>'),
    ('WORD_TYPE', r'<WORD>'),
    ('CHAR_TYPE', r'<CHAR>'),
    ('MORE_WHOLE_NUMBER', r'<MORE_WHOLE_NUMBER>'),
    ('YES_OR_NO', r'<YES_OR_NO>'),
    ('OPERATOR', r'==|!=|<=|>=|<|>'),         # Combined all comparison operators
    ('ASSIGN', r'=(?![=<>!])'),               # Modified ASSIGN pattern
    ('SEMICOLON', r';'),
    ('COLON', r':'),
    ('COMMA', r','),
    ('LBRACKET', r'\['),
    ('RBRACKET', r'\]'),
    ('LPAREN', r'\('),
    ('RPAREN', r'\)'),
    ('LBRACE', r'\{'),
    ('RBRACE', r'\}'),
    ('DOT', r'\.'),                           # DOT Token
    ('STRING', r'\".*?\"'),
    ('CHAR', r"'.*?'"),
    ('NUMBER', r'\d+(\.\d+)?([eE][-+]?\d+)?'),
    ('WORD', r'[a-zA-Z_][a-zA-Z0-9_]*'),
    ('ARITHMETIC', r'\+|\-|\*|/'),
    ('SKIP', r'[ \t\n]+'),
    ('UNKNOWN', r'.'),                        # Any other character
]