# PadScript Language Documentation

## Introduction

**PadScript** is a custom programming language designed with unique syntax and constructs to simplify control flow, mathematical operations, and object-oriented programming. It supports defining blocks of code, conditional logic, arithmetic operations, and more with a focus on readability and ease of use.

---

## Table of Contents

1. [Keywords](#keywords)
2. [Mathematical Operations](#mathematical-operations)
3. [Control Flow](#control-flow)
4. [Object-Oriented Features](#object-oriented-features)
5. [Functions](#functions)
6. [Data Types](#data-types)
7. [Examples](#examples)

---

## Keywords

- **GIVE_THIS**: Used to import or link files or utilities, similar to `import`.
- **MAKE_BIG_BLOCK**: Defines a function in PadScript. Equivalent to `function` or `def` in other languages.
- **MAKE_BIGGER_BLOCK**: Defines a class or larger block, grouping related functionality together.
- **ADOPT**: Used to return values from functions, similar to `return` in other languages.
- **PAD**: Equivalent to `print` or `console.log`, used to output values to the screen.
- **DO**: Equivalent to `if`.
- **DO_NOT_WORK**: Equivalent to `else if`.
- **DO_NOT_WORK_NOT_WORK**: Equivalent to `else`.
- **UH**: Declares a variable.
- **HUH**: Declares a constant variable.

---

## Mathematical Operations

### Supported Operators
- **add<WHOLE_NUMBER>(a, b)**: Adds two numbers `a` and `b`.
- **subtract<WHOLE_NUMBER>(a, b)**: Subtracts `b` from `a`.
- **multiply<WHOLE_NUMBER>(a, b)**: Multiplies `a` by `b`.
- **divide<WHOLE_NUMBER>(a, b)**: Divides `a` by `b`.

### Basic Example
```padscript
uh x<WHOLE_NUMBER> = 10;
uh y<WHOLE_NUMBER> = 20;
uh sum<WHOLE_NUMBER> = add<WHOLE_NUMBER>(x, y);
pad("Sum:", sum);
```
### Control Flow
```
uh score<WHOLE_NUMBER> = 85;

DO(score >= 90) {
    PAD("Grade: A");
}
DO_NOT_WORK(score >= 80) {
    PAD("Grade: B");
}
DO_NOT_WORK_NOT_WORK {
    PAD("Grade: C");
}
```
### Functions
```
MAKE_BIG_BLOCK add<WHOLE_NUMBER>(a<WHOLE_NUMBER>, b<WHOLE_NUMBER>) {
    ADOPT a + b;
}
```
### Object Oriented Features

```
MAKE_BIGGER_BLOCK Calculator(a<WHOLE_NUMBER>, b<WHOLE_NUMBER>) {
    uh current_sum<WHOLE_NUMBER> = add<WHOLE_NUMBER>(a, b);
    uh current_product<WHOLE_NUMBER> = multiply<WHOLE_NUMBER>(a, b);

    MAKE_BIG_BLOCK get_sum<WHOLE_NUMBER>() {
        ADOPT current_sum;
    }

    MAKE_BIG_BLOCK get_product<WHOLE_NUMBER>() {
        ADOPT current_product;
    }
}

uh calculator<Calculator> = Calculator(5, 10);
pad("Sum:", calculator.get_sum());
pad("Product:", calculator.get_product());
```
### Function call example
```
uh result<WHOLE_NUMBER> = add<WHOLE_NUMBER>(10, 5);
pad("Result of addition:", result);
```