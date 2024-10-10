# PadScript Documentation

## Overview

PadScript is a simple and expressive programming language that allows defining functions, classes, loops, and conditionals. It supports basic operations, user input handling, and arithmetic expressions. This document will serve as the guide to understanding and using the language effectively.

---

## 1. Function Definitions

Functions in PadScript are defined using `MAKE_BIG_BLOCK`. The `ADOPT` keyword is used to return a value from the function. Parameters and return types are declared in angle brackets (`<WHOLE_NUMBER>`, `<WORDS>`, etc.).

Example:

`MAKE_BIG_BLOCK square<WHOLE_NUMBER>(n<WHOLE_NUMBER>) { ADOPT n * n; }`

This defines a function named `square` that takes a number `n` and returns the square of that number.

Similarly, another function can be defined:

`MAKE_BIG_BLOCK greet<WORDS>(name<WORDS>) { ADOPT "Hello, " + name + "!"; }`

The `greet` function takes a string `name` and returns a greeting message.

---

## 2. Class Definitions

Classes in PadScript are defined using `MAKE_BIGGER_BLOCK`. Classes hold attributes (but no methods) and are initialized using a constructor.

Example:

`MAKE_BIGGER_BLOCK Person(name<WORDS>, age<WHOLE_NUMBER>) { }`

This creates a `Person` class with two attributes: `name` and `age`. However, the class does not have any methods.

---

## 3. Variable Declarations

Variables are declared using the `uh` keyword, followed by the variable type and value. For example:

`uh people<MORE_WORDS> = [ Person("Alice", 30), Person("Bob", 17), Person("Charlie", 25), Person("Diana", 15) ];`

This creates a list of `Person` instances, where each person has a name and an age.

---

## 4. Looping Through Lists

PadScript uses the `LOOP_COOL` block for iterating through lists. The `grab` keyword extracts an element from the list during each iteration.

Example:

`LOOP_COOL : grab person from people { pad("Name: " + person.name + ", Age: " + person.age); }`

This loop goes through each `Person` in the `people` list and prints their name and age.

---

## 5. Conditional Statements

Conditional logic is handled using `do`, `do_not_work`, and `do_not_work_not_work` blocks.

Example:

`do(person.age >= 18) { pad(person.name + " is an adult."); } do_not_work(person.age < 18) { pad(person.name + " is a minor."); } do_not_work_not_work { pad("This block should not execute."); }`

This checks whether a person is an adult or a minor and prints the corresponding message. The final `do_not_work_not_work` block handles the case where neither condition is met (though it will not execute in this example).

---

## 6. User Input

User input is captured using the `pad_in` function.

Example:

`uh new_name<WORDS> = pad_in("Enter the name of the new person: "); uh new_age_input<WORDS> = pad_in("Enter the age of the new person: "); uh new_age<WHOLE_NUMBER> = new_age_input;`

This takes a person's name and age from the user and stores it in the `new_name` and `new_age` variables.

---

## 7. Constants and Arithmetic Expressions

Constants can be defined using the `huh` keyword, and basic arithmetic expressions can be performed using standard operators.

Example:

`huh PI<NOT_NICE_NUMBER> = 3.14159; uh circumference<NOT_NICE_NUMBER> = PI * 2; pad("Circumference: " + circumference);`

This calculates the circumference of a circle with a radius of 2 using the constant `PI`.

---

## 8. Built-in Functions

PadScript allows for defining and calling functions using the `MAKE_BIG_BLOCK` keyword. Functions can return values or perform operations.

Example:

`uh greeting<WORDS> = greet("Eve"); pad(greeting);`

This calls the `greet` function, passing "Eve" as the argument, and prints the result.

---

## 9. Looping Through Numbers and Function Use

PadScript allows looping through numbers and applying functions on them within the loop.

Example:

`uh numbers<MORE_WHOLE_NUMBER> = [1, 2, 3, 4, 5]; LOOP_COOL : grab num from numbers { uh squared<WHOLE_NUMBER> = square(num); pad("Square of " + num + " is " + squared); }`

This loops through the `numbers` list and prints the square of each number by calling the `square` function.

---

## 10. Nested Functions and Control Structures

Functions can call other functions inside them, allowing for nested logic.

Example:

`MAKE_BIG_BLOCK compute_and_greet<WORDS>(num<WHOLE_NUMBER>, name<WORDS>) { uh result<WHOLE_NUMBER> = square(num); uh message<WORDS> = greet(name) + " Your number squared is " + result; ADOPT message; }`

This function `compute_and_greet` computes the square of a number and greets a person by name, combining both results.

---

## 11. Final Demonstration Loop

To demonstrate the use of conditionals and loops again, here is a final example:

`LOOP_COOL : grab person from people { pad("Name: " + person.name + ", Age: " + person.age); do(person.age >= 18) { pad(person.name + " continues to be an adult."); } do_not_work(person.age < 18) { pad(person.name + " continues to be a minor."); } do_not_work_not_work { pad("This block should not execute."); } }`

This loop iterates over the `people` list again, prints updated information, and evaluates whether the person is an adult or minor.

---

## Conclusion

PadScript provides a structured and simple way to define functions, classes, loops, and conditionals. While it has basic syntax, it can be extended to handle various types of logic and data manipulation. The constructs for handling input/output, conditionals, and loops make it suitable for small to medium-scale scripting tasks.
