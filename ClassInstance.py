from constants import DEBUG

class ClassInstance:
    def __init__(self, class_def, args, interpreter):
        self.class_def = class_def
        self.attributes = {}
        self.interpreter = interpreter  # Reference to the interpreter for executing constructor
        
        # Initialize attributes based on constructor parameters
        if 'params' not in class_def:
            raise InterpreterError(f"Class definition for '{class_def['name']}' is missing 'params'.")
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
                # Set current_class to self during execution
                interpreter.current_class = self
                interpreter.execute_tokens(class_def['body'])
            except ReturnException:
                pass  # Constructors typically do not return values
            finally:
                # Reset current_class after execution
                interpreter.current_class = None
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
