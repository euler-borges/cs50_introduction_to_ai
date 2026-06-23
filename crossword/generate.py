import sys

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        _, _, w, h = draw.textbbox((0, 0), letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """

        # Enforce node consistency by removing any words from the domain of each variable
        for var in self.crossword.variables:
            self.domains[var] = {
                word for word in self.domains[var]
                if len(word) == var.length
            }

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        revised = False
        overlap = self.crossword.overlaps[x, y]

        if overlap is None:
            return revised

        i, j = overlap

        # Check if there is a word in x's domain that has no possible match in y's domain
        for x_word in self.domains[x].copy():
            if not any(
                x_word[i] == y_word[j]
                for y_word in self.domains[y]
            ):
                self.domains[x].remove(x_word)
                revised = True

        return revised
    
    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """

        # If arcs is None, initialize it with all arcs in the problem
        if arcs is None:
            arcs = []
            for var in self.crossword.variables:
                for neighbor in self.crossword.neighbors(var):
                    arcs.append((var, neighbor))

        # Enforce arc consistency by revising each arc and adding neighboring arcs back to the list if a revision was made
        while arcs:
            x, y = arcs.pop(0)
            if self.revise(x, y):
                if not self.domains[x]:
                    return False
                for neighbor in self.crossword.neighbors(x):
                    if neighbor != y:
                        arcs.append((neighbor, x))

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """

        # Check if assignment is complete by comparing the number of assigned variables to the total number of variables
        return len(assignment) == len(self.crossword.variables)
    
    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        for var1 in assignment:
            word1 = assignment[var1]

            # Check if the assigned word fits the variable's length
            if len(word1) != var1.length:
                return False

            for var2 in assignment:
                if var1 == var2:
                    continue

                word2 = assignment[var2]

                # Check if the assigned words are different
                if word1 == word2:
                    return False

                # Check if the assigned words have a consistent overlap
                overlap = self.crossword.overlaps[var1, var2]
                if overlap is not None:
                    i, j = overlap
                    if word1[i] != word2[j]:
                        return False

        return True
    
    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        # Use the Least Constraining Value (LCV) heuristic to order the values in the domain of var
        domain_scores = []

        for value in self.domains[var]:
            eliminated_count = 0
            
            # Count how many values in the domains of neighboring variables would be eliminated by assigning value to var
            for neighbor in self.crossword.neighbors(var):
                # Skip neighbors that are already assigned, as they won't be affected by this assignment
                if neighbor in assignment:
                    continue
                
                # Get the overlap between var and neighbor, if it exists
                overlap = self.crossword.overlaps[var, neighbor]
                if overlap is None:
                    continue
                    
                i, j = overlap
                
                # Count how many values in neighbor's domain would be eliminated by this assignment
                for neighbor_value in self.domains[neighbor]:
                    if value[i] != neighbor_value[j]:
                        eliminated_count += 1
            
            # Append the value and its corresponding eliminated count to the domain_scores list
            domain_scores.append((value, eliminated_count))

        # Sort the domain_scores list by the number of values eliminated, in ascending order
        domain_scores.sort(key=lambda x: x[1])

        # Retorna apenas os valores ordenados
        return [item[0] for item in domain_scores]

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        unassigned_vars = [var for var in self.crossword.variables if var not in assignment]

        # Select unassigned variable using Minimum Remaining Values (MRV) heuristic
        mrv_vars = sorted(unassigned_vars, key=lambda var: len(self.domains[var]))
        min_remaining = len(self.domains[mrv_vars[0]])
        mrv_vars = [var for var in mrv_vars if len(self.domains[var]) == min_remaining]

        # If there is a tie, select variable with the highest degree
        if len(mrv_vars) > 1:
            return max(mrv_vars, key=lambda var: len(self.crossword.neighbors(var)))

        return mrv_vars[0]

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """

        # If assignment is complete, return it
        if self.assignment_complete(assignment):
            return assignment

        # Select an unassigned variable
        var = self.select_unassigned_variable(assignment)

        for value in self.order_domain_values(var, assignment):
            # Create a new assignment with the selected variable assigned to the current value
            new_assignment = assignment.copy()
            new_assignment[var] = value

            # If the new assignment is consistent, recursively backtrack with the new assignment
            if self.consistent(new_assignment):
                result = self.backtrack(new_assignment)
                if result is not None:
                    return result

        # If no valid assignment was found, return None
        return None


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
