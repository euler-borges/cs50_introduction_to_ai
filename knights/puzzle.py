from logic import *

AKnight = Symbol("A is a Knight")
AKnave = Symbol("A is a Knave")

BKnight = Symbol("B is a Knight")
BKnave = Symbol("B is a Knave")

CKnight = Symbol("C is a Knight")
CKnave = Symbol("C is a Knave")

base_knowledge = And(
    # A is either a knight or a knave, but not both
    Or(AKnight, AKnave),
    Not(And(AKnight, AKnave)),

    # B is either a knight or a knave, but not both
    Or(BKnight, BKnave),
    Not(And(BKnight, BKnave)),

    # C is either a knight or a knave, but not both
    Or(CKnight, CKnave),
    Not(And(CKnight, CKnave))
)

# Puzzle 0
# A says "I am both a knight and a knave."

AKnight_statement_P0 = And(AKnight, AKnave)

knowledge0 = And(
    base_knowledge,
    Implication(AKnight, AKnight_statement_P0),  # A's statement(as a knight)
    Implication(AKnave, Not(AKnight_statement_P0))  # A's statement(as a knave)
)

# Puzzle 1
# A says "We are both knaves."
# B says nothing.

AKnight_statement_P1 = And(AKnave, BKnave)

knowledge1 = And(
    base_knowledge,
    Implication(AKnight, AKnight_statement_P1),  # A's statement(as a knight)
    Implication(AKnave, Not(AKnight_statement_P1))  # A's statement(as a knave)
)

# Puzzle 2
# A says "We are the same kind."
# B says "We are of different kinds."

AKnight_statement_P2 = Or(And(AKnight, BKnight), And(AKnave, BKnave))
BKnight_statement_P2 = Or(And(AKnight, BKnave), And(AKnave, BKnight))

knowledge2 = And(
    base_knowledge,
    Implication(AKnight, AKnight_statement_P2),  # A's statement(as a knight)
    Implication(AKnave, Not(AKnight_statement_P2)),  # A's statement(as a knave)
    Implication(BKnight, BKnight_statement_P2),  # B's statement(as a knight)
    Implication(BKnave, Not(BKnight_statement_P2))  # B's statement(as a knave)
)

# Puzzle 3
# A says either "I am a knight." or "I am a knave.", but you don't know which.
# B says "A said 'I am a knave'."
# B says "C is a knave."
# C says "A is a knight."

AKnight_statement1_P3 = Or(AKnight, AKnave)
BKnight_statement1_P3 = AKnave
BKnight_statement2_P3 = CKnave
CKnight_statement1_P3 = AKnight


knowledge3 = And(
    base_knowledge,
    Implication(AKnight, AKnight_statement1_P3),  # A's statement(as a knight)
    Implication(AKnave, Not(AKnight_statement1_P3)),  # A's statement(as a knave)
    # B's statement(as a knight)
    Implication(BKnight, And(BKnight_statement1_P3, BKnight_statement2_P3)),  
    # B's statement(as a knave)
    Implication(BKnave, Not(And(BKnight_statement1_P3, BKnight_statement2_P3))),  
    Implication(CKnight, CKnight_statement1_P3),  # C's statement(as a knight)
    Implication(CKnave, Not(CKnight_statement1_P3))  # C's statement(as a knave)
)


def main():
    symbols = [AKnight, AKnave, BKnight, BKnave, CKnight, CKnave]
    puzzles = [
        ("Puzzle 0", knowledge0),
        ("Puzzle 1", knowledge1),
        ("Puzzle 2", knowledge2),
        ("Puzzle 3", knowledge3)
    ]
    for puzzle, knowledge in puzzles:
        print(puzzle)
        if len(knowledge.conjuncts) == 0:
            print("    Not yet implemented.")
        else:
            for symbol in symbols:
                if model_check(knowledge, symbol):
                    print(f"    {symbol}")


if __name__ == "__main__":
    main()
