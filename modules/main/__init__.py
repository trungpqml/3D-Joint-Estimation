import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.realpath(__file__)
        )
    )
)

if __name__ == "__main__":
    print("\n".join(sys.path))
