import os

from analysis.document import dump_analysis

if __name__ == "__main__":
    session_path = os.getenv("SESSION_PATH")
    assert session_path is not None
    dump_analysis(session_path)
