import sys

# Clear sys.path[0] if it contains this script.
# Be careful to use the most compatible Python code possible.
try:
    if sys.path[0]:
        if sys.argv[0].startswith(sys.path[0]):
            sys.path[0] = ""
        else:
            open(sys.path[0] + "/" + sys.argv[0], "rb").close()
            sys.path[0] = ""
except OSError:
    pass
except AttributeError:
    pass
except IndexError:
    pass

# Replace argv[0] with our executable instead of the script name.
try:
    if sys.argv[0][-14:].upper() == ".__SCRIPT__.PY":
        sys.argv[0] = sys.argv[0][:-14]
        sys.orig_argv[0] = sys.argv[0]
except AttributeError:
    pass
except IndexError:
    pass

from pip._internal.cli.main import main
sys.exit(main())
